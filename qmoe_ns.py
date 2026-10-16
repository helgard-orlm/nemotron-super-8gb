"""Routed-эксперты Nemotron-3.5-Super (NVFP4, relu², латент 1024 -> 2688 -> 1024) на шаге одного токена.
Слот = блок перепаковки (3 096 576 Б): up_w | up_s | dn_w | dn_s (см. repack.py).
NVFP4 (modelopt): байт = два E2M1, младший полубайт = чётный элемент; масштаб E4M3 на 16 элементов; × weight_scale_2.
Активации не квантуются (W4A16): x -> fp32, накопление fp32. Аппаратная распаковка E2M1 (как qmoe.py Qwen, sm_120)."""
import torch
import triton
import triton.language as tl

LAT, INTER = 1024, 2688
BLOCK = 3096576
O_UW, O_US, O_DW, O_DS = 0, 1376256, 1548288, 2924544
KB_UP, G_UP = LAT // 2, LAT // 16          # байт и групп в строке up (512, 64)
KB_DN, G_DN = INTER // 2, INTER // 16      # в строке down (1344, 168)
assert O_DS + LAT * G_DN == BLOCK


@triton.jit
def _e2m1_hw(c):
    r = tl.inline_asm_elementwise("{ .reg .b8 t, z; mov.b16 {t, z}, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
                                  "=r,h", [c.to(tl.uint16)], dtype=tl.uint32, is_pure=True, pack=1)
    return (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _up(slots_ptr, sidx_ptr, s2_ptr, x_ptr, h_ptr, BLOCKB: tl.constexpr, BN: tl.constexpr, GB: tl.constexpr):
    """h[j, n] = relu(up_n · x)² для эксперта в слоте sidx[j]"""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(sidx_ptr + j).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    rows = pid * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for g0 in range(0, 64, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        xe = tl.load(x_ptr + 2 * bi).to(tl.float32)
        xo = tl.load(x_ptr + 2 * bi + 1).to(tl.float32)
        w = tl.load(base + 0 + rows[:, None] * 512 + bi[None, :]).to(tl.int32)
        s = tl.load(base + 1376256 + rows[:, None] * 64 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        p = _e2m1_hw(w & 15) * xe[None, :] + _e2m1_hw(w >> 4) * xo[None, :]
        acc += tl.sum(tl.sum(tl.reshape(p, [BN, GB, 8]), axis=2) * s, axis=1)
    u = acc * tl.load(s2_ptr + j * 2 + 0)
    u = tl.maximum(u, 0.0)
    tl.store(h_ptr + j * 2688 + rows, u * u)


@triton.jit
def _down(slots_ptr, sidx_ptr, s2_ptr, coef_ptr, h_ptr, out_ptr,
          BLOCKB: tl.constexpr, BR: tl.constexpr, GB: tl.constexpr):
    """out[j, r] = coef_j * ws2d_j * Σ_n Wd[r, n] h[j, n]"""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    slot = tl.load(sidx_ptr + j).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    rows = pid * BR + tl.arange(0, BR)
    acc = tl.zeros([BR], dtype=tl.float32)
    for g0 in range(0, 168, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        he = tl.load(h_ptr + j * 2688 + 2 * bi)
        ho = tl.load(h_ptr + j * 2688 + 2 * bi + 1)
        w = tl.load(base + 1548288 + rows[:, None] * 1344 + bi[None, :]).to(tl.int32)
        s = tl.load(base + 2924544 + rows[:, None] * 168 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        p = _e2m1_hw(w & 15) * he[None, :] + _e2m1_hw(w >> 4) * ho[None, :]
        acc += tl.sum(tl.sum(tl.reshape(p, [BR, GB, 8]), axis=2) * s, axis=1)
    c = tl.load(coef_ptr + j) * tl.load(s2_ptr + j * 2 + 1)
    tl.store(out_ptr + j * 1024 + rows, acc * c)


def moe_decode(slots_u8, sidx, s2, coef, x, h_buf, out_buf):
    """slots_u8 [S, BLOCK] u8 · sidx [K] int32 · s2 [K,2] f32 (ws2 up/down) · coef [K] f32 (вес роутера) ·
    x [1024] -> fp32 [1024] (сумма по K). 168 групп down не делятся на 8 — GB=8 даёт 21 итерацию ровно."""
    K = sidx.shape[0]
    _up[(INTER // 32, K)](slots_u8, sidx, s2, x, h_buf, BLOCKB=BLOCK, BN=32, GB=8, num_warps=4)
    _down[(LAT // 32, K)](slots_u8, sidx, s2, coef, h_buf, out_buf, BLOCKB=BLOCK, BR=32, GB=8, num_warps=4)
    return out_buf[:K].sum(0)


@triton.jit
def _up_pairs(slots_ptr, pslot_ptr, ptok_ptr, s2_ptr, x_ptr, h_ptr, BLOCKB: tl.constexpr, BN: tl.constexpr,
              GB: tl.constexpr):
    """пара p = (слот эксперта, токен): h[p, n] = relu(up_n · x[tok])²"""
    pid = tl.program_id(0)
    p = tl.program_id(1)
    slot = tl.load(pslot_ptr + p).to(tl.int64)
    tok = tl.load(ptok_ptr + p).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    xb = x_ptr + tok * 1024
    rows = pid * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for g0 in range(0, 64, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        xe = tl.load(xb + 2 * bi).to(tl.float32)
        xo = tl.load(xb + 2 * bi + 1).to(tl.float32)
        w = tl.load(base + 0 + rows[:, None] * 512 + bi[None, :]).to(tl.int32)
        s = tl.load(base + 1376256 + rows[:, None] * 64 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        q = _e2m1_hw(w & 15) * xe[None, :] + _e2m1_hw(w >> 4) * xo[None, :]
        acc += tl.sum(tl.sum(tl.reshape(q, [BN, GB, 8]), axis=2) * s, axis=1)
    u = acc * tl.load(s2_ptr + p * 2 + 0)
    u = tl.maximum(u, 0.0)
    tl.store(h_ptr + p * 2688 + rows, u * u)


@triton.jit
def _down_pairs(slots_ptr, pslot_ptr, s2_ptr, coef_ptr, h_ptr, out_ptr,
                BLOCKB: tl.constexpr, BR: tl.constexpr, GB: tl.constexpr):
    pid = tl.program_id(0)
    p = tl.program_id(1)
    slot = tl.load(pslot_ptr + p).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    rows = pid * BR + tl.arange(0, BR)
    acc = tl.zeros([BR], dtype=tl.float32)
    for g0 in range(0, 168, GB):
        gi = g0 + tl.arange(0, GB)
        bi = g0 * 8 + tl.arange(0, GB * 8)
        he = tl.load(h_ptr + p * 2688 + 2 * bi)
        ho = tl.load(h_ptr + p * 2688 + 2 * bi + 1)
        w = tl.load(base + 1548288 + rows[:, None] * 1344 + bi[None, :]).to(tl.int32)
        s = tl.load(base + 2924544 + rows[:, None] * 168 + gi[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        q = _e2m1_hw(w & 15) * he[None, :] + _e2m1_hw(w >> 4) * ho[None, :]
        acc += tl.sum(tl.sum(tl.reshape(q, [BR, GB, 8]), axis=2) * s, axis=1)
    c = tl.load(coef_ptr + p) * tl.load(s2_ptr + p * 2 + 1)
    tl.store(out_ptr + p * 1024 + rows, acc * c)


def moe_pairs(slots_u8, pslot, ptok, s2p, coefp, X, h_buf, out_buf):
    """промт малыми кусками: P пар (слот, токен) -> out_buf[:P] fp32 [P,1024] (сложить по токенам — index_add)
    X [T,1024] contiguous; pslot/ptok int32 [P]; s2p [P,2] f32; coefp [P] f32"""
    P = pslot.shape[0]
    _up_pairs[(INTER // 32, P)](slots_u8, pslot, ptok, s2p, X, h_buf, BLOCKB=BLOCK, BN=32, GB=8, num_warps=4)
    _down_pairs[(LAT // 32, P)](slots_u8, pslot, s2p, coefp, h_buf, out_buf, BLOCKB=BLOCK, BR=32, GB=8, num_warps=4)
    return out_buf[:P]


# ================= пропуск нулей relu² (#399, пункт 1) =================
# Раскладка блока «T» (перестановка байтов down, размер тот же): up_w | up_s | dnT_w [2688 нейронов, 512] | dnT_s [168 групп, 1024]
#   dnT_w[n, b]: младший полубайт = строка 2b, старший = 2b+1 (столбец n исходного down);  dnT_s[g, r] = dn_s[r, g].
# На карту копируется только up (UPB байт); down ядро читает из RAM по шине (адрес по таблице) ТОЛЬКО живые нейроны.
UPB = O_DW                                  # 1 548 288: up_w + up_s
O_TW, O_TS = O_DW, O_DW + INTER * 512       # 1 548 288, 2 924 544


@triton.jit
def _down_live(tab_ptr, h_ptr, idx_ptr, cnt_ptr, s2_ptr, coef_ptr, out_ptr, BR: tl.constexpr, NB: tl.constexpr):
    """out[j, r] = coef_j·ws2d_j·Σ_{n живой} W[r, n]·S[r, n//16]·h[j, n]; блок эксперта j — по адресу tab[j] (RAM или карта)"""
    pid = tl.program_id(0)
    j = tl.program_id(1)
    base = tl.load(tab_ptr + j).to(tl.pointer_type(tl.uint8))
    cnt = tl.load(cnt_ptr + j)
    r2 = pid * (BR // 2) + tl.arange(0, BR // 2)               # байты строк [r0, r0+BR) в векторе нейрона
    rs = pid * BR + tl.arange(0, BR)
    acc_e = tl.zeros([BR // 2], dtype=tl.float32)               # чётные строки
    acc_o = tl.zeros([BR // 2], dtype=tl.float32)               # нечётные
    for k0 in range(0, cnt, NB):
        kk = k0 + tl.arange(0, NB)
        km = kk < cnt
        n = tl.load(idx_ptr + j * 2688 + kk, mask=km, other=0)
        hn = tl.load(h_ptr + j * 2688 + n, mask=km, other=0.0)
        w = tl.load(base + 1548288 + n[:, None].to(tl.int64) * 512 + r2[None, :], mask=km[:, None], other=0).to(tl.int32)
        g = (n // 16).to(tl.int64)
        se = tl.load(base + 2924544 + g[:, None] * 1024 + 2 * r2[None, :], mask=km[:, None], other=0)
        so = tl.load(base + 2924544 + g[:, None] * 1024 + 2 * r2[None, :] + 1, mask=km[:, None], other=0)
        se = se.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        so = so.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        acc_e += tl.sum(_e2m1_hw(w & 15) * se * hn[:, None], axis=0)
        acc_o += tl.sum(_e2m1_hw(w >> 4) * so * hn[:, None], axis=0)
    c = tl.load(coef_ptr + j) * tl.load(s2_ptr + j * 2 + 1)
    ob = out_ptr + j * 1024 + pid * BR
    tl.store(ob + 2 * tl.arange(0, BR // 2), acc_e * c)
    tl.store(ob + 2 * tl.arange(0, BR // 2) + 1, acc_o * c)


@triton.jit
def _down_live2(tab_ptr, h_ptr, idx_ptr, cnt_ptr, s2_ptr, coef_ptr, part_ptr, K, BR: tl.constexpr, NB: tl.constexpr,
                KS: tl.constexpr):
    """как _down_live, но список живых нейронов делится между KS программами (чередованием по NB):
    больше одновременных чтений по шине. Частичные суммы -> part[s, j, 1024]"""
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    j = tl.program_id(2)
    base = tl.load(tab_ptr + j).to(tl.pointer_type(tl.uint8))
    cnt = tl.load(cnt_ptr + j)
    r2 = pid * (BR // 2) + tl.arange(0, BR // 2)
    acc_e = tl.zeros([BR // 2], dtype=tl.float32)
    acc_o = tl.zeros([BR // 2], dtype=tl.float32)
    for k0 in range(sp * NB, cnt, NB * KS):
        kk = k0 + tl.arange(0, NB)
        km = kk < cnt
        n = tl.load(idx_ptr + j * 2688 + kk, mask=km, other=0)
        hn = tl.load(h_ptr + j * 2688 + n, mask=km, other=0.0)
        w = tl.load(base + 1548288 + n[:, None].to(tl.int64) * 512 + r2[None, :], mask=km[:, None], other=0).to(tl.int32)
        g = (n // 16).to(tl.int64)
        se = tl.load(base + 2924544 + g[:, None] * 1024 + 2 * r2[None, :], mask=km[:, None], other=0)
        so = tl.load(base + 2924544 + g[:, None] * 1024 + 2 * r2[None, :] + 1, mask=km[:, None], other=0)
        se = se.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        so = so.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        acc_e += tl.sum(_e2m1_hw(w & 15) * se * hn[:, None], axis=0)
        acc_o += tl.sum(_e2m1_hw(w >> 4) * so * hn[:, None], axis=0)
    c = tl.load(coef_ptr + j) * tl.load(s2_ptr + j * 2 + 1)
    ob = part_ptr + (sp * K + j) * 1024 + pid * BR
    tl.store(ob + 2 * tl.arange(0, BR // 2), acc_e * c)
    tl.store(ob + 2 * tl.arange(0, BR // 2) + 1, acc_o * c)


LIVE_CFG = dict(BR=256, NB=8, KS=8, num_warps=4)


@triton.jit
def _down_groups(tab_ptr, h_ptr, gidx_ptr, gcnt_ptr, s2_ptr, coef_ptr, part_ptr, K, BR: tl.constexpr,
                 KS: tl.constexpr):
    """по ЖИВЫМ ГРУППАМ из 16 нейронов: масштабы группы — один сплошной кусок [BR] (один раз), 4 бита — только живых
    нейронов группы (маска h>0: мёртвые не читаются). Группы делятся между KS программами чередованием."""
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    j = tl.program_id(2)
    base = tl.load(tab_ptr + j).to(tl.pointer_type(tl.uint8))
    gc = tl.load(gcnt_ptr + j)
    r2 = pid * (BR // 2) + tl.arange(0, BR // 2)
    rr = pid * BR + tl.arange(0, BR)
    i16 = tl.arange(0, 16)
    acc_e = tl.zeros([BR // 2], dtype=tl.float32)
    acc_o = tl.zeros([BR // 2], dtype=tl.float32)
    for k in range(sp, gc, KS):
        g = tl.load(gidx_ptr + j * 168 + k).to(tl.int64)
        n = g * 16 + i16
        hn = tl.load(h_ptr + j * 2688 + n)
        lv = hn > 0
        w = tl.load(base + 1548288 + n[:, None] * 512 + r2[None, :], mask=lv[:, None], other=0).to(tl.int32)
        sc = tl.load(base + 2924544 + g * 1024 + rr)                       # [BR] сплошь
        se, so = tl.split(tl.reshape(sc, [BR // 2, 2]))
        se = se.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        so = so.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        acc_e += tl.sum(_e2m1_hw(w & 15) * hn[:, None], axis=0) * se
        acc_o += tl.sum(_e2m1_hw(w >> 4) * hn[:, None], axis=0) * so
    c = tl.load(coef_ptr + j) * tl.load(s2_ptr + j * 2 + 1)
    ob = part_ptr + (sp * K + j) * 1024 + pid * BR
    tl.store(ob + 2 * tl.arange(0, BR // 2), acc_e * c)
    tl.store(ob + 2 * tl.arange(0, BR // 2) + 1, acc_o * c)


GROUP_CFG = dict(BR=1024, KS=2, num_warps=2)      # перебор live_tune 16.10: 634 мкс/слой (K=22, вместе с up)


def moe_decode_groups(slots_up, sidx, tab, s2, coef, x, h_buf, part_buf, gidx_buf, gcnt_buf, cfg=None):
    c = cfg or GROUP_CFG
    K = sidx.shape[0]
    _up[(INTER // 32, K)](slots_up, sidx, s2, x, h_buf, BLOCKB=UPB, BN=32, GB=8, num_warps=4)
    glive = (h_buf[:K] > 0).view(K, 168, 16).any(-1)
    torch.sum(glive, dim=1, dtype=torch.int32, out=gcnt_buf[:K])
    gidx_buf[:K] = torch.sort(glive.to(torch.uint8), dim=1, descending=True, stable=True).indices.to(torch.int32)
    KS = c["KS"]
    _down_groups[(LAT // c["BR"], KS, K)](tab, h_buf, gidx_buf, gcnt_buf, s2, coef, part_buf, K, BR=c["BR"], KS=KS,
                                          num_warps=c["num_warps"])
    return part_buf[:KS, :K].sum((0, 1))




def moe_decode_live2(slots_up, sidx, tab, s2, coef, x, h_buf, part_buf, idx_buf, cnt_buf, cfg=None):
    """part_buf [KS_max, K, 1024] f32. Сумма по KS и K — в конце (порядок фиксирован)."""
    c = cfg or LIVE_CFG
    K = sidx.shape[0]
    _up[(INTER // 32, K)](slots_up, sidx, s2, x, h_buf, BLOCKB=UPB, BN=32, GB=8, num_warps=4)
    live = h_buf[:K] > 0
    torch.sum(live, dim=1, dtype=torch.int32, out=cnt_buf[:K])
    idx_buf[:K] = torch.sort(live.to(torch.uint8), dim=1, descending=True, stable=True).indices.to(torch.int32)
    KS = c["KS"]
    _down_live2[(LAT // c["BR"], KS, K)](tab, h_buf, idx_buf, cnt_buf, s2, coef, part_buf, K, BR=c["BR"], NB=c["NB"],
                                         KS=KS, num_warps=c["num_warps"])
    return part_buf[:KS, :K].sum((0, 1))


def moe_decode_live(slots_up, sidx, tab, s2, coef, x, h_buf, out_buf, idx_buf, cnt_buf, BR=256, NB=8):
    """up — из слотов на карте (UPB байт), down — по таблице адресов tab [K] int64 (начало блока «T» в RAM/на карте),
    только живые нейроны. Без синхронизаций: список живых строится на карте (устойчивая сортировка)."""
    K = sidx.shape[0]
    _up[(INTER // 32, K)](slots_up, sidx, s2, x, h_buf, BLOCKB=UPB, BN=32, GB=8, num_warps=4)
    live = h_buf[:K] > 0
    torch.sum(live, dim=1, dtype=torch.int32, out=cnt_buf[:K])
    idx_buf[:K] = torch.sort(live.to(torch.uint8), dim=1, descending=True, stable=True).indices.to(torch.int32)
    _down_live[(LAT // BR, K)](tab, h_buf, idx_buf, cnt_buf, s2, coef, out_buf, BR=BR, NB=NB, num_warps=4)
    return out_buf[:K].sum(0)


@triton.jit
def _down_pairs_T(slots_ptr, pslot_ptr, s2_ptr, coef_ptr, h_ptr, out_ptr, BLOCKB: tl.constexpr, BR: tl.constexpr):
    """промт, раскладка «T» на карте: out[p, r] = coef·ws2d·Σ_n W[r,n]·S[r,n//16]·h[p,n] — по группам 16 нейронов
    (масштабы группы сплошь [BR], 4 бита 16 нейронов [16, BR/2]); без обратной перестановки"""
    pid = tl.program_id(0)
    p = tl.program_id(1)
    slot = tl.load(pslot_ptr + p).to(tl.int64)
    base = slots_ptr + slot * BLOCKB
    r2 = pid * (BR // 2) + tl.arange(0, BR // 2)
    rr = pid * BR + tl.arange(0, BR)
    i16 = tl.arange(0, 16)
    acc_e = tl.zeros([BR // 2], dtype=tl.float32)
    acc_o = tl.zeros([BR // 2], dtype=tl.float32)
    for g in range(0, 168):
        n = g * 16 + i16
        hn = tl.load(h_ptr + p * 2688 + n)
        w = tl.load(base + 1548288 + n[:, None].to(tl.int64) * 512 + r2[None, :]).to(tl.int32)
        sc = tl.load(base + 2924544 + g * 1024 + rr)
        se, so = tl.split(tl.reshape(sc, [BR // 2, 2]))
        se = se.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        so = so.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        acc_e += tl.sum(_e2m1_hw(w & 15) * hn[:, None], axis=0) * se
        acc_o += tl.sum(_e2m1_hw(w >> 4) * hn[:, None], axis=0) * so
    c = tl.load(coef_ptr + p) * tl.load(s2_ptr + p * 2 + 1)
    ob = out_ptr + p * 1024 + pid * BR
    tl.store(ob + 2 * tl.arange(0, BR // 2), acc_e * c)
    tl.store(ob + 2 * tl.arange(0, BR // 2) + 1, acc_o * c)


PAIRS_T_BR = 64


def moe_pairs_T(slots_u8, pslot, ptok, s2p, coefp, X, h_buf, out_buf):
    """как moe_pairs, но блоки в раскладке «T»"""
    P = pslot.shape[0]
    _up_pairs[(INTER // 32, P)](slots_u8, pslot, ptok, s2p, X, h_buf, BLOCKB=BLOCK, BN=32, GB=8, num_warps=4)
    _down_pairs_T[(LAT // PAIRS_T_BR, P)](slots_u8, pslot, s2p, coefp, h_buf, out_buf, BLOCKB=BLOCK, BR=PAIRS_T_BR,
                                          num_warps=4)
    return out_buf[:P]


@triton.jit
def _fromT_w(src_ptr, dst_ptr, BLOCKB: tl.constexpr, RB: tl.constexpr, NBk: tl.constexpr):
    """4 бита down: раскладка «T» (нейрон-строка) -> исходная (строка r, пары нейронов) в отдельный буфер dst [n, 1376256]"""
    pr = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)
    n = pn * NBk + tl.arange(0, NBk)
    rb = pr * (RB // 2) + tl.arange(0, RB // 2)
    t = tl.load(src_ptr + e * BLOCKB + 1548288 + n[:, None].to(tl.int64) * 512 + rb[None, :])   # [NBk, RB/2]
    nib = tl.reshape(tl.join(t & 15, t >> 4), [NBk, RB])                                      # (n, r)
    pairs = tl.reshape(tl.trans(nib), [RB, NBk // 2, 2])                                      # (r, n/2, 2)
    lo, hi = tl.split(pairs)
    byte = lo | (hi << 4)
    r = pr * RB + tl.arange(0, RB)
    col = pn * (NBk // 2) + tl.arange(0, NBk // 2)
    tl.store(dst_ptr + e * 1376256 + r[:, None].to(tl.int64) * 1344 + col[None, :], byte)


def from_T_fast(blks, scratch):
    """[N, BLOCK] на карте: «T» -> исходная на месте; scratch u8 >= N*1376256 (4 бита down во временном)"""
    N = blks.shape[0]
    wd = scratch[:N * (LAT * KB_DN)].view(N, LAT * KB_DN)
    _fromT_w[(LAT // 32, INTER // 64, N)](blks, wd, BLOCKB=BLOCK, RB=32, NBk=64, num_warps=4)
    sd = blks[:, O_TS:BLOCK].view(N, G_DN, LAT).transpose(1, 2).reshape(N, -1)              # копия (transpose)
    blks[:, O_DW:O_DS] = wd
    blks[:, O_DS:BLOCK] = sd
    return blks


def to_T(blk):
    """блок исходной раскладки (u8 тензор [BLOCK], любое устройство) -> раскладка «T» (новый тензор)"""
    out = blk.clone()
    dw = blk[O_DW:O_DS].view(LAT, KB_DN)                       # [1024, 1344]
    nib = torch.stack([dw & 15, dw >> 4], -1).view(LAT, INTER)  # [1024 r, 2688 n]
    t = nib.t().contiguous().view(INTER, LAT // 2, 2)           # [n, r/2, 2]
    out[O_TW:O_TS] = (t[..., 0] | (t[..., 1] << 4)).flatten()
    out[O_TS:BLOCK] = blk[O_DS:BLOCK].view(LAT, G_DN).t().contiguous().flatten()
    return out


def from_T(blk):
    """обратно: раскладка «T» -> исходная (для промта и проверки)"""
    out = blk.clone()
    tw = blk[O_TW:O_TS].view(INTER, LAT // 2)
    nib = torch.stack([tw & 15, tw >> 4], -1).view(INTER, LAT)  # [n, r]
    t = nib.t().contiguous().view(LAT, INTER // 2, 2)           # [r, n/2, 2]
    out[O_DW:O_DS] = (t[..., 0] | (t[..., 1] << 4)).flatten()
    out[O_DS:BLOCK] = blk[O_TS:BLOCK].view(G_DN, LAT).t().contiguous().flatten()
    return out


def from_T_batch(blks):
    """[N, BLOCK] на карте: раскладка «T» -> исходная, на месте (промт: дальше старые ядра)"""
    N = blks.shape[0]
    tw = blks[:, O_TW:O_TS].view(N, INTER, LAT // 2)
    nib = torch.stack([tw & 15, tw >> 4], -1).view(N, INTER, LAT).transpose(1, 2).contiguous().view(N, LAT, INTER // 2, 2)
    dw = (nib[..., 0] | (nib[..., 1] << 4)).view(N, -1)
    ds = blks[:, O_TS:BLOCK].view(N, G_DN, LAT).transpose(1, 2).contiguous().view(N, -1)
    blks[:, O_DW:O_DS] = dw
    blks[:, O_DS:BLOCK] = ds
    return blks


# ---------- эталонная распаковка (torch) ----------
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
SWAP = False   # контрольная подмена: старший полубайт = чётный элемент (заведомо неверно)


def deq(w_u8, s_e4m3_u8, ws2):
    """w_u8 [N, K/2] · s [N, K/16] (байты E4M3) · ws2 -> fp32 [N, K]"""
    t = E2M1.to(w_u8.device)
    lo = t[(w_u8 & 15).long()]
    hi = t[(w_u8 >> 4).long()]
    if SWAP:
        lo, hi = hi, lo
    w = torch.stack([lo, hi], -1).reshape(w_u8.shape[0], -1)
    s = s_e4m3_u8.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return w * s * ws2


def block_deq(blk_u8, s2):
    """блок (u8 [BLOCK]) -> up [2688,1024], down [1024,2688] fp32"""
    b = blk_u8
    u = deq(b[O_UW:O_US].view(INTER, KB_UP), b[O_US:O_DW].view(INTER, G_UP), float(s2[0]))
    d = deq(b[O_DW:O_DS].view(LAT, KB_DN), b[O_DS:BLOCK].view(LAT, G_DN), float(s2[1]))
    return u, d
