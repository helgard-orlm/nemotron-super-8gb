"""Плотная часть Nemotron в исходных форматах чекпойнта вместо bf16 (#264, 18.09; пункт 1 плана #263).

Профиль 18.09: плотные GEMV bf16 = ~10 мс/ток из ~26 (3.46 ГиБ bf16 читаются каждый токен). В чекпойнте они сжаты:
  FP8 e4m3, масштаб на тензор  — Mamba in_proj / out_proj (23+23)
  NVFP4 (E2M1 + fp8 на группу 16 + s2) — shared_experts up/down (23+23), lm_head
⇒ читать 1.58 ГиБ вместо 3.46 и освободить ~1.9 ГиБ VRAM. Внимание q/k/v/o в чекпойнте bf16 — не трогаем.

  1 токен (M==1): своё ядро, распаковка в регистрах, накопление fp32.
  M>1 (промт, logits без logits_to_keep): распаковка кусками строк в общий буфер bf16 (SCR_MIB) + F.linear.
Подмена делается на месте: у того же nn.Linear меняется forward (хуки профиля на lm_head остаются).
⚠ Числа не бит-в-бит со старым путём: там вес округлялся в bf16, здесь fp32 — сверка по логитам и 300 вопросам.
"""
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SCR_MIB = 64
_SCR = {}
MB_ON = True        # M=2 (проверка черновика MTP) — своё ядро на несколько векторов; False — распаковка, как для промта


@triton.jit
def _e2m1(c):
    """код 0..15 -> E2M1 сборкой битов float32 (без exp2: стенд 18.09 — с exp2 ядро упиралось в счёт, 165 ГБ/с).
    Модуль m=1..7 при m>=2: биты (m<<22)+(126<<23) дают 1 1.5 2 3 4 6; m=0,1 -> 0, 0.5 отдельно; знак — бит 31."""
    m = c & 7
    bits = (m << 22) + (126 << 23) + ((c & 8) << 28)
    v = bits.to(tl.float32, bitcast=True)
    return tl.where(m < 2, tl.where((c & 8) != 0, -0.5, 0.5) * (m & 1).to(tl.float32), v)


@triton.jit
def _gemv_fp8(w_ptr, x_ptr, out_ptr, s_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr,
              NSPLIT: tl.constexpr):
    """y[n] = s * sum_k fp8(w[n,k]) * x[k]; при NSPLIT>1 каждая программа берёт свою долю K и пишет частичную сумму"""
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < N
    acc = tl.zeros([BLOCK_N, BK], dtype=tl.float32)
    KS: tl.constexpr = K // NSPLIT
    for k0 in range(sp * KS, sp * KS + KS, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < K
        wb = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=rm[:, None] & km[None, :], other=0)
        wv = wb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        xv = tl.load(x_ptr + ks, mask=km, other=0.0).to(tl.float32)
        acc += wv * xv[None, :]
    y = tl.sum(acc, axis=1) * tl.load(s_ptr)
    tl.store(out_ptr + sp * N + rows, y, mask=rm)


@triton.jit
def _gemv_nvfp4(w_ptr, sc_ptr, x_ptr, out_ptr, s2_ptr, N, KB: tl.constexpr, G: tl.constexpr,
                BLOCK_N: tl.constexpr, GB: tl.constexpr, NSPLIT: tl.constexpr):
    """w uint8 [N, KB] (два E2M1 на байт, младшая тетрада — чётный), sc fp8 [N, G] на группу 16, s2 скаляр"""
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < N
    acc = tl.zeros([BLOCK_N, GB], dtype=tl.float32)
    GS: tl.constexpr = G // NSPLIT
    for g0 in range(sp * GS, sp * GS + GS, GB):
        gi = g0 + tl.arange(0, GB)
        gm = gi < G
        bi = g0 * 8 + tl.arange(0, GB * 8)
        bm = bi < KB
        wb = tl.load(w_ptr + rows[:, None] * KB + bi[None, :], mask=rm[:, None] & bm[None, :], other=0)
        wi = wb.to(tl.int32)
        xe = tl.load(x_ptr + 2 * bi, mask=bm, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + 2 * bi + 1, mask=bm, other=0.0).to(tl.float32)
        p = _e2m1(wi & 15) * xe[None, :] + _e2m1(wi >> 4) * xo[None, :]
        pg = tl.sum(tl.reshape(p, [BLOCK_N, GB, 8]), axis=2)
        sb = tl.load(sc_ptr + rows[:, None] * G + gi[None, :], mask=rm[:, None] & gm[None, :], other=0)
        acc += pg * sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    y = tl.sum(acc, axis=1) * tl.load(s2_ptr)
    tl.store(out_ptr + sp * N + rows, y, mask=rm)


@triton.jit
def _gemv_fp8_m2(w_ptr, x_ptr, out_ptr, s_ptr, N, M, K: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr):
    """#264 (проверка черновика MTP пачкой): 1..2 вектора, вес читается ОДИН раз. x [M,K], out [M,N].
    Два отдельных накопителя 2D, как в одиночном ядре (стенд 18.09: вариант с 3D [строки,векторы,K] в 6 раз медленнее)"""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < N
    acc0 = tl.zeros([BLOCK_N, BK], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_N, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < K
        wb = tl.load(w_ptr + rows[:, None] * K + ks[None, :], mask=rm[:, None] & km[None, :], other=0)
        wv = wb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        x0 = tl.load(x_ptr + ks, mask=km, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + K + ks, mask=km & (M > 1), other=0.0).to(tl.float32)
        acc0 += wv * x0[None, :]
        acc1 += wv * x1[None, :]
    s = tl.load(s_ptr)
    tl.store(out_ptr + rows, tl.sum(acc0, axis=1) * s, mask=rm)
    tl.store(out_ptr + N + rows, tl.sum(acc1, axis=1) * s, mask=rm & (M > 1))


@triton.jit
def _gemv_nvfp4_m2(w_ptr, sc_ptr, x_ptr, out_ptr, s2_ptr, N, M, KB: tl.constexpr, G: tl.constexpr,
                   BLOCK_N: tl.constexpr, GB: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < N
    K2: tl.constexpr = KB * 2
    acc0 = tl.zeros([BLOCK_N, GB], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_N, GB], dtype=tl.float32)
    for g0 in range(0, G, GB):
        gi = g0 + tl.arange(0, GB)
        gm = gi < G
        bi = g0 * 8 + tl.arange(0, GB * 8)
        bm = bi < KB
        wb = tl.load(w_ptr + rows[:, None] * KB + bi[None, :], mask=rm[:, None] & bm[None, :], other=0)
        wi = wb.to(tl.int32)
        lo = _e2m1(wi & 15)
        hi = _e2m1(wi >> 4)
        sb = tl.load(sc_ptr + rows[:, None] * G + gi[None, :], mask=rm[:, None] & gm[None, :], other=0)
        sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        xe0 = tl.load(x_ptr + 2 * bi, mask=bm, other=0.0).to(tl.float32)
        xo0 = tl.load(x_ptr + 2 * bi + 1, mask=bm, other=0.0).to(tl.float32)
        p0 = lo * xe0[None, :] + hi * xo0[None, :]
        acc0 += tl.sum(tl.reshape(p0, [BLOCK_N, GB, 8]), axis=2) * sc
        xe1 = tl.load(x_ptr + K2 + 2 * bi, mask=bm & (M > 1), other=0.0).to(tl.float32)
        xo1 = tl.load(x_ptr + K2 + 2 * bi + 1, mask=bm & (M > 1), other=0.0).to(tl.float32)
        p1 = lo * xe1[None, :] + hi * xo1[None, :]
        acc1 += tl.sum(tl.reshape(p1, [BLOCK_N, GB, 8]), axis=2) * sc
    s2 = tl.load(s2_ptr)
    tl.store(out_ptr + rows, tl.sum(acc0, axis=1) * s2, mask=rm)
    tl.store(out_ptr + N + rows, tl.sum(acc1, axis=1) * s2, mask=rm & (M > 1))


def gemv_mb(q, x):
    """x [M,K], M<=2 -> y float32 [M,N]; вес читается один раз (настройки — как у одиночного ядра той же формы)"""
    M = x.shape[0]
    assert M <= 2
    N, K = q["N"], q["K"]
    c = CFG_BY_SHAPE.get((q["fmt"], N, K), CFG[q["fmt"]])
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, c["BLOCK_N"]),)
    if q["fmt"] == "fp8":
        _gemv_fp8_m2[grid](q["w"], x, out, q["s"], N, M, K=K, BLOCK_N=c["BLOCK_N"], BK=c["BK"], num_warps=c["num_warps"])
    else:
        _gemv_nvfp4_m2[grid](q["w"], q["sc"], x, out, q["s"], N, M, KB=K // 2, G=K // 16, BLOCK_N=c["BLOCK_N"],
                             GB=c["GB"], num_warps=c["num_warps"])
    return out


# настройки по умолчанию; bench подбирает свои под форму (N, K)
CFG = {"fp8": dict(BLOCK_N=16, BK=256, NSPLIT=1, num_warps=4), "nvfp4": dict(BLOCK_N=16, GB=8, NSPLIT=1, num_warps=2)}
# лучшие по перебору bench_dense.py (18.09, gpu104, ~150 вариантов на форму)
CFG_BY_SHAPE = {
    ("fp8", 10304, 2688): dict(BLOCK_N=32, BK=128, NSPLIT=1, num_warps=8),       # in_proj
    ("fp8", 2688, 4096): dict(BLOCK_N=16, BK=256, NSPLIT=1, num_warps=4),        # out_proj
    ("nvfp4", 3712, 2688): dict(BLOCK_N=16, GB=8, NSPLIT=1, num_warps=2),        # shared up
    ("nvfp4", 2688, 3712): dict(BLOCK_N=16, GB=8, NSPLIT=1, num_warps=2),        # shared down
    ("nvfp4", 131072, 2688): dict(BLOCK_N=32, GB=8, NSPLIT=1, num_warps=2),      # lm_head
}


E2M1_T = None


def _lut(dev):
    global E2M1_T
    if E2M1_T is None or E2M1_T.device != torch.device(dev):
        v = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=dev)
        a = torch.arange(256, device=dev)
        E2M1_T = torch.stack([v[a & 15], v[a >> 4]], -1)       # [256, 2]
    return E2M1_T


def dequant_rows(q, r0, r1, out):
    """строки [r0, r1) веса в bf16 -> out [r1-r0, K]"""
    if q["fmt"] == "fp8":
        out.copy_(q["w"][r0:r1].view(torch.float8_e4m3fn).float().mul_(q["s"]))     # w хранится байтами uint8
    else:
        v = _lut(q["w"].device)[q["w"][r0:r1].int()].flatten(-2)                     # [r, K] fp32
        sh = v.shape
        sc = q["sc"][r0:r1].view(torch.float8_e4m3fn).float() * q["s"]
        out.copy_((v.view(sh[0], sh[1] // 16, 16) * sc[..., None]).view(sh))
    return out


def dequant(q):
    out = torch.empty(q["N"], q["K"], dtype=torch.bfloat16, device=q["w"].device)
    step = 4096
    for r in range(0, q["N"], step):
        dequant_rows(q, r, min(r + step, q["N"]), out[r:r + step])
    return out


def gemv(q, x):
    """x [K] (любой float) -> y float32 [N]"""
    N, K = q["N"], q["K"]
    c = CFG_BY_SHAPE.get((q["fmt"], N, K), CFG[q["fmt"]])
    ns = c["NSPLIT"]
    out = torch.empty(ns, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, c["BLOCK_N"]), ns)
    if q["fmt"] == "fp8":
        _gemv_fp8[grid](q["w"], x, out, q["s"], N, K=K, BLOCK_N=c["BLOCK_N"], BK=c["BK"], NSPLIT=ns,
                        num_warps=c["num_warps"])
    else:
        _gemv_nvfp4[grid](q["w"], q["sc"], x, out, q["s"], N, KB=K // 2, G=K // 16, BLOCK_N=c["BLOCK_N"],
                          GB=c["GB"], NSPLIT=ns, num_warps=c["num_warps"])
    return out[0] if ns == 1 else out.sum(0)


def qlinear(q, x):
    """как F.linear(x, W_bf16) без bias: вход [..., K] -> [..., N] в dtype входа"""
    sh = x.shape
    x2 = x.reshape(-1, sh[-1])
    M, N = x2.shape[0], q["N"]
    if M == 1:
        return gemv(q, x2[0].contiguous()).to(x.dtype).view(*sh[:-1], N)
    if M <= 2 and MB_ON:
        return gemv_mb(q, x2.contiguous()).to(x.dtype).view(*sh[:-1], N)
    dev = x.device
    rows = max(1, min(N, SCR_MIB * 2**20 // (2 * q["K"])))
    scr = _SCR.get(dev)
    if scr is None or scr.numel() < rows * q["K"]:
        scr = _SCR[dev] = torch.empty(SCR_MIB * 2**20 // 2, dtype=torch.bfloat16, device=dev)
    y = torch.empty(M, N, dtype=x.dtype, device=dev)
    for r in range(0, N, rows):
        r1 = min(r + rows, N)
        w = dequant_rows(q, r, r1, scr[:(r1 - r) * q["K"]].view(r1 - r, q["K"]))
        y[:, r:r1] = F.linear(x2, w)
    return y.view(*sh[:-1], N)


def load_q(ck, base, dev):
    """base без .weight -> словарь сжатого веса на карте, или None если в чекпойнте он не сжат"""
    w = ck.get(base + ".weight")
    if base + ".weight_scale_2" in ck:
        sc = ck.get(base + ".weight_scale")
        return {"fmt": "nvfp4", "w": w.to(dev), "sc": sc.view(torch.uint8).to(dev),
                "s": ck.get(base + ".weight_scale_2").float().reshape(1).to(dev), "N": w.shape[0], "K": w.shape[1] * 2}
    if base + ".weight_scale" in ck and w.dtype == torch.float8_e4m3fn:
        return {"fmt": "fp8", "w": w.view(torch.uint8).to(dev),
                "s": ck.get(base + ".weight_scale").float().reshape(1).to(dev), "N": w.shape[0], "K": w.shape[1]}
    return None


def _fwd(self, x):
    return qlinear(self._q, x)


def attach(lin, q):
    """nn.Linear -> сжатый: веса bf16 отпускаются, forward на qlinear. .weight = сжатые байты на карте (Mamba смотрит
    только .weight.device; путь с out_proj.weight напрямую — лишь при обучении)"""
    assert lin.bias is None
    lin._parameters.pop("weight", None)
    lin._q = q
    object.__setattr__(lin, "weight", q["w"])
    lin.forward = types.MethodType(_fwd, lin)


def nbytes(q):
    return sum(t.numel() * t.element_size() for k, t in q.items() if isinstance(t, torch.Tensor))
