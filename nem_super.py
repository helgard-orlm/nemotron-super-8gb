"""Движок Nemotron-3.5-Super-VL-120B-A12B (NVFP4) на gpu104 (RTX 5060 8 ГБ, RAM 31 ГБ) — только текст (#399, 16.10.2026).

Скелет — transformers NemotronH (5.9 знает латентный MoE), как у Nano (nem_stream.py):
  - постоянная часть на карте в форматах чекпойнта (dense_q: FP8 Mamba/внимание/голова, NVFP4 общие эксперты),
    латентные проекции и роутер bf16, эмбеддинги в RAM;
  - routed-эксперты (40 слоёв × 512, 3.1 МБ) — с /fast из experts.bin: RAM-кэш LRU (закреплённая память) +
    чтение промахов пулом потоков (O_DIRECT, один pread на эксперта) + копия на карту в своём потоке, две банки слотов
    (как ExpertCache у Qwen3.8, qwen_engine.py);
  - шаг токена: своё ядро qmoe_ns (relu², латент 1024); промт: распаковка экспертов в bf16 + F.linear.
Активации не квантуются (W4A16/W8A16) — веса те же, что у NVIDIA.

Переменные: NS_DIR (/fast/nemsuper), NS_RAM_GB (14), NS_THREADS (32), NS_PF (2), NS_PAIR_T (64), NS_PROMPT_CACHE_T (512)."""
import collections, concurrent.futures as cf, json, mmap, os, sys, threading, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_q
dense_q.SCR_MIB = 32                 # временная распаковка плотных весов на промте (карта 8 ГБ впритык)
import qmoe_ns as Q
from transformers import AutoTokenizer
from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
RAM_GB = float(os.environ.get("NS_RAM_GB", "14"))
THREADS = int(os.environ.get("NS_THREADS", "32"))
PF = int(os.environ.get("NS_PF", "2"))
PAIR_T = int(os.environ.get("NS_PAIR_T", "64"))            # промт до стольких токенов — ядро пар
PROMPT_CACHE_T = int(os.environ.get("NS_PROMPT_CACHE_T", "512"))  # эксперты промта до стольких токенов — в RAM-кэш
PFX = "language_model."
BLK = 4096
NE = 512
dev = "cuda"
_KEEP = []


def pinned_aligned(nbytes):
    """закреплённая память, выровненная по странице (pin_memory() не выровнен — O_DIRECT падает)"""
    nr = (nbytes + BLK - 1) // BLK * BLK
    mm = mmap.mmap(-1, nr)
    t = torch.from_numpy(np.frombuffer(mm, dtype=np.uint8, count=nbytes))
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nr, 0)
    assert int(rc) == 0, f"cudaHostRegister {rc}"
    _KEEP.append(mm)
    return t


class Ckpt:
    def __init__(self, path):
        self.f = safe_open(path, framework="pt", device="cpu")
        self.keys = set(self.f.keys())

    def __contains__(self, n):
        return n in self.keys

    def get(self, n):
        return self.f.get_tensor(n)


class _PfxCkpt:
    """имена модели HF -> имена чекпойнта (backbone.* / lm_head.* под language_model.)"""
    def __init__(self, ck):
        self.ck = ck

    def _n(self, n):
        return PFX + n

    def __contains__(self, n):
        return self._n(n) in self.ck

    def get(self, n):
        return self.ck.get(self._n(n))


# ============================================================ эксперты с диска
class ExpertCache:
    """RAM-кэш LRU + чтение промахов пулом потоков + две банки слотов на карте (чётный/нечётный MoE-слой) +
    предвыборка P экспертов следующего MoE-слоя в «пузырь» диска (как у Qwen3.8/DeepSeek v4).
    Шаг: start() -> [счёт общего эксперта и латента] -> finish() -> prefetch(m+1) -> ядро -> done() -> drop_pf(m)."""

    def __init__(self, n_layers, K, ram_gb, threads, pf=0):
        self.K, self.NL, self.pf = K, n_layers, pf
        self.fd = os.open(f"{DIR}/experts.bin", os.O_RDONLY | os.O_DIRECT)
        self.n_ram = int(ram_gb * 2**30 // Q.BLOCK)
        assert self.n_ram > 4 * K, "движок рассчитан на RAM-кэш"
        self.buf = pinned_aligned(self.n_ram * Q.BLOCK).view(self.n_ram, Q.BLOCK)
        self.where, self.lru, self.free = {}, collections.OrderedDict(), list(range(self.n_ram))
        self.row_ev, self.pend, self.inflight = {}, {}, {}
        lay = open(f"{DIR}/experts.layout").read().strip() if os.path.exists(f"{DIR}/experts.layout") else "orig"
        assert not os.path.exists(f"{DIR}/experts.layout.inprogress"), "experts.bin в середине перестановки"
        self.T = lay == "T"                    # #399 п.1: down по нейронам — на карту только up, down читается из RAM
        self.SB = Q.UPB if self.T else Q.BLOCK
        self.slots = torch.empty(2 * K, self.SB, dtype=torch.uint8, device=dev)
        self.host_base = self.buf.data_ptr()
        self.tab_h = torch.empty(64, K, dtype=torch.int64).pin_memory()     # кольцо по слою (как sidx у Qwen)
        self.tab_d = torch.empty(64, K, dtype=torch.int64, device=dev)
        self.sidx = [torch.arange(b * K, (b + 1) * K, dtype=torch.int32, device=dev) for b in (0, 1)]
        self.bank_ev = [None, None]
        self.cs = torch.cuda.Stream()
        self.pool = cf.ThreadPoolExecutor(threads)
        self.lock = threading.RLock()          # RLock: обработчик конца чтения может вызваться под замком
        scal = np.load(f"{DIR}/experts_scal.npy")
        self.s2 = torch.from_numpy(scal[:, :, :2].copy()).to(dev)          # ws2 up/down [40,512,2]
        self.stat = collections.Counter()
        self.read_s = 0.0
        print(f"эксперты: RAM-кэш {self.n_ram} ({self.n_ram * Q.BLOCK / 2**30:.1f} ГиБ, "
              f"{self.n_ram / (n_layers * NE) * 100:.0f}% всех), потоков {threads}, предвыборка P={pf}", flush=True)

    def _busy(self, v):
        return v in self.pend or (v in self.inflight and not self.inflight[v].done())

    def _alloc(self, k, protect):
        """под замком: строка RAM для нового ключа (свободная или вытеснение LRU; летящие не трогаем)"""
        if self.free:
            r = self.free.pop()
        else:
            victim = next(v for v in self.lru if v not in protect and not self._busy(v))
            r = self.where.pop(victim); del self.lru[victim]
            self.inflight.pop(victim, None)
        self.where[k] = r; self.lru[k] = None
        return r

    def _read(self, r, k):
        ev = self.row_ev.pop(r, None)
        if ev is not None:
            ev.synchronize()                      # строку ещё копируют на карту
        got = os.preadv(self.fd, [memoryview(self.buf[r].numpy())], (k[0] * NE + k[1]) * Q.BLOCK)
        assert got == Q.BLOCK, got

    def _copy(self, slot, r):
        with torch.cuda.stream(self.cs):
            self.slots[slot].copy_(self.buf[r, :self.SB], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.cs)
        self.row_ev[r] = ev

    def start(self, m, experts):
        bank = m % 2
        base = bank * self.K
        if self.bank_ev[bank] is not None:
            self.cs.wait_event(self.bank_ev[bank])                 # слоты банки свободны (ядро слоя m-2 досчитало)
        prot = {(m, e) for e in experts}
        todo = []
        with self.lock:
            for j, e in enumerate(experts):
                k = (m, e)
                if k in self.pend:
                    r, f = self.pend.pop(k); self.stat["pf_hit"] += 1
                    self.lru.move_to_end(k)
                    todo.append((j, r, f))
                elif k in self.inflight and not self.inflight[k].done():
                    self.stat["late_hit"] += 1
                    self.lru.move_to_end(k)
                    todo.append((j, self.where[k], self.inflight[k]))
                elif k in self.where:
                    self.lru.move_to_end(k); self.stat["ram_hit"] += 1
                    todo.append((j, self.where[k], None))
                else:
                    r = self._alloc(k, prot); self.stat["disk"] += 1
                    todo.append((j, r, self.pool.submit(self._read, r, k)))
        for j, r, f in todo:
            if f is None:
                self._copy(base + j, r)
        return (m, bank, todo, prot)

    def table(self, m, todo):
        """адреса строк RAM экспертов слоя (порядок j) -> на карту без синхронизации"""
        th = self.tab_h[m]
        th.copy_(torch.tensor([self.host_base + r * Q.BLOCK for _, r, _ in todo], dtype=torch.int64))
        td = self.tab_d[m]
        td.copy_(th, non_blocking=True)
        return td

    def finish(self, h):
        m, bank, todo, prot = h
        base = bank * self.K
        t = time.time()
        waits = {}
        for j, r, f in todo:
            if f is not None:
                waits.setdefault(f, []).append((j, r))
        for f in cf.as_completed(list(waits)):
            f.result()
            for j, r in waits[f]:
                self._copy(base + j, r)
        self.read_s += time.time() - t
        ev = torch.cuda.Event(); ev.record(self.cs)
        torch.cuda.current_stream().wait_event(ev)
        return self.sidx[bank][:len(todo)]

    def done(self, m, todo=None):
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
        self.bank_ev[m % 2] = ev
        if self.T and todo is not None:          # ядро down читало строки RAM напрямую — не перезаписывать до его конца
            for _, r, _ in todo:
                self.row_ev[r] = ev

    def prefetch(self, m, cands, protect=()):
        """заранее прочитать до P экспертов слоя m (кандидаты по убыванию); лежащие в RAM/летящие не считаются"""
        n = 0
        with self.lock:
            for e in cands:
                if n >= self.pf:
                    break
                k = (m, e)
                if k in self.where or k in self.pend:
                    continue
                r = self._alloc(k, set(protect)); self.stat["pf_read"] += 1
                self.pend[k] = (r, self.pool.submit(self._read, r, k)); n += 1

    def drop_pf(self, m):
        """неугаданные заранее прочитанные слоя m — без ожидания: запись остаётся в кэше, пока летит — inflight"""
        with self.lock:
            for k in [k for k in self.pend if k[0] == m]:
                r, f = self.pend.pop(k); self.stat["pf_waste"] += 1
                self.inflight[k] = f
            for k in [k for k, f in self.inflight.items() if f.done()]:
                del self.inflight[k]

    # ---- путь промта
    PB = 32

    def prefill_submit(self, m, experts, ring, cache_it):
        """пачка до PB экспертов -> [(строка RAM, future|None)]: попадания — строки кэша; промахи читаются пулом
        либо в строки кэша (cache_it: короткий промт — его эксперты почти всегда нужны и ответу), либо в своё
        кольцо промта ring (0/1), не вымывая кэш. Перед записью в кольцо ждём, пока его прошлую пачку скопировали."""
        if not hasattr(self, "pbuf"):
            self.pbuf = [pinned_aligned(self.PB * Q.BLOCK).view(self.PB, Q.BLOCK) for _ in (0, 1)]
            self.pring_ev = [None, None]
        if self.pring_ev[ring] is not None and not cache_it:
            self.pring_ev[ring].synchronize()
        res = []
        prot = {(m, e) for e in experts}
        with self.lock:
            for i, e in enumerate(experts):
                k = (m, e)
                f = self.pend.get(k, (None, None))[1] or self.inflight.get(k)
                r = self.where.get(k)
                if r is not None:
                    self.stat["p_ram"] += 1
                    if k in self.lru:
                        self.lru.move_to_end(k)
                    res.append((self.buf[r], f, r))
                elif cache_it:
                    r = self._alloc(k, prot); self.stat["p_disk"] += 1
                    f = self.pool.submit(self._read, r, k)
                    self.inflight[k] = f
                    res.append((self.buf[r], f, r))
                else:
                    self.stat["p_disk"] += 1
                    dst = self.pbuf[ring][i]
                    res.append((dst, self.pool.submit(self._pread_into, dst, k), None))
        return res

    def _pread_into(self, dst, k):
        got = os.preadv(self.fd, [memoryview(dst.numpy())], (k[0] * NE + k[1]) * Q.BLOCK)
        assert got == Q.BLOCK, got


class StreamExperts(nn.Module):
    """замена NemotronHExperts: вход — латент [T,1024], top_k_index/weights [T,22]"""

    gpu_tmp = None                     # буфер промта на карте (PB блоков), общий на все слои
    pair_buf = None                    # h/out ядра пар (короткий промт)
    t_scratch = None                   # 4 бита down при обратной перестановке на промте

    def __init__(self, m, cache):
        super().__init__()
        self.m, self.c = m, cache
        self.h = torch.empty(cache.K, Q.INTER, device=dev)
        self.o = torch.empty(cache.K, Q.LAT, device=dev)
        self.part = torch.empty(Q.GROUP_CFG["KS"], cache.K, Q.LAT, device=dev)
        self.gidx = torch.empty(cache.K, 168, dtype=torch.int32, device=dev)
        self.gcnt = torch.empty(cache.K, dtype=torch.int32, device=dev)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        c, m = self.c, self.m
        if hidden_states.shape[0] == 1:
            assert not c.T, "шаг токена идёт через moe_forward"
            experts = top_k_index[0].tolist()
            h = c.start(m, experts)
            sidx = c.finish(h)
            ei = top_k_index[0]
            y = Q.moe_decode(c.slots, sidx, c.s2[m][ei].contiguous(), top_k_weights[0].float().contiguous(),
                             hidden_states[0].contiguous(), self.h, self.o)
            c.done(m)
            return y.view(1, -1).to(hidden_states.dtype)
        # промт: по уникальным экспертам пачками, чтение пачки i+1 летит, пока считается пачка i
        T = hidden_states.shape[0]
        out = torch.zeros(hidden_states.shape, dtype=torch.float32, device=dev)
        uniq = torch.unique(top_k_index).tolist()
        PB = c.PB
        if StreamExperts.gpu_tmp is None:
            # один буфер: копия и счёт идут в одном потоке друг за другом; перекрывается только диск (кольца в RAM)
            StreamExperts.gpu_tmp = torch.empty(PB, Q.BLOCK, dtype=torch.uint8, device=dev)
        small = T <= PAIR_T
        if small and StreamExperts.pair_buf is None:
            StreamExperts.pair_buf = (torch.empty(PAIR_T * 22, Q.INTER, device=dev),
                                      torch.empty(PAIR_T * 22, Q.LAT, device=dev))
        cache_it = T <= PROMPT_CACHE_T
        X = hidden_states.contiguous()
        s2m = c.s2[m]
        s2h = s2m.cpu() if not small else None
        batches = [uniq[i:i + PB] for i in range(0, len(uniq), PB)]
        # путь пар: результат каждой пары — в свою ячейку [токен, место в top-k], сумма по k в конце (порядок
        # фиксирован). index_add_ с повторами токена шёл атомиками в случайном порядке ⇒ PPL плавала 5.19–5.29 (16.10)
        Yd = torch.zeros(T, top_k_index.shape[1], Q.LAT, dtype=torch.float32, device=dev) if small else None
        nxt = c.prefill_submit(m, batches[0], 0, cache_it)
        for bi, batch in enumerate(batches):
            cur = nxt
            if bi + 1 < len(batches):
                nxt = c.prefill_submit(m, batches[bi + 1], (bi + 1) % 2, cache_it)
            tmp = StreamExperts.gpu_tmp
            for j, (row, f, r) in enumerate(cur):
                if f is not None:
                    tw = time.time(); f.result(); c.read_s += time.time() - tw
                if r is not None:
                    ev0 = c.row_ev.get(r)
                    if ev0 is not None:
                        torch.cuda.current_stream().wait_event(ev0)
                tmp[j].copy_(row, non_blocking=True)
            if c.T:                                              # обратно в исходную раскладку одним ядром (0.8 мс/32)
                if StreamExperts.t_scratch is None:
                    StreamExperts.t_scratch = torch.empty(PB * Q.LAT * Q.KB_DN, dtype=torch.uint8, device=dev)
                Q.from_T_fast(tmp[:len(cur)], StreamExperts.t_scratch)
            ev = torch.cuda.Event(); ev.record()
            for j, (row, f, r) in enumerate(cur):
                if r is not None:
                    c.row_ev[r] = ev                              # строку RAM нельзя перечитывать, пока не скопирована
            if not cache_it:
                c.pring_ev[bi % 2] = ev
            bt = torch.tensor(batch, device=dev)
            if small:
                t_i, k_i, b_i = (top_k_index[:, :, None] == bt).nonzero(as_tuple=True)
                hb, ob = StreamExperts.pair_buf
                y = Q.moe_pairs(tmp, b_i.int().contiguous(), t_i.int().contiguous(), s2m[bt[b_i]].contiguous(),
                                top_k_weights[t_i, k_i].float().contiguous(), X, hb, ob)
                Yd[t_i, k_i] = y
            else:
                for j, e in enumerate(batch):
                    tok_idx, pos = torch.where(top_k_index == e)
                    u, d = Q.block_deq(tmp[j], s2h[e])
                    hh = torch.relu(F.linear(X[tok_idx].float(), u)) ** 2
                    out.index_add_(0, tok_idx, F.linear(hh, d) * top_k_weights[tok_idx, pos, None].float())
        if small:
            out = Yd.sum(1)
        return out.to(hidden_states.dtype)



def _route_choice(gate, x):
    """очки выбора экспертов (sigmoid + bias) — как NemotronHMoE.route_tokens_to_experts, без весов"""
    return torch.sigmoid(F.linear(x.float(), gate.weight.float())) + gate.e_score_correction_bias


def moe_forward(self, hidden_states):
    """NemotronHMoE.forward: на шаге токена — свой порядок (диск летит, пока считаются общий эксперт и латент;
    предвыборка следующего MoE-слоя в пузырь), иначе — как в transformers"""
    ex = self.experts
    if hidden_states.shape[0] * hidden_states.shape[1] != 1:
        return self._orig_forward(hidden_states)
    c, m = ex.c, ex.m
    x = hidden_states.view(1, -1)
    router_logits = self.gate(x)
    ti, tw = self.route_tokens_to_experts(router_logits)
    nxt = self._next_gate is not None and c.pf > 0
    if nxt:
        pc = torch.topk(_route_choice(self._next_gate, x), self.top_k + c.pf + 4, dim=-1).indices
        both = torch.cat([ti[0], pc[0]]).tolist()               # одна синхронизация на слой
        idx, cands = both[:self.top_k], both[self.top_k:]
    else:
        idx = ti[0].tolist()
    h = c.start(m, idx)
    sh = self.shared_experts(x)                                  # пока летит диск
    lat = self.fc1_latent_proj(x)
    sidx = c.finish(h)
    if nxt:
        c.prefetch(m + 1, cands, protect=h[3])
    if c.T:
        tab = c.table(m, h[2])
        y = Q.moe_decode_groups(c.slots, sidx, tab, c.s2[m][ti[0]].contiguous(), tw[0].float().contiguous(),
                                lat[0].contiguous(), ex.h, ex.part, ex.gidx, ex.gcnt)
    else:
        y = Q.moe_decode(c.slots, sidx, c.s2[m][ti[0]].contiguous(), tw[0].float().contiguous(),
                         lat[0].contiguous(), ex.h, ex.o)
    c.done(m, h[2])
    if m > 0 and c.pf > 0:
        c.drop_pf(m)
    out = self.fc2_latent_proj(y.to(x.dtype).view(1, -1)) + sh
    return out.view(*hidden_states.shape)


class CPUEmbedding(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, ids):
        return F.embedding(ids.cpu(), self.w).to(dev, non_blocking=True)


def set_param(root, dotted, t):
    *path, leaf = dotted.split(".")
    o = root
    for p in path:
        o = getattr(o, p)
    if leaf in o._buffers:
        o._buffers[leaf] = t
    else:
        o._parameters[leaf] = nn.Parameter(t, requires_grad=False)


def load():
    T0 = time.time()
    full = json.load(open(f"{DIR}/config.json"))
    lc = dict(full["llm_config"])
    lc.pop("quantization_config", None)
    def _fix(v):                                   # NVIDIA пишет бесконечность как {"__float__": "Infinity"}
        if isinstance(v, dict) and set(v) == {"__float__"}:
            return float(v["__float__"])
        if isinstance(v, list):
            return [_fix(x) for x in v]
        return v
    lc = {k: _fix(v) for k, v in lc.items()}
    cfg = NemotronHConfig(**lc)
    tok = AutoTokenizer.from_pretrained(f"{DIR}/tok")           # без config.json обёртки Omni (иначе спрашивает про чужой код)
    with torch.device("meta"):
        model = NemotronHForCausalLM(cfg).to(torch.bfloat16)
    model.eval()
    M = model.model
    MOE = [i for i, b in enumerate(M.layers) if b.block_type == "moe"]
    ck = _PfxCkpt(Ckpt(f"{DIR}/nonexpert.safetensors"))
    st = collections.Counter()
    with torch.no_grad():
        names = [n for n, _ in model.named_parameters() if ".mixer.experts." not in n] + \
                [n for n, _ in model.named_buffers()]
        for n in names:
            c = n.replace("model.", "backbone.", 1)
            if c.startswith("backbone.embeddings"):
                continue
            base = c[: -len(".weight")] if c.endswith(".weight") else None
            if base:
                lin = model.get_submodule(n[: -len(".weight")])
                if isinstance(lin, nn.Linear) and lin.bias is None:
                    q = dense_q.load_q(ck, base, dev)
                    if q is not None:
                        dense_q.attach(lin, q); st["dense_q_" + q["fmt"]] += 1
                        continue
            assert c in ck, f"нет в чекпойнте: {c}"
            raw = ck.get(c)
            set_param(model, n, raw.to(dev, torch.float32 if raw.dtype == torch.float32 else torch.bfloat16))
            st["plain"] += 1
    emb = ck.get("backbone.embeddings.weight").to(torch.bfloat16)
    M.embeddings = CPUEmbedding(emb)
    torch.cuda.synchronize()
    print(f"постоянная часть: {dict(st)}, VRAM {torch.cuda.memory_allocated()/2**30:.2f} ГиБ, {time.time()-T0:.0f} с",
          flush=True)
    cache = ExpertCache(len(MOE), cfg.num_experts_per_tok, RAM_GB, THREADS, PF)
    import types
    for m, li in enumerate(MOE):
        mx = M.layers[li].mixer
        mx.experts = StreamExperts(m, cache)
        object.__setattr__(mx, "_orig_forward", mx.forward)
        object.__setattr__(mx, "_next_gate", M.layers[MOE[m + 1]].mixer.gate if m + 1 < len(MOE) else None)
        mx.forward = types.MethodType(moe_forward, mx)
    torch.cuda.synchronize()
    print(f"готово: VRAM {torch.cuda.memory_allocated()/2**30:.2f} ГиБ, {time.time()-T0:.0f} с", flush=True)
    return model, tok, cache, cfg
