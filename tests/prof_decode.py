"""Куда уходит время шага и что дадут (1) пропуск нулей relu² в down, (2) кэш экспертов на карте.
  (а) доля живых нейронов h>0 после relu² по слоям (routed; из буфера ядра ex.h) — и по группам 16 (масштаб NVFP4);
  (б) повторы: эксперт слоя m нужен снова через 1, 2, 4, 8 слов; попадания LRU на карте при C мест/слой;
  (в) раскладка времени шага: обычный / без копии на карту (числа неверны, только время) / без диска и без копии.
  python -u prof_decode.py"""
import collections, json, time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS
import qmoe_ns as Q

model, tok, cache, cfg = NS.load()
dev = "cuda"
Qs = ["Explain the difference between TCP and UDP and give one use case for each.",
      "Объясни простыми словами, почему небо голубое, а закат красный.",
      "Write a Python function that merges two sorted lists into one sorted list."]
NEW = 48

live = collections.defaultdict(list)          # m -> доли живых
live16 = collections.defaultdict(list)        # m -> доли групп по 16, где есть живой
hist = []                                     # шаг -> {m: set(экспертов)}
CUR = {"step": None}
_orig_done = cache.done


def done_hook(m):
    if CUR["step"] is not None:
        mx = MX[m]
        h = mx.experts.h[:22]
        nz = (h > 0)
        live[m].append(float(nz.float().mean()))
        live16[m].append(float(nz.view(22, -1, 16).any(-1).float().mean()))
    _orig_done(m)


cache.done = done_hook
MOE = [i for i, b in enumerate(model.model.layers) if b.block_type == "moe"]
MX = [model.model.layers[i].mixer for i in MOE]
_orig_start = cache.start


def start_hook(m, experts):
    if CUR["step"] is not None:
        hist[-1].setdefault(m, set()).update(experts)
    return _orig_start(m, experts)


cache.start = start_hook


def chat(q):
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)


@torch.no_grad()
def gen(q, record):
    ids = chat(q)
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
    torch.cuda.synchronize(); t = time.time()
    for s in range(NEW):
        if record:
            CUR["step"] = s; hist.append({})
        o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=pkv, use_cache=True)
        pkv, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
    torch.cuda.synchronize()
    CUR["step"] = None
    return (time.time() - t) / NEW


res = {}
gen("Say hello.", False)
for q in Qs:
    gen(q, True)                                   # проход статистики (синхронизации искажают время)
cache.stat.clear(); cache.read_s = 0.0
ms = [gen(q, False) for q in Qs]                  # проход времени
res["normal_ms_per_tok"] = [round(x * 1e3, 1) for x in ms]
res["normal_read_wait_ms_per_tok"] = round(cache.read_s / (NEW * len(Qs)) * 1e3, 1)
res["stat"] = dict(cache.stat)

# (а) живые
lv = {m: sum(v) / len(v) for m, v in live.items()}
lv16 = {m: sum(v) / len(v) for m, v in live16.items()}
res["live_frac_mean"] = round(sum(lv.values()) / len(lv), 4)
res["live_frac_by_layer"] = {m: round(v, 3) for m, v in sorted(lv.items())}
res["live16_group_frac_mean"] = round(sum(lv16.values()) / len(lv16), 4)

# (б) повторы и LRU на карте (по истории шагов; границы вопросов игнорируем — 3×48 шагов подряд)
rep = {}
for d in (1, 2, 4, 8):
    hit = tot = 0
    for s in range(d, len(hist)):
        for m, es in hist[s].items():
            prev = hist[s - d].get(m, set())
            hit += len(es & prev); tot += len(es)
    rep[d] = round(hit / max(1, tot), 3)
res["reuse_after_d_steps"] = rep
lru = {}
for C in (2, 3, 4, 8, 16, 32):
    hit = tot = 0
    for m in range(len(MOE)):
        od = collections.OrderedDict()
        for s in range(len(hist)):
            es = hist[s].get(m, set())
            for e in es:
                tot += 1
                if e in od:
                    hit += 1; od.move_to_end(e)
                else:
                    od[e] = None
                    while len(od) > C + 22:          # C мест сверх 22 рабочих слотов слоя
                        od.popitem(last=False)
    lru[C] = {"hit": round(hit / max(1, tot), 3), "VRAM_GiB": round((C + 22) * len(MOE) * Q.BLOCK / 2**30, 2)}
res["gpu_lru_hit_by_extra_slots_per_layer"] = lru

# (в) раскладка времени: без копии / без диска и без копии
_copy = cache._copy
cache._copy = lambda slot, r: None
cache.stat.clear(); cache.read_s = 0.0
res["nocopy_ms_per_tok"] = round(sum(gen(q, False) for q in Qs) / len(Qs) * 1e3, 1)
_read = cache._read
cache._read = lambda r, k: None
cache.prefetch = lambda *a, **k: None
res["nocopy_nodisk_ms_per_tok"] = round(sum(gen(q, False) for q in Qs) / len(Qs) * 1e3, 1)
cache._copy = _copy
res["copy_nodisk_ms_per_tok"] = round(sum(gen(q, False) for q in Qs) / len(Qs) * 1e3, 1)
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
json.dump(res, open("prof_decode.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
