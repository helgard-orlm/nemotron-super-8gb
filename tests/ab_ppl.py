"""Откуда расхождение PPL EN 5.405 (test1) vs 5.141 (test2): один вход по 3 раза при разных путях промта.
  PAIR_T 0/64 — распаковка+GEMM или ядро пар; PROMPT_CACHE_T 0/512 — эксперты промта в своё кольцо или в RAM-кэш."""
import time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS

model, tok, cache, cfg = NS.load()
EN = ("The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods "
      "were produced, moving work from homes and small workshops into factories powered by water and later by steam. "
      "Cities grew rapidly as people left the countryside in search of jobs.")
ids = tok(EN, return_tensors="pt")["input_ids"].to("cuda")


@torch.no_grad()
def ppl():
    lg = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    return float(torch.exp(torch.nn.functional.cross_entropy(lg, ids[0, 1:]))), lg


ref = None
for pair_t in (0, 64):
    for pc in (0, 512):
        NS.PAIR_T, NS.PROMPT_CACHE_T = pair_t, pc
        for rep in range(3):
            t = time.time()
            p, lg = ppl()
            if ref is None:
                ref = lg
            d = (lg - ref).abs().max().item()
            print(f"PAIR_T={pair_t:3d} CACHE_T={pc:3d} повтор {rep}: PPL {p:.4f}  max|Δлогит| к первому {d:.4f}  "
                  f"{time.time()-t:.1f} с  кэш RAM занято {len(cache.where)}", flush=True)
print("DONE", flush=True)
