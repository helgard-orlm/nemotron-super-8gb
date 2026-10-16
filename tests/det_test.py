"""Гонка на шаге генерации? Один вопрос жадно 48 слов, 3 повтора в одном процессе (кэш RAM каждый раз в другом
состоянии) + вариант без предвыборки. Логиты по шагам сравниваются с первым прогоном; заодно режим «синхронно»
(после каждого MoE-слоя torch.cuda.synchronize + ожидание всех чтений) — если с ним совпадает, а без него нет, то гонка."""
import json, os, sys, time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS

model, tok, cache, cfg = NS.load()
dev = "cuda"
q = "Explain the difference between TCP and UDP and give one use case for each."
c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
ids = tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
NEW = 48
SYNC = {"on": False}
_orig_done = cache.done


def done_sync(m):
    _orig_done(m)
    if SYNC["on"]:
        torch.cuda.synchronize()
        with cache.lock:
            fs = [f for _, f in cache.pend.values()] + list(cache.inflight.values())
        for f in fs:
            f.result()
        torch.cuda.synchronize()


cache.done = done_sync


@torch.no_grad()
def run():
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv, lg = o.past_key_values, o.logits[0, -1].float()
    out, L = [], []
    for _ in range(NEW):
        t = int(lg.argmax()); out.append(t); L.append(lg.cpu())
        o = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=pkv, use_cache=True)
        pkv, lg = o.past_key_values, o.logits[0, -1].float()
    return out, torch.stack(L)


ref_ids, ref_L = None, None
RUNS = [("pf4 #1", 4, False)] if len(sys.argv) > 1 else [("pf4 #1", 4, False), ("pf4 #2", 4, False), ("pf4 #3", 4, False),
        ("pf0", 0, False), ("pf4 синхр", 4, True), ("pf4 #4", 4, False)]
for name, pf, sync in RUNS:
    cache.pf, SYNC["on"] = pf, sync
    t = time.time()
    o_ids, L = run()
    if ref_ids is None:
        ref_ids, ref_L = o_ids, L
    n = min(len(L), len(ref_L))
    # первый шаг расхождения логитов (> 1e-3) и первый шаг расхождения токена
    dl = (L[:n] - ref_L[:n]).abs().amax(-1)
    first_dl = next((i for i in range(n) if dl[i] > 1e-3), None)
    first_tok = next((i for i in range(n) if o_ids[i] != ref_ids[i]), None)
    print(f"{name:10s} {time.time()-t:5.1f} с  токены==#1: {o_ids == ref_ids}  первый шаг Δлогит>1e-3: {first_dl}  "
          f"(Δ там {float(dl[first_dl]) if first_dl is not None else 0:.4f})  первый другой токен: {first_tok}  "
          f"кэш {dict(cache.stat)}", flush=True)
    cache.stat.clear()
if len(sys.argv) > 1:
    torch.save({"ids": ref_ids, "L": ref_L}, sys.argv[1])
print("DONE", flush=True)
