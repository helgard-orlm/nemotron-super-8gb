"""Пункт 1 в движке (раскладка T): (1) PPL EN путём промта == 5.2801 (до перестановки); (2) шаг генерации —
логиты по шагам против detc_A.pt (до перестановки, те же настройки, кэш autotune); (3) скорость 3 вопроса × 64."""
import json, os, sys, time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS

model, tok, cache, cfg = NS.load()
assert cache.T, "раскладка не T"
dev = "cuda"
EN = ("The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods "
      "were produced, moving work from homes and small workshops into factories powered by water and later by steam. "
      "Cities grew rapidly as people left the countryside in search of jobs.")
with torch.no_grad():
    ids = tok(EN, return_tensors="pt")["input_ids"].to(dev)
    lg = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    print(f"PPL EN (промт, T): {float(torch.exp(torch.nn.functional.cross_entropy(lg, ids[0, 1:]))):.4f}  (до: 5.2801)", flush=True)


def chat(q):
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)


@torch.no_grad()
def run(q, new):
    ids = chat(q)
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv, lg = o.past_key_values, o.logits[0, -1].float()
    out, L = [], []
    torch.cuda.synchronize(); t = time.time()
    for _ in range(new):
        x = int(lg.argmax()); out.append(x); L.append(lg.cpu())
        o = model(input_ids=torch.tensor([[x]], device=dev), past_key_values=pkv, use_cache=True)
        pkv, lg = o.past_key_values, o.logits[0, -1].float()
    torch.cuda.synchronize()
    return out, torch.stack(L), (time.time() - t) / new


ref = torch.load("detc_A.pt")
o_ids, L, _ = run("Explain the difference between TCP and UDP and give one use case for each.", 48)
dl = (L - ref["L"]).abs().amax(-1)
print(f"ШАГ vs до перестановки: токены одинаковы {o_ids == ref['ids']}, макс Δлогит {float(dl.max()):.4f}, "
      f"шаг0 {float(dl[0]):.4f}", flush=True)
fd = next((i for i in range(len(dl)) if dl[i] > 1e-3), None)
ft = next((i for i in range(len(o_ids)) if o_ids[i] != ref["ids"][i]), None)
print(f"  первый шаг Δ>1e-3: {fd} (Δ {float(dl[fd]) if fd is not None else 0:.4f}); Δ по шагам до него: "
      f"{[round(float(v), 5) for v in dl[:(fd or 0) + 1]]}", flush=True)
if ft is not None:
    r = ref["L"][ft]
    a1, b1 = o_ids[ft], ref["ids"][ft]
    print(f"  первый другой токен на шаге {ft}: новое {tok.decode([a1])!r} vs было {tok.decode([b1])!r}; "
          f"зазор в старых логитах {float(r[b1] - r[a1]):.4f}, в новых {float(L[ft][a1] - L[ft][b1]):.4f}; "
          f"текст до: {tok.decode(o_ids[:ft])[-60:]!r}", flush=True)
if "--only-step" in sys.argv:
    sys.exit(0)
for q in ["Explain the difference between TCP and UDP and give one use case for each.",
          "Write a Python function that merges two sorted lists into one sorted list.",
          "Объясни простыми словами, почему небо голубое, а закат красный."]:
    cache.stat.clear(); cache.read_s = 0.0
    _, _, dt = run(q, 64)
    st = dict(cache.stat); hit = st.get("ram_hit", 0) + st.get("pf_hit", 0) + st.get("late_hit", 0)
    print(f"{q[:30]}: {1/dt:.2f} ток/с ({dt*1e3:.0f} мс/сл), попадания RAM {hit/max(1, hit+st.get('disk', 0)):.2f}", flush=True)
print("DONE", flush=True)
