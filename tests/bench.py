"""Замер: промт отдельно, генерация отдельно (свой жадный цикл), попадания кэша. Токены ответа -> bench_<tag>.json
для сравнения вариантов (предвыборка не должна менять ни одного токена).  NS_PF=0|2 python -u bench.py tag"""
import json, os, sys, time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS

tag = sys.argv[1] if len(sys.argv) > 1 else "x"
NEW = int(os.environ.get("NB_NEW", "64"))
model, tok, cache, cfg = NS.load()
dev = "cuda"
STOP = {2, 11}
Q = ["Explain the difference between TCP and UDP and give one use case for each.",
     "Write a Python function that merges two sorted lists into one sorted list.",
     "Объясни простыми словами, почему небо голубое, а закат красный."]


def chat(q):
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)


@torch.no_grad()
def run(q):
    ids = chat(q)
    cache.stat.clear(); cache.read_s = 0.0
    torch.cuda.synchronize(); t0 = time.time()
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv = o.past_key_values
    nxt = int(o.logits[0, -1].argmax())
    torch.cuda.synchronize(); t1 = time.time()
    pst = dict(cache.stat); cache.stat.clear(); cache.read_s = 0.0
    out = [nxt]
    while len(out) < NEW and nxt not in STOP:
        o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out.append(nxt)
    torch.cuda.synchronize(); t2 = time.time()
    st = dict(cache.stat)
    hit = st.get("ram_hit", 0) + st.get("pf_hit", 0) + st.get("late_hit", 0)
    tot = hit + st.get("disk", 0)
    r = {"q": q[:40], "prompt_tok": ids.shape[1], "prefill_s": round(t1 - t0, 2), "gen_tok": len(out),
         "gen_s": round(t2 - t1, 2), "gen_tok_s": round((len(out) - 1) / (t2 - t1), 2),
         "read_wait_s": round(cache.read_s, 2), "hit": round(hit / max(1, tot), 3), "stat": st, "prefill_stat": pst,
         "text": tok.decode(out)[:160], "ids": out}
    print(json.dumps({k: v for k, v in r.items() if k != "ids"}, ensure_ascii=False), flush=True)
    return r


run("Say hello.")                                       # прогрев (Triton, кэш)
res = [run(q) for q in Q]
res2 = [run(q) for q in Q]                              # второй круг: тёплый кэш
json.dump({"pf": NS.PF, "ram_gb": NS.RAM_GB, "cold": res, "warm": res2,
           "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30}, open(f"bench_{tag}.json", "w"),
          ensure_ascii=False, indent=1)
print("DONE", flush=True)
