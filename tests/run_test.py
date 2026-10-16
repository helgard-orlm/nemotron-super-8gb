"""Первый прогон движка: (1) перплексия EN/RU учительским прогоном (путь промта) + контроль «полубайты наоборот»;
(2) «целиком vs шагами»: логиты последнего токена из прогона целиком и из префикс+шаги (шаги = ядро qmoe_ns + кэш);
(3) жадный ответ на 2 вопроса, скорость, попадания кэша.  python -u run_test.py [--new 48]"""
import argparse, json, time
import torch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import nem_super as NS
import qmoe_ns as Q

ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=48)
ap.add_argument("--skip_ppl", action="store_true")
args = ap.parse_args()
model, tok, cache, cfg = NS.load()
dev = "cuda"
EN = ("The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods "
      "were produced, moving work from homes and small workshops into factories powered by water and later by steam. "
      "Cities grew rapidly as people left the countryside in search of jobs.")
RU = ("Байкал — самое глубокое озеро на планете. Его максимальная глубина превышает полторы тысячи метров, а в "
      "озере сосредоточено около двадцати процентов мировых запасов пресной воды. Здесь обитают сотни видов, "
      "которые больше нигде не встречаются.")


@torch.no_grad()
def ppl(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to(dev)
    t = time.time()
    lg = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    nll = torch.nn.functional.cross_entropy(lg, ids[0, 1:])
    return float(torch.exp(nll)), ids.shape[1], time.time() - t


res = {}
if not args.skip_ppl:
    for name, text in (("EN", EN), ("RU", RU)):
        p, n, dt = ppl(text)
        res[f"ppl_{name}"] = round(p, 3)
        print(f"PPL {name}: {p:.3f} ({n} ток, {dt:.1f} с), кэш {dict(cache.stat)}", flush=True)
    Q.SWAP = True
    p, n, dt = ppl(EN)
    Q.SWAP = False
    res["ppl_EN_swap_control"] = round(p, 1)
    print(f"PPL EN с полубайтами наоборот (должна взлететь): {p:.1f}", flush=True)


def chat(q):
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)


# (2) целиком vs шагами
with torch.no_grad():
    ids = chat("What is the capital of France? Answer in one word.")
    L = ids.shape[1]
    lf = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    o = model(input_ids=ids[:, :L - 3], use_cache=True)
    pkv = o.past_key_values
    for t in range(L - 3, L):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
    ls = o.logits[0, -1].float()
    d = (lf - ls).abs().max().item()
    res["full_vs_steps"] = {"max_abs_dlogit": round(d, 4), "top1_full": tok.decode(lf.argmax()),
                            "top1_steps": tok.decode(ls.argmax()),
                            "top5_full": [tok.decode(i) for i in lf.topk(5).indices],
                            "top5_steps": [tok.decode(i) for i in ls.topk(5).indices]}
    print("ЦЕЛИКОМ vs ШАГАМИ:", json.dumps(res["full_vs_steps"], ensure_ascii=False), flush=True)

# (3) ответы + скорость
STOP = {tok.eos_token_id} | set(tok.all_special_ids)
runs = []
for q in ["What is the capital of France? Answer in one word.",
          "Explain in two sentences why the sky is blue.",
          "Объясни в двух предложениях, почему зимой холодно."]:
    ids = chat(q)
    cache.stat.clear(); cache.read_s = 0.0
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        o = model.generate(input_ids=ids, max_new_tokens=args.new, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t
    g = o[0, ids.shape[1]:].tolist()
    cut = next((j for j, x in enumerate(g) if x in STOP), len(g))
    r = {"q": q, "answer": tok.decode(g[:cut]), "tokens": len(g), "sec": round(dt, 1), "tok_s": round(len(g) / dt, 2),
         "read_s": round(cache.read_s, 1), "stat": dict(cache.stat)}
    runs.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
res["runs"] = runs
res["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print("ПИК VRAM", res["peak_vram_GiB"], flush=True)
json.dump(res, open("run_test.json", "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
