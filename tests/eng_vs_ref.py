"""Шаг 1 (mkseq): последовательности для эталона: en (текст PPL) + chat_tcp / chat_ru (промт + жадный ответ движка из
bench_pf2.json). Шаг 2 (engine): логиты движка целиком на тех же токенах -> eng_logits.pt. Шаг 3 (cmp): сравнение.
  python eng_vs_ref.py mkseq | engine | cmp"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

mode = sys.argv[1]
if mode == "mkseq":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(os.environ.get("NS_DIR", "/fast/nemsuper"), "tok"))
    EN = ("The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods "
          "were produced, moving work from homes and small workshops into factories powered by water and later by steam. "
          "Cities grew rapidly as people left the countryside in search of jobs.")
    b = json.load(open("bench_pf2.json"))["warm"]
    Q = {"chat_tcp": "Explain the difference between TCP and UDP and give one use case for each.",
         "chat_ru": "Объясни простыми словами, почему небо голубое, а закат красный."}
    seq, meta = {"en": tok(EN)["input_ids"]}, {}
    for name, q in Q.items():
        c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        p = tok(c, add_special_tokens=False)["input_ids"]
        r = next(x for x in b if x["q"] == q[:40])
        assert r["prompt_tok"] == len(p), (r["prompt_tok"], len(p))
        seq[name] = p + r["ids"][:-1]                 # последний сгенерированный не подавался
        meta[name] = {"prompt": len(p), "gen": r["ids"]}
    json.dump(seq, open("ref_seq.json", "w")); json.dump(meta, open("ref_meta.json", "w"))
    print({k: len(v) for k, v in seq.items()})
elif mode == "engine":
    import nem_super as NS
    model, tok, cache, cfg = NS.load()
    seq = json.load(open("ref_seq.json"))
    out = {}
    with torch.no_grad():
        for name, ids in seq.items():
            h = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state
            out[name] = torch.cat([model.lm_head(h[:, i:i + 16]).float().cpu() for i in range(0, h.shape[1], 16)], 1)[0]
            print(name, len(ids), flush=True)
    torch.save(out, "eng_logits.pt")
    print("DONE", flush=True)
else:
    seq = json.load(open("ref_seq.json")); meta = json.load(open("ref_meta.json"))
    R = torch.load("ref_logits.pt"); E = torch.load("eng_logits.pt")
    res = {}
    for name, ids in seq.items():
        r, e = R[name].float(), E[name].float()
        tgt = torch.tensor(ids[1:])
        pr = float(torch.exp(torch.nn.functional.cross_entropy(r[:-1], tgt)))
        pe = float(torch.exp(torch.nn.functional.cross_entropy(e[:-1], tgt)))
        d = (r - e).abs()
        row = {"T": len(ids), "ppl_ref": round(pr, 4), "ppl_eng": round(pe, 4),
               "argmax_same": f"{int((r.argmax(-1) == e.argmax(-1)).sum())}/{len(ids)}",
               "max_abs_dlogit": round(float(d.max()), 3), "mean_abs_dlogit": round(float(d.mean()), 4),
               "cos_min": round(float(torch.nn.functional.cosine_similarity(r, e, dim=-1).min()), 5)}
        if name in meta:
            p, g = meta[name]["prompt"], meta[name]["gen"]
            ra = r[p - 1:p - 1 + len(g)].argmax(-1).tolist()
            row["ref_argmax_eq_engine_answer"] = f"{sum(int(a == b) for a, b in zip(ra, g))}/{len(g)}"
            gaps = []
            for j, (a, b) in enumerate(zip(ra, g)):
                if a != b:
                    lg = r[p - 1 + j]
                    gaps.append(round(float(lg[a] - lg[b]), 3))       # зазор эталона между его выбором и нашим
            row["ref_gap_where_differ"] = gaps
        res[name] = row
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(res, open("ref_cmp.json", "w"), ensure_ascii=False, indent=1)
