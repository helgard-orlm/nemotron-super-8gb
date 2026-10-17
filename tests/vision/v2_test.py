"""V2: картинка в движке nem_super — векторы башни подставляются в CPUEmbedding на места <image> (id 18) по порядку.
Вопросы с известным ответом по test1.png + контроль: тот же промт с нулями вместо картинки."""
import os, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
NS_DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
sys.path.insert(0, NS_DIR); sys.path.insert(0, os.path.join(NS_DIR, "vision"))
import torch
import torch.nn.functional as F
torch.set_num_threads(10)
from PIL import Image
import vis
IMG_ID = 18
proc = vis.processor()
V = vis.Vision("cpu", torch.float32)
img = Image.open(os.path.join(HERE, "test1.png")).convert("RGB")
QS = ["What is written in the picture?", "List every shape in the picture with its color.",
      "Опиши картинку одним предложением по-русски."]
prep = []
for q in QS:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    t = time.time(); emb = V.embed(inp["pixel_values"]); tv = time.time() - t
    prep.append((q, inp["input_ids"], emb, tv))
del V
torch.set_num_threads(8)
import nem_super as NS, mamba_cont as mc
model, tok, cache, cfg = NS.load()
mc.install(model); mc.install_flash(model)
IMG = {"emb": None, "pos": 0}
E = model.model.embeddings


def emb_fwd(ids):
    ids = ids.cpu()
    e = F.embedding(ids, E.w)
    m = ids == IMG_ID
    n = int(m.sum())
    if n:
        src = IMG["emb"][IMG["pos"]:IMG["pos"] + n]
        assert src.shape[0] == n, ("векторов картинки меньше, чем токенов <image>", src.shape[0], n)
        e[m] = src.to(e.dtype); IMG["pos"] += n
    return e.to("cuda", non_blocking=True)


E.forward = emb_fwd
res = []
with torch.no_grad():
    for q, ids, emb, tv in prep:
        for mode in ("image", "zeros"):
            IMG["emb"], IMG["pos"] = (emb if mode == "image" else torch.zeros_like(emb)), 0
            t = time.time()
            c, lg = mc.prefill(model, ids.to("cuda"), None, chunk=1024)
            torch.cuda.synchronize(); tp = time.time() - t
            assert IMG["pos"] == emb.shape[0], (IMG["pos"], emb.shape[0])
            out, nxt = [], int(lg.argmax())
            t = time.time()
            for _ in range(60):
                if nxt == 11:
                    break
                out.append(nxt)
                o = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=c, use_cache=True)
                c, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
            td = time.time() - t
            r = {"q": q, "mode": mode, "prompt_tok": ids.shape[1], "vision_s": round(tv, 1), "prefill_s": round(tp, 1),
                 "tok_s": round(len(out) / max(td, 1e-6), 2), "answer": tok.decode(out, skip_special_tokens=True)}
            print(json.dumps(r, ensure_ascii=False), flush=True); res.append(r)
            o = None; del c, lg; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(HERE, "v2.json"), "w"), ensure_ascii=False, indent=1)
print("DONE")
