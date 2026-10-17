"""Документ с мелким текстом: башня на CPU в быстром (4096) и детальном (13312) режимах — патчи, время, пик RAM."""
import os, json, random, resource, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
NS_DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
sys.path.insert(0, os.path.join(NS_DIR, "vision"))
import torch
torch.set_num_threads(10)
from PIL import Image, ImageDraw, ImageFont
random.seed(7)
W, H = 1240, 1754
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
d.text((60, 50), "INVOICE No. 2026-10-4417", fill="black", font=fb)
rows = []
y = 130
for i in range(1, 61):
    code = f"{random.choice('ABCDEFGHKMNPQRSTX')}{random.choice('ABCDEFGHKMNPQRSTX')}-{random.randint(1000, 9999)}-{random.choice('ABCDEFGH')}"
    qty = random.randint(1, 99); price = f"{random.randint(1, 9999)}.{random.randint(0, 99):02d}"
    rows.append({"n": i, "code": code, "qty": qty, "price": price})
    d.text((60, y), f"{i:02d}", fill="black", font=fs); d.text((120, y), f"Part code {code}", fill="black", font=fs)
    d.text((620, y), f"qty {qty}", fill="black", font=fs); d.text((860, y), f"price {price}", fill="black", font=fs)
    y += 26
img.save(os.path.join(HERE, "doc1.png"))
json.dump(rows, open(os.path.join(HERE, "doc1_truth.json"), "w"), indent=0)
import vis
proc = vis.processor()
v = vis.Vision("cpu", torch.bfloat16)
res = {"attn_impl": getattr(v.vm.config, "_attn_implementation", None)}
for name, mp in (("fast", 4096), ("detail", 13312)):
    proc.image_processor.max_num_patches = mp
    pv = proc.image_processor(images=[img], return_tensors="pt")["pixel_values"]
    t = time.time(); e = v.embed(pv); dt = time.time() - t
    res[name] = {"pixel_shape": list(pv.shape), "image_tokens": e.shape[0], "tower_s": round(dt, 1),
                 "peak_rss_GiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print(name, res[name], flush=True)
print(json.dumps(res, ensure_ascii=False))
