"""V1: башня+проектор на CPU — загрузка (все ключи на месте), процессор, время на картинку, форма выхода."""
import os, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
NS_DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
sys.path.insert(0, os.path.join(NS_DIR, "vision"))
import torch
torch.set_num_threads(10)
from PIL import Image, ImageDraw, ImageFont
import vis
img = Image.new("RGB", (768, 512), "white")
d = ImageDraw.Draw(img)
d.ellipse((40, 60, 240, 260), fill=(220, 30, 30))
d.rectangle((300, 80, 480, 260), fill=(30, 160, 60))
d.polygon([(600, 60), (700, 260), (500, 260)], fill=(30, 60, 220))
try:
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
except Exception:
    fnt = None
d.text((60, 340), "BLUE CRANE 47", fill="black", font=fnt)
img.save(os.path.join(HERE, "test1.png"))
res = {}
proc = vis.processor()
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is written in the picture?"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inp = proc(text=[text], images=[img], return_tensors="pt")
res["pixel_values"] = list(inp["pixel_values"].shape) if hasattr(inp["pixel_values"], "shape") else str(type(inp["pixel_values"]))
ids = inp["input_ids"][0]
res["prompt_tokens"] = len(ids); res["image_tokens"] = int((ids == 18).sum())
res["text_head"] = text[:400]
for dt in (torch.float32, torch.bfloat16):
    v = vis.Vision("cpu", dt)
    res[f"load_{dt}"] = {"s": round(v.load_s, 1), **{k: {"n": r["n"], "missing": r["missing"][:8], "unexpected": r["unexpected"][:8]}
                                                      for k, r in v.load_report.items()}}
    v.embed(inp["pixel_values"])
    t = time.time(); e = v.embed(inp["pixel_values"]); res[f"embed_{dt}_s"] = round(time.time() - t, 2)
    res[f"embed_{dt}_shape"] = list(e.shape)
    torch.save(e.float(), os.path.join(HERE, f"emb_test1_{str(dt).split('.')[-1]}.pt"))
    if dt == torch.float32:
        e32 = e.float()
    else:
        res["bf16_vs_fp32_rel"] = float((e.float() - e32).norm() / e32.norm())
    del v
torch.save({"input_ids": ids, "pixel_values": inp["pixel_values"]}, os.path.join(HERE, "inp_test1.pt"))
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
