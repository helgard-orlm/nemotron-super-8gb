"""V4: проверка службы :9809 — текст / картинка data: / другая картинка того же размера / продолжение с картинкой."""
import os, base64, io, json, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
NS_DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
from PIL import Image, ImageDraw, ImageFont
U = "http://127.0.0.1:9809/v1/chat/completions"
fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
img2 = Image.new("RGB", (768, 512), "white"); d = ImageDraw.Draw(img2)
d.regular_polygon((150, 170, 110), 5, fill=(240, 200, 0)); d.rectangle((330, 90, 690, 250), fill=(130, 0, 160))
d.text((60, 340), "GREEN FOX 12", fill="black", font=fnt); img2.save(os.path.join(HERE, "test2.png"))


def durl(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def ask(msgs, label):
    t = time.time()
    rq = urllib.request.Request(U, data=json.dumps({"model": "nemotron-3.5-super-120b", "messages": msgs,
                                                    "max_tokens": 80}).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(rq, timeout=600).read())
    a = r["choices"][0]["message"]["content"]
    print(json.dumps({"test": label, "s": round(time.time() - t, 1), "usage": r["usage"], "answer": a}, ensure_ascii=False), flush=True)
    return a


ask([{"role": "user", "content": "Сколько будет 17×23? Ответь только числом."}], "1 текст")
m1 = [{"role": "user", "content": [{"type": "text", "text": "What is written in the picture? Answer with the text only."},
                                   {"type": "image_url", "image_url": {"url": durl(os.path.join(HERE, "test1.png"))}}]}]
a1 = ask(m1, "2 картинка 1")
ask([{"role": "user", "content": [{"type": "text", "text": "What is written in the picture? Answer with the text only."},
                                  {"type": "image_url", "image_url": {"url": durl(os.path.join(HERE, "test2.png"))}}]}],
    "3 картинка 2 (тот же размер)")
ask(m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What color is the triangle?"}],
    "4 продолжение про картинку 1")
