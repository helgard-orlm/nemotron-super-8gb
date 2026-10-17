"""детальный режим: test1 (здравый смысл) + счёт-фактура doc1 быстрым и -doc: строки 21–30, сверка с doc1_truth.json."""
import os, base64, json, re, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
NS_DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
U = "http://127.0.0.1:9809/v1/chat/completions"
truth = {r["n"]: r for r in json.load(open(os.path.join(HERE, "doc1_truth.json")))}


def durl(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def ask(model, img, q, mt=500):
    t = time.time()
    body = {"model": model, "max_tokens": mt, "messages": [{"role": "user", "content": [
        {"type": "text", "text": q}, {"type": "image_url", "image_url": {"url": durl(img)}}]}]}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(body).encode(),
                                                                 headers={"Content-Type": "application/json"}), timeout=1800).read())
    return r["choices"][0]["message"]["content"], r["usage"], round(time.time() - t, 1)


a, u, s = ask("nemotron-3.5-super-120b", os.path.join(HERE, "test1.png"), "What is written in the picture? Answer with the text only.", 30)
print(json.dumps({"test": "test1 быстрый", "s": s, "usage": u, "answer": a}, ensure_ascii=False), flush=True)
Q = ("Transcribe rows 21 to 30 of the table exactly, one row per line, in the format: "
     "NN | part code | qty | price. Output only these 10 lines.")
for model in ("nemotron-3.5-super-120b", "nemotron-3.5-super-120b-doc"):
    a, u, s = ask(model, os.path.join(HERE, "doc1.png"), Q)
    ok = {"code": 0, "qty": 0, "price": 0, "rows": 0}
    for line in a.splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) < 4 or not re.match(r"^\d+$", f[0]):
            continue
        n = int(f[0])
        if n not in truth or not 21 <= n <= 30:
            continue
        t = truth[n]; ok["rows"] += 1
        ok["code"] += f[1].replace("Part code", "").strip() == t["code"]
        ok["qty"] += f[2].replace("qty", "").strip() == str(t["qty"])
        ok["price"] += f[3].replace("price", "").strip() == t["price"]
    print(json.dumps({"test": "doc1 " + model, "s": s, "usage": u, "correct_of_10": ok, "answer": a}, ensure_ascii=False), flush=True)
