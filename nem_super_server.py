#!/usr/bin/env python3
"""Nemotron-3.5-Super-VL-120B-A12B (свой движок nem_super.py, эксперты с /fast) — служба с API как у OpenAI
(/v1/models, /v1/chat/completions со стримом) для Open WebUI. Заказ helgard 16.10.2026 «делаем свой движок», «1 и 2».
Образец — qwen_server.py (Qwen3.8). Отличия:
  - состояние = кэш transformers (KV 8 слоёв внимания + состояния 40 слоёв Mamba); продолжение пачкой поверх кэша —
    mamba_cont (как у Nano); снимок = копия кэша в RAM (~170 МБ) ⇒ продолжение диалога дочитывает только хвост;
  - промт пачкой (кусками), ответ — по одному слову; жадно, если temperature не задана;
  - две модели: обычная и «-think» (<think>…</think>, Open WebUI сворачивает);
  - только текст (зрение и MTP не подключены).
Карта занята почти целиком (~7 ГиБ) ⇒ запуск из панели gpusvcs, по очереди с qwen/deepseek/nemotron-30b."""
import collections, json, os, re, sys, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("NS_PORT", "9809"))
MAX_SEQ = int(os.environ.get("NS_MAX_SEQ", "32768"))
DEFAULT_MAX_TOKENS = int(os.environ.get("NS_MAX_TOKENS", "2048"))
THINK_MAX_TOKENS = int(os.environ.get("NS_THINK_MAX_TOKENS", "8192"))
PREFILL_CHUNK = int(os.environ.get("NS_PREFILL_CHUNK", "1024"))
SNAP_N = int(os.environ.get("NS_SNAP_N", "6"))
MODEL_ID = os.environ.get("NS_MODEL_ID", "nemotron-3.5-super-120b")
THINK_ID = MODEL_ID + "-think"
STOPS = {2, 11}                                  # </s>, <|im_end|>

STATE = {"status": "loading", "error": None, "since": time.time(), "served": 0}
ENGINE = {}
GEN_LOCK = threading.Lock()
SNAPS = collections.OrderedDict()                # tuple токенов -> снимок кэша (RAM)
LIVE = {"fed": None, "cache": None}              # какие токены сейчас в кэше на карте


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_ollama():
    """модели ollama делят ту же карту 8 ГБ — выгружаем перед загрузкой"""
    try:
        ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3).read())
        for m in ps.get("models", []):
            rq = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                        data=json.dumps({"model": m["name"], "keep_alive": 0}).encode(),
                                        headers={"Content-Type": "application/json"})
            urllib.request.urlopen(rq, timeout=30).read()
            log("ollama выгружена:", m["name"])
    except Exception as e:
        log("ollama: пропуск выгрузки:", e)


def load_engine():
    try:
        free_ollama()
        sys.path.insert(0, HERE)
        import torch
        torch.set_num_threads(8)
        import nem_super as NS
        import mamba_cont as mc
        model, tok, cache, cfg = NS.load()
        mc.install(model)
        try:
            mc.install_flash(model)
            log("внимание: FlashAttention-2 (GQA без размножения KV)")
        except Exception as e:
            log("FlashAttention-2 не поставлен, остаётся sdpa:", e)
        ENGINE.update(model=model, tok=tok, cache=cache, torch=torch, mc=mc, NS=NS, im_start=10)
        STATE["status"] = "ready"
        log(f"готово: VRAM {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ, RAM-кэш {NS.RAM_GB} ГБ, P={NS.PF}, "
            f"max_seq {MAX_SEQ}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE.update(status="error", error=repr(e))


def conv_msg(m):
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    role = m.get("role", "user")
    if role not in ("system", "user", "assistant"):
        role = "user"
    c = c or ""
    if role == "assistant" and "</think>" in c:                       # прошлое рассуждение в контекст не кладём
        c = c.split("</think>", 1)[1].lstrip("\n")
    if role == "assistant" and "<details" in c and "</details>" in c:  # так Open WebUI может вернуть свёрнутое рассуждение
        c = re.sub(r"<details[^>]*>.*?</details>\s*", "", c, flags=re.S)
    return {"role": role, "content": c}


def snap_take(fed, cache):
    mc = ENGINE["mc"]
    SNAPS[fed] = mc.snapshot(cache); SNAPS.move_to_end(fed)
    while len(SNAPS) > SNAP_N:
        SNAPS.popitem(last=False)


def pick(lg, samp, gen):
    """samp None — жадно; иначе (T, top_k, top_p)"""
    torch = ENGINE["torch"]
    if samp is None:
        return int(lg.argmax())
    T, k, p = samp
    v, i = torch.topk(lg.float() / max(T, 1e-5), k)
    pr = torch.softmax(v, -1)
    cum = pr.cumsum(-1)
    pr = torch.where(cum - pr > p, torch.zeros_like(pr), pr)          # top_p (первое слово остаётся всегда)
    return int(i[torch.multinomial(pr / pr.sum(), 1, generator=gen)])


def generate(messages, max_tokens, on_text, cancelled, think=False, samp=None):
    torch, model, tok, mc = ENGINE["torch"], ENGINE["model"], ENGINE["tok"], ENGINE["mc"]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think, tokenize=False)
    gen = torch.Generator(device="cuda"); gen.manual_seed(int(time.time() * 1000) % 2**31)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) + max_tokens > MAX_SEQ:
        max_tokens = MAX_SEQ - len(ids)
        if max_tokens < 16:
            on_text(f"[диалог слишком длинный: {len(ids)} токенов при пределе {MAX_SEQ} — начните новый чат]")
            return 0, len(ids), 0.0, 0.0
    g, shown = [], ""
    with torch.inference_mode():
        torch.cuda.synchronize(); t0 = time.time()
        start, src, cache = 0, "с нуля", None
        if not STATE.get("nocache"):
            best = None
            for fed in [LIVE["fed"]] + list(reversed(list(SNAPS))):
                if fed and len(fed) < len(ids) and tuple(ids[:len(fed)]) == tuple(fed) and \
                        (best is None or len(fed) > len(best)):
                    best = fed
            if best is not None:
                cache = LIVE["cache"] if best == LIVE["fed"] else mc.restore(SNAPS[best])
                start, src = len(best), f"продолжение {len(best)} ток + хвост {len(ids) - len(best)}"
        LIVE["fed"], LIVE["cache"] = None, None
        # снимок на начале последнего «<|im_start|>» (заголовок ответа): этот префикс совпадает с тем, как следующий
        # запрос отрисует историю в любом режиме (рассуждения прошлых ответов шаблон вырезает)
        im = ENGINE["im_start"]
        cut = max((i for i, t in enumerate(ids) if t == im), default=0)
        try:
            lg = None
            spans = [(start, cut), (cut, len(ids))] if start < cut else [(start, len(ids))]
            for a0, b0 in spans:
                for a in range(a0, b0, PREFILL_CHUNK):
                    piece = torch.tensor([ids[a:min(a + PREFILL_CHUNK, b0)]], device="cuda")
                    cache, lg = mc.prefill(model, piece, cache, chunk=PREFILL_CHUNK)
                if b0 == cut and cut > 0:
                    snap_take(tuple(ids[:cut]), cache)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"промт {len(ids)} ток: нехватка видеопамяти ({src})")
            on_text("[Nemotron: не хватило видеопамяти на чтение промта — сократите сообщение или начните новый чат]")
            return 0, len(ids), 0.0, 0.0
        torch.cuda.synchronize(); t_pre = time.time() - t0
        log(f"промт {len(ids)} ток: {src}, {t_pre:.1f} с" + (", думает" if think else ""))
        t1 = time.time()
        if think:
            on_text("<think>\n")                                       # шаблон открыл <think> внутри промта
        fed = list(ids)
        while True:
            t = pick(lg.float(), samp, gen)
            if t in STOPS:
                break
            g.append(t)
            s = tok.decode(g, skip_special_tokens=True)
            if not s.endswith("�") and len(s) > len(shown):          # не резать многобайтный символ пополам
                on_text(s[len(shown):]); shown = s
            if len(g) >= max_tokens or cancelled():
                break
            o = model(input_ids=torch.tensor([[t]], device="cuda"), past_key_values=cache, use_cache=True)
            cache, lg = o.past_key_values, o.logits[0, -1]
            fed.append(t)
        s = tok.decode(g, skip_special_tokens=True)
        if len(s) > len(shown):
            on_text(s[len(shown):])
        # в кэше — ids + g без последнего (последний ещё не подан); шаблон добавит <|im_end|> — его дочитает хвост
        LIVE["fed"], LIVE["cache"] = tuple(fed), cache
    return len(g), len(ids), t_pre, time.time() - t1


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            pass

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/health"):
            st = {**STATE, "uptime_s": round(time.time() - STATE["since"]), "busy": GEN_LOCK.locked()}
            if ENGINE.get("cache") is not None:
                st["experts"] = dict(ENGINE["cache"].stat)
            return self._json(200 if STATE["status"] == "ready" else 503, st)
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "gpu104", "created": 0} for m in (MODEL_ID, THINK_ID)]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        waited = 0
        while STATE["status"] == "loading" and waited < 600:
            time.sleep(1); waited += 1
        if STATE["status"] != "ready":
            return self._json(503, {"error": {"message": f"модель не готова: {STATE['status']} {STATE['error'] or ''}"}})
        msgs = [conv_msg(m) for m in req.get("messages", [])]
        STATE["nocache"] = bool(req.get("ns_nocache"))
        model_id = req.get("model") or MODEL_ID
        think = model_id.endswith("-think")
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or
                         (THINK_MAX_TOKENS if think else DEFAULT_MAX_TOKENS))
        temp = req.get("temperature")
        if think and temp is None:
            samp = (1.0, 50, 0.95)                         # generation_config авторов: T=1.0, top_p 0.95
        elif temp is not None and float(temp) > 0:
            samp = (float(temp), int(req.get("top_k") or 50), float(req.get("top_p") or 0.95))
        else:
            samp = None                                    # жадно
        cid, created, state = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time()), {"broken": False}
        with GEN_LOCK:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n").encode()
                    try:
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        state["broken"] = True

                def emit(t):
                    send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                          "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})

                n, plen, t_pre, dt = generate(msgs, max_tokens, emit, lambda: state["broken"], think, samp)
                send({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if n >= max_tokens else "stop"}],
                      "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
                send("[DONE]")
                try:
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                except Exception:
                    pass
            else:
                parts = []
                n, plen, t_pre, dt = generate(msgs, max_tokens, parts.append, lambda: False, think, samp)
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                              "finish_reason": "length" if n >= max_tokens else "stop"}],
                                 "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}})
        STATE["served"] += 1
        log(f"промт {plen} ток за {t_pre:.1f} с, ответ {n} ток за {dt:.1f} с = {max(n - 1, 0) / max(dt, 1e-6):.2f} ток/с"
            + (" (прерван)" if state["broken"] else ""))


if __name__ == "__main__":
    os.chdir(HERE)
    threading.Thread(target=load_engine, daemon=True).start()
    log(f"nemotron-super на :{PORT}, грузится модель…")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
