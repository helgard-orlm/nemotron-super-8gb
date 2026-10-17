#!/usr/bin/env python3
"""Nemotron-3.5-Super-VL-120B-A12B (свой движок nem_super.py, эксперты с /fast) — служба с API как у OpenAI
(/v1/models, /v1/chat/completions со стримом) для Open WebUI. Заказ helgard 16.10.2026 «делаем свой движок», «1 и 2».
Образец — qwen_server.py (Qwen3.8). Отличия:
  - состояние = кэш transformers (KV 8 слоёв внимания + состояния 40 слоёв Mamba); продолжение пачкой поверх кэша —
    mamba_cont (как у Nano); снимок = копия кэша в RAM (~170 МБ) ⇒ продолжение диалога дочитывает только хвост;
  - промт пачкой (кусками), ответ — по одному слову; жадно, если temperature не задана;
  - две модели: обычная и «-think» (<think>…</think>, Open WebUI сворачивает);
  - картинки (#401, 17.10): image_url (data:/http) → башня C-RADIO (vision/vis.py, BF16 как у NVIDIA,
    грузится при первой картинке; NS_VIS_DEVICE=cuda — веса в RAM, на карту по модулю; cpu — на процессоре) → векторы на места <image> (id 18) в CPUEmbedding; ключ кэша промта на этих
    местах = отпечаток картинки (иначе две картинки одного размера совпали бы); MTP не подключён.
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
DOC_ID = MODEL_ID + "-doc"                       # картинки в детальном разрешении (документы, мелкий текст)
MODEL_IDS = (MODEL_ID, THINK_ID, DOC_ID, DOC_ID + "-think")
STOPS = {2, 11}                                  # </s>, <|im_end|>
IMG_ID = 18                                      # <image>
IMG_MAX_PATCHES = int(os.environ.get("NS_IMG_MAX_PATCHES", "4096"))   # быстрый: 4096 патчей = 1024 токена
IMG_DOC_PATCHES = int(os.environ.get("NS_IMG_DOC_PATCHES", "13312"))  # документы (-doc / detail=high): предел NVIDIA
VIS_DEVICE = os.environ.get("NS_VIS_DEVICE", "cuda")                  # cuda: веса башни по модулю из RAM; cpu
IMG_MAX_BYTES = 20 * 2**20
VIS = {"v": None, "proc": None, "emb": __import__("collections").OrderedDict(), "q": None, "pos": 0,
       "lock": threading.Lock()}

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


def _img_src(p):
    u = p.get("image_url")
    if isinstance(u, dict):
        u = u.get("url")
    return u or p.get("url") or p.get("image")


def load_image(u):
    import base64, io
    from PIL import Image
    if u.startswith("data:"):
        raw = base64.b64decode(u.split(",", 1)[1])
    elif u.startswith(("http://", "https://")):
        raw = urllib.request.urlopen(u, timeout=20).read(IMG_MAX_BYTES + 1)
    else:
        raise ValueError("картинка: нужен data: или http(s) адрес")
    if len(raw) > IMG_MAX_BYTES:
        raise ValueError("картинка больше 20 МБ")
    return raw, Image.open(io.BytesIO(raw)).convert("RGB")


def conv_msg(m):
    c = m.get("content")
    role = m.get("role", "user")
    if isinstance(c, list) and role == "user" and any(isinstance(p, dict) and p.get("type") in
                                                      ("image_url", "input_image", "image") for p in c):
        parts = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and p.get("text"):
                parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") in ("image_url", "input_image", "image") and _img_src(p):
                u = p.get("image_url")
                parts.append({"type": "image", "src": _img_src(p),
                              "detail": (u.get("detail") if isinstance(u, dict) else None) or p.get("detail")})
        return {"role": "user", "content": parts}
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    if role not in ("system", "user", "assistant"):
        role = "user"
    c = c or ""
    if role == "assistant" and "</think>" in c:                       # прошлое рассуждение в контекст не кладём
        c = c.split("</think>", 1)[1].lstrip("\n")
    if role == "assistant" and "<details" in c and "</details>" in c:  # так Open WebUI может вернуть свёрнутое рассуждение
        c = re.sub(r"<details[^>]*>.*?</details>\s*", "", c, flags=re.S)
    return {"role": role, "content": c}


def vision():
    """башня на CPU при первой картинке; подстановка в CPUEmbedding ставится один раз"""
    with VIS["lock"]:
        if VIS["v"] is None:
            torch = ENGINE["torch"]
            sys.path.insert(0, os.path.join(HERE, "vision"))
            import vis
            t = time.time()
            VIS["proc"] = vis.processor()
            VIS["v"] = vis.Vision("cpu", torch.bfloat16)
            if VIS_DEVICE == "cuda":
                vis.stream_cuda(VIS["v"])
            E = ENGINE["model"].model.embeddings
            F = torch.nn.functional

            def emb_fwd(ids):
                ids = ids.cpu()
                e = F.embedding(ids, E.w)
                m = ids == IMG_ID
                n = int(m.sum())
                q = VIS["q"]
                if n and q is not None:                 # без очереди (сгенерированный id 18) — обычная строка словаря
                    assert VIS["pos"] + n <= q.shape[0], "векторов картинки меньше, чем <image>"
                    e[m] = q[VIS["pos"]:VIS["pos"] + n].to(e.dtype); VIS["pos"] += n
                return e.to("cuda", non_blocking=True)

            E.forward = emb_fwd
            log(f"зрение загружено: {time.time() - t:.1f} с, башня на {VIS_DEVICE}, предел {IMG_MAX_PATCHES} / "
                f"документы {IMG_DOC_PATCHES} патчей")
    return VIS["v"], VIS["proc"]


def image_embeds(raw, img, mp):
    """векторы одной картинки [n, 4096] в RAM (кэш по sha256 байтов + предел патчей) + отпечаток"""
    import hashlib
    torch = ENGINE["torch"]
    h = hashlib.sha256(raw).hexdigest() + f"/{mp}"           # режим входит в ключ: та же картинка ≠ те же векторы
    if h in VIS["emb"]:
        VIS["emb"].move_to_end(h)
        return VIS["emb"][h], h, 0.0
    v, proc = vision()
    if v.device == "cuda" and LIVE["cache"] is not None:   # карта впритык: живой кэш диалога — снимком в RAM
        snap_take(LIVE["fed"], LIVE["cache"])
        LIVE["fed"], LIVE["cache"] = None, None
    if v.device == "cuda":                                   # буферы промта (~0.2 ГиБ) создадутся заново при чтении
        SE = ENGINE["NS"].StreamExperts
        SE.gpu_tmp = SE.pair_buf = SE.t_scratch = None
        torch.cuda.empty_cache()
    t = time.time()
    proc.image_processor.max_num_patches = mp
    pv = proc.image_processor(images=[img], return_tensors="pt")["pixel_values"]
    torch.set_num_threads(10)
    try:
        e = v.embed(pv).float().cpu()
        if v.device == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()   # куски башни не держать в кэше распределителя
    finally:
        torch.set_num_threads(8)
    VIS["emb"][h] = e
    while len(VIS["emb"]) > 8:
        VIS["emb"].popitem(last=False)
    return e, h, time.time() - t


def build_prompt(messages, think, doc=False):
    """-> (ids, ключ, векторы|None, сек башни). Без картинок — как раньше (tok.apply_chat_template)."""
    torch, tok = ENGINE["torch"], ENGINE["tok"]
    imgs = [p for m in messages if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image"]
    if not imgs:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return ids, list(ids), None, 0.0, 0
    v, proc = vision()
    mp = IMG_DOC_PATCHES if doc or any(p.get("detail") == "high" for p in imgs) else IMG_MAX_PATCHES
    loaded = [load_image(p["src"]) for p in imgs]
    embs, hs, tv = [], [], 0.0
    for raw, img in loaded:
        e, h, dt = image_embeds(raw, img, mp); embs.append(e); hs.append(h); tv += dt
    proc.image_processor.max_num_patches = mp                 # число <image> в промте считается тем же пределом
    msgs = [{"role": m["role"], "content": [{"type": "image"} if p["type"] == "image" else p for p in m["content"]]}
            if isinstance(m["content"], list) else m for m in messages]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    ids = proc(text=[text], images=[img for _, img in loaded], return_tensors="pt")["input_ids"][0].tolist()
    q = torch.cat(embs)
    assert sum(1 for t in ids if t == IMG_ID) == q.shape[0], ("токенов <image>", sum(1 for t in ids if t == IMG_ID), q.shape[0])
    # ключ: на местах <image> — отпечаток своей картинки (отрицательные числа, с токенами не пересекаются)
    key, k, left = [], 0, [e.shape[0] for e in embs]
    for t in ids:
        if t == IMG_ID:
            while left[k] == 0:
                k += 1
            key.append(-1 - int(__import__("hashlib").sha256(hs[k].encode()).hexdigest()[:12], 16)); left[k] -= 1
        else:
            key.append(t)
    return ids, key, q, tv, mp


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


def generate(messages, max_tokens, on_text, cancelled, think=False, samp=None, doc=False):
    torch, model, tok, mc = ENGINE["torch"], ENGINE["model"], ENGINE["tok"], ENGINE["mc"]
    try:
        ids, key, q, tv, mp = build_prompt(messages, think, doc)
    except Exception as e:
        log("картинка: ошибка", repr(e))
        on_text(f"[Nemotron: не удалось прочитать картинку — {e}]")
        return 0, 0, 0.0, 0.0
    gen = torch.Generator(device="cuda"); gen.manual_seed(int(time.time() * 1000) % 2**31)
    if len(ids) + max_tokens > MAX_SEQ:
        max_tokens = MAX_SEQ - len(ids)
        if max_tokens < 16:
            on_text(f"[диалог слишком длинный: {len(ids)} токенов при пределе {MAX_SEQ} — начните новый чат]")
            return 0, len(ids), 0.0, 0.0
    g, shown = [], ""
    with torch.inference_mode():
        torch.cuda.synchronize(); t0 = time.time()
        start, src, cache = 0, "с нуля", None
        best, LIVE_BEST = None, [object()]
        if not STATE.get("nocache"):
            best = None
            for fed in [LIVE["fed"]] + list(reversed(list(SNAPS))):
                if fed and len(fed) < len(key) and tuple(key[:len(fed)]) == tuple(fed) and \
                        (best is None or len(fed) > len(best)):
                    best = fed
            if best is not None:
                LIVE_BEST[0] = LIVE["fed"]
                cache = LIVE["cache"] if best == LIVE["fed"] else mc.restore(SNAPS[best])
                start, src = len(best), f"продолжение {len(best)} ток + хвост {len(ids) - len(best)}"
        LIVE["fed"], LIVE["cache"] = None, None
        # снимок на начале последнего «<|im_start|>» (заголовок ответа): этот префикс совпадает с тем, как следующий
        # запрос отрисует историю в любом режиме (рассуждения прошлых ответов шаблон вырезает)
        im = ENGINE["im_start"]
        cut = max((i for i, t in enumerate(ids) if t == im), default=0)
        from_snap = best if (best is not None and best != LIVE_BEST[0]) else None
        chunk = min(PREFILL_CHUNK, 512) if q is not None else PREFILL_CHUNK   # картинки: куски 512 (VRAM впритык)
        while True:
            VIS["q"], VIS["pos"] = q, sum(1 for t in ids[:start] if t == IMG_ID)   # картинки в кэше уже прочитаны
            try:
                lg = None
                spans = [(start, cut), (cut, len(ids))] if start < cut else [(start, len(ids))]
                for a0, b0 in spans:
                    for a in range(a0, b0, chunk):
                        piece = torch.tensor([ids[a:min(a + chunk, b0)]], device="cuda")
                        cache, lg = mc.prefill(model, piece, cache, chunk=chunk)
                    if b0 == cut and cut > 0:
                        snap_take(tuple(key[:cut]), cache)
                break
            except torch.OutOfMemoryError:
                cache = lg = None
                torch.cuda.empty_cache()
                if chunk <= 128:
                    log(f"промт {len(ids)} ток: нехватка видеопамяти ({src}, куски {chunk})")
                    on_text("[Nemotron: не хватило видеопамяти на чтение промта — сократите сообщение или начните новый чат]")
                    VIS["q"] = None
                    return 0, len(ids), 0.0, 0.0
                chunk //= 2
                # упавший проход мог частично изменить кэш на месте ⇒ заново: с целого снимка в RAM или с нуля
                if from_snap is not None and from_snap in SNAPS:
                    cache = mc.restore(SNAPS[from_snap])
                else:
                    start, cache, src = 0, None, "с нуля (повтор)"
                log(f"промт {len(ids)} ток: нехватка видеопамяти — повтор кусками {chunk}")
        if chunk != PREFILL_CHUNK:
            src += f", куски {chunk}"
        torch.cuda.synchronize(); t_pre = time.time() - t0
        if q is not None:
            assert VIS["pos"] == q.shape[0], ("подставлено векторов", VIS["pos"], q.shape[0])
            src += f", картинок {q.shape[0]} ток (предел {mp} патчей, башня {tv:.1f} с)"
        VIS["q"] = None
        log(f"промт {len(ids)} ток: {src}, {t_pre:.1f} с" + (", думает" if think else ""))
        t1 = time.time()
        if think:
            on_text("<think>\n")                                       # шаблон открыл <think> внутри промта
        fed = list(key)
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
        if self.path in ("/", "/health") or self.path.startswith("/health?"):
            st = {**STATE, "uptime_s": round(time.time() - STATE["since"]), "busy": GEN_LOCK.locked()}
            if ENGINE.get("cache") is not None:
                st["experts"] = dict(ENGINE["cache"].stat)
                tc = ENGINE["torch"].cuda
                st["vram_GiB"] = {"allocated": round(tc.memory_allocated() / 2**30, 3),
                                  "reserved": round(tc.memory_reserved() / 2**30, 3),
                                  "peak": round(tc.max_memory_allocated() / 2**30, 3)}
                if "reset_peak" in self.path:
                    tc.reset_peak_memory_stats()
            return self._json(200 if STATE["status"] == "ready" else 503, st)
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "gpu104", "created": 0} for m in MODEL_IDS]})
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
        doc = model_id.startswith(DOC_ID)
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

                n, plen, t_pre, dt = generate(msgs, max_tokens, emit, lambda: state["broken"], think, samp, doc)
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
                n, plen, t_pre, dt = generate(msgs, max_tokens, parts.append, lambda: False, think, samp, doc)
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
