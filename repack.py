"""Nemotron-3.5-Super-VL-120B-A12B NVFP4 -> файлы движка на /fast (#399, 16.10.2026).

1) experts.bin — routed-эксперты одним куском на эксперта (одно чтение с диска на эксперта), блок (m*512+e),
   m = номер MoE-слоя по порядку (0..39), 3 096 576 Б = 756 страниц по 4 КиБ:
     up_w [2688,512]u8 | up_s [2688,64]e4m3 | dn_w [1024,1344]u8 | dn_s [1024,168]e4m3
   (up: 1024 -> 2688, down: 2688 -> 1024; эксперты работают в латенте 1024)
   скаляры -> experts_scal.npy float32 [40,512,4]: ws2 up/down, input_scale up/down
2) nonexpert.safetensors — все тензоры language_model.* кроме routed-экспертов и MTP, байты как есть
   (копия диапазонов без загрузки в RAM).
Источник читается по возрастанию смещений в каждом файле."""
import json, os, re, struct, sys, time
import numpy as np

SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
DST = os.environ.get("NS_DIR", "/fast/nemsuper")
NE = 512
PARTS = [("up_proj.weight", 2688 * 512), ("up_proj.weight_scale", 2688 * 64),
         ("down_proj.weight", 1024 * 1344), ("down_proj.weight_scale", 1024 * 168)]
OFF = {}
o = 0
for n, s in PARTS:
    OFF[n] = (o, s); o += s
BLOCK = o
assert BLOCK == 3096576 and BLOCK % 4096 == 0
SCAL = {"up_proj.weight_scale_2": 0, "down_proj.weight_scale_2": 1, "up_proj.input_scale": 2, "down_proj.input_scale": 3}
pat = re.compile(r"^language_model\.backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(.+)$")

cfg = json.load(open(f"{SRC}/config.json"))["llm_config"]
MOE = [i for i, t in enumerate(cfg["layers_block_type"]) if t == "moe"]
assert len(MOE) == 40, len(MOE)
MI = {L: m for m, L in enumerate(MOE)}
NL = len(MOE)
os.makedirs(DST, exist_ok=True)

wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
files = sorted(set(wm.values()))

# ---------------------------------------------------------------- 1. эксперты
scal = np.full((NL, NE, 4), np.nan, dtype=np.float32)
fd = os.open(f"{DST}/experts.bin", os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, NL * NE * BLOCK)
seen = 0; nbytes = 0; t0 = time.time()
nonexp = []                                   # (имя, файл, dtype, shape, a, b, base)
for f in files:
    with open(f"{SRC}/{f}", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    items = []
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        m = pat.match(k)
        if m:
            items.append((a, b, int(m.group(1)), int(m.group(2)), m.group(3), v["dtype"]))
        elif k.startswith("language_model.") and ".mtp." not in k:
            nonexp.append((k, f, v["dtype"], v["shape"], a, b, base))
    items.sort()
    if not items:
        continue
    src = os.open(f"{SRC}/{f}", os.O_RDONLY)
    for a, b, L, e, part, dt in items:
        buf = os.pread(src, b - a, base + a)
        assert len(buf) == b - a
        if part in SCAL:
            assert dt == "F32" and len(buf) == 4, (part, dt, len(buf))
            scal[MI[L], e, SCAL[part]] = struct.unpack("<f", buf)[0]
        else:
            po, ps = OFF[part]
            assert ps == len(buf), (part, len(buf))
            os.pwrite(fd, buf, (MI[L] * NE + e) * BLOCK + po)
            seen += 1; nbytes += len(buf)
    os.close(src)
    el = time.time() - t0
    print(f"{f}: частей {seen}/{NL*NE*4}, {nbytes/2**30:.1f} ГиБ, {nbytes/2**20/max(el,1e-9):.0f} МиБ/с", flush=True)
os.fsync(fd); os.close(fd)
assert seen == NL * NE * 4, seen
assert not np.isnan(scal).any(), int(np.isnan(scal).sum())
np.save(f"{DST}/experts_scal.npy", scal)
print("EXPERTS_DONE", seen, f"{time.time()-t0:.0f} с", flush=True)

# ---------------------------------------------------------------- 2. постоянная часть
nonexp.sort(key=lambda r: r[0])
hdr, off = {}, 0
for k, f, dt, sh, a, b, base in nonexp:
    hdr[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + (b - a)]}
    off += b - a
hb = json.dumps(hdr, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(f"{DST}/nonexpert.safetensors", "wb") as out:
    out.write(struct.pack("<Q", len(hb))); out.write(hb)
    srcs = {}
    for k, f, dt, sh, a, b, base in nonexp:
        if f not in srcs:
            srcs[f] = os.open(f"{SRC}/{f}", os.O_RDONLY)
        rem, pos = b - a, base + a
        while rem:
            chunk = os.pread(srcs[f], min(rem, 64 << 20), pos)
            assert chunk
            out.write(chunk); rem -= len(chunk); pos += len(chunk)
    out.flush(); os.fsync(out.fileno())
print(f"NONEXPERT_DONE {len(nonexp)} тензоров, {off/2**30:.2f} ГиБ", flush=True)
for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    if os.path.exists(f"{SRC}/{fn}"):
        os.system(f"cp -p '{SRC}/{fn}' '{DST}/'")
os.makedirs(f"{DST}/tok", exist_ok=True)                 # токенизатор без config.json обёртки Omni
for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    os.system(f"cp -p '{SRC}/{fn}' '{DST}/tok/'")
import json as _j
_p = f"{DST}/tok/tokenizer_config.json"; _c = _j.load(open(_p))
_c.pop("auto_map", None); _c.pop("processor_class", None)        # иначе вопрос trust_remote_code
_j.dump(_c, open(_p, "w"), indent=1, ensure_ascii=False)
print("REPACK_DONE", f"{time.time()-t0:.0f} с", flush=True)
