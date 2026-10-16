"""experts.bin: исходная раскладка down -> «T» (по нейронам; только перестановка байтов, размер тот же) НА МЕСТЕ.
Пачками по 64 блока через карту. Метка: experts.layout = "T". Повторный запуск при метке T — отказ.
Обратно: python repack_T.py --back (from_T), метка удаляется.
Сверка после: 200 экспертов from_T(блок) == оригинал на /data (бит-в-бит)."""
import json, os, random, sys, time
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmoe_ns as Q

DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
back = "--back" in sys.argv
mark = f"{DIR}/experts.layout"
cur = open(mark).read().strip() if os.path.exists(mark) else "orig"
if back and cur != "T" or not back and cur == "T":
    sys.exit(f"раскладка уже {cur}, отказ")
N = 40 * 512
B = 64
fd = os.open(f"{DIR}/experts.bin", os.O_RDWR)
assert os.fstat(fd).st_size == N * Q.BLOCK
buf = torch.empty(B, Q.BLOCK, dtype=torch.uint8).pin_memory()
t0 = time.time()


def to_T_batch(g):
    n = g.shape[0]
    dw = g[:, Q.O_DW:Q.O_DS].view(n, Q.LAT, Q.KB_DN)
    nib = torch.stack([dw & 15, dw >> 4], -1).view(n, Q.LAT, Q.INTER).transpose(1, 2).contiguous().view(n, Q.INTER, Q.LAT // 2, 2)
    tw = (nib[..., 0] | (nib[..., 1] << 4)).view(n, -1)
    ts = g[:, Q.O_DS:Q.BLOCK].view(n, Q.LAT, Q.G_DN).transpose(1, 2).contiguous().view(n, -1)
    g[:, Q.O_TW:Q.O_TS] = tw
    g[:, Q.O_TS:Q.BLOCK] = ts
    return g


with open(mark + ".inprogress", "w") as f:
    f.write("back" if back else "T")
for i in range(0, N, B):
    n = min(B, N - i)
    got = os.preadv(fd, [memoryview(buf[:n].numpy())], i * Q.BLOCK)
    assert got == n * Q.BLOCK
    g = buf[:n].to("cuda")
    g = Q.from_T_batch(g) if back else to_T_batch(g)
    buf[:n].copy_(g.cpu())
    os.pwritev(fd, [memoryview(buf[:n].numpy())], i * Q.BLOCK)
    if i % (B * 40) == 0:
        el = time.time() - t0
        print(f"{i + n}/{N} блоков, {(i + n) * Q.BLOCK / 2**30 / max(el, 1e-9):.2f} ГиБ/с", flush=True)
os.fsync(fd)
if back:
    os.remove(mark)
else:
    open(mark, "w").write("T\n")
os.remove(mark + ".inprogress")
print(f"ПЕРЕСТАНОВКА {'ОБРАТНО' if back else 'В T'} ГОТОВА, {time.time()-t0:.0f} с", flush=True)

# сверка с оригиналом
cfg = json.load(open(f"{SRC}/config.json"))["llm_config"]
MOE = [i for i, t in enumerate(cfg["layers_block_type"]) if t == "moe"]
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = wm[n]
    if f not in H:
        H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)
random.seed(7); bad = 0
picks = [(0, 0), (39, 511)] + [(random.randrange(40), random.randrange(512)) for _ in range(198)]
mm = np.memmap(f"{DIR}/experts.bin", dtype=np.uint8, mode="r")
for m, e in picks:
    p = f"language_model.backbone.layers.{MOE[m]}.mixer.experts.{e}"
    ref = torch.cat([get(f"{p}.{n}").view(torch.uint8).flatten() for n in
                     ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")])
    blk = torch.from_numpy(np.array(mm[(m * 512 + e) * Q.BLOCK:(m * 512 + e + 1) * Q.BLOCK]))
    got = blk if back else Q.from_T(blk)
    if not torch.equal(got, ref):
        bad += 1; print("BAD", m, e, flush=True)
print(f"СВЕРКА с оригиналом: {len(picks)} экспертов, расхождений {bad}", flush=True)
