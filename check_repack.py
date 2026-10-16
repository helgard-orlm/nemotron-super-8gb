"""200 случайных экспертов: блок experts.bin == склейка частей из оригинала; скаляры == оригинал"""
import json, os, random, sys, numpy as np, torch
from safetensors import safe_open
import qmoe_ns as Q
DIR = os.environ.get("NS_DIR", "/fast/nemsuper")
SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
cfg = json.load(open(f"{SRC}/config.json"))["llm_config"]
MOE = [i for i, t in enumerate(cfg["layers_block_type"]) if t == "moe"]
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = wm[n]
    if f not in H: H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)
mm = np.memmap(f"{DIR}/experts.bin", dtype=np.uint8, mode="r")
sc = np.load(f"{DIR}/experts_scal.npy")
random.seed(1); bad = 0
picks = [(m, e) for m in (0, 39) for e in (0, 511)] + [(random.randrange(40), random.randrange(512)) for _ in range(196)]
for m, e in picks:
    p = f"language_model.backbone.layers.{MOE[m]}.mixer.experts.{e}"
    ref = torch.cat([get(f"{p}.{n}").view(torch.uint8).flatten() for n in
                     ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")]).numpy()
    blk = mm[(m * 512 + e) * Q.BLOCK:(m * 512 + e + 1) * Q.BLOCK]
    s = [get(f"{p}.{n}").float().item() for n in ("up_proj.weight_scale_2", "down_proj.weight_scale_2", "up_proj.input_scale", "down_proj.input_scale")]
    if not (np.array_equal(ref, blk) and np.allclose(s, sc[m, e], rtol=0, atol=0)): bad += 1; print("BAD", m, e)
print(f"сверено {len(picks)} экспертов, расхождений {bad}")
