"""moe_pairs_T (раскладка T) против moe_pairs (исходная) на 32 настоящих экспертах; время; подбор BR"""
import json, os, sys, time
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qmoe_ns as Q
SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = wm[n]
    if f not in H: H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)
dev = "cuda"; NB = 32; T = 32
torch.manual_seed(0)
experts = sorted(torch.randperm(512)[:NB].tolist())
blocks = torch.empty(NB, Q.BLOCK, dtype=torch.uint8); s2 = torch.empty(NB, 2)
for j, e in enumerate(experts):
    p = f"language_model.backbone.layers.3.mixer.experts.{e}"
    blocks[j] = torch.cat([get(f"{p}.{n}").view(torch.uint8).flatten() for n in ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")])
    s2[j, 0] = get(f"{p}.up_proj.weight_scale_2").float(); s2[j, 1] = get(f"{p}.down_proj.weight_scale_2").float()
slots = blocks.to(dev); slotsT = torch.stack([Q.to_T(b) for b in blocks]).to(dev); s2 = s2.to(dev)
X = (torch.randn(T, 1024, device=dev) * 0.5).to(torch.bfloat16)
sel = torch.stack([torch.randperm(NB)[:22] for _ in range(T)]).to(dev); w = torch.rand(T, 22, device=dev)
t_i = torch.arange(T, device=dev).repeat_interleave(22); k_i = torch.arange(22, device=dev).repeat(T); b_i = sel[t_i, k_i]
args = (b_i.int().contiguous(), t_i.int().contiguous(), s2[b_i].contiguous(), w[t_i, k_i].float().contiguous(), X.contiguous())
hb = torch.empty(T * 22, 2688, device=dev); ob = torch.empty(T * 22, 1024, device=dev)
y0 = Q.moe_pairs(slots, *args, hb, ob).clone()
def tm(fn, R=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t) / R * 1e3
print(f"старое moe_pairs: {tm(lambda: Q.moe_pairs(slots, *args, hb, ob)):.2f} мс (P={T*22})")
def conv():
    g = slotsT.clone()
    for i in range(0, NB, 4): Q.from_T_batch(g[i:i + 4])
    return g
print(f"обратная перестановка 32 блоков (как в движке сейчас): {tm(conv):.2f} мс")
for BR in (32, 64, 128, 256):
    Q.PAIRS_T_BR = BR
    y = Q.moe_pairs_T(slotsT, *args, hb, ob)
    err = float((y - y0).abs().max() / y0.abs().max())
    print(f"moe_pairs_T BR={BR}: {tm(lambda: Q.moe_pairs_T(slotsT, *args, hb, ob)):.2f} мс, отн {err:.1e}")
scratch = torch.empty(NB * 1376256, dtype=torch.uint8, device=dev)
g = slotsT.clone(); Q.from_T_fast(g, scratch)
print(f"from_T_fast == исходный бит-в-бит: {torch.equal(g, slots)}")
print(f"from_T_fast 32 блоков: {tm(lambda: Q.from_T_fast(slotsT.clone(), scratch)):.2f} мс (с clone)")
print(f"clone отдельно: {tm(lambda: slotsT.clone()):.2f} мс")
