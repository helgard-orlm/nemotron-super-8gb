"""Проверка qmoe_ns.moe_decode против torch fp32 на 22 настоящих экспертах слоя 1 (блоки собираются из оригинала)."""
import os, sys, time, json
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qmoe_ns as Q

SRC = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = wm[n]
    if f not in H:
        H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)

dev = "cuda"
K = 22
torch.manual_seed(0)
experts = sorted(torch.randperm(512)[:K].tolist())
blocks = torch.empty(K, Q.BLOCK, dtype=torch.uint8)
s2 = torch.empty(K, 2)
for j, e in enumerate(experts):
    p = f"language_model.backbone.layers.1.mixer.experts.{e}"
    parts = [get(f"{p}.up_proj.weight").view(torch.uint8), get(f"{p}.up_proj.weight_scale").view(torch.uint8),
             get(f"{p}.down_proj.weight").view(torch.uint8), get(f"{p}.down_proj.weight_scale").view(torch.uint8)]
    blocks[j] = torch.cat([t.flatten() for t in parts])
    s2[j, 0] = get(f"{p}.up_proj.weight_scale_2").float(); s2[j, 1] = get(f"{p}.down_proj.weight_scale_2").float()
slots = blocks.to(dev)
x = torch.randn(1024, device=dev) * 0.5
coef = torch.rand(K, device=dev)
sidx = torch.arange(K, dtype=torch.int32, device=dev)
h = torch.empty(K, 2688, device=dev); out = torch.empty(K, 1024, device=dev)
y = Q.moe_decode(slots, sidx, s2.to(dev), coef, x, h, out)
ref = torch.zeros(1024, device=dev, dtype=torch.float64)
for j in range(K):
    u, d = Q.block_deq(slots[j], s2[j])
    hh = torch.relu(u.double() @ x.double()) ** 2
    ref += coef[j].double() * (d.double() @ hh)
err = (y.double() - ref).abs().max().item(); sc = ref.abs().max().item()
print(f"ядро vs torch: max|Δ| {err:.3e}, масштаб {sc:.3e}, отн {err/sc:.2e}")
Q.SWAP = True
ref2 = torch.zeros(1024, device=dev, dtype=torch.float64)
for j in range(K):
    u, d = Q.block_deq(slots[j], s2[j])
    ref2 += coef[j].double() * (d.double() @ (torch.relu(u.double() @ x.double()) ** 2))
print(f"контроль (полубайты наоборот) vs ядро: отн {(y.double()-ref2).abs().max().item()/sc:.2e}")
Q.SWAP = False
# перестановка слотов: sidx в обратном порядке, coef тоже
y2 = Q.moe_decode(slots, sidx.flip(0).contiguous(), s2.flip(0).contiguous().to(dev), coef.flip(0).contiguous(), x, h, out)
print(f"перестановка слотов: max|Δ| {(y2-y).abs().max().item():.2e}")
torch.cuda.synchronize(); t = time.time()
for _ in range(200):
    Q.moe_decode(slots, sidx, s2.to(dev), coef, x, h, out)
torch.cuda.synchronize(); dt = (time.time() - t) / 200
print(f"время: {dt*1e6:.0f} мкс/слой (K={K}) ⇒ {dt*40*1e3:.1f} мс/слово на 40 слоёв, {K*Q.BLOCK/dt/1e9:.0f} ГБ/с")
