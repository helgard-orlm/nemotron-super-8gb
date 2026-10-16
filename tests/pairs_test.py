"""moe_pairs против moe_decode (уже сверено с torch) на настоящих экспертах: T токенов × 22 эксперта из пачки."""
import json, os, sys
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qmoe_ns as Q

SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}
def get(n):
    f = wm[n]
    if f not in H:
        H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)

dev = "cuda"
NB = 32                     # экспертов в пачке (слотов)
T = 7
torch.manual_seed(0)
experts = sorted(torch.randperm(512)[:NB].tolist())
blocks = torch.empty(NB, Q.BLOCK, dtype=torch.uint8); s2 = torch.empty(NB, 2)
for j, e in enumerate(experts):
    p = f"language_model.backbone.layers.1.mixer.experts.{e}"
    parts = [get(f"{p}.{n}").view(torch.uint8).flatten() for n in
             ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")]
    blocks[j] = torch.cat(parts)
    s2[j, 0] = get(f"{p}.up_proj.weight_scale_2").float(); s2[j, 1] = get(f"{p}.down_proj.weight_scale_2").float()
slots = blocks.to(dev); s2 = s2.to(dev)
X = (torch.randn(T, 1024, device=dev) * 0.5).to(torch.bfloat16)
# каждому токену — 22 разных слота из пачки и веса
sel = torch.stack([torch.randperm(NB)[:22] for _ in range(T)]).to(dev)           # [T,22] номера слотов
w = torch.rand(T, 22, device=dev)
# эталон: moe_decode по токенам
ref = torch.stack([Q.moe_decode(slots, sel[t].int().contiguous(), s2[sel[t]].contiguous(), w[t].contiguous(),
                                X[t].contiguous(), torch.empty(22, 2688, device=dev), torch.empty(22, 1024, device=dev))
                   for t in range(T)])
# пары, как в nem_super: (t, k) -> слот sel[t,k]
t_i = torch.arange(T, device=dev).repeat_interleave(22); k_i = torch.arange(22, device=dev).repeat(T)
b_i = sel[t_i, k_i]
hb = torch.empty(T * 22, 2688, device=dev); ob = torch.empty(T * 22, 1024, device=dev)
y = Q.moe_pairs(slots, b_i.int().contiguous(), t_i.int().contiguous(), s2[b_i].contiguous(),
                w[t_i, k_i].float().contiguous(), X.contiguous(), hb, ob)
out = torch.zeros(T, 1024, device=dev).index_add_(0, t_i, y)
d = (out - ref).abs().max().item(); sc = ref.abs().max().item()
print(f"пары vs шаг: max|Δ| {d:.3e}, масштаб {sc:.3e}, отн {d/sc:.2e}")
# как в движке: x bf16, но эталон шага получает тот же bf16
for t in range(T):
    dt = (out[t] - ref[t]).abs().max().item()
    print(f"  токен {t}: {dt/sc:.2e}")
