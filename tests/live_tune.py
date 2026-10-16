"""перебор BR/NB/KS для _down_live2 на настоящих экспертах при реальной доле живых (подгоняем x, чтобы ~7–11%)"""
import json, os, sys, time, itertools
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
dev = "cuda"; K = 22
torch.manual_seed(0)
experts = sorted(torch.randperm(512)[:K].tolist())
blocks = torch.empty(K, Q.BLOCK, dtype=torch.uint8); s2 = torch.empty(K, 2)
for j, e in enumerate(experts):
    p = f"language_model.backbone.layers.3.mixer.experts.{e}"
    blocks[j] = torch.cat([get(f"{p}.{n}").view(torch.uint8).flatten() for n in ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")])
    s2[j, 0] = get(f"{p}.up_proj.weight_scale_2").float(); s2[j, 1] = get(f"{p}.down_proj.weight_scale_2").float()
T = torch.stack([Q.to_T(b) for b in blocks]); host = T.pin_memory()
tab = torch.tensor([host[j].data_ptr() for j in range(K)], dtype=torch.int64, device=dev)
slots_up = T[:, :Q.UPB].contiguous().to(dev); slots_old = blocks.to(dev)
s2d = s2.to(dev); sidx = torch.arange(K, dtype=torch.int32, device=dev); coef = torch.rand(K, device=dev)
h = torch.empty(K, 2688, device=dev); part = torch.empty(64, K, 1024, device=dev)
idx = torch.empty(K, 2688, dtype=torch.int32, device=dev); cnt = torch.empty(K, dtype=torch.int32, device=dev)
x = (torch.randn(1024, device=dev) * 0.5).to(torch.bfloat16)
y0 = Q.moe_decode(slots_old, sidx, s2d, coef, x, torch.empty_like(h), torch.empty(K, 1024, device=dev)).clone()
Q.moe_decode_live2(slots_up, sidx, tab, s2d, coef, x, h, part, idx, cnt)
print(f"живых {float((h>0).float().mean())*100:.1f}%", flush=True)
def tm(fn, R=50):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t) / R * 1e6
print(f"старое (всё на карте, без копии): {tm(lambda: Q.moe_decode(slots_old, sidx, s2d, coef, x, torch.empty_like(h), torch.empty(K,1024,device=dev))):.0f} мкс", flush=True)
res = []
for BR, NB, KS, nw in itertools.product((256, 512, 1024), (4, 8, 16), (4, 8, 16, 32, 64), (2, 4, 8)):
    cfg = dict(BR=BR, NB=NB, KS=KS, num_warps=nw)
    try:
        y = Q.moe_decode_live2(slots_up, sidx, tab, s2d, coef, x, h, part, idx, cnt, cfg)
        err = float((y - y0).abs().max() / y0.abs().max())
        us = tm(lambda: Q.moe_decode_live2(slots_up, sidx, tab, s2d, coef, x, h, part, idx, cnt, cfg))
        res.append((us, cfg, err))
    except Exception as e:
        print("пропуск", cfg, repr(e)[:80], flush=True)
gidx = torch.empty(K, 168, dtype=torch.int32, device=dev); gcnt = torch.empty(K, dtype=torch.int32, device=dev)
resg = []
for BR, KS, nw in itertools.product((128, 256, 512, 1024), (2, 4, 8, 16, 32), (1, 2, 4, 8)):
    cfg = dict(BR=BR, KS=KS, num_warps=nw)
    try:
        y = Q.moe_decode_groups(slots_up, sidx, tab, s2d, coef, x, h, part, gidx, gcnt, cfg)
        err = float((y - y0).abs().max() / y0.abs().max())
        us = tm(lambda: Q.moe_decode_groups(slots_up, sidx, tab, s2d, coef, x, h, part, gidx, gcnt, cfg))
        resg.append((us, cfg, err))
    except Exception as e:
        print("пропуск", cfg, repr(e)[:120], flush=True)
resg.sort(key=lambda r: r[0])
print("ГРУППЫ:", flush=True)
for us, cfg, err in resg[:6]:
    print(f"{us:7.0f} мкс  {cfg}  отн {err:.1e}", flush=True)
print(f"групп с живым: {float((h>0).view(K,168,16).any(-1).float().mean())*100:.0f}%", flush=True)
res.sort(key=lambda r: r[0])
for us, cfg, err in res[:8]:
    print(f"{us:7.0f} мкс  {cfg}  отн {err:.1e}", flush=True)
# составляющие лучшего
best = res[0][1]
up_only = tm(lambda: Q._up[(Q.INTER // 32, K)](slots_up, sidx, s2d, x, h, BLOCKB=Q.UPB, BN=32, GB=8, num_warps=4))
print(f"из них up-ядро: {up_only:.0f} мкс; лучший cfg {best}", flush=True)
