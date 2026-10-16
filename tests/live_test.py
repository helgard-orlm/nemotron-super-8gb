"""Пункт 1: moe_decode_live (up с карты, down из закреплённой RAM по таблице, только живые) против moe_decode
(исходная раскладка, всё на карте) на 22 настоящих экспертах; to_T/from_T туда-обратно бит-в-бит; время."""
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
    if f not in H:
        H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)

dev = "cuda"
K = 22
torch.manual_seed(0)
for layer in (1, 3):
    experts = sorted(torch.randperm(512)[:K].tolist())
    blocks = torch.empty(K, Q.BLOCK, dtype=torch.uint8); s2 = torch.empty(K, 2)
    for j, e in enumerate(experts):
        p = f"language_model.backbone.layers.{layer}.mixer.experts.{e}"
        blocks[j] = torch.cat([get(f"{p}.{n}").view(torch.uint8).flatten() for n in
                               ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")])
        s2[j, 0] = get(f"{p}.up_proj.weight_scale_2").float(); s2[j, 1] = get(f"{p}.down_proj.weight_scale_2").float()
    T = torch.stack([Q.to_T(b) for b in blocks])
    back = torch.stack([Q.from_T(b) for b in T])
    print(f"слой {layer}: туда-обратно бит-в-бит: {torch.equal(back, blocks)}; up не тронут: "
          f"{torch.equal(T[:, :Q.UPB], blocks[:, :Q.UPB])}", flush=True)
    gb = T.to(dev).clone(); Q.from_T_batch(gb)
    print(f"  from_T_batch на карте == исходный: {torch.equal(gb.cpu(), blocks)}", flush=True)
    host = T.pin_memory()
    tab = torch.tensor([host[j].data_ptr() for j in range(K)], dtype=torch.int64, device=dev)
    slots_up = T[:, :Q.UPB].contiguous().to(dev)
    slots_old = blocks.to(dev)
    s2d = s2.to(dev); sidx = torch.arange(K, dtype=torch.int32, device=dev)
    h = torch.empty(K, 2688, device=dev); o = torch.empty(K, 1024, device=dev)
    idx = torch.empty(K, 2688, dtype=torch.int32, device=dev); cnt = torch.empty(K, dtype=torch.int32, device=dev)
    for trial in range(3):
        x = (torch.randn(1024, device=dev) * (0.5 + trial)).to(torch.bfloat16)
        coef = torch.rand(K, device=dev)
        y0 = Q.moe_decode(slots_old, sidx, s2d, coef, x, torch.empty_like(h), torch.empty_like(o)).clone()
        y1 = Q.moe_decode_live(slots_up, sidx, tab, s2d, coef, x, h, o, idx, cnt).clone()
        live = float((h > 0).float().mean())
        d = (y1 - y0).abs().max().item(); sc = y0.abs().max().item()
        print(f"  проба {trial}: живых {live*100:.1f}%  live vs старое: отн {d/sc:.2e}", flush=True)
    # контроль: таблица указывает на ЧУЖИЕ эксперты (сдвиг) — должно разойтись
    tab_bad = torch.roll(tab, 1)
    y2 = Q.moe_decode_live(slots_up, sidx, tab_bad, s2d, coef, x, h, o, idx, cnt)
    print(f"  контроль (down чужих экспертов): отн {(y2 - y0).abs().max().item()/sc:.2e}", flush=True)
    # время
    for name, fn in (("старое (всё на карте)", lambda: Q.moe_decode(slots_old, sidx, s2d, coef, x, torch.empty_like(h), torch.empty_like(o))),
                     ("live (down из RAM)", lambda: Q.moe_decode_live(slots_up, sidx, tab, s2d, coef, x, h, o, idx, cnt))):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        print(f"  {name}: {(time.time()-t)*1e4:.0f} мкс/слой", flush=True)
    # копия на карту: целый блок vs только up
    for nb, lab in ((Q.BLOCK, "копия целого блока"), (Q.UPB, "копия только up")):
        dst = torch.empty(K, nb, dtype=torch.uint8, device=dev)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            for j in range(K):
                dst[j].copy_(host[j, :nb], non_blocking=True)
        torch.cuda.synchronize()
        print(f"  {lab}: {(time.time()-t)/20*1e3:.2f} мс/слой ({K*nb/((time.time()-t)/20)/1e9:.1f} ГБ/с)", flush=True)
