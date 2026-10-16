"""ЭТАЛОН: transformers NemotronHForCausalLM целиком на процессоре в fp32, слой за слоем, веса — из ОРИГИНАЛА на /data
(не из перепаковки), распаковка — функцией transformers (Fp8Dequantize._unpack_fp4: младший полубайт первым) +
двухуровневый масштаб NVFP4 (E4M3 на 16 весов × weight_scale_2; так же пакует конвертер llama.cpp _nvfp4_pack);
FP8 — w × weight_scale. Активации не квантуются (как и в нашем движке; у NVIDIA на vLLM — W4A4/W8A8 по input_scale).
Ничего из нашего движка (repack, qmoe_ns, dense_q, ExpertCache, moe_forward) не используется.

Слой материализуется перед своим forward и освобождается после (хуки). У MoE-слоя распаковываются только
эксперты, выбранные роутером на этом входе (роутер — код transformers, вызывается в хуке на том же входе).

  python -u ref_full.py seq.json out.pt   (seq.json: {"ids": [...]})  -> логиты fp32 [T, V] по всем позициям"""
import json, sys, time
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

SRC = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
torch.set_num_threads(10)
def unpack_fp4(_, w):                                    # метод transformers; таблица берётся с класса
    return Fp8Dequantize._unpack_fp4(Fp8Dequantize, w)
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
H = {}


def raw(n):
    f = wm[n]
    if f not in H:
        H[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
    return H[f].get_tensor(n)


def weight(base):
    """base без .weight (имя чекпойнта) -> fp32"""
    w = raw(base + ".weight")
    if base + ".weight_scale_2" in wm:                       # NVFP4
        v = unpack_fp4(None, w)                              # [N, K] значения E2M1
        s = raw(base + ".weight_scale").float()              # [N, K/16]
        return (v.view(v.shape[0], -1, 16) * s[..., None]).view(v.shape) * raw(base + ".weight_scale_2").float()
    if w.dtype == torch.float8_e4m3fn:                       # FP8 на тензор
        return w.float() * raw(base + ".weight_scale").float()
    return w.float()


cfg_full = json.load(open(f"{SRC}/config.json"))
lc = dict(cfg_full["llm_config"]); lc.pop("quantization_config", None)
fx = lambda v: float(v["__float__"]) if isinstance(v, dict) and set(v) == {"__float__"} else (
    [fx(x) for x in v] if isinstance(v, list) else v)
lc = {k: fx(v) for k, v in lc.items()}
cfg = NemotronHConfig(**lc)
cfg.use_mamba_kernels = False                                # на процессоре — путь torch
with torch.device("meta"):
    model = NemotronHForCausalLM(cfg)
model.eval()
M = model.model
P = "language_model."


def ck_name(n):
    return P + n.replace("model.", "backbone.", 1)


def materialize(mod, prefix, skip_experts=True):
    mod.to_empty(device="cpu")
    for n, p in list(mod.named_parameters(recurse=True)):
        full = f"{prefix}.{n}"
        if skip_experts and ".experts." in f".{n}":
            continue
        c = ck_name(full)
        base = c[:-len(".weight")] if c.endswith(".weight") else None
        t = weight(base) if base and (base + ".weight") in wm else raw(c).float()
        assert t.shape == p.shape, (full, t.shape, p.shape)
        p.data = t
    for n, b in list(mod.named_buffers(recurse=True)):
        c = ck_name(f"{prefix}.{n}")
        if c in wm:
            b.data = raw(c).float()


def free(mod):
    mod.to("meta")


T0 = time.time()
stat = {"experts": 0}


def pre(block, args, kwargs):
    i = block.layer_idx
    materialize(block, f"model.layers.{i}")
    if block.block_type == "moe":
        mx = block.mixer
        h = args[0] if args else kwargs["hidden_states"]
        x = block.norm(h.to(block.norm.weight.dtype))
        ti, _ = mx.route_tokens_to_experts(mx.gate(x))
        used = sorted(set(ti.flatten().tolist()))
        E = mx.experts
        E.up_proj = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.moe_latent_size),
                                 requires_grad=False)
        E.down_proj = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.moe_latent_size, cfg.moe_intermediate_size),
                                   requires_grad=False)
        for e in used:
            b = f"{P}backbone.layers.{i}.mixer.experts.{e}"
            E.up_proj.data[e] = weight(b + ".up_proj")
            E.down_proj.data[e] = weight(b + ".down_proj")
        stat["experts"] += len(used)
    return None


def post(block, args, kwargs, out):
    free(block)
    print(f"  слой {block.layer_idx:2d} {block.block_type:9s} {time.time()-T0:6.0f} с", flush=True)
    return None


for b in M.layers:
    b.register_forward_pre_hook(pre, with_kwargs=True)
    b.register_forward_hook(post, with_kwargs=True)
with torch.no_grad():
    M.embeddings.to_empty(device="cpu"); M.embeddings.weight.data = raw(P + "backbone.embeddings.weight").float()
    M.norm_f.to_empty(device="cpu"); M.norm_f.weight.data = raw(P + "backbone.norm_f.weight").float()
    model.lm_head.to_empty(device="cpu"); model.lm_head.weight.data = weight(P + "lm_head")
    seq = json.load(open(sys.argv[1]))
    out = {}
    for name, ids in seq.items():
        t = time.time()
        lg = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
        out[name] = lg
        print(f"{name}: {len(ids)} ток, {time.time()-t:.0f} с, экспертов распаковано {stat['experts']}", flush=True)
    torch.save(out, sys.argv[2])
print("DONE", flush=True)
