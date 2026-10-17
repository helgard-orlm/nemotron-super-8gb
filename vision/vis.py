"""#401 зрение Nemotron-3.5-Super-VL: башня C-RADIO + проектор (код NVIDIA из nvl/, веса BF16 из первого файла оригинала).
  load(device, dtype) -> Vision; v.embed(pixel_values) -> [N_img_tokens, 4096]
Ключи переименовываются как register_radio_conversion_mapping (qkv режется на q/k/v по dim 0)."""
import os, re, sys, time
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvl.configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config
from nvl.modeling_radio import RadioModel
from nvl.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3VisionProjector

ORIG = os.environ.get("NS_ORIG", "/data/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
REN = [("radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"),
       ("radio_model.model.patch_generator.embedder", "embeddings.patch_projection"),
       ("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding"),
       ("radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"),
       ("radio_model.model.blocks", "encoder.layer"),
       ("attn.proj", "attention.output.dense"),
       ("radio_model.input_conditioner", "input_conditioner"),
       ("radio_model.summary_idxs", "summary_idxs")]
PROJ = {"mlp1.0.": "mlp1.norm.", "mlp1.1.": "mlp1.linear1.", "mlp1.3.": "mlp1.linear2.",
        "vision_projector.vision_final_layernorm.": "vision_final_layernorm."}


class Vision:
    def __init__(self, device="cpu", dtype=torch.float32):
        t0 = time.time()
        cfg = NemotronH_Omni_Reasoning_V3_Config.from_pretrained(ORIG)
        self.cfg = cfg
        with torch.device("meta"):
            vm = RadioModel(cfg.vision_config)
            pj = NemotronH_Omni_Reasoning_V3VisionProjector(cfg)
        vm.make_preprocessor_external()
        f = safe_open(f"{ORIG}/model-00001-of-00088.safetensors", "pt")
        sv, sp = {}, {}
        for k in f.keys():
            if k.startswith("vision_model."):
                n = k[len("vision_model."):]
                for a, b in REN:
                    n = n.replace(a, b)
                t = f.get_tensor(k)
                if "attn.qkv" in n:
                    for part, x in zip(("query", "key", "value"), t.chunk(3, dim=0)):
                        sv[n.replace("attn.qkv", f"attention.attention.{part}")] = x
                else:
                    sv[n] = t
            else:
                for a, b in PROJ.items():
                    if k.startswith(a):
                        sp[b + k[len(a):]] = f.get_tensor(k)
        self.load_report = {}
        # layer_scale*.lambda1 нет в чекпойнте (_keys_to_ignore_on_load_missing) — HF ставит layerscale_value
        for n, p in vm.named_parameters():
            if re.search(r"layer_scale\d+\.lambda1$", n) and n not in sv:
                sv[n] = torch.full(p.shape, float(cfg.vision_config.layerscale_value), dtype=torch.bfloat16)
        for name, mod, sd in (("vision", vm, sv), ("projector", pj, sp)):
            want = set(mod.state_dict().keys())
            got = set(sd)
            self.load_report[name] = {"missing": sorted(want - got), "unexpected": sorted(got - want), "n": len(got & want)}
            mod.load_state_dict({k: v for k, v in sd.items() if k in want}, strict=False, assign=True)
        bad = {k: r for k, r in self.load_report.items() if r["missing"]}
        if bad:
            raise RuntimeError(f"ключи не сошлись: { {k: (r["missing"][:12], r["unexpected"][:12]) for k, r in bad.items()} }")
        self.vm = vm.to(device, dtype).eval()
        self.pj = pj.to(device, dtype).eval()
        self.vm.config.torch_dtype = dtype
        self.device, self.dtype = device, dtype
        self.load_s = time.time() - t0

    @torch.no_grad()
    def embed(self, pixel_values):
        """pixel_values [n_tiles,3,H,W] (после процессора NVIDIA) -> [n_tiles*H*W/16²/4, 4096]"""
        pv = pixel_values.to(self.device, self.dtype)
        return self.pj(pv, self.vm).reshape(-1, self.pj.mlp1.linear2.out_features)


def processor():
    """процессор из локальных классов nvl/ (без trust_remote_code): настройки — processor_config.json оригинала"""
    import json
    from transformers import AutoTokenizer
    from nvl.processing_nemotron_h_omni import NemotronH_Omni_Reasoning_V3Processor
    from nvl.image_processing_nemotron_h_omni import NemotronH_Omni_Reasoning_V3ImageProcessor
    pc = json.load(open(f"{ORIG}/processor_config.json"))
    ic = {k: v for k, v in pc.pop("image_processor").items() if k not in ("auto_map", "image_processor_type")}
    ip = NemotronH_Omni_Reasoning_V3ImageProcessor(**ic)
    tok = AutoTokenizer.from_pretrained(os.path.join(os.environ.get("NS_DIR", "/fast/nemsuper"), "tok"))
    tpl = open(f"{ORIG}/chat_template.jinja").read()
    kw = {k: v for k, v in pc.items() if k not in ("auto_map", "processor_class")}
    return NemotronH_Omni_Reasoning_V3Processor(image_processor=ip, tokenizer=tok, chat_template=tpl, **kw)


def _pre(mod, args):
    for p in mod.parameters(recurse=False):
        p.data = p._host.to("cuda", non_blocking=True)


def _post(mod, args, out):
    for p in mod.parameters(recurse=False):
        p.data = p._host


BIG = 64 * 2**20


def _chunked(m):
    w = m.weight.data.pin_memory(); b = m.bias.data.pin_memory() if m.bias is not None else None
    m.weight.data = w
    if b is not None:
        m.bias.data = b
    rows = max(1, (50 * 2**20) // (w.shape[1] * w.element_size()))

    def fwd(x):
        outs = []
        for i in range(0, w.shape[0], rows):
            wc = w[i:i + rows].to("cuda", non_blocking=True)
            bc = b[i:i + rows].to("cuda", non_blocking=True) if b is not None else None
            outs.append(torch.nn.functional.linear(x, wc, bc))
        return torch.cat(outs, -1)

    m.forward = fwd


def stream_cuda(v):
    """башню и проектор считать на карте, веса держать в закреплённой RAM и подавать по модулю (hook до/после):
    на карте одновременно веса одного модуля (≤210 МБ, mlp1.linear1) + активации; буферы (мелкие) — на карте."""
    import types
    for mod in (v.vm, v.pj):
        for m in mod.modules():
            ps = list(m.parameters(recurse=False))
            if isinstance(m, torch.nn.Linear) and m.weight.numel() * m.weight.element_size() > BIG:
                _chunked(m)                     # mlp1.linear1/2 (200/168 МБ): веса кусками ~50 МБ по выходам
                continue
            if ps:
                for p in ps:
                    p._host = p.data.pin_memory()
                    p.data = p._host
                m.register_forward_pre_hook(_pre)
                m.register_forward_hook(_post)
            for n, b in list(m._buffers.items()):
                if b is not None:
                    m._buffers[n] = b.to("cuda")
    v.device = "cuda"
    return v
