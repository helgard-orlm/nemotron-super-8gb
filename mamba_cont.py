"""Nemotron-H: дочитывание НЕСКОЛЬКИХ токенов поверх заполненного кэша + снимки кэша.

transformers 5.9 с заполненным кэшем умеет только 1 токен (in_proj(h.squeeze(1)) + selective_state_update).
Здесь: при кэше и L>1 — свёртка с тремя прошлыми входами из conv_states и скан с initial_states.
Проверено на ядрах (14.09): продолжение 120+80 = скан целиком до 0.003.

  install(model)            — ставит путь (нужны быстрые ядра: PYTHONPATH с kernels 0.12)
  snap = snapshot(cache)    — копия кэша в RAM (pinned)
  cache = restore(snap)     — копия обратно на GPU (снимок остаётся цел)
"""
import copy
import sys

import torch

_ATTRS = ("keys", "values", "conv_states", "recurrent_states")


def install(model):
    layers = model.model.layers
    mixers = [b.mixer for b in layers if b.block_type == "mamba"]
    cls = type(mixers[0])
    mod = sys.modules[cls.__module__]
    if not getattr(mod, "is_fast_path_available", False):
        raise RuntimeError("быстрые ядра Mamba не загружены — продолжение пачкой не поставлено")
    if getattr(cls, "_cont_installed", False):
        return
    orig = cls.cuda_kernels_forward

    def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
        bsz, L, _ = hidden_states.shape
        if cache_params is None or L == 1 or not cache_params.has_previous_state(self.layer_idx):
            return orig(self, hidden_states, cache_params, attention_mask)
        layer = cache_params.layers[self.layer_idx]
        proj = self.in_proj(hidden_states)
        gate, xBC, dt = torch.split(proj, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        K = self.conv_kernel_size
        # conv_states = последние K входов (включая последний обработанный) ⇒ прошлые K-1 = [..., 1:]
        xin = torch.cat([layer.conv_states[..., 1:].to(xBC.dtype), xBC.transpose(1, 2)], dim=-1)
        layer.conv_states.copy_(xin[..., -K:])
        xBC = mod.causal_conv1d_fn(x=xin, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                                   activation=self.activation)[..., K - 1:].transpose(1, 2)
        gs = self.n_groups * self.ssm_state_size
        x, B, C = torch.split(xBC, [self.intermediate_size, gs, gs], dim=-1)
        A = -torch.exp(self.A_log.float())
        dt_limit = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
        st0 = layer.recurrent_states
        y, st = mod.mamba_chunk_scan_combined(
            x.view(bsz, L, -1, self.head_dim), dt, A,
            B.view(bsz, L, self.n_groups, -1), C.view(bsz, L, self.n_groups, -1),
            chunk_size=self.chunk_size, D=self.D, z=None, dt_bias=self.dt_bias,
            initial_states=st0, seq_idx=None, return_final_states=True, dt_softplus=True, **dt_limit)
        layer.recurrent_states.copy_(st.to(st0.dtype))
        y = self.norm(y.view(bsz, L, -1), gate)
        return self.out_proj(y)

    cls.cuda_kernels_forward = cuda_kernels_forward
    cls._cont_installed = True


def install_flash(model, name="fa2_gqa"):
    """Внимание через FlashAttention-2 (HF kernels-community/flash-attn2, сборка torch211-cu128 работает на sm_120:
    vs float32-эталон 0.26%, 14.09). Без маски и без repeat_kv: GQA 32/2 и causal «снизу-справа» (кусок поверх кэша)
    FA2 делает сам. sdpa с маской раздувал KV 2→32 головы ⇒ OOM на 128k. Только batch=1 без паддинга."""
    from kernels import get_kernel
    from transformers import AttentionInterface
    fa = get_kernel("kernels-community/flash-attn2")

    def fa2_gqa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if attention_mask is not None:
            raise RuntimeError("fa2_gqa: маска не поддерживается (batch=1 без паддинга)")
        o = fa.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                               dropout_p=0.0, softmax_scale=scaling, causal=True)
        o = o[0] if isinstance(o, tuple) else o
        return o, None

    AttentionInterface.register(name, fa2_gqa)
    model.config._attn_implementation = name
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = name
    return fa


def prefill(model, ids, cache=None, chunk=4096):
    """Читать ids кусками поверх cache (None = с нуля). Возвращает (cache, логиты последнего токена)."""
    from transformers import DynamicCache
    if cache is None:
        cache = DynamicCache(config=model.config)
    lg = None
    for i in range(0, ids.shape[1], chunk):
        o = model(input_ids=ids[:, i:i + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, lg = o.past_key_values, o.logits[0, -1]
        del o
    return cache, lg


def snapshot(cache, pin=True):
    snap = copy.copy(cache)
    snap.layers = []
    for lay in cache.layers:
        c = copy.copy(lay)
        for a in _ATTRS:
            t = getattr(lay, a, None)
            if isinstance(t, torch.Tensor):
                cpu = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin)
                cpu.copy_(t, non_blocking=False)
                setattr(c, a, cpu)
        snap.layers.append(c)
    return snap


def restore(snap, device="cuda"):
    cache = copy.copy(snap)
    cache.layers = []
    for lay in snap.layers:
        c = copy.copy(lay)
        for a in _ATTRS:
            t = getattr(lay, a, None)
            if isinstance(t, torch.Tensor):
                setattr(c, a, t.to(device, copy=True))
        if hasattr(c, "device") and isinstance(getattr(lay, "device", None), torch.device):
            c.device = torch.device(device)
        cache.layers.append(c)
    return cache


def nbytes(snap):
    return sum(getattr(l, a).numel() * getattr(l, a).element_size()
               for l in snap.layers for a in _ATTRS if isinstance(getattr(l, a, None), torch.Tensor))
