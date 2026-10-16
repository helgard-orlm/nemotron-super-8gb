# Nemotron-3.5-Super 120B (75 GB NVFP4) on an 8 GB GPU — experts streamed from NVMe

A from-scratch inference engine that runs `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`
(88 layers: 40 Mamba-2 + 40 MoE + 8 attention; 512 experts, top-22, latent MoE) on a single
**RTX 5060 8 GB** at **~7.3–7.6 tokens/s** in chat. Only the dense part of the model lives on the GPU;
the 59 GiB of routed experts are read on demand from an NVMe drive, through a RAM cache, one layer at a time.

Output is checked against the Hugging Face reference implementation (`transformers` 5.9 `NemotronHForCausalLM`,
fp32 on the CPU, weights taken from the original checkpoint and dequantized with transformers' own FP4 code) —
see [Correctness](#correctness).

Batch size 1, text only (the vision tower and the MTP head are not wired in). Code comments are in Russian.

Same family of engines: [Qwen3.8-Flash-Next on 8 GB](https://github.com/helgard-orlm/qwen-flash-next-8gb).

## Hardware it was built on

| part | value |
|---|---|
| GPU | RTX 5060, 8 GB (Blackwell, sm_120), PCIe 5.0 ×8 — measured host→GPU copy ~28 GB/s, zero-copy read ~26 GB/s |
| CPU | Intel Core Ultra 5 225F |
| RAM | 31 GiB |
| model disk | PCIe Gen5 NVMe, used only for model files |

## The model, in bytes

Measured from the safetensors headers of the NVFP4 checkpoint:

| part | size | where it lives |
|---|---|---|
| routed experts (40 layers × 512) | 59.1 GiB | NVMe → RAM cache → GPU, on demand |
| Mamba `in_proj`/`out_proj` (FP8, per-tensor scale) | 4.1 GiB | GPU, kept in FP8 |
| shared experts (NVFP4) | 0.9 GiB | GPU, kept in NVFP4 |
| latent projections 4096↔1024, router (BF16) | 0.8 GiB | GPU |
| attention q/k/v/o (FP8), lm_head (FP8) | 0.8 GiB | GPU |
| embeddings (BF16) | 1.0 GiB | RAM (row lookup on the CPU) |

One routed expert works in the 1024-wide latent space: `up 2688×1024` + `down 1024×2688`, ReLU², no gate =
**3.10 MB in NVFP4** (block of exactly 3,096,576 bytes = 756 pages of 4 KiB). 22 experts × 40 layers ⇒
**2.7 GB of experts per token** before caching.

## The method

1. **Split the model** (`repack.py`). All non-expert text tensors are copied byte-for-byte into
   `nonexpert.safetensors` (7.6 GiB). On the GPU they stay in the checkpoint's own formats (FP8 / NVFP4) and are
   multiplied by our Triton GEMV kernels (`dense_q.py`), so the resident part is **6.6 GiB VRAM**.
2. **One contiguous block per expert** in `experts.bin`: `up_w | up_s | down_w | down_s`; one `pread` with
   `O_DIRECT` into page-aligned pinned RAM per expert. Repacking 59 GiB takes ~2.5 min.
3. **Skeleton from transformers.** Mamba-2 mixers, attention, the router (sigmoid + correction bias, top-22,
   ×5.0) and the latent projections are the transformers `NemotronH` code. We replace the experts module, the
   dense linears and, on the decode step, the MoE forward itself (so disk reads overlap the shared expert and
   the latent projection).
4. **Expert cache** (`ExpertCache`): RAM LRU (14 GiB = 24% of all experts), 32 reader threads, two banks of GPU
   slots. Because routing is concentrated, **24% of the experts in RAM give 81–84% hits** on the decode step.
5. **Prefetch into the gap** (`NS_PF=4`): the next MoE layer's router applied to the current input guesses its
   experts (~72% right); the misses among the guesses are read while the GPU works.
6. **Skip the ReLU² zeros** (the part that gave the last +12%). After ReLU², **92.5% of the expert's neurons are
   exactly zero** (live 7.5% on average, 1–16.5% by layer), and a zero neuron does not need its column of
   `down`. The NVFP4 scale is shared by 16 consecutive inputs, so `down` is re-laid-out once on disk
   (`repack_T.py`, a pure byte permutation, same size): the 4-bit values of each neuron become contiguous, the
   scales are grouped per 16-neuron group. On the decode step only `up` (1.55 MB) is copied to the GPU; the
   `down` kernel (`_down_groups`) reads straight from pinned RAM over PCIe (address table, no copy) — only the
   groups that contain a live neuron, and inside them only the live neurons' bytes.
7. **Prompt path.** Experts are processed in batches of 32 with the disk read of batch *i+1* in flight while
   batch *i* computes. Prompts up to 64 tokens use a "pairs" kernel (expert × token straight from 4-bit
   weights); longer ones dequantize to BF16 + GEMM. In the re-laid-out file the prompt path converts blocks back
   on the GPU with one Triton kernel (0.84 ms per 32 blocks).
8. **Chat server** (`nem_super_server.py`): OpenAI-compatible `/v1/chat/completions` with streaming; dialog
   snapshots of the cache in RAM (~170 MB: 8 KV layers + 40 Mamba states), so the next turn only reads its new
   tail (a multi-token continuation of the Mamba state, `mamba_cont.py`); FlashAttention-2 for attention;
   a `-think` model id turns on the reasoning mode of the chat template.

## Results

Greedy decode, chat prompts of 31–40 tokens, warm cache:

| version | decode tok/s | prompt (32–40 tok) |
|---|---|---|
| first run (no prefetch) | 6.1 | 11 s |
| + prefetch P=2 / P=4 | 6.45 / 6.5 | — |
| + pipelined prompt + pairs kernel | 6.5 | 3.2–3.8 s |
| **+ ReLU² zero skip (current)** | **7.3–7.6** | **4.7–5.0 s** (server, incl. template) |

Where the decode time goes (before the zero skip; measured while another job loaded the CPU, so absolute values
are approximate): ~180 ms/token total, **~100 ms host→GPU copies**, ~3 ms waiting for the disk (the readers hide
it), ~75 ms compute + Python. So the PCIe bus, not the NVMe, is the bottleneck on this machine — which is why the
zero skip helped and a bigger RAM cache would help less than one would expect.

Load time 7–13 s; VRAM 6.6 GiB (peak ~7.2 GiB); RAM: 14 GiB pinned cache + 1 GiB embeddings.

## Correctness

**Against the reference** (`tests/ref_full.py` + `tests/eng_vs_ref.py`): transformers `NemotronHForCausalLM` in
fp32 on the CPU, layer by layer, weights read from the **original** checkpoint (not our repack), NVFP4 unpacked
with transformers' `Fp8Dequantize._unpack_fp4` (low nibble first — the same convention as llama.cpp's
`_nvfp4_pack`), E4M3 scale per 16 values × `weight_scale_2`; FP8 = weight × `weight_scale`. Activations are not
quantized in either (NVIDIA's `input_scale` is for W4A4/W8A8 on vLLM).

| text | PPL reference | PPL engine | same argmax | engine's greedy answer = reference choice | KL mean / max |
|---|---|---|---|---|---|
| English paragraph, 51 tok | 5.233 | 5.280 | 50/51 | — | 0.0028 / 0.025 |
| chat answer (TCP vs UDP), 95 tok | 6.592 | 6.714 | 93/95 | **63/64** | 0.0015 / 0.013 |
| chat answer (Russian), 103 tok | 5.873 | 5.854 | 103/103 | **64/64** | 0.0020 / 0.023 |

The one differing answer token is a true tie: the reference itself scores `-layer` 68.916 vs ` layer` 68.807
(gap 0.109, smaller than one BF16 step at that magnitude).

**Internal checks:** expert kernels vs torch fp64 on real experts (rel. 1.9e-8); pairs kernel vs step kernel
(1.4e-7); zero-skip kernel vs full kernel (4e-8…1.7e-7) with a negative control (down of other experts → 12–88%
off); controlled swap of nibbles → PPL 18,784 (the unpacking order is tested, not assumed); repack and re-layout
checked against the original checkpoint byte-for-byte on 200 experts each; prefetch on/off gives identical tokens.

**Determinism.** Within one process every run is bit-identical (different cache states, prefetch off, fully
synchronous mode). Across processes the `mamba-ssm` Triton kernels autotune by timing and may pick different
configs → different rounding; `TRITON_CACHE_AUTOTUNING=1` stores the choice on disk and makes separate processes
bit-identical. A bug of ours found this way: one `index_add_` with repeated token indices (atomics in random
order) made the PPL of the same input drift 5.19–5.29; fixed with a per-(token, k) buffer summed in fixed order.

## What did not work / was rejected

- Putting the prompt's experts into the RAM cache: the prompt got slower (+0.6 s) and decode did not gain.
- Prefetch P=6: more wasted reads, slower than P=4.
- Zero-skip per neuron (scales re-read per neuron): 1.2–1.9 ms/layer — slower than copying the whole block.
  Per 16-neuron group with masked loads: 0.63 ms/layer (incl. the up kernel). Used.
- A pairs kernel reading the re-laid-out format directly: correct, but slower (12.5 ms) than "convert back + old
  kernel" once the conversion became one Triton kernel.
- Keeping recently used experts on the GPU: 40% of a layer's experts are needed again on the next token, but
  holding them takes ~2.7 GiB of VRAM that does not exist on an 8 GB card next to the 6.6 GiB dense part.
  Not implemented.
- llama.cpp as the reference: it supports this model (PR #30520), but converting to GGUF needs another ~77 GB of
  disk; the transformers CPU reference needs none.

## Running it

Not tested end-to-end from a clean clone; these are the steps used on the machine above.

```bash
pip install -r requirements.txt            # see the file for exact versions; mamba/conv1d/FA2 kernels come from HF kernels-community
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --local-dir /data/nemotron-super-nvfp4
export NS_ORIG=/data/nemotron-super-nvfp4 NS_DIR=/fast/nemsuper   # NS_DIR: fast NVMe, ~67 GB free
python repack.py                            # experts.bin + nonexpert.safetensors + tok/  (~2.5 min)
python check_repack.py                      # 200 experts vs the original, byte-for-byte
python repack_T.py                          # re-layout down for the zero skip (~2 min, in place; --back undoes it)
cd "$NS_DIR" && cp /path/to/repo/*.py .     # the server is started from NS_DIR
NS_RAM_GB=14 NS_PF=4 NS_PROMPT_CACHE_T=0 TRITON_CACHE_AUTOTUNING=1 NS_PORT=9809 python -u nem_super_server.py
curl localhost:9809/v1/models
```

Environment: `NS_DIR` (engine files), `NS_ORIG` (original checkpoint, used by the repack/check scripts and the
reference), `NS_RAM_GB` (RAM expert cache), `NS_PF` (prefetch), `NS_PAIR_T` (prompt length for the pairs
kernel), `NS_PROMPT_CACHE_T` (prompt experts into the RAM cache up to this length; 0 = never),
`NS_PORT`, `NS_MAX_SEQ`, `NS_MODEL_ID`.

Versions used: torch 2.11.0+cu128, triton 3.6.0, transformers 5.9.0, kernels 0.12.3, CUDA driver 595.71.

## Files

| file | what |
|---|---|
| `nem_super.py` | engine: loading, `ExpertCache`, experts module, decode-step MoE forward |
| `qmoe_ns.py` | Triton kernels: experts step / pairs / zero-skip down, layout conversions |
| `dense_q.py` | FP8 / NVFP4 GEMV kernels for the dense part (shared with our Nemotron Nano engine) |
| `mamba_cont.py` | multi-token continuation of the Mamba state over a filled cache, cache snapshots, FA2 |
| `nem_super_server.py` | OpenAI-compatible chat server |
| `repack.py`, `check_repack.py`, `repack_T.py` | build and verify the engine files |
| `tests/` | the measurements behind the numbers above (reference, kernels, determinism, profiles) |

## License

Code: Apache-2.0. The model weights are not included and are under NVIDIA's license (OpenMDW-1.1 per the model
card).
