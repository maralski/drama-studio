# Drama Studio design

A native macOS app that runs Fish Audio Drama 3 locally. The 35B hybrid MoE
backbone and the codebook tower run on hand-written Metal compute kernels. The
DAC audio codec runs through Core ML so the Neural Engine can take it. No MLX,
no PyTorch at runtime.

## Model facts (from `config.json` and fish-speech v3.0.0-beta)

Backbone `FishOmniTextModel`, hidden 2048, vocab 252288, 40 layers.

- Layer types: 3 x linear attention then 1 x full attention, repeated (30 + 10).
- Every layer has a MoE MLP: 256 routed experts, top 8, intermediate 512,
  softmax router, renormalized top-k weights, plus a shared expert (512) with a
  sigmoid gate.
- Full attention: 16 query heads, 2 KV heads, head dim 256, q/k zero-centered
  RMSNorm per head, partial NeoX RoPE on the first 64 dims (theta 1e7), sigmoid
  output gate from a second half of `q_proj` (rows are [q | gate] per head),
  scale 1/16.
- Linear attention (Gated DeltaNet): `in_proj_qkv` -> depthwise causal conv
  (kernel 4) -> SiLU -> split q,k (16 heads x 128) and v (32 heads x 128), q/k
  repeated 2x to 32 heads, L2-normalized, q scaled by 128^-0.5. beta =
  sigmoid(in_proj_b), decay = -exp(A_log) * softplus(in_proj_a + dt_bias).
  State per head is 128 x 128 fp32. Output goes through a gated RMSNorm per
  128-dim head: weight * rmsnorm(core) * silu(z), then `out_proj`.
- All RMSNorms are zero-centered: scale by (1 + weight), computed in fp32.
- Experts are FP8 e4m3 with fp32 block scales (128 x 128 blocks). Everything
  else is bf16. `quant_method` is `bsgemm-moe`.

Audio tower `FishOmniAudioDecoder`, dim 2048, 13 layers, each one transformer
block (16 heads, 2 KV heads, head dim 256, interleaved partial RoPE on 64 dims,
q/k zero-centered norm, output gate; MLP w1/w3/w2 with intermediate 6144).

- 4 shared layers then 9 residual layers; residual layer i has a head
  (2048 -> 4096) predicting codebook i+1.
- Each tower layer mixes an embedding e into the hidden h as
  `eh_proj(cat(enorm(e), hnorm(h)))`, then runs the block, then a final norm.
- A frame row is `[main_token, c0, ..., c9]`. For an audio row the main token
  is `c0 + 248099`. Text rows are `[token, 0 x 10]`.
- Backbone input for an audio row is `(embed(main) + codebook_emb[1][c1]) /
  sqrt(2)`; the first shared tower layer adds the sum of codebook embeddings
  2..9.
- Sampling: the main token is drawn from the 4097 candidates (eos 248046 and
  semantic ids 248099..252194) with top-k/top-p; if it repeats a main token in
  the last 16 frames it is redrawn with temperature 1.0, top-k 30, top-p 0.95.
  Defaults: temperature 0.7, top-k 30, top-p 0.7.

Codec `DAC` at 44.1 kHz. Encoder strides [2,4,8,8] (hop 512) then the
quantizer downsamples 2 x 2, so one frame is 2048 samples (21.53 frames/s).
Latent dim 1024. Quantizer: one semantic codebook (4096 entries, dim 8) plus 9
residual codebooks (1024 entries, dim 8), an 8-layer causal windowed
transformer before and after quantization, ConvNeXt blocks in the resampler.
Decoder: conv stack with Snake activations, decoder dim 1536, a 4-layer
transformer in the first decoder block. Everything is causal.

Prompt format (`build_drama3_prompt`):

```
<|im_start|>system\n
convert the provided text to speech              (no references)
convert the provided text to speech reference to the following:
\n\n[Reference 1]\nText: <|speaker:0|> transcript\nSpeech: <codes>   (per reference)
<|im_end|>\n
<|im_start|>user\n<|speaker:0|> text<|im_end|>\n
<|im_start|>assistant\n<|voice|>
```

Each literal is tokenized on its own. Long text is chunked at about 300 UTF-8
bytes and previous chunks are fed back as history turns.

## Components

```
Sources/DramaEngine        Swift library: loader, tokenizer, Metal graph, sampler
Sources/DramaEngine/Kernels/*.metal
Sources/drama              CLI: generate, dump (for the eval suite), bench, encode-ref
App/DramaStudio            SwiftUI app target (SwiftPM executable, bundled by tools/make_app.sh)
eval/                      Python oracle (vendored fish-speech modules) and the faithfulness suite
tools/convert_codec.py     coremltools export of the DAC encoder and decoder
```

## Data shapes (decided before any logic)

- `Frame`: 11 x Int32 `[main, c0..c9]`. The prompt and the generated sequence
  are `[Frame]`. The tower cache row p holds frame p+1 as its label.
- `ModelLayout`: parsed from the safetensors headers once. A `Tensor` is
  `(bufferIndex, byteOffset, shape, dtype)` into one of the 8 mmap'd shard
  buffers. Experts are addressed through a per-layer GPU-address table
  (`uint64[256][6]`: gate/up/down weight and scale) so no expert is copied.
- `DecodeState`: KV caches for the 10 full-attention layers and 13 tower
  layers, GDN conv and recurrent states for the 30 linear layers, the main-token
  history window (16) for repetition-aware sampling, and the current position.
  All GPU buffers, allocated once for the requested capacity.
- `GenerationConfig`: temperature, topK, topP, maxNewFrames, seed.
- `Precision` knob: fp32 activations everywhere, with an optional `mimicBF16`
  mode that rounds at the points where the reference rounds (Linear outputs,
  residual stream). The eval suite decides which mode ships.

## Kernel list

Elementwise and norms: `rmsnorm_zc` (also per-head for q/k), `gated_rmsnorm`,
`silu_mul`, `sigmoid_gate_mul`, `residual_add`, `embed_mix`.
Projections: `gemv_bf16` (decode), `gemm_bf16` (prefill, M <= 128), both read
bf16 as `ushort` and widen to fp32 exactly.
Attention: `rope_neox_partial`, `rope_interleaved_partial`, `attn_decode`
(GQA over the KV cache), `attn_prefill` (causal).
Gated DeltaNet: `gdn_conv_silu` (kernel 4 with carried state),
`gdn_recurrent` (one thread per head x value column, loops over the chunk's
tokens; the recurrent and chunked rules are the same math).
MoE: `moe_router` (softmax, top-8, renormalize), `moe_gate_up_fp8`,
`moe_down_fp8` (expert-major: one threadgroup per expert x row tile, it scans
the routing table for its tokens and exits if it has none; the FP8 byte is
widened through a 256-entry constant table and multiplied by its block scale).
Head: `logits_candidates` (lm_head rows for the 4097 candidates only).
Sampling runs on the CPU with a seeded generator, mirroring the reference's
sort / mask / exponential-noise argmax.

## Verification

Faithfulness is measured against the fish-speech PyTorch code running on the
CPU (oracle). Levels:

1. Tokenizer: Swift BPE vs `tokenizers` on a corpus, exact match.
2. Kernels: each kernel against a NumPy/PyTorch formula on random inputs.
3. Teacher-forced parity: the same prompt and the same frames through both
   implementations; per step compare backbone hidden state, candidate logits,
   and each tower head's logits (max abs error, KL, top-1 agreement).
4. End to end: generate a fixed sentence set with and without cloning, decode
   with the codec, and score with the oracle's teacher-forced NLL of the
   engine's frames, ASR word error rate (Apple Speech framework), and speaker
   similarity for cloned voices.
5. Hill-climb: iterate on the precision knobs and kernel numerics until the
   parity metrics stop moving and the end-to-end scores match the oracle's own
   samples.

## Decisions

- No MLX (user instruction). Own Metal kernels for the transformer stacks.
- Core ML for the codec because the Neural Engine has no public kernel API.
- Weights read in place from the Hugging Face cache; disk is nearly full.
- Experts stay FP8 in memory; exact to the release and half the bytes.
- Only candidate logits are computed; the reference discards the rest.
