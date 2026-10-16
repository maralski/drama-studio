# Drama Studio design

Author: Peter Marelas.

A native macOS app that runs Fish Audio Drama 3 locally. The 35B hybrid MoE
backbone and the codebook tower run on hand-written Metal compute kernels. The
DAC audio codec runs through Core ML, the only route to the Neural Engine; in
practice it runs on the CPU there (see the README). No MLX, no PyTorch at
runtime.

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
Sources/drama              CLI: generate, dump (for the eval suite), bench, tune, profile, kbench, decode
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
Projections: `gemv_bf16_t<tile>` for one row (and for outputs of 64 columns or
fewer at any row count), `gemm_bf16_<bm>x<bn>x<bk>` for two rows or more: a
simdgroup-matrix GEMM with one output tile per threadgroup, the bf16 weights
widened to fp32 in threadgroup memory. Both form the same fp32 products; only
the summation order differs, and a row's result never depends on which chunk
it is in (one tile shape for every M keeps it so). The token tile, the rows
per threadgroup and the GEMM tile are fields of the `KernelSchedule` below.
Attention: `rope_neox`, `rope_interleaved`, `attention` (GQA over the KV cache,
online softmax, one threadgroup per query row and head, the keys split over 4,
8 or 16 simdgroups by the schedule).
Gated DeltaNet: `gdn_conv_silu` (kernel 4 with carried state),
`gdn_recurrent_p<parts>` (128 x parts threads per head: `parts` threads per
state column, the two per-token dot products combined from `parts` partial
sums in a fixed order; the recurrence runs token by token and is
chunk-invariant; parts is 1, 2, 4 or 8 by the schedule).
MoE: `moe_router` (softmax, top-8, renormalize), `moe_bucket` (per-expert row
lists and the active-expert list), `moe_gate_up_fp8` and `moe_down_fp8` for one
row (expert-major, one simdgroup per output row, the FP8 byte widened through a
256-entry constant table and multiplied by its block scale), and for two rows
or more `moe_gate_up_fp8_tiled_<bn>x<rows>x<bk>` and
`moe_down_fp8_tiled_<bn>x<rows>x<bk>`: one threadgroup per (active expert, bn
output columns), the expert's rows gathered in blocks of 8 through simdgroup
matrices, each weight tile dequantized once per pass of up to `rows` rows.
Every tile accumulates an output in the same 8-wide order over K, so the
tiles are bit-identical with each other. At 128 rows nearly every expert is active, so a
prefill chunk reads all 30 GB of expert weights once; the prefill chunk is 512
rows so a 10 s reference clip is one chunk.
Head: the candidate logits are two `gemv_bf16` calls over the lm_head rows of
eos and the semantic band, into one 4097-float buffer.
Kernel schedule: `KernelSchedule` (`Model/KernelSchedule.swift`) holds one value
per process for each of these choices, with the domain of each written next
to its kernel: the GEMV token tile and simdgroups per threadgroup (one pair
for decode, one for narrow prefill projections), the GEMM tile, the MoE tile,
the MoE row-kernel simdgroups, the attention split (decode and prefill) and
the GDN parts (decode and prefill). Path choice between kernel families
depends on M == 1 against M > 1 and on N only, so the prefix-cache invariant
holds whatever the schedule. `Autotune` (`Autotune.swift`) fills it on first
load for a (chip, macOS version, kernel source digest, model snapshot) key:
nine entries, each one kernel family's real workload on the resident weights
rotated over ten layers, every candidate validated against the default at a
relative L2 of 1e-5 on the same seeded inputs (the MoE entries on a real
hidden state, since noise routes evenly and misleads the tile choice), the
valid ones timed in five interleaved rounds, the fastest kept when it beats
the default by 5 percent. Then a whole-forward check times 128- and 300-row
and one-row forwards on both schedules and reverts a side unless the tuned
schedule wins by 2 percent at every row count. `TuningCache` stores the
report as JSON under `~/Library/Application Support/DramaStudio/Tuning/`; a
hit measures nothing. `drama schedule-ab` compares schedule files in one
process, the instrument for isolating one field.
Sampling: `sample_main` and `sample_tower` (one threadgroup per draw) run the
reference's sort / mask / exponential-noise argmax on the GPU with the same
xoshiro256** stream as the CPU `Sampler`, which stays as the reference the
parity test checks against. The CPU works in double; the kernels work in
double-float (hi + lo fp32 pairs, exp by range reduction and a degree-13
series, log by Newton steps on that exp) so every top-p cutoff and argmax
decision matches. The top of the distribution is extracted one element at a
time with simdgroup reductions, since at most top-k elements are kept.
A decode frame is one command buffer: main sampler, tower with a sampler after
each head, backbone step for the new row. The samplers write the frame row and
the tower's code inputs on the GPU, and the next frame's command buffer is
committed before the current one is waited for. The generator state chains
through two GPU slots so the CPU copies back the state of each finished frame
without a race; the draws of a frame still in flight when a session ends are
never observed, as on the CPU.

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
  Measured outcome: the ANE compile did not finish for the shipped shape and
  most of the decoder is CPU-only in the compute plan, so the codec ships on
  the CPU.
- The engine runs in fp32 after widening bf16 weights; the `mimicBF16` knob
  exists but was not evaluated, because the fp32 path already matches the fp32
  oracle at 1e-6 and sits far below the bf16 noise floor.
- Weights read in place from the Hugging Face cache; disk is nearly full.
- Experts stay FP8 in memory; exact to the release and half the bytes.
- Only candidate logits are computed; the reference discards the rest.
- Kernel schedules are tuned per machine, not shipped as constants, after a
  review of Magnitude's approach: borrow the tune, validate and cache idea
  without the dependency, on our own kernels. Only dispatch shapes and
  template instantiations are searched; every fp32 product stays the same and
  only the summation order may change, so the tuned engine is held to the same
  faithfulness table as the default one.
