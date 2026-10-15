# Drama Studio

A native macOS app that runs Fish Audio's [Drama 3](https://huggingface.co/fishaudio/drama-3)
text-to-speech model locally on Apple silicon. The 35B mixture-of-experts
backbone and the codebook tower run on hand-written Metal compute kernels that
read the released safetensors weights in place. The DAC audio codec runs
through Core ML so the Neural Engine can take it. No MLX, no PyTorch at runtime.

Built with Fish Audio. The model weights are under the
[Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE);
cloning a real person's voice without their consent is not permitted.

## What you get

- **Script editor** with import of `.txt`, `.md`, `.rtf`, `.rtfd`, `.html`,
  `.docx`, `.srt` and `.vtt`, drag and drop, and a screenplay mode that turns
  `Name: line` dialogue into Drama 3 speaker tags (up to 16 speakers).
- **Voice tuning**: temperature, top-p, top-k, frames per chunk, chunking
  policy (one sentence per turn, or the reference's 300-byte chunks), seed, and
  an insert menu for performance cues such as `[laughs]`, `[sighs]` and
  `<whisper>…</whisper>`.
- **Voice cloning** from one or more reference clips (any format AVFoundation
  reads) plus their transcripts, kept in a local voice library.
- **Streaming playback** as chunks finish, scrubbing, and export to WAV or M4A.

## Requirements

- Apple silicon Mac with 64 GB of unified memory or more (the weights are
  41.5 GB and are mapped, not copied; 128 GB was used for development).
- macOS 15 or newer, Xcode 26 command line tools.
- The model in the Hugging Face cache:

```bash
uvx --from huggingface_hub hf download fishaudio/drama-3
```

- The Core ML codec models, built once from `codec.pth` with
  `tools/convert_codec.py` (see `tools/README` in that script's header). They
  land in `~/Library/Application Support/DramaStudio/Codec/`.

## Build and run

```bash
swift build -c release
tools/make_app.sh            # assembles dist/DramaStudio.app
open dist/DramaStudio.app
```

Command line:

```bash
.build/release/drama generate --text "Hello there." --out frames.npy
.build/release/drama bench
```

## How it is built

- `Sources/DramaEngine/Weights`: the eight safetensors shards are memory-mapped
  (`MAP_SHARED`) and wrapped as Metal buffers without copying. Experts stay in
  FP8 E4M3 with their 128x128 block scales and are decoded inside the kernel
  through a 256-entry table; every other weight is bf16 widened to fp32 in the
  kernel.
- `Sources/DramaEngine/Kernels`: zero-centered RMSNorm, bf16 GEMV with a
  register-resident token tile, the Gated DeltaNet depthwise conv and
  recurrence, GQA attention with an online softmax over an fp32 KV cache,
  partial RoPE (NeoX for the backbone, interleaved for the tower), the MoE
  router and expert-major FP8 kernels that dispatch only over active experts,
  and the codebook tower's embedding mixing.
- `Sources/DramaEngine/Model`: the backbone and tower runners, the reference's
  sampler (top-k, top-p, exponential-noise argmax, repetition-aware retry of the
  main token), prefill and decode sessions, and the chunked frame generator.
- `Sources/DramaEngine/Prompt`: the `reference-bracket-v1` prompt builder, the
  reference's chunker and history policy, plus a sentence-per-turn mode.
- `Sources/DramaEngine/Tokenizer`: a byte-level BPE tokenizer that matches
  Hugging Face `tokenizers` on a 661-case corpus.
- `Sources/DramaEngine/Codec`: Core ML wrappers for the DAC encoder (reference
  clips to codes) and decoder (codes to 44.1 kHz audio).
- `App/DramaStudio`: the SwiftUI app.
- `eval/`: the faithfulness suite (see `docs/EVAL.md`).

Design notes and decisions are in `docs/DESIGN.md`; the decision trail of the
build is `.audit/drama-studio.tsv`.

## Faithfulness

The engine is checked against the fish-speech v3.0.0-beta PyTorch
implementation running on the CPU in float32, teacher-forced on the same
prompt and frames. Numbers from `eval/compare.py`:

| quantity | relative L2 vs float32 oracle | top-1 agreement | mean KL |
| --- | --- | --- | --- |
| backbone hidden state (121 positions) | 6.5e-7 | | |
| main-token candidate logits | 3.3e-7 | 1.000 | 1.5e-12 |
| nine tower head logits | 5.0e-7 | 1.000 | 4.9e-12 |
| residual stream after each of the 40 layers | under 1e-6 each | | |

For scale, the released bf16 numerics differ from the float32 oracle by a
relative L2 of 5.2e-2 on the hidden state with a top-1 agreement of 0.846. The
Metal engine sits at float32 precision.

The sampler is checked separately: over 1174 sampled picks, none falls outside
the reference's top-k / top-p set, and the count of rank-0 picks matches the
expected value within noise.

End-to-end, generations that finish transcribe word-perfectly under Whisper.
The reference implementation and the engine share the model's habit of
stopping early or trailing into silence on multi-sentence prompts without a
reference voice; the sentence-per-turn chunking mode exists for that reason.
See `eval/README.md` for how to reproduce every number.

## Performance (M3 Max, 128 GB)

| step | time |
| --- | --- |
| weights mapped and kernels compiled | 0.4 s |
| first touch of the 41.5 GB (parallel pre-read) | 4 s |
| backbone decode, one frame | 25 ms GPU |
| tower decode, one frame | 9 ms GPU |
| end-to-end decode | about 32 frames/s (1.5x real time) |
| prefill, 128 rows | 1.1 s backbone, 0.3 s tower |

## Licenses

The app code is MIT. The vendored reference code under `eval/oracle/` and the
model weights are under the Fish Audio Community License; see
`eval/oracle/LICENSE_NOTE.md`.
