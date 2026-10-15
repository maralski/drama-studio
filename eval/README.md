# Faithfulness oracle

The oracle is the fish-speech `v3.0.0-beta` PyTorch code for Drama 3, vendored
under `oracle/` (see `oracle/LICENSE_NOTE.md`). It runs on the CPU and writes
the dump schema in `docs/EVAL.md`. `compare.py` scores the Metal engine's dump
against it.

## Setup

The scripts use the virtual environment at `eval/.venv` of the main checkout
(torch 2.8.0, safetensors, tokenizers, loguru, numpy, soundfile,
descript-audio-codec, scipy). Run every command from the repository root:

```sh
PY=eval/.venv/bin/python
```

The weights are read in place from the Hugging Face cache
(`models--fishaudio--drama-3`, snapshot `e0a9218a`). Pass `--checkpoint DIR` to
use another copy.

## Make the prompt specs

```sh
$PY eval/make_spec.py
```

This writes two specs:

- `specs/plain.json` has one sentence and no reference.
- `specs/clone.json` has one reference clip. The script speaks the clip with
  macOS `say`, converts it to 44.1 kHz mono with `ffmpeg`, and encodes it with
  the vendored DAC codec on the CPU. The clip and its codes (`[10, T]` int64)
  go to `data/`, which git ignores.

Codes paths in a spec are relative to the spec file.

## Dump the oracle

Sample frames and dump them (the first run of a new prompt):

```sh
$PY eval/dump_reference.py --spec eval/specs/plain.json --steps 12 \
    --dtype float32 --out eval/out/plain_ref_f32.npz
```

Force given frames instead of sampling (`--frames` is an `[F, 11]` int64
`.npy`):

```sh
$PY -c "import numpy as np; np.save('eval/out/plain_frames.npy', np.load('eval/out/plain_ref_f32.npz')['frames'])"
$PY eval/dump_reference.py --spec eval/specs/plain.json --frames eval/out/plain_frames.npy \
    --dtype bfloat16 --out eval/out/plain_ref_bf16.npz
```

Options:

- `--dtype float32` is the sharp reference for finding engine bugs.
  `--dtype bfloat16` is the precision the released model runs in. The FP8
  experts stay FP8 in both cases. Each forward dequantizes only the experts it
  selects, into `--dtype`.
- `--seed`, `--temperature`, `--top-k`, `--top-p` control sampling. The
  defaults are the reference defaults (0.7, 30, 0.7) and seed 0.

The oracle runs on the CPU only. MPS has no float8 dtype, so the FP8 expert
buffers cannot live there.

The script prints the wall time for loading, prefill and each step. The dump's
`meta` key holds the same timings, the peak RSS and the settings. On an M3 Max
a run takes about 40 s to load, 40 GiB of RAM, and 10 s per step. Most of a
step is the depthwise `conv1d` of the 30 Gated DeltaNet layers, which PyTorch
runs on its slow per-group CPU path.

## Compare two dumps

```sh
$PY eval/compare.py eval/out/plain_ref_f32.npz eval/out/plain_eng.npz
```

The first dump is the reference. For each float key the script prints the
max and mean absolute error, the relative L2 error `|a - b| / |a|`, and the
cosine. For `cand_logits` and `tower_logits` it also prints the top-1
agreement and the mean KL(a || b) of the softmax at temperature 1.
`--detail` adds the relative L2 error per row of axis 0 (per step, per layer).

`prompt_rows` and `frames` must match exactly, or the dumps are not
comparable. The script exits 1 when a limit in `thresholds.json` fails.
A limit is named `<metric>_max` or `<metric>_min`. The shipped limits are for
a float32 engine against the float32 oracle. Tighten them as the engine
improves, or pass `--thresholds FILE` for another set.
