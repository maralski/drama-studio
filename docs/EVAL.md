# Faithfulness evaluation

The oracle is the fish-speech v3.0.0-beta PyTorch implementation, vendored under
`eval/oracle/`, run on the CPU. The engine under test is the Metal engine, driven
through the `drama` CLI. Both write the same dump schema; `eval/compare.py`
scores one against the other.

## Prompt spec (JSON)

```json
{
  "text": "<|speaker:0|> Hello there.",
  "references": [
    {"transcript": "<|speaker:0|> The quick brown fox.", "codes": "ref1_codes.npy"}
  ],
  "history": []
}
```

`codes` is a path to an int64 `.npy` of shape `[10, T]` (codec codes of the
clip). `history` is a list of `{"text": ..., "codes": ...}` pairs, oldest first.

## Dump schema (`.npz`)

Positions: the prompt has `R` rows at positions `0..R-1`. Forced frames
`f_0..f_{F-1}` sit at positions `R..R+F-1`. Step `i` predicts `f_i` from the
hidden state at position `R-1+i`.

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `prompt_rows` | int64 | `[R, 11]` | the prompt as frame rows |
| `frames` | int64 | `[F, 11]` | the forced (or sampled) frames |
| `hidden` | float32 | `[F+1, 2048]` | final-norm backbone output at positions `R-1 .. R+F-1` |
| `cand_logits` | float32 | `[F+1, 4097]` | lm_head logits for the candidate ids, in ascending id order (eos first) |
| `tower_logits` | float32 | `[F, 9, 4096]` | head `k` logits for codebook `k+1` at step `i`, given the forced `c0..c_k` of `f_i` |
| `layer_out_step0` | float32 | `[40, 2048]` | residual stream after each backbone layer at position `R-1` |
| `tower_hidden_step0` | float32 | `[13, 2048]` | output of each tower layer at step 0 |
| `meta` | json string | | dtype, seed, sampling settings, timings |

Producers:

- `eval/dump_reference.py --spec spec.json --frames frames.npy --out ref.npz --dtype float32|bfloat16`
  When `--frames` is absent the oracle samples `--steps` frames itself and
  dumps them, so they can be forced through the engine.
- `drama dump --spec spec.json --frames frames.npy --out eng.npz`

## Metrics (`eval/compare.py a.npz b.npz`)

Per key: max abs error, mean abs error, relative L2 error, cosine. For logits:
top-1 agreement rate and mean KL(a || b) after the oracle's temperature. The
script prints a table and exits non-zero when a threshold in `eval/thresholds.json`
fails.

## End to end (`eval/e2e.py`)

A fixed sentence set (`eval/sentences.json`), with and without a reference
voice, three seeds each. For each run: engine frames via the `drama` CLI,
audio via the reference PyTorch codec on the CPU (`eval/decode_codes.py`),
Whisper small.en transcription and word error rate against the input text,
eos reached or not, trailing silent frames, and (`eval/speaker_similarity.py`)
the resemblyzer cosine between the reference clip and the output. Results land
in `eval/out/e2e/<tag>/report.json`; the README's hill-climb table is built from
those summaries. `eval/check_sampler.py` checks a sampled trace against the
engine's own logits.
