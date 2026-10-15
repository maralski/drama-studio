"""Score dump b against reference dump a (docs/EVAL.md); exit 1 if a threshold fails."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

EXACT_KEYS = ("prompt_rows", "frames")
FLOAT_KEYS = ("hidden", "cand_logits", "tower_logits", "layer_out_step0", "tower_hidden_step0")
LOGIT_KEYS = ("cand_logits", "tower_logits")
COLUMNS = ("max_abs", "mean_abs", "rel_l2", "cosine", "top1", "kl")


def metrics(a: np.ndarray, b: np.ndarray, logits: bool) -> dict[str, float]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = a - b
    out = dict(
        max_abs=np.abs(diff).max(),
        mean_abs=np.abs(diff).mean(),
        rel_l2=np.linalg.norm(diff) / np.linalg.norm(a),
        cosine=(a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)),
    )
    if logits:
        rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        out["top1"] = (rows_a.argmax(-1) == rows_b.argmax(-1)).mean()
        log_a, log_b = log_softmax(rows_a, -1), log_softmax(rows_b, -1)
        out["kl"] = (np.exp(log_a) * (log_a - log_b)).sum(-1).mean()
    return {k: float(v) for k, v in out.items()}


def failures(values: dict[str, float], limits: dict[str, float]) -> list[str]:
    failed = []
    for name, limit in limits.items():
        metric, bound = name.rsplit("_", 1)
        value = values[metric]
        if (bound == "max" and not value <= limit) or (bound == "min" and not value >= limit):
            failed.append(f"{metric} {value:.3g} vs {bound} {limit:g}")
    return failed


def per_row_rel_l2(a: np.ndarray, b: np.ndarray) -> list[float]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return [float(np.linalg.norm(x - y) / np.linalg.norm(x)) for x, y in zip(a, b)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="reference dump")
    parser.add_argument("b", type=Path, help="dump under test")
    parser.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json")
    )
    parser.add_argument("--detail", action="store_true", help="relative L2 per row of axis 0")
    args = parser.parse_args()

    a, b = np.load(args.a), np.load(args.b)
    thresholds = json.loads(args.thresholds.read_text())
    for name, dump in (("a", a), ("b", b)):
        meta = json.loads(str(dump["meta"]))
        print(f"{name}: {meta.get('producer')} {meta.get('dtype')} {meta.get('mode')} ({getattr(args, name)})")

    failed = False
    for key in EXACT_KEYS:
        if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
            print(f"{key}: differ ({a[key].shape} vs {b[key].shape}); the dumps are not comparable")
            failed = True
    if failed:
        sys.exit(1)

    print(f"\n{'key':<20}" + "".join(f"{c:>11}" for c in COLUMNS) + "  status")
    for key in FLOAT_KEYS:
        if a[key].shape != b[key].shape:
            print(f"{key:<20} shape {a[key].shape} vs {b[key].shape}  FAIL")
            failed = True
            continue
        values = metrics(a[key], b[key], key in LOGIT_KEYS)
        cells = "".join(
            f"{values[c]:>11.3e}" if c in values else f"{'-':>11}" for c in COLUMNS
        )
        problems = failures(values, thresholds.get(key, {}))
        failed |= bool(problems)
        print(f"{key:<20}{cells}  {'FAIL ' + '; '.join(problems) if problems else 'ok'}")

    if args.detail:
        print("\nrelative L2 per row of axis 0")
        for key in FLOAT_KEYS:
            rows = per_row_rel_l2(a[key], b[key])
            print(f"{key:<20}" + " ".join(f"{r:.1e}" for r in rows))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
