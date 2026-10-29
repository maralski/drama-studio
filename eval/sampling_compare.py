# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Pairs a reliability variant against the shipped-default rows by condition
and seed, and reports what a sampling change does to prose: rows lost or
won, the word error rate shift with its standard error, the sentences
present (the cue study's check), speed, and how many rows would have run
into the app's text-derived frame cap.

  eval/.venv/bin/python -I eval/sampling_compare.py eval/out/reliability/rescored eval/out/reliability-space/rescored
"""
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "cues"))
from run import sentences_present  # noqa: E402  (eval/cues/run.py)

CONTROLS = {"no-retry", "reference-chunking"}
CUES = {"cues", "cloned cues"}
SLIDER = {"high-expressive", "low-expressive"}
BRACKET = re.compile(r"\[[^\]]*\]")


def load(out_dir, mode=None):
    rows = [json.loads(l) for p in sorted(Path(out_dir).glob("*.jsonl")) for l in p.read_text().splitlines() if l.strip()]
    return {(r["id"], r["seed"]): r for r in rows if mode is None or r.get("mode", "full") == mode}


def app_cap_seconds(item, chunks):
    """The app's cap summed over a row's chunks: 1.5 x expected plus 2 s each,
    plus 1.5 s per bracket cue. Row level, so an approximation."""
    expected = len(re.sub(r"\s+", " ", BRACKET.sub(" ", re.sub(r"<\|[^|]*\|>", " ", item["text"]))).strip()) / 16
    return 1.5 * expected + 2 * max(chunks, 1) + 1.5 * len(BRACKET.findall(item["text"]))


def describe(name, pairs, items):
    if not pairs:
        return
    a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
    ok_a = sum(r["verdict"] == "ok" for r in a); ok_b = sum(r["verdict"] == "ok" for r in b)
    lost = [(ra["name"], rb["verdict"], rb.get("wer")) for ra, rb in pairs if ra["verdict"] == "ok" and rb["verdict"] != "ok"]
    won = [(ra["name"], ra["verdict"]) for ra, rb in pairs if ra["verdict"] != "ok" and rb["verdict"] == "ok"]
    scored = [(ra, rb) for ra, rb in pairs if "wer" in ra and "wer" in rb]
    d = np.array([rb["wer"] - ra["wer"] for ra, rb in scored])
    sent_a = [all(sentences_present(items[ra["id"]]["text"], ra["hyp"])) for ra, _ in scored]
    sent_b = [all(sentences_present(items[rb["id"]]["text"], rb["hyp"])) for _, rb in scored]
    ratio_a = np.array([ra["spoken_seconds"] / ra["expected_seconds"] for ra, _ in scored])
    ratio_b = np.array([rb["spoken_seconds"] / rb["expected_seconds"] for _, rb in scored])
    dr = ratio_b - ratio_a
    frames_a = np.array([ra["frames"] for ra, _ in scored]); frames_b = np.array([rb["frames"] for _, rb in scored])
    cap_a = sum(ra["seconds"] > app_cap_seconds(items[ra["id"]], ra["chunks"]) for ra, _ in scored)
    cap_b = sum(rb["seconds"] > app_cap_seconds(items[rb["id"]], rb["chunks"]) for _, rb in scored)
    fps = [float(m.group(1)) for _, rb in scored for line in rb.get("chunk_log", []) for m in [re.search(r"\(([\d.]+) frames/s", line)] if m]
    print(f"\n{name}: {len(pairs)} pairs")
    print(f"  ok            default {ok_a} / {len(pairs)}   variant {ok_b} / {len(pairs)}   lost {len(lost)}   won {len(won)}")
    print(f"  all sentences default {sum(sent_a)} / {len(scored)}   variant {sum(sent_b)} / {len(scored)}")
    print(f"  WER           default {np.mean([ra['wer'] for ra, _ in scored]):.3f}   variant {np.mean([rb['wer'] for _, rb in scored]):.3f}   "
          f"paired shift {d.mean():+.3f} (standard error {d.std(ddof=1) / np.sqrt(len(d)):.3f}), {sum(d > 0)} rows worse, {sum(d < 0)} better")
    print(f"  spoken/expected default {ratio_a.mean():.2f}   variant {ratio_b.mean():.2f}   paired shift {dr.mean():+.3f} (standard error {dr.std(ddof=1) / np.sqrt(len(dr)):.3f})")
    print(f"  frames per row  default {frames_a.mean():.0f}   variant {frames_b.mean():.0f}   ({(frames_b.sum() / frames_a.sum() - 1) * 100:+.0f}%)")
    print(f"  retries       default {sum(ra.get('retries', 0) for ra, _ in scored)}   variant {sum(rb.get('retries', 0) for _, rb in scored)};   "
          f"eos default {sum(ra['eos'] for ra, _ in scored)}   variant {sum(rb['eos'] for _, rb in scored)}")
    print(f"  over the app's cap (row-level estimate) default {cap_a}   variant {cap_b}")
    print(f"  wall s/row    default {np.mean([ra['wall_seconds'] for ra, _ in scored]):.1f}   variant {np.mean([rb['wall_seconds'] for _, rb in scored]):.1f}"
          + (f"   variant load {np.mean([rb['load'] for _, rb in scored if 'load' in rb]):.1f}" if any('load' in rb for _, rb in scored) else "")
          + (f"   variant decode {np.mean(fps):.1f} frames/s" if fps else ""))
    for n, v, w in lost: print(f"    lost  {n:36s} {v:8s} wer={w}")
    for n, v in won: print(f"    won   {n:36s} (default was {v})")


def main():
    base_dir, var_dir = Path(sys.argv[1]), Path(sys.argv[2])
    items = {item["id"]: item for p in (HERE / "reliability/slices").glob("*.json") for item in json.loads(p.read_text())}
    base = load(base_dir, "full"); var = load(var_dir)
    keys = sorted(k for k in var if k in base)
    missing = sorted(k for k in base if k not in var)
    pairs = [(base[k], var[k]) for k in keys]
    print(f"{len(pairs)} pairs ({len(missing)} default rows without a variant row yet)")
    cat = lambda r: "cloned cues" if r["id"] == "clone-cues" else r["category"]
    describe("prose, defaults only (no cues, no controls, no slider rows)", [p for p in pairs if cat(p[0]) not in CONTROLS | CUES | SLIDER], items)
    describe("prose, all 130 (slider rows included)", [p for p in pairs if cat(p[0]) not in CONTROLS | CUES], items)
    describe("performance cues", [p for p in pairs if cat(p[0]) in CUES], items)
    describe("controls (retries off, reference chunking)", [p for p in pairs if cat(p[0]) in CONTROLS], items)
    describe("every condition", pairs, items)
    print("\nper category, ok default -> variant:")
    for c in sorted({cat(p[0]) for p in pairs}):
        sel = [p for p in pairs if cat(p[0]) == c]
        print(f"  {c:20s} {sum(a['verdict'] == 'ok' for a, _ in sel):2d} -> {sum(b['verdict'] == 'ok' for _, b in sel):2d} of {len(sel):2d}   "
              f"WER {np.mean([a['wer'] for a, _ in sel if 'wer' in a]):.3f} -> {np.mean([b['wer'] for _, b in sel if 'wer' in b]):.3f}   "
              f"spoken/expected {np.mean([a['spoken_seconds'] / a['expected_seconds'] for a, _ in sel if 'wer' in a]):.2f} -> {np.mean([b['spoken_seconds'] / b['expected_seconds'] for _, b in sel if 'wer' in b]):.2f}")


if __name__ == "__main__":
    main()
