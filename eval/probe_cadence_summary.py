#!/usr/bin/env python3
"""Summarises eval/probe_cadence.sh runs per condition: frames per second and
GPU ms per frame from the --trace-frames CSVs, the probe's share of the decode
from the chunk log, where each chunk ended, and whether the frames matched the
shipped cadence's.

  python3 eval/probe_cadence_summary.py <outdir> [--skip 2]"""
import re, statistics, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_trace_summary import load, metrics  # noqa: E402

CHUNK = re.compile(r"(\d+) frames in ([\d.]+)s \(([\d.]+) frames/s, GPU ([\d.]+) ms/frame\).*?spoken=([\d.]+)s cap=(\d+) end=(\w+) silence_at=(\S+)(?: probe=([\d.]+)s)?")


def runs(out):
    by = {}
    for p in sorted(Path(out).glob("gen_*_*.txt")):
        name, i = p.stem[4:].rsplit("_", 1)
        by.setdefault(name, []).append((int(i), p))
    return by


def main():
    args = sys.argv[1:]
    skip = 2
    if "--skip" in args:
        k = args.index("--skip"); skip = int(args[k + 1]); del args[k:k + 2]
    out = Path(args[0])
    by = runs(out)
    ref = {}
    for name, items in by.items():
        for i, p in items:
            ref.setdefault(i, {})[name] = np.load(out / f"gen_{name}_{i}.npy") if (out / f"gen_{name}_{i}.npy").exists() else None
    print(f"{'condition':8s} {'runs':>4s} {'frames/s':>14s} {'GPU ms/frame':>14s} {'interval ms':>12s} {'frames':>7s} {'decode s':>9s} {'probe s':>8s} {'probe duty':>10s} {'ends':>22s}")
    for name, items in by.items():
        fps, gpu, iv, frames, dec, probe, ends, lates = [], [], [], [], [], [], {}, []
        for i, p in items:
            trace = out / f"trace_{name}_{i}.csv"
            if trace.exists():
                m = metrics(load(trace), skip)
                fps.append(1000 / statistics.fmean(m["interval"]))
                gpu.append(statistics.median(m["gpu"]))
                iv.append(statistics.median(m["interval"]))
            n = d = pr = 0
            for line in p.read_text().splitlines():
                mm = CHUNK.search(line)
                if not mm: continue
                n += int(mm[1]); d += float(mm[2]); pr += float(mm[9] or 0)
                ends[mm[7]] = ends.get(mm[7], 0) + 1
                if mm[8] != "-" and mm[7] == "silence":
                    trailing = int(mm[1]) - round(float(mm[5]) * 44100 / 2048)
                    lates.append(trailing - 44)
            frames.append(n); dec.append(d); probe.append(pr)
        def med(v, f="{:.1f}"):
            return (f.format(statistics.median(v)) + " [" + f.format(min(v)) + ".." + f.format(max(v)) + "]") if v else "-"
        duty = [p / d for p, d in zip(probe, dec) if d]
        print(f"{name:8s} {len(items):4d} {med(fps):>14s} {med(gpu, '{:.2f}'):>14s} {med(iv, '{:.2f}'):>12s} {med(frames, '{:.0f}'):>7s} {med(dec):>9s} {med(probe, '{:.2f}'):>8s} {med(duty, '{:.0%}'):>10s} {str(ends):>22s}" + (f"  stop late {sorted(lates)}" if lates else ""))
    print("\nframes against p16, same round (identical / differ at chunk boundaries):")
    for i, conds in sorted(ref.items()):
        base = conds.get("p16")
        if base is None: continue
        cells = []
        for name, arr in conds.items():
            if name == "p16" or arr is None: continue
            same = arr.shape == base.shape and bool((arr == base).all())
            cells.append(f"{name}={'identical' if same else f'{arr.shape[0]} vs {base.shape[0]} frames'}")
        print(f"  round {i}: " + ", ".join(cells))


if __name__ == "__main__":
    main()
