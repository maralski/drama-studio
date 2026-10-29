#!/usr/bin/env python3
"""GPU ms per frame by frame position within the chunk, per condition, from
eval/probe_cadence.sh traces: the frames before a gated probe's first reading
run with no probe, so the same positions across conditions in the same round
show the probe's cost with the machine's drift cancelled.

  python3 eval/probe_cadence_positions.py <outdir> [<outdir> ...] --gates 72,171,128"""
import statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_trace_summary import load  # noqa: E402


def main():
    args = sys.argv[1:]
    gates = [72, 171, 128]
    if "--gates" in args:
        k = args.index("--gates"); gates = [int(x) for x in args[k + 1].split(",")]; del args[k:k + 2]
    rows = {}
    for out in args:
        for p in sorted(Path(out).glob("trace_*_*.csv")):
            name, i = p.stem[6:].rsplit("_", 1)
            by = {}
            for r in load(p): by.setdefault(r["session"], []).append(r)
            before, after = [], []
            for s, (sid, rs) in enumerate(sorted(by.items())):
                rs.sort(key=lambda r: r["position"])
                gate = gates[s] if s < len(gates) else gates[-1]
                for j, r in enumerate(rs):
                    if j < 2: continue
                    gpu = r["gpu_end"] - r["gpu_start"]
                    (before if r["position"] - rs[0]["position"] + 1 < gate - 16 else after).append(gpu)
            rows.setdefault(name, []).append((f"{Path(out).name}/{i}", statistics.median(before), statistics.median(after), len(before), len(after)))
    print(f"{'condition':9s} {'run':12s} {'before gate':>12s} {'after gate':>11s}   (median GPU ms/frame; before: chunk positions under gate-16, after: the rest)")
    for name, items in rows.items():
        for run, b, a, nb, na in items:
            print(f"{name:9s} {run:12s} {b:12.2f} {a:11.2f}   ({nb}/{na} frames)")
        print(f"{name:9s} {'median':12s} {statistics.median(x[1] for x in items):12.2f} {statistics.median(x[2] for x in items):11.2f}")


if __name__ == "__main__":
    main()
