"""Summarize eval/perf_ab.sh output: median and range per side of the bench
rows and of the end-to-end prefill seconds and frames per second."""
import re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])


def collect(side):
    rows, pre, fps, frames = {}, [], [], []
    for f in sorted(out.glob(f"bench_{side}_*.txt")):
        for line in f.read_text().splitlines():
            m = re.match(r"rows (\d+):\s+all=([\d.]+)ms.*?tower=([\d.]+)ms", line)
            if m:
                rows.setdefault(int(m[1]), []).append((float(m[2]), float(m[3])))
    for f in sorted(out.glob(f"gen_{side}_*.txt")):
        t = f.read_text()
        for m in re.finditer(r"(\d+) rows prefilled in ([\d.]+)s; (\d+) frames in ([\d.]+)s \(([\d.]+) frames/s[^)]*\)", t):
            pre.append((int(m[1]), float(m[2]))); frames.append(int(m[3])); fps.append(float(m[5]))
    return rows, pre, fps, frames


def fmt(v):
    return f"{statistics.median(v):.1f} [{min(v):.1f}..{max(v):.1f}] n={len(v)}"


for side in "AB":
    rows, pre, fps, frames = collect(side)
    print(f"side {side}")
    for r in sorted(rows):
        print(f"  bench rows {r:4d}: backbone ms {fmt([a for a, _ in rows[r]])}  tower ms {fmt([b for _, b in rows[r]])}")
    if pre:
        print(f"  e2e prefill of {pre[0][0]} rows: s {statistics.median([p for _, p in pre]):.2f} [{min(p for _, p in pre):.2f}..{max(p for _, p in pre):.2f}] n={len(pre)}")
        print(f"  e2e decode frames/s: {fmt(fps)}; frames per run {sorted(set(frames))}")
