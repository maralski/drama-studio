"""Stall tail of an interleaved bench A/B (eval/bench_ab.sh or perf_ab.sh
output): per side, the rows-1 cells more than `--above` ms (default 2) over
their column's median, with the column medians and ranges, so a binary whose
frame stalls under another GPU client shows as a count, not a wider range.

  python3 eval/bench_tail.py <outdir> [--rows 1] [--above 2]"""
import re, statistics, sys
from pathlib import Path

args = sys.argv[1:]
rows_wanted = int(args[args.index("--rows") + 1]) if "--rows" in args else 1
above = float(args[args.index("--above") + 1]) if "--above" in args else 2.0
out = Path(args[0])

for side in "AB":
    cols = {}
    files = sorted(out.glob(f"bench_{side}_*.txt"), key=lambda p: int(p.stem.split("_")[-1]))
    for f in files:
        for line in f.read_text().splitlines():
            if not line.startswith(f"rows {rows_wanted}:"):
                continue
            for name, ms in re.findall(r"(\S+)=([\d.]+)ms", line):
                cols.setdefault(name, []).append(float(ms))
    if not cols:
        continue
    total = sum(len(v) for v in cols.values())
    tail = {k: [x for x in v if x > statistics.median(v) + above] for k, v in cols.items()}
    n_tail = sum(len(v) for v in tail.values())
    print(f"side {side}: {len(files)} runs, {n_tail} of {total} rows-{rows_wanted} cells more than {above:g} ms over their column median")
    for k, v in cols.items():
        flag = f"  tail {len(tail[k])}: {' '.join(f'{x:.1f}' for x in tail[k])}" if tail[k] else ""
        print(f"  {k:22s} median {statistics.median(v):6.1f} [{min(v):.1f}..{max(v):.1f}]{flag}")
