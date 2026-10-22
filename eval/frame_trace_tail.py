"""Tail and drift of `drama generate --trace-frames` CSVs: per file, the frame
interval's median, p90, p99 and max, the frames more than `--above` ms (default
2) over the median, and the median of the first and last thirds of each
session, so a stall tail and a clock drift within a run read as numbers.

  python3 eval/frame_trace_tail.py a.csv [b.csv ...] [--above 2] [--skip 2]"""
import csv, statistics, sys


def main():
    args = sys.argv[1:]
    above = 2.0; skip = 2
    if "--above" in args:
        i = args.index("--above"); above = float(args[i + 1]); del args[i:i + 2]
    if "--skip" in args:
        i = args.index("--skip"); skip = int(args[i + 1]); del args[i:i + 2]
    for path in args:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        by = {}
        for r in rows: by.setdefault(int(r["session"]), []).append(r)
        intervals, backbone, first, last = [], [], [], []
        for rs in by.values():
            rs.sort(key=lambda r: int(r["position"]))
            iv = [float(rs[i]["ready"]) - float(rs[i - 1]["ready"]) for i in range(skip, len(rs))]
            intervals += iv
            backbone += [float(r["stage:backbone"]) for r in rs[skip:] if "stage:backbone" in r]
            third = max(1, len(iv) // 3)
            first += iv[:third]; last += iv[-third:]
        iv = sorted(intervals); med = statistics.median(iv)
        tail = [x for x in iv if x > med + above]
        q = lambda p: iv[min(len(iv) - 1, int(p * (len(iv) - 1)))]
        print(f"{path}: {len(iv)} frames in {len(by)} sessions")
        print(f"  interval ms median {med:.2f} p90 {q(0.9):.2f} p99 {q(0.99):.2f} max {iv[-1]:.2f}; mean {statistics.fmean(iv):.2f}; frames/s {1000 / statistics.fmean(iv):.1f}")
        print(f"  tail: {len(tail)} frames more than {above:g} ms over the median" + (f": {' '.join(f'{x:.1f}' for x in tail[:12])}{' ...' if len(tail) > 12 else ''}" if tail else ""))
        print(f"  drift: first third median {statistics.median(first):.2f}, last third {statistics.median(last):.2f}")
        if backbone:
            print(f"  backbone stage ms median {statistics.median(backbone):.2f} p90 {sorted(backbone)[int(0.9 * (len(backbone) - 1))]:.2f} max {max(backbone):.2f}")


if __name__ == "__main__":
    main()
