#!/usr/bin/env python3
"""Summarizes `drama generate --trace-frames` CSVs: where each decode frame's
wall time goes, in milliseconds per frame, median and mean over the frames of
every session after its first `--skip` frames.

  python3 eval/frame_trace_summary.py trace_1.csv [trace_2.csv ...] [--skip 2]

interval     ready[i] - ready[i-1]: wall time per frame as the caller sees it
gpu          gpu_end - gpu_start of frame i's command buffer
kernel       kernel_end - kernel_start
gpu_idle     gpu_start[i] - gpu_end[i-1]: GPU time with no frame buffer running
queued_lead  gpu_end[i-1] - commit[i]: how long frame i was committed before
             the previous buffer finished; negative means the GPU waited on
             the CPU's commit
sched        gpu_start - commit, when the buffer was committed after the GPU
             went idle: the commit-to-start latency
wake         ready - gpu_end: GPU completion to the CPU's wait returning
cpu_between  step_enter[i+1] - step_exit[i]: the caller's work between frames
encode       commit - encode_start: CPU time to encode a frame buffer
Stage columns, when present, are summed GPU ms per stage encoder."""
import csv, statistics, sys


def load(path):
    with open(path) as f:
        return [{k: (float(v) if k not in ("frame", "session", "position") else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def metrics(rows, skip):
    out = {}
    def add(k, v): out.setdefault(k, []).append(v)
    by_session = {}
    for r in rows: by_session.setdefault(r["session"], []).append(r)
    for rs in by_session.values():
        rs.sort(key=lambda r: r["position"])
        for i, r in enumerate(rs):
            if i < skip: continue
            p = rs[i - 1]
            add("interval", r["ready"] - p["ready"])
            add("gpu", r["gpu_end"] - r["gpu_start"])
            add("kernel", r["kernel_end"] - r["kernel_start"])
            add("gpu_idle", r["gpu_start"] - p["gpu_end"])
            lead = p["gpu_end"] - r["commit"]
            add("queued_lead", lead)
            if lead < 0: add("sched", r["gpu_start"] - r["commit"])
            add("wake", r["ready"] - r["gpu_end"])
            add("encode", r["commit"] - r["encode_start"])
            if i + 1 < len(rs): add("cpu_between", rs[i + 1]["step_enter"] - r["step_exit"])
            for k, v in r.items():
                if k.startswith("stage:"): add(k, v)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    skip = 2
    if "--skip" in args:
        i = args.index("--skip"); skip = int(args[i + 1]); del args[i:i + 2]
    if not args: sys.exit(__doc__)
    for path in args:
        m = metrics(load(path), skip)
        n = len(m.get("interval", []))
        print(f"{path}: {n} frames")
        for k, v in m.items():
            extra = f"  ({len(v)} frames)" if k == "sched" else ""
            print(f"  {k:24s} median {statistics.median(v):8.3f}  mean {statistics.fmean(v):8.3f}  p90 {sorted(v)[int(0.9 * (len(v) - 1))]:8.3f}{extra}")
        if "interval" in m:
            print(f"  frames/s from mean interval: {1000 / statistics.fmean(m['interval']):.1f}")


if __name__ == "__main__":
    main()
