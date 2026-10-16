"""Summarize eval/precision_ab.sh and eval/precision_memory.sh output per
mode: medians and ranges of the one-row backbone and tower GPU times, the
128-row prefill, the end-to-end frames per second and GPU ms per frame,
wired memory after warm-up (from the mem_ files, whose runs were quiesced),
and whether every run of a mode produced the same frames."""
import re, statistics, sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
modes = sys.argv[2:] or ["full", "balanced", "compact"]


def fmt(v, digits=1):
    if not v:
        return "n/a"
    return f"{statistics.median(v):.{digits}f} [{min(v):.{digits}f}..{max(v):.{digits}f}] n={len(v)}"


def collect(mode):
    rows, wired, resident, footprint, cache = {}, [], [], [], None
    for f in sorted(out.glob(f"bench_{mode}_*.txt")):
        text = f.read_text()
        for line in text.splitlines():
            m = re.match(r"rows (\d+):\s+all=([\d.]+)ms.*?tower=([\d.]+)ms", line)
            if m:
                rows.setdefault(int(m[1]), []).append((float(m[2]), float(m[3])))
        c = re.search(r"quantized cache: (.*)", text)
        if c:
            cache = c[1]
    for f in sorted(out.glob(f"mem_{mode}_*.txt")):
        mem = re.search(r"memory after warm-up: wired ([-+][\d.]+) GB since process start, resident ([\d.]+) GB, footprint ([\d.]+) GB", f.read_text())
        if mem:
            wired.append(float(mem[1])); resident.append(float(mem[2])); footprint.append(float(mem[3]))
    pre, fps, gpu_ms, frames, eos, arrays = [], [], [], [], [], []
    for f in sorted(out.glob(f"gen_{mode}_*.txt")):
        t = f.read_text()
        for m in re.finditer(r"(\d+) rows prefilled in ([\d.]+)s; (\d+) frames in ([\d.]+)s \(([\d.]+) frames/s, GPU ([\d.]+) ms/frame\) eos=(\w+)", t):
            pre.append(float(m[2])); frames.append(int(m[3])); fps.append(float(m[5])); gpu_ms.append(float(m[6])); eos.append(m[7] == "yes")
        npy = f.with_suffix(".npy")
        if npy.exists():
            arrays.append(np.load(npy))
    identical = all(a.shape == arrays[0].shape and np.array_equal(a, arrays[0]) for a in arrays[1:]) if arrays else None
    return rows, wired, resident, footprint, cache, pre, fps, gpu_ms, frames, eos, identical


for mode in modes:
    rows, wired, resident, footprint, cache, pre, fps, gpu_ms, frames, eos, identical = collect(mode)
    print(f"mode {mode}")
    if cache:
        print(f"  {cache}")
    print(f"  wired after warm-up GB {fmt(wired, 2)}  resident GB {fmt(resident, 2)}  footprint GB {fmt(footprint, 2)}")
    for r in sorted(rows):
        print(f"  bench rows {r:4d}: backbone ms {fmt([a for a, _ in rows[r]])}  tower ms {fmt([b for _, b in rows[r]])}")
    if pre:
        print(f"  e2e prefill s {fmt(pre, 2)}")
        print(f"  e2e decode frames/s {fmt(fps)}; GPU ms/frame {fmt(gpu_ms)}; frames per run {sorted(set(frames))}; eos {sum(eos)}/{len(eos)}; runs identical: {identical}")
