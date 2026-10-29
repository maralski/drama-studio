# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""How late the silence stop fires, from the cue harness's rows.

Each chunk line the drama CLI logs carries `spoken=<s>` (the chunk's audio
less its trailing silence), `end=<why>` and `silence_at=<frame>` (the frame at
which the audio first held the configured silence, or `-`). A chunk that ended
with at least 2 s (44 frames) of trailing silence met the stop's condition
`trailing - 44` frames before it ended: for a chunk the stop ended that is how
late the stop fired, and for one that eos or the frame cap ended it is how long
the stop had already failed to act. A chunk whose audio held the silence
earlier (`silence_at`) but ended with less trailing silence paused for 2 s and
spoke again; the stop firing there is by design, and the retry rule covers it.

  eval/.venv/bin/python -I eval/cues/probe_latency.py eval/out/cues-before eval/out/cues-after
"""
import json, re, sys
from pathlib import Path
from statistics import median

CHUNK = re.compile(r"(\d+) frames in .*? spoken=([\d.]+)s cap=(\d+) end=(\w+) silence_at=(\S+)")
FRAMES_PER_SECOND = 44100 / 2048
STOP_FRAMES = 44


def chunks(out_dir):
    for p in sorted(Path(out_dir).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for entry in row.get("chunk_log", []):
                m = CHUNK.search(entry)
                if not m:
                    continue
                frames, spoken, cap, end, at = int(m[1]), float(m[2]), int(m[3]), m[4], m[5]
                trailing = frames - round(spoken * FRAMES_PER_SECOND)
                yield row, frames, cap, end, trailing, (None if at == "-" else int(at))


def report(out_dir):
    rows = list(chunks(out_dir))
    if not rows:
        print(f"{out_dir}: no rows"); return
    for variant in sorted({r[0]["variant"] for r in rows}):
        sel = [r for r in rows if r[0]["variant"] == variant]
        runs = {r[0]["name"]: r[0] for r in sel}
        present = sum(1 for r in runs.values() if r.get("all_sentences"))
        loads = [r["load"] for r in runs.values() if "load" in r]
        load = f", load {min(loads):.1f} to {max(loads):.1f}" if loads else ""
        ends = {}
        for _, _, _, end, _, _ in sel:
            ends[end] = ends.get(end, 0) + 1
        print(f"{out_dir} variant {variant}: {len(runs)} runs, {len(sel)} chunks (the kept attempt of each), ends {ends}{load}")
        print(f"  all sentences present: {present} of {len(runs)}")
        met = [(row["name"], frames, cap, end, trailing) for row, frames, cap, end, trailing, _ in sel if trailing >= STOP_FRAMES]
        paused = sum(1 for _, _, _, _, trailing, at in sel if at is not None and trailing < STOP_FRAMES)
        print(f"  chunks that ended with 2 s of trailing silence: {len(met)}; chunks that paused 2 s and spoke again: {paused}")
        for why in sorted({end for _, _, _, end, _ in met}):
            late = sorted(trailing - STOP_FRAMES for _, _, _, end, trailing in met if end == why)
            print(f"    ended by {why}: {len(late)}, frames late median {median(late):.0f}, range {late[0]} to {late[-1]}, all {late}")
        for name, frames, cap, end, trailing in met:
            print(f"      {name:28s} {end:9s} {frames} frames, cap {cap}, trailing silence {trailing} frames, {trailing - STOP_FRAMES} late")


if __name__ == "__main__":
    for out_dir in sys.argv[1:] or ["eval/out/cues-before", "eval/out/cues-after"]:
        report(out_dir)
