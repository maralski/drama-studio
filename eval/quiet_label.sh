#!/bin/sh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Labels a measurement with the GPU's state right before it: a 120-frame
# generation on a drama binary with 256-thread samplers, traced per frame, and
# the sampler stages summed. Under another GPU client the samplers are the
# stages that stall first, so their GPU ms per frame says whether the machine
# is quiet (about 0.7 ms) in a way the ioreg utilization counter does not.
# usage: eval/quiet_label.sh <drama> <outdir> <label> [--schedule-file s.json]
set -e
BIN=$1; OUT=$2; LABEL=$3; shift 3
mkdir -p "$OUT"
TEXT="When the old lighthouse keeper finally retired, the town threw him a party on the pier. Children ran between the tables, gulls circled overhead, and someone had baked a cake shaped like the lighthouse itself, complete with a tiny candle at the top."
REF="<|speaker:0|> Every morning I walk to the little bakery at the end of our street."
CSV="$OUT/label_$LABEL.csv"
uptime >> "$OUT/label_load.txt"
ps -Ao %cpu,comm | grep -E 'Resources/WindowServer|MacOS/UniversalControl' | grep -v grep >> "$OUT/label_load.txt"
"$BIN" generate "$@" --warm --no-codec --seed 1 --max-frames 120 --chunking reference --trace-frames "$CSV" --trace-stages --text "$TEXT" --ref eval/out/perf/ref_215.npy --ref-text "$REF" --out "$OUT/label_$LABEL.npy" > "$OUT/label_$LABEL.txt" 2>&1
python3 - "$CSV" "$LABEL" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows = [r for r in rows if int(r["position"]) >= 2]
med = lambda k: statistics.median(float(r[k]) for r in rows)
samplers = med("stage:sample.main") + med("stage:sample.tower")
frame = statistics.median(float(b["ready"]) - float(a["ready"]) for a, b in zip(rows, rows[1:]) if a["session"] == b["session"])
print(f"label {sys.argv[2]}: samplers {samplers:.2f} ms/frame (main {med('stage:sample.main'):.2f} + tower {med('stage:sample.tower'):.2f}), backbone {med('stage:backbone'):.2f}, frame {frame:.2f} ms, {len(rows)} frames")
PY
