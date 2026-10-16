#!/bin/sh
# Interleaved timing of the precision modes on one drama binary: for each run,
# every mode in turn does a GPU bench (one-row and prefill GPU times, wired
# memory after load) and one end-to-end generation with a 10 s reference clip
# (about 300 prompt rows, 600 frames, --no-codec so the GPU is the only limit).
# eval/precision_summary.py reads the output.
# usage: eval/precision_ab.sh <drama> <runs> <outdir> [modes...]
set -e
BIN=$1; RUNS=$2; OUT=$3; shift 3
MODES=${*:-"full balanced compact"}
mkdir -p "$OUT"
TEXT="When the old lighthouse keeper finally retired, the town threw him a party on the pier. Children ran between the tables, gulls circled overhead, and someone had baked a cake shaped like the lighthouse itself, complete with a tiny candle at the top."
REF="<|speaker:0|> Every morning I walk to the little bakery at the end of our street."
i=1
while [ "$i" -le "$RUNS" ]; do
  for mode in $MODES; do
    uptime >> "$OUT/load.txt"
    "$BIN" bench --precision-mode "$mode" > "$OUT/bench_${mode}_$i.txt" 2>&1
    "$BIN" generate --precision-mode "$mode" --seed 1 --text "$TEXT" --ref eval/out/perf/ref_215.npy --ref-text "$REF" --out "$OUT/gen_${mode}_$i.npy" --no-codec --warm --chunking reference --max-frames 600 > "$OUT/gen_${mode}_$i.txt" 2>&1
  done
  i=$((i + 1))
done
