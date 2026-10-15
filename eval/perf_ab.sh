#!/bin/sh
# Interleaved A/B timing of two drama binaries: GPU bench plus one end-to-end
# generation with a 10 s reference clip (about 300 prompt rows, 600 frames).
# usage: eval/perf_ab.sh <baseline drama> <candidate drama> <runs> <outdir>
set -e
A=$1; B=$2; RUNS=$3; OUT=$4
mkdir -p "$OUT"
TEXT="When the old lighthouse keeper finally retired, the town threw him a party on the pier. Children ran between the tables, gulls circled overhead, and someone had baked a cake shaped like the lighthouse itself, complete with a tiny candle at the top."
REF="<|speaker:0|> Every morning I walk to the little bakery at the end of our street."
i=1
while [ "$i" -le "$RUNS" ]; do
  for side in A B; do
    if [ "$side" = A ]; then BIN=$A; else BIN=$B; fi
    uptime >> "$OUT/load.txt"
    "$BIN" bench > "$OUT/bench_${side}_$i.txt" 2>&1
    "$BIN" generate --seed 1 --text "$TEXT" --ref eval/out/perf/ref_215.npy --ref-text "$REF" --out "$OUT/gen_${side}_$i.npy" --no-codec --warm --chunking reference --max-frames 600 > "$OUT/gen_${side}_$i.txt" 2>&1
  done
  i=$((i + 1))
done
