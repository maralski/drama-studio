#!/bin/sh
# Interleaved A/B of `drama bench` between two binaries, no generation. Writes
# bench_<side>_<i>.txt in perf_ab.sh's layout so eval/perf_ab_summary.py
# summarizes it; load.txt gets an uptime line before every run. Pass a third
# binary with 256-thread samplers as LABEL to run eval/quiet_label.sh before
# the first round and after every fifth.
# usage: eval/bench_ab.sh <A drama> <B drama> <rounds> <outdir> [LABEL=<drama>]
set -e
A=$1; B=$2; RUNS=$3; OUT=$4; LABEL=${5#LABEL=}
mkdir -p "$OUT"
i=1
while [ "$i" -le "$RUNS" ]; do
  if [ -n "$LABEL" ] && [ $(( (i - 1) % 5 )) -eq 0 ]; then eval/quiet_label.sh "$LABEL" "$OUT" "round$i" | tee -a "$OUT/labels.txt"; fi
  for side in A B; do
    if [ "$side" = A ]; then BIN=$A; else BIN=$B; fi
    uptime >> "$OUT/load.txt"
    "$BIN" bench > "$OUT/bench_${side}_$i.txt" 2>&1
  done
  i=$((i + 1))
done
if [ -n "$LABEL" ]; then eval/quiet_label.sh "$LABEL" "$OUT" "end" | tee -a "$OUT/labels.txt"; fi
