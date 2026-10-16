#!/bin/sh
# Wired memory of each precision mode after load and warm-up, from the
# process's own delta of the host wired count (SystemMemory). Metal releases a
# process's wired pages a few seconds after it exits, so each run waits for
# the previous one to settle; a back-to-back series reads the previous
# process's teardown instead. Writes <outdir>/mem_<mode>_<i>.txt, which
# eval/precision_summary.py reads.
# usage: eval/precision_memory.sh <drama> <runs> <outdir> [modes...]
set -e
BIN=$1; RUNS=$2; OUT=$3; shift 3
MODES=${*:-"full balanced compact"}
mkdir -p "$OUT"
i=1
while [ "$i" -le "$RUNS" ]; do
  for mode in $MODES; do
    sleep 20
    vm_stat | grep "wired" >> "$OUT/mem_before.txt"
    "$BIN" bench --warm --precision-mode "$mode" > "$OUT/mem_${mode}_$i.txt" 2>&1
  done
  i=$((i + 1))
done
grep -H "memory after" "$OUT"/mem_*_*.txt
