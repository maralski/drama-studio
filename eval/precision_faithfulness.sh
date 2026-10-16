#!/bin/sh
# Teacher-forced faithfulness of every precision mode against the float32
# oracle: the 120-frame plain2 prompt (the README's table) and the 12-frame
# plain prompt, whose bf16 oracle run (eval/out/plain_ref_bf16.npz) places
# each mode against the bf16 noise floor. Writes <outdir>/<mode>_*.npz and
# <outdir>/<mode>_*.txt (the compare.py tables; their exit codes are ignored
# because the shipped thresholds are for the float32 engine).
# usage: eval/precision_faithfulness.sh <drama> <outdir> [modes...]
set -e
BIN=$1; OUT=$2; shift 2
MODES=${*:-"full balanced compact"}
PY=eval/.venv/bin/python
mkdir -p "$OUT"
PLAIN2="The quick brown fox jumps over the lazy dog. She sells sea shells by the sea shore."
PLAIN="<|speaker:0|> The quick brown fox jumps over the lazy dog."
for mode in $MODES; do
  "$BIN" dump --precision-mode "$mode" --text "$PLAIN2" --frames eval/out/plain2_frames.npy --out "$OUT/${mode}_plain2.npz" > "$OUT/${mode}_plain2_dump.txt" 2>&1
  $PY -I eval/compare.py eval/out/plain2_ref_f32.npz "$OUT/${mode}_plain2.npz" > "$OUT/${mode}_plain2_vs_f32.txt" 2>&1 || true
  "$BIN" dump --precision-mode "$mode" --text "$PLAIN" --frames eval/out/plain_frames.npy --out "$OUT/${mode}_plain.npz" > "$OUT/${mode}_plain_dump.txt" 2>&1
  $PY -I eval/compare.py eval/out/plain_ref_f32.npz "$OUT/${mode}_plain.npz" > "$OUT/${mode}_plain_vs_f32.txt" 2>&1 || true
  $PY -I eval/compare.py eval/out/plain_ref_bf16.npz "$OUT/${mode}_plain.npz" > "$OUT/${mode}_plain_vs_bf16.txt" 2>&1 || true
done
$PY -I eval/compare.py eval/out/plain_ref_f32.npz eval/out/plain_ref_bf16.npz > "$OUT/oracle_bf16_vs_f32.txt" 2>&1 || true
for f in "$OUT"/*_vs_*.txt; do echo "== $f"; cat "$f"; done
