#!/bin/sh
# The end-to-end speech suite for every precision mode: 36 runs each (six
# texts, plain and cloned, three seeds), scored by Whisper and resemblyzer.
# Results land in eval/out/e2e/<tag>/report.json with tag "precision-<mode>".
# usage: eval/precision_e2e.sh <drama> [modes...]
set -e
BIN=$1; shift
MODES=${*:-"full balanced compact"}
PY=eval/.venv/bin/python
for mode in $MODES; do
  $PY -I eval/e2e.py --drama "$BIN" --tag "precision-$mode" --extra chunking=sentences precision-mode="$mode" > "eval/out/e2e/precision-$mode.log" 2>&1
  $PY -I eval/speaker_similarity.py "eval/out/e2e/precision-$mode" eval/data/clip.wav >> "eval/out/e2e/precision-$mode.log" 2>&1
  tail -3 "eval/out/e2e/precision-$mode.log"
done
