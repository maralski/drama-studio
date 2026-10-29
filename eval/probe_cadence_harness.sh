#!/bin/sh
# The cue-line and prose harness per silence-probe cadence: variant a (the
# pre-study defaults, where the stop matters most), seeds 1 to 5, for every
# condition, each whole harness run under tools/with-gpu-lock.
# usage: eval/probe_cadence_harness.sh <drama> <outroot> [variant] [condition ...]
# Read with eval/cues/probe_latency.py <outroot>/cues-<cond> ... and
# eval/cues/run.py --summary --out <outroot>/cues-<cond>.
set -e
BIN=$1; ROOT=$2; VARIANT=${3:-a}; [ $# -ge 3 ] && shift 3 || shift 2
if [ $# -eq 0 ]; then
  set -- "p16=probe-after-expected no" "p32=probe-every 32|probe-after-expected no" "p64=probe-every 64|probe-after-expected no" "gate16=" "gate32=probe-every 32"
fi
PY=eval/.venv/bin/python
for cond in "$@"; do
  name=${cond%%=*}; flags=${cond#*=}
  args=""
  IFS='|'; for f in $flags; do [ -n "$f" ] && args="$args --flag \"$f\""; done; unset IFS
  for set in lines prose-lines; do
    out="$ROOT/cues-$name"; [ "$set" = prose-lines ] && out="$out-prose"
    if pgrep -x "Drama Studio" >/dev/null || pgrep -x DramaStudio >/dev/null; then echo "Drama Studio is running; quit it first" >&2; exit 1; fi
    eval tools/with-gpu-lock "$PY" -u eval/cues/run.py --variants "$VARIANT" --seeds 1 2 3 4 5 --lines "eval/cues/$set.json" --out "$out" --drama "$BIN" $args
  done
done
