#!/bin/sh
# Interleaved end-to-end timing of silence-probe cadences on the app's path:
# sentence chunking, the text-derived cap, the Core ML codec and its probe,
# one process at a time under tools/with-gpu-lock, with Drama Studio quit.
# usage: eval/probe_cadence.sh <drama> <rounds> <outdir> [condition ...]
# A condition is name=flags, for example p32="--probe-every 32"; the default
# set is off, p16, p32, p64 (each read from the chunk's start), gate16 (shipped)
# and gate32.
# Summarise with eval/probe_cadence_summary.py <outdir>.
set -e
BIN=$1; ROUNDS=$2; OUT=$3; shift 3
mkdir -p "$OUT"
if [ $# -eq 0 ]; then
  set -- "off=--probe-every 0" "p16=--probe-after-expected no" "p32=--probe-every 32 --probe-after-expected no" "p64=--probe-every 64 --probe-after-expected no" "gate16=" "gate32=--probe-every 32"
fi
TEXT="When the old lighthouse keeper finally retired, the town threw him a party on the pier. Children ran between the tables, gulls circled overhead, and someone had baked a cake shaped like the lighthouse itself, complete with a tiny candle at the top. He gave a short speech, thanked everyone twice, and then stood for a long while looking out at the water. Nobody hurried him."
REF="<|speaker:0|> Every morning I walk to the little bakery at the end of our street."
i=1
while [ "$i" -le "$ROUNDS" ]; do
  for cond in "$@"; do
    name=${cond%%=*}; flags=${cond#*=}
    if pgrep -x "Drama Studio" >/dev/null || pgrep -x DramaStudio >/dev/null; then echo "Drama Studio is running; quit it first" >&2; exit 1; fi
    uptime >> "$OUT/load.txt"
    # shellcheck disable=SC2086
    tools/with-gpu-lock "$BIN" generate --seed 1 --text "$TEXT" --ref eval/out/perf/ref_215.npy --ref-text "$REF" \
      --chunking sentences --text-cap --warm --trace-frames "$OUT/trace_${name}_$i.csv" --out "$OUT/gen_${name}_$i.npy" $flags \
      > "$OUT/gen_${name}_$i.txt" 2>&1
  done
  i=$((i + 1))
done
