#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=${0:A:h:h:h}
here=$root/tools/word_check_eval
self=$0
cd "$root"

usage() {
    print -u2 "usage: $self replay
       $self audio [--archive <word-check-eval-audio-v1.zip>]
       $self mutations [--workers N] [--check] [name ...]
       $self regenerate [--dry-run] [set ...]"
    exit 2
}

build() {
    bin=$(swift build -c release --show-bin-path)
    $run python3 -I tools/bootstrap_word_recognition.py
    $run swift build -c release --product word-check-lab
    $run swift build -c release --product drama-word-recognizer
    $run python3 -I tools/stage_word_recognition.py --bin "$bin"
}

run=()
(( $# )) || usage
layer=$1; shift
case $layer in
replay)
    python3 -I tools/bootstrap_word_recognition.py --framework-only
    swift test -c release --filter WordCheckCorpusTests
    ;;
audio)
    build
    audio=$(python3 -I "$here/fetch_audio.py" "$@" | tail -1)
    python3 -I "$here/decide.py" --bin "$bin" --audio "$audio"
    ;;
mutations)
    python3 -I tools/bootstrap_word_recognition.py
    python3 -I "$here/mutate.py" "$@"
    ;;
regenerate)
    if [[ ${1:-} == --dry-run ]]; then run=(print --); shift; fi
    sets=("$@")
    (( $#sets )) || sets=(${(f)"$(python3 -I -c 'import json, sys; print("\n".join(s["name"] for s in json.load(open(sys.argv[1]))))' Tests/DramaEngineTests/Fixtures/word-check-sets.json)"})
    if [[ -z $run && -z ${WORD_CHECK_VOICE_NEWSREADER:-} ]]; then
        print -u2 "Set WORD_CHECK_VOICE_NEWSREADER to the Newsreader voice JSON exported from the Drama Studio library."
        exit 1
    fi
    for set in $sets; do
        [[ -f $here/sets/$set/items.json ]] || { print -u2 "unknown set $set"; exit 1; }
    done
    build
    for set in $sets; do
        items=$here/sets/$set/items.json
        out=$here/audio/$set
        $run mkdir -p "$out"
        $run tools/with-gpu-lock "$bin/word-check-lab" gen "$items" "$out"
        $run "$bin/word-check-lab" recog "$items" "$out"
        $run python3 -I "$here/record.py" "$set" "$out/recog.json"
    done
    ;;
*)
    usage
    ;;
esac
