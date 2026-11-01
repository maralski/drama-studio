#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=${0:A:h:h}
cd "$root"
python3 tools/bootstrap_word_recognition.py
swift build -c release --product drama
swift build -c release --product drama-word-recognizer
bin=$(swift build -c release --show-bin-path)
output="$root/dist/drama-cli"
rm -rf "$output"
python3 tools/stage_word_recognition.py --bin "$bin" --cli "$output"
for artifact in "$output/whisper.framework" "$output/drama-word-recognizer" "$output/drama"; do
    codesign --force --sign - "$artifact"
    codesign --verify --strict "$artifact"
done
python3 tools/check_word_recognition.py --bin "$output" --audio Tests/DramaEngineTests/Fixtures/terminal-monday.wav --expect-transcript "The meeting begins on Monday."
python3 tools/check_word_recognition.py --bin "$output" --audio Tests/DramaEngineTests/Fixtures/golden-instead-of-garden.wav --expect-transcript "The Gordon gate is closed."
if [[ ${DS_VERIFY_NATIVE_WORD_CHECKS:-0} == 1 ]]; then
    "$output/drama" check-words Tests/DramaEngineTests/Fixtures/terminal-monday.wav "The meeting begins on Monday." accepted
    "$output/drama" check-words Tests/DramaEngineTests/Fixtures/golden-instead-of-garden.wav "The garden gate is closed." rejected
fi
print "Built and checked $output"
