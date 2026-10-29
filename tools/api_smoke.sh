#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Drives a running Drama Studio API end to end with curl and checks each answer.
# Usage: tools/api_smoke.sh [base-url]   (default http://127.0.0.1:8780)
# Creates two voices named "API smoke ..." and deletes them again.
set -euo pipefail
zmodload zsh/mathfunc

base=${1:-http://127.0.0.1:8780}
token_file="$HOME/Library/Application Support/DramaStudio/api-token"
auth=()
[[ -f $token_file ]] && auth=(-H "Authorization: Bearer $(<"$token_file")")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0

step() { print "\n== $*" }
check() {
    if eval "$2"; then print "ok: $1"; else print "FAIL: $1"; failures=$((failures + 1)); fi
}
header() { grep -i "^$1:" "$2" | head -1 | cut -d' ' -f2- | tr -d '\r' }

step "GET /v1/health (no token needed)"
health=$(curl -sS "$base/v1/health")
print "$health"
check "health reports ok" '[[ $health == *"\"status\":\"ok\""* ]]'

step "GET /v1/models"
models=$(curl -sS "${auth[@]}" "$base/v1/models")
print "$models"
check "lists fishaudio/drama-3" '[[ $models == *"fishaudio/drama-3"* ]]'

step "GET /v1/voices without the token"
code=$(curl -sS -o "$work/denied.txt" -w '%{http_code}' "$base/v1/voices")
print "$code $(<"$work/denied.txt")"
check "401 without a bearer token" '[[ $code == 401 || ${#auth[@]} == 0 ]]'

text="The quick brown fox jumps over the lazy dog. Then it naps in the sun for a while."
step "POST /v1/audio/speech, wav"
curl -sS "${auth[@]}" -D "$work/wav.h" -o "$work/speech.wav" "$base/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "{\"input\":\"$text\",\"seed\":1}"
grep -i '^x-drama\|^content-type' "$work/wav.h" | tr -d '\r'
afinfo "$work/speech.wav" | grep -E 'File type|Data format|estimated duration|audio bytes'
wav_seconds=$(header x-drama-seconds "$work/wav.h")
afinfo_seconds=$(afinfo "$work/speech.wav" | awk '/estimated duration/ {print $3}')
check "afinfo duration $afinfo_seconds matches X-Drama-Seconds $wav_seconds" '(( fabs(afinfo_seconds - wav_seconds) < 0.01 ))'

step "POST /v1/audio/speech, stream:true pcm"
curl -sS "${auth[@]}" -N -D "$work/pcm.h" -o "$work/speech.pcm" "$base/v1/audio/speech" \
    -d "{\"input\":\"$text\",\"stream\":true,\"response_format\":\"pcm\"}"
grep -i '^transfer-encoding\|^x-drama\|^content-type' "$work/pcm.h" | tr -d '\r'
bytes=$(stat -f %z "$work/speech.pcm")
stream_seconds=$(( bytes / 2.0 / 44100 ))
print "streamed $bytes bytes = $stream_seconds s of 16-bit mono at 44.1 kHz"
check "chunked transfer" '[[ $(header transfer-encoding "$work/pcm.h") == chunked ]]'
check "stream length matches the file's X-Drama-Seconds $wav_seconds" '(( fabs(stream_seconds - wav_seconds) < 0.001 ))'

step "POST /v1/tts alias with text"
code=$(curl -sS "${auth[@]}" -o "$work/alias.wav" -w '%{http_code}' "$base/v1/tts" -d '{"text":"Alias route."}')
check "alias answers 200 with a wav ($code)" '[[ $code == 200 && $(head -c 4 "$work/alias.wav") == RIFF ]]'

step "POST /v1/audio/speech with a bad body"
code=$(curl -sS "${auth[@]}" -o "$work/bad.txt" -w '%{http_code}' "$base/v1/audio/speech" -d '{"input":"Hi","response_format":"flac"}')
print "$code $(<"$work/bad.txt")"
check "400 with a plain message" '[[ $code == 400 ]]'

step "Voices: create (JSON and multipart), list, delete"
clip_b64=$(base64 -i "$work/alias.wav")
print "{\"name\":\"API smoke JSON\",\"transcript\":\"Alias route.\",\"audio_base64\":\"$clip_b64\"}" > "$work/voice.json"
created=$(curl -sS "${auth[@]}" "$base/v1/voices" -H 'Content-Type: application/json' --data-binary "@$work/voice.json")
print "$created"
json_id=$(print -r -- "$created" | sed -E 's/.*"id":"([^"]+)".*/\1/')
created=$(curl -sS "${auth[@]}" "$base/v1/voices" -F "file=@$work/alias.wav" -F "name=API smoke multipart" -F "transcript=Alias route.")
print "$created"
multipart_id=$(print -r -- "$created" | sed -E 's/.*"id":"([^"]+)".*/\1/')
listed=$(curl -sS "${auth[@]}" "$base/v1/voices")
check "both voices listed" '[[ $listed == *"$json_id"* && $listed == *"$multipart_id"* ]]'
print "DELETE $json_id: $(curl -sS "${auth[@]}" -X DELETE "$base/v1/voices/$json_id")"
print "DELETE $multipart_id: $(curl -sS "${auth[@]}" -X DELETE "$base/v1/voices/$multipart_id")"
code=$(curl -sS "${auth[@]}" -o "$work/gone.txt" -w '%{http_code}' -X DELETE "$base/v1/voices/$json_id")
print "$code $(<"$work/gone.txt")"
check "deleting again is 404" '[[ $code == 404 ]]'

step "429 while a generation runs"
long=$(printf 'This sentence keeps the engine busy for a little while. %.0s' {1..12})
curl -sS "${auth[@]}" -N -o /dev/null "$base/v1/audio/speech" -d "{\"input\":\"$long\",\"stream\":true}" &
busy_pid=$!
sleep 0.5
code=$(curl -sS "${auth[@]}" -D "$work/busy.h" -o "$work/busy.txt" -w '%{http_code}' "$base/v1/tts" -d '{"text":"Me too."}')
print "$code Retry-After: $(header retry-after "$work/busy.h") $(<"$work/busy.txt")"
check "second request gets 429 with Retry-After" '[[ $code == 429 && -n $(header retry-after "$work/busy.h") ]]'
wait $busy_pid

print "\n$failures failure(s)"
exit $(( failures > 0 ))
