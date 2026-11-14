#!/bin/zsh
# Launches and drives Drama Studio in a throwaway home. Source it from the repository root:
#   source .claude/skills/verify/verify.sh
#   ds_build                      # debug build, word-check models and the dsax helper
#   ds_home ming                  # a fresh home on Ming (or: ds_home drama3)
#   ds_app                        # launch the app; or ds_serve for the API without the app
#   ds_doctor                     # is this instance worth driving?
#   ds_ax tree | ds_ax press "Generate speech (⌘Return)" | ds_ax set AXTextArea "Hello." | ds_ax menu Speech Engine list
#   ds_wait; ds_shot after-generate; ds_tree after-generate; ds_takes; ds_api /v1/voices; ds_say -o out.wav "Hello."
#   ds_cleanup                    # quits what this shell started and removes the home; evidence stays
# The running instance is recorded in $DS_HOME.state, so sourcing again in a new shell picks it up.
# Never uses the real library or port 8780, never clicks by coordinates and never sends keystrokes.
DS_REPO=${DS_REPO:-$(git rev-parse --show-toplevel)}
DS_BIN=$DS_REPO/.build/debug
DS_WORK=$DS_REPO/.build/verify
DS_HOME=${DS_HOME:-$DS_WORK/home}
DS_PORT=${DS_PORT:-8791}
DS_EVIDENCE=${DS_EVIDENCE:-$DS_WORK/evidence/$(date +%Y%m%d-%H%M%S)}
DS_REAL="$HOME/Library/Application Support/DramaStudio"
DS_PID=
DS_STATE=$DS_HOME.state   # the pid and evidence folder of the running instance, so any shell can pick it up
_ds_load() { [[ -z $DS_PID && -f $DS_STATE ]] && read DS_PID DS_EVIDENCE < $DS_STATE; kill -0 ${DS_PID:-0} 2>/dev/null || { DS_PID=; rm -f $DS_STATE } }
_ds_load; [[ -n $DS_PID ]] && echo "picked up pid $DS_PID, evidence $DS_EVIDENCE"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
# Where the Ming weights are on this Mac: MING_WEIGHTS, else local.env beside this script (untracked,
# one line: MING_WEIGHTS=/path), else the newest snapshot in the Hugging Face cache.
[[ -f ${0:A:h}/local.env ]] && source ${0:A:h}/local.env
: ${MING_WEIGHTS:=$(find ~/.cache/huggingface/hub/models--inclusionAI--Ming-3.0-tts-0.5B/snapshots -mindepth 1 -maxdepth 1 -type d 2>/dev/null | tail -1)}
export MING_WEIGHTS

ds_build() {
  (cd $DS_REPO && swift build 2>&1 | tail -1 && python3 tools/stage_word_recognition.py --bin $DS_BIN | tail -1) || return 1
  mkdir -p $DS_WORK && swiftc -O $DS_REPO/.claude/skills/verify/dsax.swift -o $DS_WORK/dsax 2>&1 | grep -v warning
  [[ -x $DS_WORK/dsax ]] && echo "built $DS_BIN and $DS_WORK/dsax"
}

ds_home() { # ds_home ming|drama3 ; a fresh home with setup already answered for that engine
  [[ $DS_PORT == 8780 ]] && { echo "8780 is the user's own app; choose another DS_PORT"; return 1 }
  [[ $DS_HOME == $DS_WORK/* ]] || { echo "DS_HOME must be under $DS_WORK"; return 1 }
  _ds_load; [[ -n $DS_PID ]] && { echo "pid $DS_PID still runs from this home; ds_cleanup first"; return 1 }
  rm -rf $DS_HOME; mkdir -p $DS_HOME $DS_EVIDENCE
  local plist=$DS_HOME/defaults.plist
  defaults write $plist usageSharing.v1 -string disabled
  defaults write $plist apiServerPort -int $DS_PORT
  defaults write $plist apiServerStartsWithApp -bool true
  defaults write $plist automaticallyUpdatesApp -bool false
  defaults write $plist streamsAudio -bool false
  defaults write $plist setupEngine -string ${1:-ming}
  defaults write $plist speechEngine -string ${1:-ming}
  case ${1:-ming} in
    ming) [[ -f $MING_WEIGHTS/model.safetensors ]] || { echo "no Ming weights at '$MING_WEIGHTS'; set MING_WEIGHTS or write local.env, and do not download them"; return 1 }
      defaults write $plist mingDirectory -string $MING_WEIGHTS ;;
    drama3) # the user approved recording their existing Fish Audio agreement in test homes only
      local snapshot=($(find ~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots -mindepth 1 -maxdepth 1 -type d 2>/dev/null))
      [[ -n $snapshot && -d $DS_REAL/Codec ]] || { echo "Drama 3 is not installed on this Mac; do not download it"; return 1 }
      cp -Rc "$DS_REAL/Codec" "$DS_REAL/Tuning" $DS_HOME/
      defaults write $plist modelDirectory -string $snapshot[1]/
      defaults write $plist termsAcceptedVersion -string fish-audio-community-license-1.0 ;;
    *) echo "ds_home ming|drama3"; return 1 ;;
  esac
  echo "home $DS_HOME on ${1:-ming}, port $DS_PORT, evidence $DS_EVIDENCE"
}

_ds_start() { # _ds_start <log> <command...>
  [[ -d $DS_HOME ]] || { echo "run ds_home first"; return 1 }
  _ds_load; [[ -n $DS_PID ]] && { echo "already running as $DS_PID; ds_quit first"; return 1 }
  lsof -nP -iTCP:$DS_PORT -sTCP:LISTEN >/dev/null && { echo "port $DS_PORT is taken; set DS_PORT"; return 1 }
  env -u DRAMA_ENGINE DRAMA_STUDIO_HOME=$DS_HOME MING_WEIGHTS=$MING_WEIGHTS "${@:2}" > $DS_EVIDENCE/$1 2>&1 &
  DS_PID=$!
  echo "$DS_PID $DS_EVIDENCE" > $DS_STATE
  for i in {1..90}; do curl -s -m 1 http://127.0.0.1:$DS_PORT/v1/health >/dev/null && break; kill -0 $DS_PID 2>/dev/null || break; sleep 1; done
  ds_doctor
}
ds_app() { _ds_start app.log $DS_BIN/DramaStudio }
# --voices is required: without it drama serve uses the real voice library, whatever DRAMA_STUDIO_HOME says
ds_serve() { mkdir -p $DS_HOME/Voices; _ds_start serve.log $DS_BIN/drama serve --port $DS_PORT --engine ${1:-ming} --ming-weights $MING_WEIGHTS --voices $DS_HOME/Voices }

ds_doctor() { # read-only: our process, our port, our build, our home
  _ds_load
  local ok=0
  kill -0 ${DS_PID:-0} 2>/dev/null && echo "ok: process $DS_PID runs $(ps -o comm= -p $DS_PID)" || { echo "FAIL: no process started by this shell"; ok=1 }
  local owner=$(lsof -nP -iTCP:$DS_PORT -sTCP:LISTEN -t 2>/dev/null | head -1)
  [[ -n $owner && $owner == $DS_PID ]] && echo "ok: port $DS_PORT belongs to it" || { echo "FAIL: port $DS_PORT is owned by '${owner:-nobody}'"; ok=1 }
  local health=$(curl -s -m 5 http://127.0.0.1:$DS_PORT/v1/health)
  [[ $health == *'"status":"ok"'* ]] && echo "ok: $health" || { echo "FAIL: health says '$health'"; ok=1 }
  [[ $DS_BIN/DramaStudio -nt $DS_REPO/Package.swift && -z $(find $DS_REPO/App $DS_REPO/Sources -name '*.swift' -newer $DS_BIN/DramaStudio | head -1) ]] \
    && echo "ok: the build is newer than every source file" || { echo "FAIL: sources changed since the build; ds_build"; ok=1 }
  [[ -f $DS_HOME/api-token ]] && echo "ok: token in $DS_HOME" || echo "note: no api-token in $DS_HOME (server started with no token, or not yet)"
  return $ok
}

ds_ax() { _ds_load; $DS_WORK/dsax $1 $DS_PID "${@:2}" }
ds_api() { curl -sS -m ${DS_TIMEOUT:-300} -H "Authorization: Bearer $(cat $DS_HOME/api-token 2>/dev/null)" "${@:2}" http://127.0.0.1:$DS_PORT$1 }
ds_say() { DRAMA_STUDIO_HOME=$DS_HOME DRAMA_API_URL=http://127.0.0.1:$DS_PORT $DS_BIN/drama-say "$@" }   # always with -o FILE, or it plays aloud
ds_busy() { curl -s -m 5 http://127.0.0.1:$DS_PORT/v1/health }
ds_wait() { # ds_wait [seconds] ; waits for a generation in the app window to start, then to end
  local limit=${1:-300} i=0
  while (( i < 10 )) && [[ $(ds_busy) != *'"busy":true'* ]]; do sleep 1; ((i++)); done
  i=0; while (( i < limit )) && [[ $(ds_busy) == *'"busy":true'* ]]; do sleep 2; ((i+=2)); done
  echo "idle after about $i s: $(ds_busy)"
}
ds_shot() { # ds_shot <name> ; the app's largest on-screen window, captured by window id
  _ds_load; local id=$($DS_WORK/dsax windows $DS_PID | awk '$3=="on"{split($2,a,"x"); print a[1]*a[2], $1}' | sort -n | tail -1 | cut -d' ' -f2)
  [[ -n $id ]] || { echo "no on-screen window"; return 1 }
  screencapture -x -o -l$id $DS_EVIDENCE/$1.png && echo $DS_EVIDENCE/$1.png
}
ds_takes() { # copies the app window's take WAVs to the evidence folder; the app saves them a few seconds after a take ends
  local wavs; for i in {1..15}; do wavs=($(find $DS_HOME/Recovery.dramaproj/audio -name '*.wav' 2>/dev/null)); (( $#wavs )) && break; sleep 1; done
  (( $#wavs )) || { echo "no take audio in $DS_HOME/Recovery.dramaproj"; return 1 }
  cp $wavs $DS_EVIDENCE/ && for wav in $wavs; do echo "$DS_EVIDENCE/${wav:t} $(afinfo $wav | awk '/estimated duration/ {print $3}') s"; done
}
ds_tree() { ds_ax tree > $DS_EVIDENCE/$1.tree.txt && echo $DS_EVIDENCE/$1.tree.txt }      # ds_tree <name> ; the window's elements as text

ds_quit() { # only the process this shell started
  _ds_load
  [[ -n $DS_PID ]] || return 0
  kill $DS_PID 2>/dev/null
  for i in {1..20}; do kill -0 $DS_PID 2>/dev/null || break; sleep 0.5; done
  kill -9 $DS_PID 2>/dev/null; DS_PID=; rm -f $DS_STATE
}
ds_cleanup() { ds_quit; [[ $DS_HOME == $DS_WORK/* ]] && rm -rf $DS_HOME; echo "evidence kept in $DS_EVIDENCE" }
