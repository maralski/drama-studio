---
name: verify
description: Launch and drive the real Drama Studio Mac app, its HTTP API on 127.0.0.1 and its drama-say command in a throwaway home, and capture proof (window screenshots, the window's elements as text, WAV files, response headers). Use it to confirm a change works in the running app or API, to reproduce a bug a user would see, or whenever a task says to verify, run or screenshot Drama Studio.
---

# Verify Drama Studio

Drama Studio is a SwiftUI Mac app that turns a script into speech with a local engine, Ming or Drama 3. The same process serves an HTTP API, and `drama serve` serves that API without the app. This skill runs a debug build in a throwaway home and drives it the way a user or an API client does.

Everything goes through one script. Source it from the repository root in every shell that uses it:

```sh
source .claude/skills/verify/verify.sh
```

Each Bash tool call is a new shell. The script records the running instance's pid and evidence folder in `.build/verify/home.state`, so sourcing it again picks the instance up and says so. Do not pipe `ds_app` or `ds_serve` into another command.

## Rules that are not negotiable

- Never read-modify or write `~/Library/Application Support/DramaStudio`. It is the user's real voice library and projects. `ds_home` makes a home under `.build/verify/` and every launch sets `DRAMA_STUDIO_HOME` to it.
- Never use port 8780. The user's own app listens there. The script uses 8791 and refuses 8780. `drama-say` and `drama-mcp` default to 8780 and start the user's app when nothing answers, so call `drama-say` only through `ds_say`.
- Never download model weights and never press a Download button. `ds_home ming` points at weights already on this Mac: `MING_WEIGHTS`, else the untracked `local.env` beside the script (one line, `MING_WEIGHTS=/path`), else the newest snapshot in the Hugging Face cache. It stops when none has `model.safetensors`. `ds_home drama3` reuses the installed Drama 3 snapshot and codec and stops if they are missing.
- Never click "I Agree" on the Fish Audio terms. `ds_home drama3` records the user's existing agreement in the test home, which is the only thing the user approved.
- Never click by coordinates, send keystrokes or bring the app to the front. The user is working on this Mac. `ds_ax` acts through the accessibility API on the app's process id, and `ds_shot` captures by window id.
- A second instance is fine as long as it has its own home and port: set `DS_HOME=$DS_WORK/home2 DS_PORT=8792` before sourcing. Two instances share the GPU, so do not time anything while both generate.

## Launch

```sh
ds_build            # swift build, stage the word-check models into .build/debug, compile dsax
ds_home ming        # or: ds_home drama3
ds_app              # the app, with its API on $DS_PORT
```

`ds_app` returns when `GET /v1/health` answers, then runs the doctor. The main window is titled "Drama Studio Take" until a script is typed. To drive the API without the app, use `ds_serve` (or `ds_serve drama3`) in place of `ds_app`; it runs `drama serve`.

A home made by `ds_home` skips the setup window, turns off update checks and usage sharing, and turns off "Stream audio while generating" so a generation in the app stays silent. To test the setup window itself, delete `setupEngine` and `speechEngine` from `$DS_HOME/defaults.plist` before `ds_app` and export `DRAMA_STUDIO_MEMORY_GB=16` to see the low-memory offer. `export DRAMA_PREVIEW_ENGINE=1` runs the app on a mock engine that makes placeholder audio, for layout checks that need no model.

Use `tools/with-gpu-lock <command>` around `swift test` and around anything you time. Other agents share the GPU.

## Doctor

```sh
ds_doctor
```

It is read-only and prints one `ok:` or `FAIL:` line each for: the process this shell started is alive, the port belongs to that process, `/v1/health` says `ok` (the line shows the selected engine and whether the model is loaded), and the build is newer than every Swift source. Run it first whenever something looks off. A `FAIL` on the build line means you are about to verify old code: run `ds_build` and relaunch.

## Drive

In the app, through accessibility:

```sh
ds_ax tree                                    # every element: role | title | description | help | value
ds_ax set AXTextArea "The harbor lights came on at dusk."   # the script editor is the first AXTextArea
ds_ax press "Generate"                        # by title, description, help or identifier, exact match
ds_wait 120                                   # until /v1/health stops reporting busy
ds_ax menu Speech Engine list                 # list a menu; a check mark shows the selection
ds_ax menu Speech Engine Ming                 # press a menu item by its path
```

Labels that exist today: `Generate` (help `Generate speech (⌘Return)`), `Play`, `Export`, `Add Voice`, `Remove Voice`, `Inspector`, `Import Script`, `Clear Script`, `Speech engine`, `Playback speed`, `Takes height`, `Engine notices`, `Dismiss engine notices`, `Delete Take N`, and a take tile labelled like `Take 1 · 2.5 s, used for export`. Run `ds_ax tree` to find the rest; do not guess. A sheet is a child of its window in the tree. System Events and AppleScript cannot read SwiftUI button titles, so use `ds_ax` and do not conclude a button is unlabelled from AppleScript.

Open panels, save panels and drag and drop cannot be driven this way. Reach the same code through the API or a hosted test in `Tests/DramaStudioTests` and say so in the report.

Through the API (the token is `$DS_HOME/api-token`, which `ds_api` sends):

```sh
ds_api /v1/voices
ds_api /v1/audio/speech -D $DS_EVIDENCE/speech.headers -o $DS_EVIDENCE/speech.wav \
  -H 'Content-Type: application/json' -d '{"input":"Tickets go on sale at nine.","voice":"UK_George","seed":1}'
ds_say --verbose -o $DS_EVIDENCE/say.wav "Doors open at seven."
```

`tools/api_smoke.sh http://127.0.0.1:$DS_PORT` runs a longer API pass, but it reads the token from the real library, so it only works against `ds_serve` started with `--no-token`; prefer `ds_api`.

The feature map in [features/README.md](features/README.md) says how to reach and prove each feature. A proof that drives one entry point is incomplete when the map lists others.

## Evidence

Everything goes in `$DS_EVIDENCE` (`.build/verify/evidence/<time>/`), which `ds_home` prints. `app.log` or `serve.log` is there already.

```sh
ds_shot 02-generated      # PNG of the app's largest on-screen window, by window id
ds_tree 02-generated      # the same window as text, which is what to grep and quote
```

What counts as proof:

- Drive the path a user takes: type in the editor and press Generate, or call the public API. Do not call model methods from a test and call it verified.
- Capture the action and the state after it: a screenshot or tree before and after, not only the last screen.
- Check the side effect as well as the screen. A take the app made is a WAV in `$DS_HOME/Recovery.dramaproj/audio/`, saved a few seconds after the take ends; `ds_takes` waits for it, copies it to the evidence folder and prints its duration. An API take has `X-Drama-Take-Id`, `X-Drama-Engine`, `X-Drama-Seconds` and `X-Drama-Word-Coverage` headers; save them with `-D`. A saved voice is a JSON file and a clip in `$DS_HOME/Voices/`.
- Nobody hears the audio in these runs. `X-Drama-Word-Coverage: recognized` means the word check heard the scripted words; it says nothing about how the voice sounds. Say that in the report.
- After the run, confirm the real library is untouched: `find "$HOME/Library/Application Support/DramaStudio" -mmin -30 -not -name gpu.lock` prints nothing.

## Cleanup

```sh
ds_cleanup
```

It quits only the process this shell started (by pid, never by name), removes `$DS_HOME` and leaves `$DS_EVIDENCE`. Run it after a failed attempt too, so no app or port is left behind. If the state file is gone but something still listens, find it with `lsof -nP -iTCP:8791 -sTCP:LISTEN`, check with `ps` that its command is this checkout's `.build/debug` binary, and kill that pid. Never `killall DramaStudio`: the user's own app has the same name.

## Helpers

- `verify.sh`: the functions above. Its header lists them.
- `dsax.swift`: the accessibility tool behind `ds_ax` and `ds_shot`. `ds_build` compiles it to `.build/verify/dsax`. Run directly as `.build/verify/dsax tree|press|set|menu|windows <pid> ...`. It needs Accessibility permission for the terminal and says so when it is missing; `ds_shot` needs Screen Recording permission.
