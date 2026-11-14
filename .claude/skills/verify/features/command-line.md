# drama-say and the MCP tools

`drama-say` speaks text from a terminal through the running app's API, like `say`. `drama-mcp` gives the same to an MCP client (`synthesize_speech`, `list_voices`, `studio_status`). `drama` is the engine's own tool: `drama setup` reports what each engine's install has and lacks, and `drama serve` runs the API without the app.

## Sub-features

- `drama-say "text"`, `-f FILE`, `-o FILE` (.wav, .m4a, .pcm), `-v VOICE`, `-v '?'`, `-e ENGINE`, `-l LANGUAGE`, `--note`, `--retake TAKE_ID`, `--verbose`.
- Exit status: 0 spoken, 64 bad usage, 75 the studio is busy, 1 other errors. Notices go to standard error.
- `drama setup [--model DIR] [--ming-weights DIR]`, `drama serve [--port N] [--engine drama3|ming] [--no-token]`, `drama --help`.

## How to get to it (user POV)

In Terminal, after installing the command-line tools from the app or building with `swift build`. `drama-say` needs the app or `drama serve` running; with neither, it starts the installed app.

## Driving it with verify.sh

```sh
ds_home ming && ds_serve               # or ds_app
ds_say --verbose -o $DS_EVIDENCE/say.wav "Doors open at seven." 2> $DS_EVIDENCE/say.stderr; echo "exit $?"
cat $DS_EVIDENCE/say.stderr; afinfo $DS_EVIDENCE/say.wav | grep duration
ds_say -v '?' | head
DRAMA_STUDIO_HOME=$DS_HOME MING_WEIGHTS=$MING_WEIGHTS $DS_BIN/drama setup
```

Proof: exit 0, a line such as "drama-say: take ID on ming after 9.49 s" on standard error, and a WAV with a duration.

## Gotchas

- Always go through `ds_say`. A bare `drama-say` talks to port 8780, which is the user's own app, and launches that app when nothing answers.
- Always pass `-o FILE`. Without it `drama-say` plays through the speakers.
- The MCP `synthesize_speech` tool writes to the real `Output` folder by default, whatever `DRAMA_STUDIO_HOME` says. Pass an explicit output path inside `$DS_EVIDENCE`.
- `drama serve` with no `--port` uses 8780. `ds_serve` always passes the port.
