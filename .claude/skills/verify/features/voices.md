# Voices: built-in, library and designed

A line is spoken in a voice. The user can clone a voice from a clip and save it to the library, pick one of Ming's 241 built-in voices, or describe a character and let the engine design one.

## Sub-features

- Add a voice from a reference clip (`Add Voice`, 5 to 15 seconds for Ming, 5 to 20 for Drama 3) and remove it.
- Choose a built-in voice in the Inspector's voice picker: "Choose a Built-in Voice…" opens a searchable sheet (`Search built-in voices`).
- Save the voice of a take to the library (`Save Voice to Library`).
- Over the API: `voice` takes a library name or id, or a built-in name such as `UK_George`. `ming:US_Emma` always means the built-in voice. `GET /v1/voices` returns `voices` and `built_in`.
- A library voice wins a name both have. A built-in voice on the other engine fails with `400`.

## How to get to it (user POV)

The sidebar on the left lists library voices, with `+` and `-` at the bottom. Click the Inspector button in the toolbar; under Cast each speaker has a voice picker labelled "Voice for NAME".

## Driving it with verify.sh

```sh
ds_home ming && ds_app
ds_api /v1/voices | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["voices"]), len(d["built_in"]))'   # 0 241
ds_api /v1/audio/speech -D $DS_EVIDENCE/george.headers -o $DS_EVIDENCE/george.wav \
  -H 'Content-Type: application/json' -d '{"input":"Tickets go on sale at nine.","voice":"UK_George","seed":1}'
ds_say -v '?' | head
ds_ax press "Inspector"; ds_tree inspector; grep -n "Voice for\|Speech engine" $DS_EVIDENCE/inspector.tree.txt
```

Proof for a built-in voice: `200` with `X-Drama-Engine: ming` and a WAV. Proof for a saved voice: a JSON file and a clip in `$DS_HOME/Voices/`, and the name in `GET /v1/voices`.

## Gotchas

- Under `ds_serve`, `POST /v1/voices` writes to `$DS_HOME/Voices` only because `ds_serve` passes `--voices`. A hand-started `drama serve` writes to the user's real library.
- A fresh verify home has no library voices. To test with one, `POST /v1/voices` with a clip from `Tests/DramaEngineTests/Fixtures`, never by copying the user's real `Voices` folder for anything that writes.
- Choosing a file in the Add Voice sheet goes through an open panel, which accessibility cannot drive here. Use `POST /v1/voices` or the hosted tests in `Tests/DramaStudioTests/AppBuiltInVoiceTests.swift`.
- Nobody hears the result. A `200` and a passed word check do not show that the voice is the one named.
