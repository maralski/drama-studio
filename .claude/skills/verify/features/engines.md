# Engine choice: Ming and Drama 3

Drama Studio speaks with one of two engines. Ming is a 3.0 GB download for Macs with less memory; Drama 3 is 41.5 GB and needs agreement to the Fish Audio terms. Only one engine is in memory at a time.

## Sub-features

- The setup window's first step, "Choose a speech engine", recommends Ming below 64 GB of memory.
- The Inspector's `Speech engine` picker and Speech > Engine switch engines. An engine that is not installed is listed as "(not installed)" and disabled.
- Settings > General adds the other engine later: "Download (3.0 GB)" or "Choose Folder…" for Ming, "Set Up Drama 3…" for Drama 3.
- A take's tile names its engine when both are installed. A retake runs on the engine that made the take.
- Over the API: `GET /v1/models` lists installed engines, and `model` in a speech request picks one.

## How to get to it (user POV)

On first launch the setup window asks for the engine. Afterwards: Speech menu > Engine, or the Inspector (toolbar button at the right) > Engine picker.

## Driving it with verify.sh

```sh
ds_home ming && ds_app
ds_ax menu Speech Engine list          # "✓ Ming" and "Drama 3 (not installed)  (disabled)"
ds_api /v1/models
ds_api /v1/health                      # speech_engine and model_loaded
```

For both engines in one home, run `ds_home drama3`, then `defaults write $DS_HOME/defaults.plist mingDirectory -string $MING_WEIGHTS` before `ds_app`, and switch with `ds_ax menu Speech Engine Ming`.

For the setup window: `ds_home ming`, then `defaults delete $DS_HOME/defaults.plist setupEngine; defaults delete $DS_HOME/defaults.plist speechEngine`, `export DRAMA_STUDIO_MEMORY_GB=16`, `ds_app`, and read the steps with `ds_ax tree`.

Proof: the menu's check mark, `speech_engine` in `/v1/health`, and `X-Drama-Engine` on a take made after the switch.

## Gotchas

- Never press Download, "I Agree" or "Continue" on a step that starts a download. Read the setup window; do not complete it.
- `ds_home drama3` needs the Drama 3 snapshot in the Hugging Face cache and the codec in the real library, which it copies by APFS clone and never changes. Drama 3 maps about 41 GB and takes far longer to load than Ming.
- `ds_app` waits for `/v1/health`. While the setup window is up the server may not have started, so `ds_app` waits its full 90 s and the doctor reports a health failure; the app is still running and `ds_ax tree` works.
- `MING_WEIGHTS` in the environment overrides a folder chosen in Settings, and the launch always sets it. Unset it in `verify.sh` to test the Settings folder path.
