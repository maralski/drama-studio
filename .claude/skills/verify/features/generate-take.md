# Generate a take from a script

The user types or pastes a script, presses Generate, and gets a take: audio with a waveform, listed under Takes. Each line is checked by a speech recognizer and retried when words are missing.

## Sub-features

- Generate a first take, and further takes of the same script.
- Retake with a director's note (the note field and `Retake` under the takes). On Ming the note chips are happy, sad, angry, fearful and surprised.
- Cues in the script such as `[laughs]`. An engine that cannot perform one leaves it out and shows a notice banner (`Engine notices`).
- The take grid: whole rows only, scrolling rests on a row, and the `Takes height` handle raises it.
- Stop a generation (`Stop Generating`).

## How to get to it (user POV)

Open the app. Type in the editor in the middle of the window. Click Generate in the bar under the editor, or press ⌘Return, or choose Speech > Generate. The status line under the waveform shows progress and then a summary such as "2.5 s of audio · 0.2x real time · generated in 13.4 s". The take appears under "Takes (1)".

## Driving it with verify.sh

```sh
ds_home ming && ds_app
ds_shot 01-before
ds_ax set AXTextArea "The harbor lights came on at dusk."
ds_ax press "Generate"            # or: ds_ax menu Speech Generate
ds_wait 120
ds_shot 02-after; ds_tree 02-after
grep -E "Takes \(|Take 1|s of audio" $DS_EVIDENCE/02-after.tree.txt
ds_takes
```

Proof: the tree has `Takes (1)`, a button `Take 1 · N s, used for export` and the summary line, and the WAV in the recovery project has the same duration. A take that failed its word check is labelled `unchecked` or `incomplete` on its tile.

For a retake, set the note with `ds_ax tree` to find the note field (an `AXTextField`), then `ds_ax press "Retake"`.

## Gotchas

- The first generation loads the model, about 10 s on Ming and longer on Drama 3. `ds_wait` covers it.
- `ds_ax set AXTextArea` sets the first text area, which is the script editor only while no sheet is open.
- The home has "Stream audio while generating" off, so nothing plays. Pressing `Play` does play through the speakers; do not press it while the user is at the Mac.
- Quitting with a take leaves `Recovery.dramaproj` in the home, and the next launch from that home shows a "Recover unsaved work?" alert. `ds_cleanup` removes the home; use a fresh `ds_home` per run.
- The grid's resize handle takes accessibility increment and decrement, not a drag. A pointer drag is covered only by the hosted tests in `Tests/DramaStudioTests/TakeGridRestTests.swift`.
