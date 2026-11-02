# Changelog

DramaEngine follows [semantic versioning](https://semver.org). The version is
`DramaEngineVersion.string` in `Sources/DramaEngine/Version.swift`. A major
release removes a public symbol or changes what one means, a minor release
adds one, and a patch release changes neither.

## Unreleased

## 2.1.0

- Keep assigned library voices matched to their speakers across screenplay
  sentences. Untagged reference clips now use their cast index instead of all
  being labeled speaker 0. Preserve explicit reference speaker tags.
- Add regression coverage for multi-voice prompts and the Herald and Engineer
  screenplay. Verify playback with the saved Herald and Newsreader voices.

## 2.0.0

- Save a voice from a selected complete, word-checked take, including Model
  default takes. Name the voice and select it automatically for future reads.
  Preserve the selected take's original words and codes after script edits
  and project reopening, without changing the export selection.
- Use Metal for the independent word recognizer. Preserve model integrity,
  native confidence checks, and independent transcript verification.
- Accept guarded English ordinal and clock formatting equivalents during
  word verification. Preserve decimal values, signs, negation, word order,
  and uncertainty checks.
- Remove Quick Preview from the app, HTTP API, MCP tool, and say command.
  Existing projects retain their audio and drop the obsolete preview setting
  when saved.
- Breaking API changes: replace
  `GenerationConfig.make(expressiveness:quickPreview:)` with
  `GenerationConfig.make(expressiveness:)`. Remove
  `SpeechParameters.quickPreview`, `SayCommand.quickPreview`, and the
  `drama-say --quick` option. Normal app and API generation use word checks.
- Preserve reference speaker tags when a single-speaker script uses a
  nonzero speaker number.

## 1.1.0

- Display takes in a resizable, scrolling card panel and add take deletion.
  Remove the redundant dice generation action.
- Check generated words locally before releasing normal speech, including
  designed-voice previews and adjacent sentence boundaries. Retry rejected
  attempts and report exhaustion instead of creating a completed take.
  Quick Preview remains unchecked.
- Keep voice-design auditions and main generation on the same checked path,
  improve retry directions, and preserve cue reactions during word comparison.
- Checked generation requires macOS 26 and an installed on-device speech
  model for the script language. Bundle the independent word recognizer.
- Known limitations: some Mia and Hila requests can still exhaust retries.
  Voice and cue reliability is not guaranteed for every input.
- Keep model status across settings tabs and show a neutral state while checking.
  Use Settings > Updates for all update controls and installation confirmation.
- Add Clear Script and Clear Section actions, resizable take cards,
  and screenplay errors with format examples and help.
- Reuse parsed script status across UI controls and calculate waveform peaks
  away from the main thread. Refresh waveforms when equal-length takes change.
- Document system requirements and distinguish recommended memory from tested
  hardware. Correct model-location help and partial-streaming examples.

## 1.0.3

- Fit setup windows to their content instead of inheriting the editor's large
  default or restored size. Long setup messages scroll while action buttons
  remain visible; completing setup returns to a resizable editor. (#10)
- Include the usage-sharing screen in layout coverage and add a regression
  for short setup content and long errors. (#10)

## 1.0.2

- Retry incomplete chunks before normal playback for every voice. Keep short
  sentences separate, allow pauses before speech finishes, and give slow
  reference voices more generation time. Quick Preview retains partial streaming. (#8)
- Make optional silence-stop checkpoints deterministic for seeded generation. (#8)
- Preserve all takes, voice assignments, and export selections in projects.
  Add recovery drafts, Save As, unsaved-work prompts, and regeneration of
  changed sections. Project format 2 requires this app version. (#9)
- Separate auditioning from export selection and warn about outdated or
  incomplete audio. Adapt the studio and dialogs to smaller windows. (#9)
- Download verified app updates in the background and ask before restarting.
  Wait for project quit approval before starting the installer. (#7, #9)
- Add optional anonymous usage counts, sent only after explicit consent.
  Scripts, audio, transcripts, and voice information are excluded.

## 1.0.1

- Preserve retry safeguards for streamed laughter, emotion cues, passage tags,
  and director's notes. These requests send completed sentences when retries
  are enabled, so failed attempts can be replaced before playback. Plain text
  and Quick Preview retain partial streaming. (#6)
- Exclude passage markup from speech timing estimates and frame budgets. (#1)
- Stop showing a false "stopped early" warning when reopening saved projects. (#2)
- Decode form-encoded spaces in API query text while preserving encoded plus
  signs and URL paths. (#5)
- Compare streamed audio with its own take in the API smoke test, and make
  session progress tests independent of polling timing. (#3, #4)

- `GenerationConfig.topP` at 1 or above now applies no nucleus cut: every
  candidate up to `topK` stays eligible. Before, the cut compared the
  cumulative mass against 1 on rounding noise, so on peaked rows the CPU
  reference and the GPU sampler cut at different places and their random
  streams drifted apart from the next row on (1073 of 2000 picks differed on
  real tower logits at topK 600; now 0 of 60,000). Top-p below 1, which is
  the whole range of the app's Expressiveness slider, is unchanged, and the
  default-setting parity tests still pass.

## 1.0.0

The first release of DramaEngine as a library for other macOS apps.

- One package dependency gives the engine: `ModelLocation`,
  `DramaSynthesizer` and the `SpeechSynthesizer` protocol, `SpeechRequest`,
  `SpeechEvent`, `SpeechResult`, `Take`, `ReferenceVoice`, `VoiceDesign`,
  `GenerationConfig` with `Chunking`, `StreamingMode` and `FrameCap`,
  `GenerationEstimate`, `StreamedAudio`, `Drama3Chunking.splitSentencesTagged`,
  `Drama3Prompt.render`, `DramaCodec`, and `MockSynthesizer`.
- `DramaSetup` checks an install without loading the model and prepares it:
  it builds the Core ML codec models with the bundled `convert_codec.py`
  through a Python you name, and tunes the kernels for the Mac.
- `PronunciationDictionary` and `PronunciationRule` moved from `DramaServer`
  to `DramaEngine`. Code that used them through `import DramaServer` now
  needs `import DramaEngine`.
- Engine internals (the model, the frame generator, the tokenizer, the
  autotuner, the Metal context, the prompt builder's token rows) are
  `package`: the repository's own targets use them, other packages cannot.
- Every public symbol has a doc comment, and every public value type is
  `Sendable`. A DocC catalog covers getting the model, voices, and
  streaming.
- `tools/make_xcframework.sh` builds `DramaEngine.xcframework` for macOS on
  Apple silicon, with the Metal kernel sources and the codec conversion
  script inside the framework.
- `GenerationConfig` adds three defaults for lines with performance cues.
  `stopOnLoop` (true) stops a chunk whose main tokens repeat in a cycle, as
  a laughter loop does. `retryWithoutEOSBelow` (1.0) generates a chunk
  again when the loop guard stopped it, or when it stopped before the end
  of speech with less audio than its text expects. `capSecondsPerCue` (1.5)
  gives a text-derived frame cap 1.5 s for each bracket cue. On 8 cue lines
  every sentence came out in 95% of 40 runs, against 80% of 55 runs
  before. Set the three to false, 0 and 0 for the earlier behaviour.
- Of two attempts at a chunk, the engine now keeps one that reached the
  end of speech over one that did not, then the one with more speech.
- `GenerationConfig.make(expressiveness:quickPreview:)` runs from steady
  (0) through the vendor's sampling (0.5, the default) to the Hugging Face
  Space's (1: temperature 1.0, top-p 0.9, no top-k), linear between. Above
  the middle `topK` is 4097, every candidate. Measured over the 165-row
  reliability study: no prose row lost, every sentence of the cue lines,
  fewer retries, 0.007 more word error rate on prose.
- The silence stop reads the newest frames through a third Core ML model,
  `DramaCodecProbe`, the fp16 decoder exported at 64 frames (353 MB):
  `DramaCodec.probeFrames`, `probeContext`, `probeName` and `modelNames`,
  and `CodecError.wrongShape` for a model file built at another frame count.
  A reading costs 190 ms instead of a 1.6 s 512-frame decode, so the stop
  fires 9 to 21 frames after the audio holds 2 s of silence where before it
  never fired on the cue lines; `convert_codec.py` writes the probe beside
  the other two models, `DramaSetup` requires it, and the conversion venv
  recipe pins `numpy<2.4`.
