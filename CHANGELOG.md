# Changelog

DramaEngine follows [semantic versioning](https://semver.org). The version is
`DramaEngineVersion.string` in `Sources/DramaEngine/Version.swift`. A major
release removes a public symbol or changes what one means, a minor release
adds one, and a patch release changes neither.

## Unreleased

Drama Studio adds a second engine, Ming: inclusionAI's Ming-3.0-tts-0.5B
through MingTTS 0.1.0, a Swift and Metal port. Its checkpoint is 3.0 GB
against Drama 3's 41.5 GB, so on a Mac with less than 64 GB of memory, use
Ming instead of Drama 3.

- The setup window starts with **Choose a speech engine**. It recommends and
  selects Ming on a Mac with less than 64 GB of memory, and Drama 3
  otherwise. Choosing Drama 3 below 64 GB shows a warning first. The Ming
  path has no terms, codec or tuning step and never downloads Drama 3.
  **Choose Another Engine** goes back from the warning, the terms, a
  download or a failure, and keeps partial downloads. An install that already
  agreed to the Fish Audio terms counts as a Drama 3 install.
- **Settings > General** adds Ming with **Download (3.0 GB)** or
  **Choose Folder…**, and adds Drama 3 with **Set Up Drama 3…**, which runs
  the Drama 3 setup steps, terms included, in the main window.
- The Fish Audio terms gate Drama 3 only. Ming's MIT License asks for no
  agreement. **Settings > General > About** names both licenses.
- The Inspector's **Engine** picker and the Speech menu switch engines. Only
  one engine is in memory at a time. A retake runs on the engine that made
  its take. A take's tile names the engine that made it when both engines are
  installed or the take is from the engine that is not selected.
- Ming performs 14 fixed cue tags and five emotions. Drama Studio maps common
  cues such as `[laughs]` and `[sighs]` to them and leaves out the rest. A
  passage tag's words are spoken plainly. Each change is a notice that names
  the engine, such as "Ming can't perform [shouting], so it was left out.",
  in the app's notice banner and take tile, the HTTP API
  (`X-Drama-Notices`, and `notices` in the `done` event), `drama-say` and the
  MCP tools.
- On Ming, **Insert Cue** disables the cues Ming can't perform and marks them
  "(not on Ming)", the Expressiveness slider is disabled, and the note
  suggestions are Ming's five emotions. A request that sets Expressiveness
  away from the default gets the notice "Ming ignores the Expressiveness
  setting."
- **Add Voice** advises 5 to 15 seconds of speech for Ming and 5 to 20 for
  Drama 3. Ming encodes the whole clip, and a clip over 15 seconds gets a
  warning, not a refusal.
- Ming takes Chinese (`zh-Hans` and `zh-Hant`), and a script in a language
  Ming does not speak fails before generation with "Ming can't speak
  Italian; it speaks Chinese, English, Japanese, Korean, German, French, and
  Spanish". The API returns it with `400`.
- A voice saved before voices kept their recording can't speak on Ming until
  Drama 3 is set up, because only Drama 3's codec can recreate its clip.
- The status line reads "4.2 s of audio · 1.8x real time · generated in
  2.3 s" for the selected take.
- The HTTP API lists every installed engine in `GET /v1/models`, takes the
  engine in the speech body's `model` field or the `model` query parameter,
  and names the engine in `X-Drama-Engine` and `X-Drama-Model`. A file
  response carries `X-Drama-Samples`; `X-Drama-Frames` and
  `X-Drama-Generated-Frames` are Drama 3 only. `GET /v1/takes/{id}` returns
  `samples`, with `frames` and `codes_base64` for a Drama 3 take.
- `drama setup` reports both engines and takes `--ming-weights`. `drama serve`
  serves every installed engine and takes `--engine drama3|ming` (or
  `DRAMA_ENGINE`), `--ming-weights` and `--voices`.
- `drama-say` takes `-e` or `--engine`, and `--verbose` names the engine that
  made the take.
- The API's `401` message names the token file the server reads, which is in
  `DRAMA_STUDIO_HOME` when that is set.
- The MCP `synthesize_speech` tool takes an `engine`, and `studio_status`
  lists every installed engine.
- Every speech route takes `language`, a BCP 47 code such as `en` or
  `zh-Hans`, in the JSON body or the query string. The engine speaks and
  checks the text in that language and skips detection, so a wrong code fails
  the word check. A value that is not a language code gets `400`, and so does
  a language Ming does not speak on Ming. The MCP tools do not take it.
- A line whose language the engine cannot tell is read in the macOS user's
  language, when the text is in a script that language uses. A line cannot be
  told when the recognizer is uncertain, or when the line has fewer than four
  words and the recognizer is less than 0.95 sure. "Hi." no longer fails as
  Catalan, and "Goodbye." no longer fails with "the script's language is
  uncertain". In a live session, such a sentence takes the language of the
  sentence before it.
- A failed event stream, from `GET /v1/audio/speech/stream` or the live route
  with `Accept: text/event-stream`, ends as a complete chunked body after its
  `error` event. Before, it closed without the last chunk and curl exited
  with code 18. A failed raw PCM stream still ends without the last chunk,
  because raw PCM has no other way to report the failure.
- The app's API server answers while the **Recover unsaved work?** alert is
  open. Before, `GET /v1/health`, the voice routes and every speech request
  waited until the alert was answered, so a launcher polling `/v1/health`
  timed out. `/v1/health` now also answers while the app's main thread is
  busy, and its `busy` field still reports a generation in the window.
- New public API: `MingSynthesizer`, `MingLocation`, `MingVoiceEncoding`,
  `SpeechEngines`, `EngineCapabilities`, `CueSupport`, `EngineInstall`,
  `HuggingFaceCache`, `SpeechNotice`, `SpeechOutline`, `EngineDescription`,
  `VoiceEncodingStore`, `DramaSetup.check(ming:)`, `SpeechResult.notices` and
  `SpeechRequest.language`.
  `EngineInstall` names each engine's `publisher`, `license` (an
  `EngineInstall.License` with `requiresAgreement`), `recommendedMemory`
  and `downloadBytes`. `EngineCapabilities` has `expressiveness` and
  `noteEmotions`. `SpeechNotice.message(on:)` names the engine that ran the
  request, and `SpeechNotice.ignoredExpressiveness` reports an Expressiveness
  setting the engine ignores. `SpeechEngines.Factory.placeholder(_:)` holds
  the selection before any engine is installed.
- DramaEngine depends on the MingTTS package. `MING_PACKAGE_PATH` builds
  against a local checkout.

Breaking: voices, takes and sentence timing no longer assume Drama 3, so a
second engine can share them.

- `Take` holds `text` and a `TakeContext`; `.drama3(codes:)` carries the
  codes. `frameCount` and `frames` are gone; `sampleCount` and
  `durationSeconds` remain. Takes saved with `codes` still decode.
- `ReferenceVoice` holds `encodings`, one `VoiceEncoding` per
  `SpeechEngineID`, and `clipSampleCount`. Voices saved with `codes` decode
  with a Drama 3 encoding.
- `SpokenSentence` counts `samples` instead of `frames`. Records saved with
  `frames` still decode.
- `SpeechSynthesizer` replaces `encodeReference(audioURL:transcript:name:)`
  with `encodeVoice(samples:transcript:)` and the `cloneVoice` extension, drops
  `decode(frames:)`, and adds `engineDescription` and `isTuned`.
- `SpeechResult.frames` is gone. `Drama3Prompt.HistoryTurn` is its own type.
- `SetupReport.location` is now `directory: URL?`, and the report names its
  `engine`. `DramaSetup.downloadCommand` is now
  `SpeechEngineID.install.downloadCommand`.
- `DramaCodec.loadAudio` is now `ReferenceAudio.load(url:)`, which reads the
  whole file in blocks. `ReferenceAudio.write(_:to:)` stores a clip.
  `DramaCodec.recreateClip(of:directory:)` decodes a voice's Drama 3 codes.
- The app's project format is 5. It still opens formats 1 to 4. The voice
  library keeps each voice's clip beside its JSON file.
- The word check hears a take that ends in silence with 0.75 s of silence
  appended, so a complete final word on a take with little trailing silence
  clears the 0.5 floor on the first pass. A take still sounding at its last
  sample is heard as it is. The separate end-of-audio confirmation pass and its
  0.9 floor are gone. The acceptance policy is revision 8, so takes checked
  under an earlier revision show "earlier word check".
- The word check takes a word with a final s and the same word without it as
  the same before a word that starts with s, when both have three letters or
  more: "beam sweep" matches "beams sweep". The two s sounds run together, and
  Apple's transcriber and whisper wrote different forms for the same takes.
- The word check's `homophones` table in `EnglishSpokenForms.json` accepts
  "fairy" for "ferry" and "fairies" for "ferries".
- Those two fixes end false rejections of correct takes that were already on
  `main`. A 20-second Drama 3 paragraph that failed for some seeds now passes
  for seeds 100 to 104. "The ferry docked at dawn." failed all six attempts on
  both engines because whisper heard "fairy". The acceptance policy stays at
  revision 8.

## 2.1.3

- Accept the different ways the same words are spoken and written in the
  word check. English numbers match by value and kind: digits, cardinal,
  paired ("nine thirty"), decimal, decade, money and units. A different value
  or kind never matches. UK and US spellings, unit abbreviations, possessive
  apostrophes and compound spacing also match.
- Accept Apple's lossy number formats, such as "130" for "one thirty", only
  from the Apple transcript. The independent recognizer still needs an exact
  reading.
- Check currency. "£12" no longer matches speech that says "twelve dollars".
- Keep leading zeros. "Agent 007" no longer matches "Agent seven".
- Accept every cue, including typed `[cues]` and directions such as
  `(laughing)`. On a cued line, ignore short recognition labels and inserted
  sounds such as "ugh"; every scripted word is still required.
- Lower Apple's per-word confidence floor from 0.5 to 0.3 for interior words.
  The final word keeps 0.5 and end-of-audio confirmation.
- Bump the acceptance policy to revision 7. Takes checked under an earlier
  revision show "earlier word check" until they are generated again.
- Match the expected words about 16 times faster by compiling patterns once.
- Add `tools/word_check_eval`, which measures the check on 841 recorded
  Drama 3 clips. Across them, correct clips accepted rose from 255/508 to
  426/508, and wrong clips accepted fell from 3/241 to 0/241.

## 2.1.2

- Compare the installed codec archive's recorded SHA-256 with the release's
  digest at setup. Reuse matching models across app releases and download
  changed or missing codecs. Older installs without a recorded digest need
  one verified download. Verify cached archives before reuse and record the
  new digest only after all three models are installed.

## 2.1.1

- Show background app downloads as a horizontal progress bar that follows
  the download task's actual progress. Clear the bar when the download ends
  or fails, and use the existing spinner for checking and verification.

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
