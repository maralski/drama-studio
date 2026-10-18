# Changelog

DramaEngine follows [semantic versioning](https://semver.org). The version is
`DramaEngineVersion.string` in `Sources/DramaEngine/Version.swift`. A major
release removes a public symbol or changes what one means, a minor release
adds one, and a patch release changes neither.

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
