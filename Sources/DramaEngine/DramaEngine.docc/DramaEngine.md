# ``DramaEngine``

Run Fish Audio's Drama 3 text-to-speech model on Apple silicon from your own macOS app.

## Overview

DramaEngine is the inference engine behind Drama Studio, packaged as a Swift library.
It loads the released Drama 3 weights in place, clones a voice from a short clip,
and streams speech as it generates. The 35B mixture-of-experts backbone and the
codebook tower run on hand-written Metal compute kernels. The DAC audio codec runs
through Core ML. Nothing runs through MLX or PyTorch at run time.

The model weights are under the
[Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE).
Cloning a real person's voice without their consent is not permitted.

### Requirements

- An Apple silicon Mac with a lot of unified memory. The weights are 41.5 GB and are
  memory-mapped, not copied, plus about 2 GB of state and codec. Only a 128 GB M3 Max has been tested.
- macOS 15 or later, and Xcode 26 or its command line tools.
- The model in the Hugging Face cache and the Core ML codec models built once from
  it. See <doc:GettingTheModel>.

### Speak a sentence in a minute

Add the package, check the install, and create the engine:

```swift
import DramaEngine

let location = ModelLocation.huggingFaceCache()
let report = DramaSetup.check(location: location)
guard let location, report.isReady else {
    report.fixes.forEach { print($0) }
    exit(1)
}
let engine = DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory)
```

Then clone a voice from a clip and speak, streaming the audio as it is
decoded. The function takes any ``SpeechSynthesizer``, so the same code runs
against ``MockSynthesizer`` in previews and tests:

```swift
func speak(_ text: String, cloning clip: URL, transcript: String,
           with engine: some SpeechSynthesizer) async throws -> (result: SpeechResult, streamed: [Float]) {
    try await engine.load { phase in print(phase) }
    let voice = try await engine.encodeReference(audioURL: clip, transcript: transcript, name: "Narrator")
    var request = SpeechRequest(text: text, references: [voice])
    request.config.streaming = .defaultLowLatency
    var audio = StreamedAudio()
    var streamed: [Float] = []
    for try await event in engine.synthesize(request) {
        streamed += audio.newSamples(from: event)
        if case .finished(let result) = event { return (result, streamed) }
    }
    throw CancellationError()
}

let (result, _) = try await speak("Hello there. Good to see you.", cloning: URL(filePath: "clip.wav"),
                                  transcript: "What the clip says, word for word.", with: engine)
print("\(result.take.durationSeconds) s of audio")
```

`streamed` grows by about 3 s of 44.1 kHz mono audio at a time while the
model generates. In a player, schedule each `newSamples(from:)` result as it
arrives instead of collecting them.

The first load on a Mac also tunes the kernel schedule for that Mac, about 30 s,
and caches the result. Run ``DramaSetup/prepare(location:codecDirectory:python:tuningDirectory:progress:)``
ahead of time to move that cost out of the first request.

### What the package exposes

The public surface is small on purpose. The table names every public type and
what it is for.

| Area | Types |
| --- | --- |
| Setup | ``ModelLocation``, ``DramaSetup``, ``SetupReport``, ``SetupItem`` |
| Engines | ``SpeechSynthesizer``, ``DramaSynthesizer``, ``MockSynthesizer`` |
| Requests | ``SpeechRequest``, ``ReferenceVoice``, ``VoiceDesign``, ``Take``, ``GenerationConfig``, ``StreamingMode``, ``FrameCap`` |
| Results | ``SpeechEvent``, ``SpeechResult``, ``SpokenSentence``, ``SpeechTimings``, ``StreamedAudio``, ``GenerationEstimate`` |
| Text | ``Drama3Chunking``, ``Drama3Prompt``, ``PronunciationDictionary``, ``PronunciationRule`` |
| Codec | ``DramaCodec`` |
| Constants and errors | ``Frame``, ``ModelConstants``, `DramaEngineVersion`, ``PromptError``, ``CodecError``, ``EngineError``, ``TokenizerError``, ``DramaSetupError`` |

Every value type is `Sendable`. ``DramaSynthesizer``, ``MockSynthesizer`` and
``DramaCodec`` are actors, so you can share one instance across tasks.
``DramaSynthesizer`` runs one request at a time. A second request waits for the
first to finish.

`DramaEngineVersion.string` is the library's semantic version, `"1.0.2"`
for this release. `CHANGELOG.md` at the repository root lists what each
release changed.

## Topics

### Essentials

- <doc:GettingTheModel>
- ``ModelLocation``
- ``DramaSetup``
- ``SetupReport``
- ``SetupItem``
- ``DramaSetupError``

### Generating speech

- ``SpeechSynthesizer``
- ``DramaSynthesizer``
- ``MockSynthesizer``
- ``SpeechRequest``
- ``GenerationConfig``
- ``FrameCap``

### Voices and takes

- <doc:Voices>
- ``ReferenceVoice``
- ``VoiceDesign``
- ``Take``

### Streaming audio

- <doc:Streaming>
- ``StreamingMode``
- ``SpeechEvent``
- ``SpeechResult``
- ``SpokenSentence``
- ``SpeechTimings``
- ``StreamedAudio``
- ``GenerationEstimate``

### Text

- ``Drama3Chunking``
- ``Drama3Prompt``
- ``PronunciationDictionary``
- ``PronunciationRule``

### Codec

- ``DramaCodec``

### Constants and errors

- ``Frame``
- ``ModelConstants``
- ``PromptError``
- ``CodecError``
- ``EngineError``
- ``TokenizerError``
