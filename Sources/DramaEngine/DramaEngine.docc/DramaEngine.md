# ``DramaEngine``

Run Fish Audio's Drama 3 or inclusionAI's Ming-3.0-tts text-to-speech model on Apple silicon from your own macOS app.

## Overview

DramaEngine is the inference engine behind Drama Studio, packaged as a Swift library.
It loads the released Drama 3 weights in place, clones a voice from a short clip,
and streams speech as it generates. The 35B mixture-of-experts backbone and the
codebook tower run on hand-written Metal compute kernels. The DAC audio codec runs
through Core ML. Nothing runs through MLX or PyTorch at run time.

The model weights are under the
[Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE).
Cloning a real person's voice without their consent is not permitted.

DramaEngine also runs a second, smaller engine. ``MingSynthesizer`` runs
inclusionAI's [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
through [MingTTS](https://github.com/maralski/ming-tts), a Swift and Metal port
that the package depends on. Its checkpoint is 3.0 GB, and its weights are under
the MIT License. On a Mac with less than 64 GB of memory, use Ming instead of
Drama 3. Ming takes the same ``SpeechRequest`` behind the same
``SpeechSynthesizer`` protocol. ``SpeechEngineID/capabilities`` lists what each
engine can do, such as the cues it performs and the languages it speaks.

### Requirements

- An Apple silicon Mac. For Drama 3, use 64 GB or more of unified memory; this
  is a recommendation, not a tested minimum. Only a 128 GB M3 Max has been
  tested. Allow about 49 GB of free disk for model files and setup caches. For
  Ming, allow about 3 GB of free disk. Ming states no memory minimum and has
  also run only on the 128 GB M3 Max, where its largest resident size was
  5.9 GB. No Mac with less memory has been tested.
- Xcode 26 or its command line tools. Checked generation requires macOS 26 and the bundled Whisper word recognizer. Apple's on-device speech model for the request's language also checks the words when it is installed; without it, ``SpeechResult/notices`` holds ``SpeechNotice/wordsCheckedByOneRecognizer(language:missing:)``. Unchecked preview targets macOS 15.
- For Drama 3, the model in the Hugging Face cache and the Core ML codec models
  built once from it. For Ming, its checkpoint. See <doc:GettingTheModel>.

### Generate speech

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
    let (voice, _) = try await engine.cloneVoice(audioURL: clip, transcript: transcript, name: "Narrator")
    var request = SpeechRequest(text: text, references: [voice])
    request.config.wordVerification = .uncheckedPreview
    request.config.streaming = .defaultLowLatency
    request.config.maxRetries = 0
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

This example turns off the word check and retries so partial audio can
arrive before a chunk finishes. With either one on, the engine holds each
chunk until it passes. Keep the defaults when completeness matters more than
early playback.

`streamed` grows by about 3 s of 44.1 kHz mono audio at a time while the
model generates. In a player, schedule each `newSamples(from:)` result as it
arrives instead of collecting them.

The first load on a Mac also tunes the kernel schedule for that Mac, about 30 s,
and caches the result. Run ``DramaSetup/prepare(location:codecDirectory:python:tuningDirectory:progress:)``
ahead of time to move that cost out of the first request.

### What the package exposes

The table lists the main public types and their uses.

| Area | Types |
| --- | --- |
| Setup | ``ModelLocation``, ``MingLocation``, ``DramaSetup``, ``SetupReport``, ``SetupItem``, ``EngineInstall``, ``HuggingFaceCache`` |
| Engines | ``SpeechSynthesizer``, ``SpeechEngineID``, ``EngineDescription``, ``EngineCapabilities``, ``DramaSynthesizer``, ``MingSynthesizer``, ``SpeechEngines``, ``MockSynthesizer`` |
| Requests | ``SpeechRequest``, ``ReferenceVoice``, ``VoiceEncoding``, ``ReferenceAudio``, ``VoiceDesign``, ``Take``, ``TakeContext``, ``GenerationConfig``, ``StreamingMode``, ``FrameCap`` |
| Results | ``SpeechEvent``, ``SpeechResult``, ``SpeechNotice``, ``SpokenSentence``, ``SpeechTimings``, ``StreamedAudio``, ``GenerationEstimate`` |
| Text | ``Drama3Chunking``, ``Drama3Prompt``, ``PronunciationDictionary``, ``PronunciationRule`` |
| Codec | ``DramaCodec`` |
| Constants and errors | ``Frame``, ``ModelConstants``, `DramaEngineVersion`, ``PromptError``, ``CodecError``, ``EngineError``, ``TokenizerError``, ``DramaSetupError`` |

Every value type is `Sendable`. ``DramaSynthesizer``, ``MingSynthesizer``,
``SpeechEngines``, ``MockSynthesizer`` and ``DramaCodec`` are actors, so you can share one instance across tasks.
``DramaSynthesizer`` runs one request at a time. A second request waits for the
first to finish.

`DramaEngineVersion.string` is the library's semantic version, `"2.1.3"`
for this release. `CHANGELOG.md` at the repository root lists what each
release changed.

## Topics

### Essentials

- <doc:GettingTheModel>
- ``ModelLocation``
- ``MingLocation``
- ``DramaSetup``
- ``SetupReport``
- ``SetupItem``
- ``DramaSetupError``

### Generating speech

- ``SpeechSynthesizer``
- ``DramaSynthesizer``
- ``MingSynthesizer``
- ``SpeechEngines``
- ``MockSynthesizer``
- ``SpeechEngineID``
- ``EngineDescription``
- ``EngineCapabilities``
- ``EngineInstall``
- ``SpeechRequest``
- ``GenerationConfig``
- ``FrameCap``

### Voices and takes

- <doc:Voices>
- ``ReferenceVoice``
- ``VoiceEncoding``
- ``ReferenceAudio``
- ``VoiceDesign``
- ``BuiltInVoice``
- ``Take``
- ``TakeContext``

### Streaming audio

- <doc:Streaming>
- ``StreamingMode``
- ``SpeechEvent``
- ``SpeechResult``
- ``SpeechNotice``
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
