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

Add the package, then check the setup, load the engine, clone a voice, and speak:

```swift
import DramaEngine

let location = ModelLocation.huggingFaceCache()
let report = DramaSetup.check(location: location)
guard let location, report.isReady else {
	report.items.forEach { print($0.summary) }
	exit(1)
}

let engine = DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory)
try await engine.load { phase in print(phase) }

let voice = try await engine.encodeReference(
	audioURL: URL(filePath: "clip.wav"),
	transcript: "What the clip says, word for word.",
	name: "Narrator")

var request = SpeechRequest(text: "Hello there. Good to see you.", references: [voice])
request.config.streaming = .defaultLowLatency

var audio = StreamedAudio()
for try await event in engine.synthesize(request) {
	switch event {
	case .frame(let index, let maxFrames): print("frame \(index + 1) of at most \(maxFrames)")
	case .finished(let result): print("\(result.take.durationSeconds) s of audio")
	default: break
	}
	let samples = audio.newSamples(from: event)
	// Play or store `samples`: 44.1 kHz mono Float.
}
```

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
| Constants and errors | ``Frame``, ``ModelConstants``, ``DramaEngineVersion``, ``PromptError``, ``CodecError``, ``EngineError``, ``DramaSetupError`` |

Every value type is `Sendable`. ``DramaSynthesizer``, ``MockSynthesizer`` and
``DramaCodec`` are actors, so you can share one instance across tasks.
``DramaSynthesizer`` runs one request at a time. A second request waits for the
first to finish.

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

### Voices

- <doc:Voices>
- ``ReferenceVoice``
- ``VoiceDesign``
- ``Take``

### Streaming

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
- ``DramaEngineVersion``
- ``PromptError``
- ``CodecError``
- ``EngineError``
