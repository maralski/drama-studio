# Streaming

Play speech while the model is still generating it, and show the time left.

## Overview

``SpeechSynthesizer/synthesize(_:)`` returns an `AsyncThrowingStream` of
``SpeechEvent``s. Every request ends with ``SpeechEvent/finished(_:)``, which
carries the whole result. ``GenerationConfig/streaming`` decides when the
audio arrives before that.

### Complete normal takes

Normal takes wait for the model's end of speech, subject to the text-derived
frame cap and loop guard. ``GenerationConfig/stopAfterSilenceSeconds`` is
`nil` by default because a silence stop can cut words after a quiet pause.
Quick preview explicitly uses a 2 s silence stop and disables retries.

A slow reference voice extends the frame budget and the progress estimate.
The retry check uses the script's length at 16 characters per second instead,
because reference audio may include pauses or trailing silence. If you opt
into silence stopping, probe readings follow the voice-aware allowance.

### Pick when audio leaves the engine

When ``GenerationConfig/maxRetries`` and ``GenerationConfig/minSpokenRatio``
are positive, every request uses sentence mode, regardless of the requested
streaming mode or voice. The engine checks each complete attempt before
releasing audio. Explicitly disable retries to receive low-latency partials.

| Mode | Audio events | A chunk that comes out short | Use for |
| --- | --- | --- | --- |
| ``StreamingMode/sentence``, the default | one ``SpeechEvent/audio(samples:chunkIndex:)`` per chunk, once the chunk is generated | generated again, up to ``GenerationConfig/maxRetries`` times | exports, captions, anything kept |
| ``StreamingMode/defaultLowLatency``, with retries disabled, 64-frame steps | ``SpeechEvent/partialAudio(samples:chunkIndex:frameRange:)`` about every 3 s of audio, then the chunk's `audio` | kept, because its start may already be playing | live playback |

Under low latency the CPU decodes the newest 64 frames while the GPU
generates the next ones. A chunk's partials, in order, join into exactly the
samples of its `audio` event: on the Core ML codec, 600 frames decoded in
64-frame steps differ from one full decode by a maximum absolute error of 0.

Measured on an M3 Max with a two-sentence line through the HTTP API, the
first audio byte came after 4.49 s in low-latency mode against 12.95 s when
every chunk was decoded after the whole request. The CPU decode slows
generation by about 6 percent while it overlaps the GPU. For a short first
sentence, low latency is no faster than sentence mode to the first audio. On
a 12.5 s first sentence it started at 6.19 s against 23.5 s. Steps of 32
frames started 0.7 s sooner but ran out of audio before the next partial in
every measured trace, so 64 is the default.

### Join the audio events

``StreamedAudio`` turns the events into one signal under either mode: it
returns each partial as it arrives, and from each chunk's final `audio` only
the samples its partials did not carry.

```swift
var request = SpeechRequest(text: script, references: [voice])
request.config.streaming = .defaultLowLatency
request.config.maxRetries = 0
var audio = StreamedAudio()
for try await event in engine.synthesize(request) {
    let samples = audio.newSamples(from: event)
    if !samples.isEmpty { player.schedule(samples) }
}
```

All audio is mono `Float` at 44.1 kHz. Every frame is 2048 samples, and every
audio event holds a whole number of frames.

### Show progress and the time left

``SpeechEvent/chunkStarted(index:count:text:)`` names each chunk, and
``SpeechEvent/frame(index:maxFrames:)`` counts its frames against the chunk's
frame cap. ``GenerationEstimate`` turns the events into a countdown:

```swift
let start = ContinuousClock.now
var estimate = GenerationEstimate(lines: Drama3Chunking.splitSentencesTagged(script))
for try await event in engine.synthesize(request) {
    let elapsed = (ContinuousClock.now - start) / .seconds(1)
    estimate.apply(event, at: elapsed)
    status = estimate.label(at: elapsed)
}
```

The label reads "about 12 s left", rounded to 5 s above 20 s, then
"finishing…" once the last chunk passes its expected length, or "retrying"
while a chunk is generated again.

### Time captions and highlights

``SpeechResult/sentences`` lists every chunk with its script text and its
frame count, in order. A chunk starts at the sum of the earlier chunks'
frames times 2048 samples, so captions and a highlight of the sentence being
spoken need no other timing.

### Cancel a request

Cancel the task that reads the stream. The engine stops after the frame it is
generating, and the next request can start.
