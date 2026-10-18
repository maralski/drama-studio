# Voices

Clone a voice from a clip, design one from a description, and keep it the
same across takes.

## Overview

A request takes its voice from at most one source: reference clips, a
designed character, or the first of its earlier takes. With none of them the
model picks a voice of its own, which changes from request to request.

### Clone a voice from a clip

Give ``SpeechSynthesizer/encodeReference(audioURL:transcript:name:)`` a clip
and what it says. It reads any format AVFoundation opens, mixes the channels
to mono, resamples to 44.1 kHz, and encodes the clip to codec codes. A 10 s
clip takes about 2.1 s on an M3 Max.

```swift
let voice = try await engine.encodeReference(
    audioURL: clipURL,
    transcript: "What the clip says, word for word.",
    name: "Narrator")
let request = SpeechRequest(text: "A new line in the same voice.", references: [voice])
```

The transcript matters: the model reads the clip as that text being spoken.
The project's performance measurements use a 10 s clip. On the speech
evaluation, cloned outputs scored a speaker similarity of 0.87 to the
clip, against 0.54 to 0.56 for outputs without the clip.

Cloning a real person's voice without their consent is not permitted by the
[Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE).

### Keep a voice library

``ReferenceVoice`` is `Codable` and holds the codes, so store it and skip the
encoder next time. ``ReferenceVoice/durationSeconds`` gives the clip's length.
Every frame of a reference is one prompt row, and a prompt holds at most
``Drama3Prompt/maxRows`` (8192) rows, so keep clips short.

### Record a clip

Record with AVFoundation, for example `AVAudioRecorder` writing a WAV, then
pass the file to `encodeReference`. Recording needs the microphone: add
`NSMicrophoneUsageDescription` to the app's `Info.plist`, and under the App
Sandbox or the Hardened Runtime the `com.apple.security.device.audio-input`
entitlement. Nothing else in DramaEngine needs the microphone.

### Voice several speakers

Tag the script with `<|speaker:N|>`, N from 0 to 15, and pass one reference
per speaker in speaker order, with no gaps from speaker 0:

```swift
let request = SpeechRequest(
    text: "<|speaker:0|> Are you coming? <|speaker:1|> In a minute.",
    references: [narrator, guest])
```

### Design a voice from a description

A ``VoiceDesign`` replaces the clip with a name, a description, and an
optional scene:

```swift
let captain = VoiceDesign(name: "Captain", description: "gravelly old male voice, weathered and slow")
let request = SpeechRequest(text: "[low and menacing] Pray ye never find out.", character: captain)
```

A request cannot have both a character and references: the stream throws
``PromptError/voiceConflict``.

### Keep the voice across takes

Every request returns its take in ``SpeechResult/take``. Pass earlier takes,
oldest first, as ``SpeechRequest/previousTakes``. The last two become
history. With no references, the first take is also the voice reference, so
a designed or model-chosen voice stays the same from take to take:

```swift
var takes: [Take] = []
for line in lines {
    var request = SpeechRequest(text: line, character: captain, previousTakes: takes)
    request.config.seed = 7
    // Read the stream; keep the result's take.
    takes.append(result.take)
}
```

A ``Take`` is `Codable`, so a project can save it and continue later in the
same voice.

### Direct a retake with a note

Set ``SpeechRequest/note`` to a director's note, such as "softer and
warmer", and pass the line's takes. The line is generated again as one turn,
with the note as a direction and as a cue before every sentence. The engine
cuts the note to 120 characters and drops `[ ] < > |` from it.

```swift
let retake = SpeechRequest(text: line, note: "softer and warmer", previousTakes: [firstTake])
```

### Shape the delivery

Write performance cues into the text: `[laughs]`, `[sighs]`,
`<whisper>…</whisper>`. In the reliability study the cue lines were the one
area where the model dropped content a few times in twenty, and `[laughs]`
can turn into a laughter loop that the frame cap ends. 
``GenerationConfig/make(expressiveness:quickPreview:)`` maps one 0-to-1
control onto temperature, top-p and top-k, with the vendor's sampling at 0.5
and the Hugging Face Space's (temperature 1.0, top-p 0.9, no top-k) at 1.
The Space's sampling kept every sentence of the cue lines in the reliability
study and needed fewer retries, at a cost of 0.007 of word error rate on
prose.
