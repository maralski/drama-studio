// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import DramaEngine

let usage = """
    usage: SpeakCLI <clip> <clip transcript> <text> <out.wav> [--mock]
           SpeakCLI --check
    """
var args = Array(CommandLine.arguments.dropFirst())
let mock = args.contains("--mock")
args.removeAll { $0 == "--mock" }

print("DramaEngine \(DramaEngineVersion.string)")
let report = DramaSetup.check(location: .huggingFaceCache())
if args == ["--check"] {
    for item in report.items { print("\(item.summary) \(item.fix ?? "")") }
    print("conversion script: \(DramaSetup.conversionScript?.path ?? "missing")")
    exit(report.isReady ? 0 : 1)
}
guard args.count == 4 else { print(usage); exit(2) }
let (clip, transcript, text, out) = (URL(filePath: args[0]), args[1], args[2], URL(filePath: args[3]))

let engine: any SpeechSynthesizer
if mock {
    engine = MockSynthesizer(pace: .instant)
} else {
    guard let location = report.location, report.isReady else {
        report.fixes.forEach { print($0) }
        exit(1)
    }
    engine = DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory)
}

try await engine.load { phase in print("load: \(phase)") }
let voice = try await engine.encodeReference(audioURL: clip, transcript: transcript, name: "Clip")
print("voice: \(String(format: "%.1f", voice.durationSeconds)) s clip")

var request = SpeechRequest(text: text, references: [voice])
request.config.streaming = .defaultLowLatency
var estimate = GenerationEstimate(lines: Drama3Chunking.splitSentencesTagged(text))
var audio = StreamedAudio()
var samples: [Float] = []
let start = ContinuousClock.now

for try await event in engine.synthesize(request) {
    let elapsed = (ContinuousClock.now - start) / .seconds(1)
    estimate.apply(event, at: elapsed)
    let new = audio.newSamples(from: event)
    samples += new
    switch event {
    case .chunkStarted(let index, let count, let line): print("sentence \(index + 1) of \(count): \(line)")
    case .frame(let index, _) where (index + 1) % 32 == 0: print("  \(index + 1) frames, \(estimate.label(at: elapsed))")
    case .retry(let chunk, let attempt): print("  sentence \(chunk + 1) again, attempt \(attempt)")
    case .finished(let result):
        print(String(format: "done: %.2f s of audio in %.2f s, %.1f frames/s", result.take.durationSeconds, elapsed, result.timings.framesPerSecond))
    default: break
    }
    if !new.isEmpty { print(String(format: "  +%.2f s of audio streamed", Double(new.count) / Double(ModelConstants.sampleRate))) }
}

let format = AVAudioFormat(standardFormatWithSampleRate: Double(ModelConstants.sampleRate), channels: 1)!
let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
buffer.frameLength = buffer.frameCapacity
samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: ModelConstants.sampleRate,
                               AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16]
try AVAudioFile(forWriting: out, settings: settings).write(from: buffer)
print(String(format: "wrote %@: %.2f s", out.path, Double(samples.count) / Double(ModelConstants.sampleRate)))
