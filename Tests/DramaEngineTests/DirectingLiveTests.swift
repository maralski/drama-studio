import XCTest
@testable import DramaEngine

/// Voice design and a director's-note retake on the real model. Opt in with
/// DRAMA_DIRECTING_OUT=<directory>; each take is written there as a WAV.
final class DirectingLiveTests: XCTestCase {
    func testDesignedLineThenRetakesWithNotes() async throws {
        guard let out = ProcessInfo.processInfo.environment["DRAMA_DIRECTING_OUT"] else { throw XCTSkip("set DRAMA_DIRECTING_OUT to run") }
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let directory = URL(filePath: out, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tuning = directory.appending(path: "tuning")
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        let captain = VoiceDesign(name: "Captain", description: "gravelly old male voice, about seventy, weathered and slow, salt in every word",
                                  scene: "a storm-battered tavern at the edge of the harbor")
        let line = "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out."

        func run(_ name: String, _ request: SpeechRequest) async throws -> SpeechResult {
            var result: SpeechResult?
            for try await event in synth.synthesize(request) {
                if case .finished(let r) = event { result = r }
            }
            let r = try XCTUnwrap(result)
            try Self.wav(r.samples).write(to: directory.appending(path: "\(name).wav"))
            try JSONEncoder().encode(r.take.text).write(to: directory.appending(path: "\(name).turn.json"))
            print("live \(name): \(r.take.frameCount) frames, \(String(format: "%.2f", r.take.durationSeconds)) s, turn \(r.take.text.debugDescription)")
            return r
        }

        func config(_ seed: UInt64) -> GenerationConfig {
            var c = GenerationConfig()
            c.seed = seed
            return c
        }

        let design = try await run("1-design", SpeechRequest(text: line, character: captain, config: config(11)))
        let plain = try await run("2-retake-plain", SpeechRequest(text: line, character: captain, previousTakes: [design.take], config: config(12)))
        let history = [design.take, plain.take]
        _ = try await run("3-note-whispered", SpeechRequest(text: line, character: captain, note: "whispered", previousTakes: history, config: config(13)))
        _ = try await run("4-note-slower", SpeechRequest(text: line, character: captain, note: "slower, with more pauses", previousTakes: history, config: config(13)))
        _ = try await run("5-note-angrier", SpeechRequest(text: line, character: captain, note: "angrier", previousTakes: history, config: config(13)))
    }

    static func wav(_ samples: [Float]) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let rate = UInt32(ModelConstants.sampleRate)
        data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + samples.count * 2))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(rate); append(rate * 2); append(UInt16(2)); append(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); append(UInt32(samples.count * 2))
        for s in samples { append(Int16(max(-1, min(1, s)) * 32767)) }
        return data
    }
}
