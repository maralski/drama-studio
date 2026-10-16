import XCTest
@testable import DramaEngine

/// The app's engine path end to end: text in, streamed events and 44.1 kHz
/// audio out, through the Metal engine and the Core ML codec.
final class SynthesizerIntegrationTests: XCTestCase {
    func testShortSentenceProducesSpeech() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        // A temp tuning directory keeps the test from writing the user's cache
        // with a debug-process measurement; the first load tunes into it.
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 2
        let request = SpeechRequest(text: "Hello, my name is Drama and I live inside a computer.", config: config)
        var audioChunks = 0
        var frames = 0
        var result: SpeechResult?
        for try await event in synth.synthesize(request) {
            switch event {
            case .audio(let samples, _): audioChunks += 1; XCTAssertGreaterThan(samples.count, 44_100)
            case .frame: frames += 1
            case .finished(let r): result = r
            default: break
            }
        }
        let r = try XCTUnwrap(result)
        XCTAssertEqual(audioChunks, 1)
        XCTAssertEqual(r.frames.count, frames)
        XCTAssertEqual(r.samples.count, r.frames.count * ModelConstants.samplesPerFrame)
        let rms = (r.samples.reduce(0) { $0 + $1 * $1 } / Float(r.samples.count)).squareRoot()
        XCTAssertGreaterThan(rms, 0.02, "decoded audio should contain speech")
        XCTAssertGreaterThan(Double(r.frames.count) * 2048 / 44100, 2.0)
        print("integration: \(r.frames.count) frames, \(String(format: "%.1f", Double(r.samples.count) / 44100))s audio, rms \(rms), \(String(format: "%.1f", r.timings.framesPerSecond)) frames/s, codec \(String(format: "%.2f", r.timings.codecSeconds))s")
    }
}
