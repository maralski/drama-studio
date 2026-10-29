// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import DramaEngine
import XCTest

// The landing page's example, verbatim; testLandingPageShowsThisExample keeps the two equal.
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

final class SDKUsageTests: XCTestCase {
    func testExampleSpeaksWithTheMock() async throws {
        let clip = FileManager.default.temporaryDirectory.appending(path: "sdk-usage-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: clip) }
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 44_100))
        buffer.frameLength = 44_100
        try AVAudioFile(forWriting: clip, settings: format.settings).write(from: buffer)

        let text = "Hello there, it is good to see you again. The train leaves at nine."
        let (result, streamed) = try await speak(text, cloning: clip, transcript: "One second of silence.",
                                                 with: MockSynthesizer(pace: .instant))

        XCTAssertEqual(result.sentences.map(\.text), Drama3Chunking.splitSentencesTagged(text))
        XCTAssertEqual(streamed, result.samples)
        XCTAssertEqual(result.samples.count, result.take.frameCount * ModelConstants.samplesPerFrame)
        XCTAssertEqual(result.sentences.map(\.frames).reduce(0, +), result.take.frameCount)
    }

    func testLandingPageShowsThisExample() throws {
        let here = URL(filePath: #filePath)
        let page = here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Sources/DramaEngine/DramaEngine.docc/DramaEngine.md")
        let source = try String(contentsOf: here, encoding: .utf8)
        let start = try XCTUnwrap(source.range(of: "func speak("))
        let end = try XCTUnwrap(source.range(of: "\n}\n", range: start.upperBound..<source.endIndex))
        let example = String(source[start.lowerBound..<end.upperBound])
        XCTAssertTrue(try String(contentsOf: page, encoding: .utf8).contains(example), "DramaEngine.md must show speak(_:cloning:transcript:with:) exactly as this test runs it")
    }
}
