// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class CueStreamingTests: XCTestCase {
    private let line = "The garden gate is closed, and we can all go back to the house together now."

    private func partials(for request: SpeechRequest) async throws -> Int {
        var partials = 0
        var samples = 0
        var finished = false
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            switch event {
            case .partialAudio: partials += 1
            case .audio(let audio, _): samples += audio.count
            case .finished(let result):
                XCTAssertEqual(result.samples.count, samples)
                finished = true
            default: break
            }
        }
        XCTAssertTrue(finished)
        XCTAssertGreaterThan(samples, 0)
        return partials
    }

    func testCueBearingRequestsEmitOnlyCompletedChunks() async throws {
        var config = GenerationConfig()
        config.streaming = .lowLatency(stepFrames: 16)
        let requests = [
            SpeechRequest(text: "[laughs] " + line, config: config),
            SpeechRequest(text: "[with bright eager excitement] " + line, config: config),
            SpeechRequest(text: "<whisper>" + line + "</whisper>", config: config),
            SpeechRequest(text: line, note: "more amused", config: config),
        ]
        for request in requests {
            let count = try await partials(for: request)
            XCTAssertEqual(count, 0, request.text)
        }
    }

    func testQuickPreviewAndExplicitlyDisabledRetriesKeepCuePartials() async throws {
        var quick = GenerationConfig.make(expressiveness: 0.5, quickPreview: true)
        quick.streaming = .lowLatency(stepFrames: 16)
        var noAttempts = GenerationConfig()
        noAttempts.streaming = .lowLatency(stepFrames: 16)
        noAttempts.maxRetries = 0
        var noThreshold = GenerationConfig()
        noThreshold.streaming = .lowLatency(stepFrames: 16)
        noThreshold.minSpokenRatio = 0
        for config in [quick, noAttempts, noThreshold] {
            let count = try await partials(for: SpeechRequest(text: "[laughs] " + line, config: config))
            XCTAssertGreaterThan(count, 0)
        }
    }

    func testSpeakerTagsAloneKeepLowLatency() async throws {
        var config = GenerationConfig()
        config.streaming = .lowLatency(stepFrames: 16)
        let count = try await partials(for: SpeechRequest(text: "<|speaker:0|> " + line, config: config))
        XCTAssertGreaterThan(count, 0)
    }
}
