// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import DramaEngine

final class ProgressiveDecodeTests: XCTestCase {
    func testMockStreamsPartialsThatJoinIntoEachChunk() async throws {
        var config = GenerationConfig()
        config.seed = 1
        config.streaming = .lowLatency(stepFrames: 16)
        config.maxRetries = 0
        config.wordVerification = .uncheckedPreview
        let request = SpeechRequest(text: "The first sentence runs a while. The second one is here too.", config: config)
        var partials: [Int: [Float]] = [:]
        var finals: [Int: [Float]] = [:]
        var ranges: [Range<Int>] = []
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            switch event {
            case .partialAudio(let samples, let chunk, let range):
                partials[chunk, default: []] += samples
                ranges.append(range)
                XCTAssertEqual(samples.count, range.count * ModelConstants.samplesPerFrame)
            case .audio(let samples, let chunk): finals[chunk] = samples
            default: break
            }
        }
        XCTAssertFalse(finals.isEmpty)
        XCTAssertEqual(partials, finals)
        XCTAssertTrue(ranges.contains { $0.count == 16 })
    }

    func testSentenceModeSendsNoPartials() async throws {
        var config = GenerationConfig()
        config.seed = 1
        let request = SpeechRequest(text: "Only whole sentences arrive here.", config: config)
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            if case .partialAudio = event { XCTFail("sentence mode streamed a partial") }
        }
    }
}
