// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class SpokenSentenceTests: XCTestCase {
    func testTheResultNamesEachChunkWithTheFramesItSpoke() async throws {
        var config = GenerationConfig()
        config.seed = 3
        config.streaming = .defaultLowLatency
        let request = SpeechRequest(text: "The first sentence is here. <|speaker:1|>And a second one follows it.", config: config)
        var chunkFrames: [Int: Int] = [:]
        var result: SpeechResult?
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            switch event {
            case .audio(let samples, let chunk): chunkFrames[chunk] = samples.count
            case .finished(let r): result = r
            default: break
            }
        }
        let r = try XCTUnwrap(result)
        XCTAssertEqual(r.sentences.map(\.text), try GenerationPlan(request).chunks.map(\.line))
        XCTAssertEqual(r.sentences.map(\.samples), [0, 1].map { chunkFrames[$0] ?? -1 })
        XCTAssertEqual(r.sentences.reduce(0) { $0 + $1.samples }, r.take.sampleCount)
    }
}
