// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class ChunkAcceptanceTests: XCTestCase {
    func testCheckedReferenceChunkingUsesSentenceUnitsEvenWithoutRetries() throws {
        var config = GenerationConfig()
        config.chunking = .reference
        config.chunkBytes = 1024
        config.maxRetries = 0
        let line = "<whisper>The garden gate is closed. The meeting begins on Monday.</whisper>"
        let plan = try GenerationPlan(text: line, references: [], config: config)
        XCTAssertEqual(plan.chunks.map(\.line), [
            "<whisper>The garden gate is closed.</whisper>",
            "<whisper>The meeting begins on Monday.</whisper>"
        ])
        config.wordVerification = .uncheckedPreview
        XCTAssertEqual(try GenerationPlan(text: line, references: [], config: config).chunks.map(\.line), [line])
    }

    private func result(_ attempt: Int, end: FrameGenerator.ChunkEnd = .eos, spoken: Double = 10) -> FrameGenerator.ChunkResult {
        FrameGenerator.ChunkResult(index: 0, count: 1, text: "[coughs] The garden gate is closed.",
                                   frames: [[Int32(attempt)]], promptRows: 1, end: end, frameBudget: 200,
                                   prefillSeconds: 0, decodeSeconds: 0, gpuSeconds: 0, spokenSeconds: spoken,
                                   attempts: attempt + 1, samples: [Float(attempt)], codecSeconds: 0)
    }

    func testOnlyTheWordCheckedAttemptCanLeaveTheRetryBoundary() throws {
        var generated: [Int] = [], checked: [Int] = [], retries: [Int] = []
        let accepted = try ChunkAttempts.run(index: 0, maxRetries: 2, expectedSeconds: 2, config: GenerationConfig(),
                                             shouldContinue: { true }, attempt: { number in
            generated.append(number); return self.result(number)
        }, verify: { result in checked.append(result.attempts); return result.attempts == 3 }, retry: { number, _ in retries.append(number) })
        XCTAssertEqual(generated, [0, 1, 2])
        XCTAssertEqual(checked, [1, 2, 3], "Long EOS audio must still undergo word checks.")
        XCTAssertEqual(retries, [1, 2])
        XCTAssertEqual(accepted.samples, [2], "Only accepted audio can be emitted or used as history.")
    }

    func testExhaustionDoesNotReturnTheBestFailedAudio() {
        var attempts = 0
        XCTAssertThrowsError(try ChunkAttempts.run(index: 1, maxRetries: 2, expectedSeconds: 2, config: GenerationConfig(),
                                                   shouldContinue: { true }, attempt: { number in
            attempts += 1; return self.result(number)
        }, verify: { _ in false }, retry: { _, _ in })) { error in
            guard case SpeechVerificationError.exhausted(sentence: 2, attempts: 3) = error else { return XCTFail("Unexpected \(error)") }
        }
        XCTAssertEqual(attempts, 3)
    }

    func testDisablingHeuristicsOrRetriesDoesNotBypassWordChecks() {
        var config = GenerationConfig()
        config.minSpokenRatio = 0
        config.maxRetries = 0
        XCTAssertThrowsError(try ChunkAttempts.run(index: 0, maxRetries: 0, expectedSeconds: 2, config: config,
                                                   shouldContinue: { true }, attempt: { self.result($0) },
                                                   verify: { _ in false }, retry: { _, _ in XCTFail("No retry allowed") }))
    }

    func testCheckedFastSpeechCanPassWithoutMeetingDurationHeuristic() throws {
        let accepted = try ChunkAttempts.run(index: 0, maxRetries: 2, expectedSeconds: 10, config: GenerationConfig(),
                                             shouldContinue: { true }, attempt: { self.result($0, spoken: 1) },
                                             verify: { _ in true }, retry: { _, _ in XCTFail("All words were recognized") })
        XCTAssertEqual(accepted.samples, [0])
    }

    func testCancellationDuringRecognitionCannotReturnAudio() {
        var active = true
        XCTAssertThrowsError(try ChunkAttempts.run(index: 0, maxRetries: 2, expectedSeconds: 2, config: GenerationConfig(),
                                                   shouldContinue: { active }, attempt: { self.result($0) }, verify: { _ in
            active = false; return true
        }, retry: { _, _ in XCTFail("Cancelled") })) { XCTAssertTrue($0 is CancellationError) }
    }

    func testRecognizerFailureDoesNotBecomeSuccessfulFallbackAudio() {
        XCTAssertThrowsError(try ChunkAttempts.run(index: 0, maxRetries: 2, expectedSeconds: 2, config: GenerationConfig(),
                                                   shouldContinue: { true }, attempt: { self.result($0) },
                                                   verify: { _ in throw SpeechVerificationError.timedOut }, retry: { _, _ in XCTFail("Operational failure") }))
    }

    func testOnlyRejectedCueAttemptsBroadenSamplingWithoutChangingTheRequestPolicy() {
        let original = GenerationConfig.make(expressiveness: 0.25, quickPreview: false)
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "[coughs] Keep the gate closed.", attempt: 0), original)
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "Keep the gate closed.", attempt: 1), original)
        var replacement = original
        replacement.temperature = 1
        replacement.topP = 0.9
        replacement.topK = 4097
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "[coughs] Keep the gate closed.", attempt: 1), replacement)
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "[coughs] Keep the gate closed.", attempt: 2), original)
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "[coughs] Keep the gate closed.", attempt: 3), replacement)
        XCTAssertEqual(FrameGenerator.sampling(config: original, text: "<whisper>Keep the gate closed.</whisper>", attempt: 4), original)
        let preview = GenerationConfig.make(expressiveness: 0.25, quickPreview: true)
        XCTAssertEqual(FrameGenerator.sampling(config: preview, text: "[coughs] Keep the gate closed.", attempt: 1), preview)
    }

    func testTimingsCountRejectedWorkAsWellAsTheAcceptedAudio() throws {
        var timings = SpeechTimings()
        let accepted = try ChunkAttempts.run(index: 0, maxRetries: 2, expectedSeconds: 2, config: GenerationConfig(),
                                             shouldContinue: { true }, attempt: { number in
            var result = self.result(number)
            result.frames = Array(repeating: [Int32(number)], count: number + 2)
            result.prefillSeconds = 0.25
            result.decodeSeconds = 1
            result.codecSeconds = 0.5
            timings.include(result)
            return result
        }, verify: { $0.attempts == 3 }, retry: { _, _ in })
        XCTAssertEqual(accepted.frames.count, 4)
        XCTAssertEqual(timings.framesGenerated, 9)
        XCTAssertEqual(timings.promptRows, 3)
        XCTAssertEqual(timings.prefillSeconds, 0.75)
        XCTAssertEqual(timings.decodeSeconds, 3)
        XCTAssertEqual(timings.codecSeconds, 1.5)
    }

    func testExplicitPreviewKeepsUncheckedStreamingBehavior() throws {
        let config = GenerationConfig.make(expressiveness: 0.5, quickPreview: true)
        let accepted = try ChunkAttempts.run(index: 0, maxRetries: 0, expectedSeconds: 2, config: config,
                                             shouldContinue: { true }, attempt: { self.result($0) },
                                             verify: { _ in XCTFail("Preview is unchecked"); return false }, retry: { _, _ in })
        XCTAssertEqual(accepted.samples, [0])
    }
}
