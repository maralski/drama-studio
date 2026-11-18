// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
import DramaTTS
@testable import DramaEngine

final class Drama3BridgeTests: XCTestCase {
    func testEveryConfigurationFieldReachesTheEngine() {
        var config = GenerationConfig()
        config.wordVerification = .uncheckedPreview
        config.temperature = 0.91
        config.topK = 17
        config.topP = 0.83
        config.frameCap = .fixed(123)
        config.seed = 99
        config.chunkBytes = 211
        config.chunking = .reference
        config.stopAfterSilenceSeconds = 1.25
        config.minSpokenRatio = 0.4
        config.maxRetries = 2
        config.stopOnLoop = false
        config.retryWithoutEOSBelow = 0.7
        config.capSecondsPerCue = 2.5
        config.streaming = .lowLatency(stepFrames: 32)
        let options = config.drama
        XCTAssertFalse(options.reviewsAttempts)
        XCTAssertEqual(options.temperature, 0.91)
        XCTAssertEqual(options.topK, 17)
        XCTAssertEqual(options.topP, 0.83)
        XCTAssertEqual(options.frameCap, .fixed(123))
        XCTAssertEqual(options.seed, 99)
        XCTAssertEqual(options.chunkBytes, 211)
        XCTAssertEqual(options.chunking, .reference)
        XCTAssertEqual(options.stopAfterSilenceSeconds, 1.25)
        XCTAssertEqual(options.minSpokenRatio, 0.4)
        XCTAssertEqual(options.maxRetries, 2)
        XCTAssertFalse(options.stopOnLoop)
        XCTAssertEqual(options.retryWithoutEOSBelow, 0.7)
        XCTAssertEqual(options.capSecondsPerCue, 2.5)
        XCTAssertEqual(options.streaming, .lowLatency(stepFrames: 32))
    }

    func testDefaultsMatchTheEngineExceptForTheReview() {
        var expected = DramaSynthesisOptions()
        expected.reviewsAttempts = true
        XCTAssertEqual(GenerationConfig().drama, expected)
        XCTAssertEqual(GenerationConfig().effectiveStreaming.drama, expected.effectiveStreaming)
    }

    func testEngineErrorsArriveAsDramaEngineErrors() {
        XCTAssertEqual(Drama3Errors.translate(DramaPromptError.tooLong(9000)) as? PromptError, .tooLong(9000))
        XCTAssertEqual(Drama3Errors.translate(DramaPromptError.voiceConflict) as? PromptError, .voiceConflict)
        guard case .exhausted(sentence: 2, attempts: 6)? = Drama3Errors.translate(DramaReviewError.exhausted(sentence: 2, attempts: 6)) as? SpeechVerificationError else {
            return XCTFail("a rejected sentence must surface as SpeechVerificationError.exhausted")
        }
        guard case .badModel("x")? = Drama3Errors.translate(DramaEngineError.badModel("x")) as? EngineError else {
            return XCTFail("a bad snapshot must surface as EngineError.badModel")
        }
        XCTAssertThrowsError(try Drama3Prompt.render(text: "<|speaker:16|> Hi")) { error in
            XCTAssertEqual(error as? PromptError, .badSpeakerTag("text"))
        }
    }

    func testConstantsAgreeWithTheEngine() {
        XCTAssertEqual(ModelConstants.sampleRate, DramaConstants.sampleRate)
        XCTAssertEqual(ModelConstants.samplesPerFrame, DramaConstants.samplesPerFrame)
        XCTAssertEqual(ModelConstants.numCodebooks, DramaConstants.numCodebooks)
        XCTAssertEqual(ModelConstants.codebookSize, DramaConstants.codebookSize)
        XCTAssertEqual(ModelConstants.semanticStart, DramaConstants.semanticStart)
        XCTAssertEqual(ModelConstants.maxSpeakerID, DramaConstants.maxSpeakerID)
        XCTAssertEqual(DramaCodec.windowFrames, DramaAudioCodec.windowFrames)
        XCTAssertEqual(DramaCodec.modelNames, DramaAudioCodec.modelNames)
        XCTAssertEqual(FrameCap.maximumFrames, DramaFrameCap.maximumFrames)
        XCTAssertEqual(FrameCap.minimumFrames, DramaFrameCap.minimumFrames)
    }
}
