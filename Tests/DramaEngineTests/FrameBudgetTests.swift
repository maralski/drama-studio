// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class FrameBudgetTests: XCTestCase {
    func testSlowReferenceSetsThePoemTiming() throws {
        let reference = Drama3Prompt.Reference(
            transcript: "Hey there my name is Joey and I love you",
            codes: Array(repeating: Array(repeating: 1, count: 142), count: 10))
        let plan = try GenerationPlan(
            text: "Here is a poem for you.\nThe morning light begins to break, Across the quiet, misty lake.",
            references: [reference], config: GenerationConfig())
        XCTAssertEqual(GenerationEstimate(plan: plan).expectedFrames, [82, 227],
                       "A slow reference must not use the generic 16 characters/second timing.")
        let line = try XCTUnwrap(plan.chunks.last).line
        let expected = plan.expectedSeconds(for: line)
        XCTAssertEqual(FrameGenerator.frameBudget(for: line, config: GenerationConfig(), expectedSeconds: expected), 384)
        let probe = SilenceProbePolicy.shipped.schedule(expectedSeconds: expected, silenceFrames: 44)
        XCTAssertFalse(probe.reads(at: 96), "A pause within the poem sentence must not end this slow chunk.")
        XCTAssertFalse(probe.reads(at: 256))
        XCTAssertTrue(probe.reads(at: 272))
        var fixed = GenerationConfig()
        fixed.frameCap = .fixed(100)
        XCTAssertEqual(FrameGenerator.frameBudget(for: line, config: fixed, expectedSeconds: expected), 100)
    }

    func testReferenceSilenceExtendsTheBudgetWithoutRaisingTheSpokenRetryThreshold() throws {
        let line = "Always verify important details"
        // A roughly 7.3 s reference can contain pauses or a long silent tail.
        // The complete 3.07 s EOS take must not be judged against that duration.
        let reference = Drama3Prompt.Reference(transcript: line,
            codes: Array(repeating: Array(repeating: 1, count: 157), count: 10))
        let config = GenerationConfig()
        let plan = try GenerationPlan(text: line, references: [reference], config: config)
        let allowance = plan.expectedSeconds(for: line)
        XCTAssertEqual(allowance, 7.291065759637188, accuracy: 1e-9)
        XCTAssertEqual(GenerationEstimate(plan: plan).expectedFrames, [157])
        XCTAssertEqual(FrameGenerator.frameBudget(for: line, config: config, expectedSeconds: allowance), 279)
        let expected = FrameGenerator.expectedSeconds(for: line)
        XCTAssertEqual(expected, 1.9375)
        XCTAssertFalse(FrameGenerator.shouldRetry(end: .eos, spokenSeconds: 3.07, expectedSeconds: expected, config: config))
        XCTAssertTrue(FrameGenerator.shouldRetry(end: .eos, spokenSeconds: 1, expectedSeconds: expected, config: config),
                      "The text-only threshold must still reject a short attempt.")
        let probe = SilenceProbePolicy.shipped.schedule(expectedSeconds: allowance, silenceFrames: 44)
        XCTAssertFalse(probe.reads(at: 192))
        XCTAssertTrue(probe.reads(at: 208), "Opt-in silence probes must keep the full reference-aware allowance.")
    }

    func testFastReferencesDoNotShortenTheDefaultAllowance() throws {
        let reference = Drama3Prompt.Reference(transcript: "Twelve chars",
            codes: Array(repeating: Array(repeating: 1, count: 10), count: 10))
        for references in [[], [reference]] {
            let plan = try GenerationPlan(text: "Twelve chars", references: references, config: GenerationConfig())
            XCTAssertEqual(plan.expectedSeconds(for: "Twelve chars"), 0.75)
            XCTAssertEqual(GenerationEstimate(plan: plan).expectedFrames, [16])
        }
    }

    func testVoiceTimingAppliesWhenAnEarlierTakeSuppliesTheReference() throws {
        let take = Take(text: "Hey there my name is Joey and I love you",
                        codes: Array(repeating: Array(repeating: 1, count: 142), count: 10))
        let plan = try GenerationPlan(text: "Hey there my name is Joey and I love you", references: [],
                                      previousTakes: [take], config: GenerationConfig())
        XCTAssertEqual(GenerationEstimate(plan: plan).expectedFrames, [142])
    }

    func testPassageMarkupDoesNotCountAsSpokenText() {
        for text in ["<whisper>Twelve chars</whisper>", "<slow>Twelve chars</slow>",
                     "<emphasis>Twelve chars</emphasis>", "<whisper><slow>Twelve chars</slow></whisper>"] {
            XCTAssertEqual(FrameGenerator.expectedSeconds(for: text), 0.75)
            XCTAssertEqual(FrameGenerator.frameBudget(for: text, config: GenerationConfig()), 68)
            XCTAssertEqual(GenerationEstimate(lines: [text]).expectedFrames, [16])
        }
    }

    func testDefaultCapFollowsTheText() {
        XCTAssertEqual(GenerationConfig().frameCap, .fromText(margin: 1.5, floorSeconds: 2.0))
    }

    func testFromTextIsExpectedTimesMarginPlusFloorRoundedUpToFrames() {
        let cap = FrameCap.fromText(margin: 1.5, floorSeconds: 2.0)
        XCTAssertEqual(cap.frames(expectedSeconds: 3), 140, "(3 * 1.5 + 2) s * 44100 / 2048 = 139.97")
        XCTAssertEqual(cap.frames(expectedSeconds: 0), 44, "2 s = 43.07 frames")
        let sentence = "Twelve chars"
        XCTAssertEqual(FrameGenerator.expectedSeconds(for: sentence), 0.75)
        XCTAssertEqual(cap.frames(expectedSeconds: FrameGenerator.expectedSeconds(for: sentence)), 68)
    }

    func testFromTextClampsToTwoSecondsAndTheOldFixedCap() {
        let preview = FrameCap.fromText(margin: 1.2, floorSeconds: 1.0)
        XCTAssertEqual(preview.frames(expectedSeconds: 0), 43)
        XCTAssertEqual(preview.frames(expectedSeconds: 1), 48)
        XCTAssertEqual(FrameCap.fromText(margin: 1.5, floorSeconds: 2.0).frames(expectedSeconds: 60), 600)
    }

    func testFixedIsTakenAsGiven() {
        XCTAssertEqual(FrameCap.fixed(2048).frames(expectedSeconds: 1), 2048)
        XCTAssertEqual(FrameCap.fixed(10).frames(expectedSeconds: 30), 10)
    }

    func testMaxNewFramesPinsAFixedCapForTheCLI() {
        var config = GenerationConfig()
        config.maxNewFrames = 300
        XCTAssertEqual(config.frameCap, .fixed(300))
        XCTAssertEqual(config.maxNewFrames, 300)
    }
}
