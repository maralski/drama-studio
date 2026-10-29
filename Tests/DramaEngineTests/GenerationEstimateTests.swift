// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class GenerationEstimateTests: XCTestCase {
    private let lines = ["The first sentence takes a few seconds to say.", "A second sentence that stands on its own."]

    func testUpFrontEstimateUsesTheTypicalRateAndOnePrefillPerChunk() throws {
        let estimate = GenerationEstimate(lines: lines)
        let expected = lines.map { Int((FrameGenerator.expectedSeconds(for: $0) * 44_100 / 2048).rounded()) }
        XCTAssertEqual(estimate.expectedFrames, expected)
        let remaining = try XCTUnwrap(estimate.remainingSeconds(at: 0))
        XCTAssertEqual(remaining, Double(expected.reduce(0, +)) / 35 + 2 * 0.7, accuracy: 1e-9)
    }

    func testChunksComeFromTheGeneratorsPlan() throws {
        let plan = try GenerationPlan(text: lines.joined(separator: " "), references: [], config: GenerationConfig())
        XCTAssertEqual(GenerationEstimate(plan: plan).expectedFrames, GenerationEstimate(lines: lines).expectedFrames)
    }

    /// Frames at a steady 20 frames/s, far from the 35 the estimate starts with:
    /// the predicted finish time settles on the true one.
    func testEstimateConvergesAtASteadyRate() throws {
        let rate = 20.0
        let prefill = 0.4
        var estimate = GenerationEstimate(lines: lines)
        let frames = estimate.expectedFrames
        let trueEnd = Double(frames.reduce(0, +)) / rate + prefill * Double(frames.count)
        var t = 0.0
        var errors: [Double] = []
        for (index, count) in frames.enumerated() {
            estimate.apply(.chunkStarted(index: index, count: frames.count, text: lines[index]), at: t)
            t += prefill
            estimate.apply(.prefill(rows: 100), at: t)
            for frame in 0 ..< count {
                t += 1 / rate
                estimate.apply(.frame(index: frame, maxFrames: 600), at: t)
                if let remaining = estimate.remainingSeconds(at: t), frame < count - 1 || index < frames.count - 1 {
                    errors.append(abs(t + remaining - trueEnd))
                }
            }
        }
        XCTAssertEqual(estimate.framesPerSecond, rate, accuracy: 0.5)
        XCTAssertEqual(estimate.framesDone, estimate.framesExpected)
        XCTAssertGreaterThan(errors.first!, 1, "the typical rate misjudges a 20 frames/s run at first")
        XCTAssertLessThan(errors.suffix(errors.count / 2).max()!, 0.25, "the second half predicts the end within a quarter second")
        XCTAssertEqual(estimate.phase, .finishing)
    }

    func testCountsDownBetweenEvents() throws {
        var estimate = GenerationEstimate(lines: lines)
        estimate.apply(.chunkStarted(index: 0, count: 2, text: lines[0]), at: 1)
        let atStart = try XCTUnwrap(estimate.remainingSeconds(at: 1))
        XCTAssertEqual(try XCTUnwrap(estimate.remainingSeconds(at: 1.5)), atStart - 0.5, accuracy: 1e-9)
    }

    func testSilenceTailOfTheLastChunkAndRetriesStopTheEstimate() {
        var estimate = GenerationEstimate(lines: lines)
        estimate.apply(.chunkStarted(index: 0, count: 2, text: lines[0]), at: 0)
        estimate.apply(.frame(index: 3, maxFrames: 600), at: 1)
        estimate.apply(.retry(chunkIndex: 0, attempt: 1), at: 2)
        XCTAssertEqual(estimate.phase, .retrying)
        XCTAssertEqual(estimate.label(at: 2), "retrying")
        XCTAssertNil(estimate.remainingSeconds(at: 2))
        estimate.apply(.chunkStarted(index: 1, count: 2, text: lines[1]), at: 3)
        XCTAssertEqual(estimate.phase, .estimating)
        estimate.apply(.frame(index: estimate.expectedFrames[1] - 1, maxFrames: 600), at: 4)
        XCTAssertEqual(estimate.phase, .finishing)
        XCTAssertEqual(estimate.label(at: 4), "finishing\u{2026}")
    }

    func testLabelRoundsToFiveSecondsAboveTwentyAndToOneBelow() {
        XCTAssertEqual(GenerationEstimate.describe(remaining: 12.4), "about 12 s left")
        XCTAssertEqual(GenerationEstimate.describe(remaining: 19.6), "about 20 s left")
        XCTAssertEqual(GenerationEstimate.describe(remaining: 23.4), "about 25 s left")
        XCTAssertEqual(GenerationEstimate.describe(remaining: 47.0), "about 45 s left")
        XCTAssertEqual(GenerationEstimate.describe(remaining: 0.2), "about 1 s left")
    }
}
