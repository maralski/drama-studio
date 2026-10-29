// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class SilenceStopTests: XCTestCase {
    private let n = ModelConstants.samplesPerFrame

    /// Frames of audio at the given per-frame RMS values, as a flat sample array.
    private func audio(rms: [Float]) -> [Float] {
        rms.flatMap { [Float](repeating: $0, count: n) }
    }

    func testTwoSecondsIsFortyFourFrames() {
        XCTAssertEqual(FrameGenerator.silenceFrames(for: 2.0), 44, "43 frames are 1.997 s")
        XCTAssertEqual(FrameGenerator.silenceFrames(for: 0), 0)
    }

    func testStopFrameIsTheFirstPrefixThatEndsInEnoughSilence() {
        let samples = audio(rms: [Float](repeating: 0.5, count: 100) + [Float](repeating: 0, count: 60))
        XCTAssertEqual(FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: 44), 144)
        XCTAssertEqual(FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: 60), 160)
        XCTAssertNil(FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: 61))
    }

    func testALoudFrameRestartsTheCount() {
        let rms = [Float](repeating: 0, count: 30) + [0.2] + [Float](repeating: 0, count: 44)
        XCTAssertEqual(FrameGenerator.silenceStopFrame(samples: audio(rms: rms), silenceFrames: 44), 75)
    }

    func testTheThresholdMatchesTheTrailingSilenceReading() {
        let rms = [Float](repeating: 0.3, count: 10) + [0.0099, 0.0101] + [Float](repeating: 0, count: 50)
        let samples = audio(rms: rms)
        let stop = FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: 44)
        XCTAssertEqual(stop, 12 + 44, "the 0.0101 frame is loud and the 0.0099 frame before it is silent")
        let k = stop!
        XCTAssertGreaterThanOrEqual(DramaSynthesizer.trailingSilenceSeconds(Array(samples.prefix(k * n))), 2.0)
        XCTAssertLessThan(DramaSynthesizer.trailingSilenceSeconds(Array(samples.prefix((k - 1) * n))), 2.0)
    }

    func testEmptyAudioHasNoStopFrame() {
        XCTAssertNil(FrameGenerator.silenceStopFrame(samples: [], silenceFrames: 44))
        XCTAssertNil(FrameGenerator.silenceStopFrame(samples: [Float](repeating: 0, count: n - 1), silenceFrames: 1))
    }
}
