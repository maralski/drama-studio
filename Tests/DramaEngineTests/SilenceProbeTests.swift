// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class SilenceProbeTests: XCTestCase {
    private let n = ModelConstants.samplesPerFrame

    private static func frames(_ count: Int) -> [Frame] {
        (0 ..< count).map { t in [Int32(t) + ModelConstants.semanticStart] + [Int32](repeating: Int32(t), count: ModelConstants.numCodebooks) }
    }

    /// A decode whose frame `t` is silent when `silent(t)`, which records every
    /// window it was asked for.
    private func decoder(silent: @escaping (Int) -> Bool, windows: @escaping (DecodeWindow) -> Void = { _ in }) -> SilenceProbe.Decode {
        { _, w in
            windows(w)
            return (w.keep ..< w.end).flatMap { t in [Float](repeating: silent(t) ? 0 : 0.5, count: ModelConstants.samplesPerFrame) }
        }
    }

    func testProbingWindowsKeepTheNewFramesAfterTheContext() {
        typealias W = DecodeWindow
        XCTAssertEqual(W.probing(0 ..< 48, shape: 64, context: 48), [W(start: 0, keep: 0, end: 48)])
        XCTAssertEqual(W.probing(48 ..< 64, shape: 64, context: 48), [W(start: 0, keep: 48, end: 64)])
        XCTAssertEqual(W.probing(160 ..< 176, shape: 64, context: 48), [W(start: 112, keep: 160, end: 176)])
        XCTAssertEqual(W.probing(160 ..< 200, shape: 64, context: 48), [W(start: 112, keep: 160, end: 176), W(start: 128, keep: 176, end: 192), W(start: 144, keep: 192, end: 200)],
                       "a probe that fell behind catches up in windows of the usual size")
        XCTAssertEqual(W.probing(0 ..< 0), [])
        XCTAssertEqual(W.probing(10 ..< 100, shape: 128, context: 64), [W(start: 0, keep: 10, end: 100)], "the chunk start has less context than asked")
        for w in W.probing(500 ..< 700, shape: DramaCodec.probeFrames, context: DramaCodec.probeContext) {
            XCTAssertLessThanOrEqual(w.end - w.start, DramaCodec.probeFrames)
            XCTAssertEqual(w.keep - w.start, DramaCodec.probeContext)
        }
    }

    func testTheReadingGrowsWithTheSilentTailAndResetsOnSpeech() {
        let all = Self.frames(200)
        var asked: [DecodeWindow] = []
        let probe = SilenceProbe(shape: 64, context: 48, decoder(silent: { $0 >= 100 && $0 != 130 }, windows: { asked.append($0) }))
        XCTAssertEqual(probe.latest, 0)
        var landed = 0
        while landed < all.count {
            landed = min(all.count, landed + 16)
            probe.measure(Array(all.prefix(landed)))
            let silentTail = landed <= 100 ? 0 : landed <= 131 ? (landed > 130 ? landed - 131 : landed - 100) : landed - 131
            XCTAssertEqual(probe.latest, Double(silentTail * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9, "after \(landed) frames")
            XCTAssertEqual(probe.classifiedFrames, landed)
        }
        XCTAssertTrue(asked.allSatisfy { $0.end - $0.start <= 64 && $0.keep - $0.start <= 48 }, "every call fits the probe shape")
        XCTAssertEqual(asked.map(\.kept.count).reduce(0, +), all.count, "every frame is classified once")
    }

    func testAnUnevenCadenceClassifiesEveryFrameOnce() {
        let all = Self.frames(300)
        var kept = 0
        let probe = SilenceProbe(shape: 64, context: 48, decoder(silent: { $0 >= 250 }, windows: { kept += $0.kept.count }))
        for landed in [5, 48, 49, 120, 121, 299, 300] { probe.measure(Array(all.prefix(landed))) }
        probe.measure(all)
        XCTAssertEqual(kept, 300)
        XCTAssertEqual(probe.latest, Double(50 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9)
    }

    func testAFailedDecodeLeavesTheReadingWhereItWas() {
        let all = Self.frames(128)
        var fail = false
        struct Broken: Error {}
        let good = decoder(silent: { _ in true })
        let probe = SilenceProbe(shape: 64, context: 48) { codes, w in
            if fail { throw Broken() }
            return try good(codes, w)
        }
        probe.measure(Array(all.prefix(64)))
        let before = probe.latest
        XCTAssertEqual(before, Double(64 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9)
        fail = true
        probe.measure(all)
        XCTAssertEqual(probe.latest, before)
        XCTAssertEqual(probe.classifiedFrames, 64, "the failed frames are read again next time")
        fail = false
        probe.measure(all)
        XCTAssertEqual(probe.classifiedFrames, 128)
        XCTAssertEqual(probe.latest, Double(128 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9)

        // A reading that would skip ahead and then fails keeps the earlier tail count and position too.
        let behind = SilenceProbe(shape: 64, context: 48, silenceFrames: 44) { codes, w in
            if fail { throw Broken() }
            return try good(codes, w)
        }
        behind.measure(Array(all.prefix(64)))
        let saturated = Double(44 * n) / Double(ModelConstants.sampleRate)
        XCTAssertEqual(behind.latest, saturated, accuracy: 1e-9, "64 frames at once read as the newest 44")
        fail = true
        behind.measure(all)
        XCTAssertEqual(behind.classifiedFrames, 64)
        XCTAssertEqual(behind.latest, saturated, accuracy: 1e-9)
    }

    func testTheStopFiresWithinOneCadenceOfTheIdealFrame() {
        // Speech for 90 frames, then silence: the audio holds 2 s of silence at frame 134, and a reading every 16 frames sees it at 144.
        let all = Self.frames(200)
        let probe = SilenceProbe(shape: 64, context: 48, decoder(silent: { $0 >= 90 }))
        let samples = (0 ..< 200).flatMap { t in [Float](repeating: t >= 90 ? 0 : 0.5, count: n) }
        let ideal = FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: FrameGenerator.silenceFrames(for: 2.0))
        XCTAssertEqual(ideal, 134)
        var fired: Int?
        var landed = 0
        while landed < all.count, fired == nil {
            landed += SilenceProbePolicy.shipped.everyFrames
            probe.measure(Array(all.prefix(landed)))
            if probe.latest >= 2.0 { fired = landed }
        }
        XCTAssertEqual(fired, 144)
    }

    func testAReadingFarBehindClassifiesOnlyTheFramesTheStopNeeds() {
        // 200 frames land at once. Only the newest 44 decide whether the tail holds 2 s of silence, so only they are read.
        let all = Self.frames(200)
        var asked: [DecodeWindow] = []
        let silent = SilenceProbe(shape: 64, context: 48, silenceFrames: 44, decoder(silent: { $0 >= 100 }, windows: { asked.append($0) }))
        silent.measure(all)
        XCTAssertEqual(asked.first?.keep, 156)
        XCTAssertEqual(asked.map(\.kept.count).reduce(0, +), 44)
        XCTAssertEqual(silent.classifiedFrames, 200)
        XCTAssertEqual(silent.latest, Double(44 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9, "the reading saturates at the silence the stop needs")
        XCTAssertGreaterThanOrEqual(silent.latest, 2.0)

        let spoke = SilenceProbe(shape: 64, context: 48, silenceFrames: 44, decoder(silent: { $0 != 190 }))
        spoke.measure(all)
        XCTAssertEqual(spoke.latest, Double(9 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9, "a loud frame among the newest 44 is seen")

        // A reading on time never skips: 16 new frames are fewer than 44.
        let onTime = SilenceProbe(shape: 64, context: 48, silenceFrames: 44, decoder(silent: { $0 >= 10 }))
        var landed = 0
        while landed < 64 { landed += 16; onTime.measure(Array(all.prefix(landed))) }
        XCTAssertEqual(onTime.latest, Double(54 * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9, "every frame counted, past the saturation point")
    }

    func testAScheduleFromTheStartReadsEverySixteenFramesOnceTheSilenceFits() {
        let s = SilenceProbePolicy(everyFrames: 16, afterExpectedEnd: false).schedule(expectedSeconds: 10, silenceFrames: 44)
        XCTAssertEqual(s, SilenceProbeSchedule(everyFrames: 16, firstFrame: 45))
        XCTAssertEqual((0 ... 100).filter(s.reads(at:)), [48, 64, 80, 96])
        XCTAssertFalse(SilenceProbePolicy(everyFrames: 0, afterExpectedEnd: false).schedule(expectedSeconds: 10, silenceFrames: 44).reads(at: 48), "a cadence of 0 never reads")
    }

    func testTheShippedScheduleWaitsForExpectedSpeechThenSilence() {
        let policy = SilenceProbePolicy.shipped
        XCTAssertEqual(policy, SilenceProbePolicy(everyFrames: 16, afterExpectedEnd: true))
        // A pause within the expected ten seconds of speech must not end the chunk.
        let long = policy.schedule(expectedSeconds: 10, silenceFrames: 44)
        XCTAssertEqual(long.firstFrame, 259)
        XCTAssertEqual((0 ... 300).filter(long.reads(at:)), [272, 288])
        let short = policy.schedule(expectedSeconds: 1.5, silenceFrames: 44)
        XCTAssertEqual(short, SilenceProbeSchedule(everyFrames: 16, firstFrame: 76))
        XCTAssertEqual((0 ... 100).filter(short.reads(at:)), [80, 96])
        XCTAssertFalse(SilenceProbePolicy(everyFrames: 32, afterExpectedEnd: true).schedule(expectedSeconds: 10, silenceFrames: 44).reads(at: 256))
        XCTAssertTrue(SilenceProbePolicy(everyFrames: 32, afterExpectedEnd: true).schedule(expectedSeconds: 10, silenceFrames: 44).reads(at: 288))
    }

    func testExpectedEndCheckpointsReadTheBoundedTailThenEachCadence() {
        let schedule = SilenceProbePolicy.shipped.schedule(expectedSeconds: 6, silenceFrames: 44)
        let checkpoints = (1 ... 300).filter(schedule.reads(at:))
        XCTAssertEqual(Array(checkpoints.prefix(3)), [176, 192, 208])

        var asked: [DecodeWindow] = []
        let probe = SilenceProbe(shape: 64, context: 48, silenceFrames: 44,
                                 decoder(silent: { $0 >= 160 }, windows: { asked.append($0) }))
        let all = Self.frames(300)
        var windowsPerReading: [Int] = []
        for (checkpoint, silentFrames) in zip(checkpoints, [16, 32, 48]) {
            let before = asked.count
            probe.measure(Array(all.prefix(checkpoint)))
            windowsPerReading.append(asked.count - before)
            XCTAssertEqual(probe.classifiedFrames, checkpoint)
            XCTAssertEqual(probe.latest, Double(silentFrames * n) / Double(ModelConstants.sampleRate), accuracy: 1e-9)
            XCTAssertEqual(probe.latest >= 2.0, checkpoint == 208)
        }
        XCTAssertEqual(windowsPerReading, [3, 1, 1])
        XCTAssertEqual(asked.map(\.kept), [132 ..< 148, 148 ..< 164, 164 ..< 176, 176 ..< 192, 192 ..< 208])
    }

    func testTheProbeShapeHoldsTheCadenceAndContext() {
        XCTAssertEqual(DramaCodec.probeContext + SilenceProbePolicy.shipped.everyFrames, DramaCodec.probeFrames,
                       "a reading on time classifies exactly one cadence of new frames after the full context")
        XCTAssertEqual(DramaCodec.modelNames, ["DramaCodecEncoder", "DramaCodecDecoder", "DramaCodecProbe"])
    }
}
