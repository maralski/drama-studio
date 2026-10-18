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
            landed += FrameGenerator.silenceProbeEveryFrames
            probe.measure(Array(all.prefix(landed)))
            if probe.latest >= 2.0 { fired = landed }
        }
        XCTAssertEqual(fired, 144)
    }

    func testTheProbeShapeHoldsTheCadenceAndContext() {
        XCTAssertEqual(DramaCodec.probeContext + FrameGenerator.silenceProbeEveryFrames, DramaCodec.probeFrames,
                       "a reading on time classifies exactly one cadence of new frames after the full context")
        XCTAssertEqual(DramaCodec.modelNames, ["DramaCodecEncoder", "DramaCodecDecoder", "DramaCodecProbe"])
    }
}
