import XCTest
@testable import DramaEngine

final class CueGuardTests: XCTestCase {
    private func cycle(period: Int, frames: Int) -> [Int32] {
        (0..<frames).map { Int32(100 + $0 % period) }
    }

    func testLoopGuardShipsOnIncludingQuickPreview() {
        XCTAssertTrue(GenerationConfig().stopOnLoop)
        XCTAssertTrue(GenerationConfig.make(expressiveness: 0.5, quickPreview: true).stopOnLoop)
    }

    func testAPeriodSeventeenCycleIsALoopOnceWindowAndLagFit() {
        XCTAssertTrue(MainTokenLoop.cycles(cycle(period: 17, frames: 70)))
        XCTAssertFalse(MainTokenLoop.cycles(cycle(period: 17, frames: 60)), "48 + 17 = 65 frames are needed before any lag can be checked")
    }

    func testRandomTokensAreNotALoop() {
        var g = SeededGenerator(seed: 42)
        let tokens = (0..<200).map { _ in Int32(g.next() % 4096) }
        XCTAssertFalse(MainTokenLoop.cycles(tokens))
    }

    func testAConstantTokenIsALoopFromSixtyFiveFrames() {
        // A silent tail repeats one code. The silence stop may miss it, so the loop guard is the backstop.
        XCTAssertFalse(MainTokenLoop.cycles(Array(repeating: 7, count: 64)))
        XCTAssertTrue(MainTokenLoop.cycles(Array(repeating: 7, count: 65)))
    }

    func testNinetyPercentOfTheWindowMustRepeat() {
        func perturbed(_ count: Int) -> [Int32] {
            var tokens = cycle(period: 17, frames: 70)
            for k in 0..<count { tokens[69 - k] = Int32(5000 + k) }
            return tokens
        }
        XCTAssertTrue(MainTokenLoop.cycles(perturbed(4)), "44 of 48 = 0.917")
        XCTAssertFalse(MainTokenLoop.cycles(perturbed(6)), "42 of 48 = 0.875")
    }

    func testRetryRuleFollowsTheCueStudyEvidence() {
        let expected = 3.0
        let config = GenerationConfig()
        let cases: [(FrameGenerator.ChunkEnd, Double, Bool, String)] = [
            (.eos, 0.62, false, "reached eos and spoke past minSpokenRatio"),
            (.eos, 0.5, true, "reached eos but spoke under minSpokenRatio"),
            (.frameCap, 0.9, true, "cut at the cap with words missing"),
            (.frameCap, 1.19, false, "a trailing cue that never reaches eos, every word spoken"),
            (.silence, 0.7, true, "the silence stop ended it early"),
            (.silence, 1.5, false, "the silence stop after every word"),
            (.loopGuard, 2.9, true, "a laughter loop speaks long and still lost the line"),
            (.cancelled, 0.1, false, "the caller stopped generation"),
        ]
        for (end, ratio, retry, why) in cases {
            XCTAssertEqual(FrameGenerator.shouldRetry(end: end, spokenSeconds: ratio * expected, expectedSeconds: expected, config: config), retry, why)
        }
    }

    func testRetriesOffNeverRetryEvenALoop() {
        let preview = GenerationConfig.make(expressiveness: 0.5, quickPreview: true)
        XCTAssertFalse(FrameGenerator.shouldRetry(end: .loopGuard, spokenSeconds: 9, expectedSeconds: 3, config: preview))
        XCTAssertFalse(FrameGenerator.shouldRetry(end: .frameCap, spokenSeconds: 0.3, expectedSeconds: 3, config: preview))
    }

    private func attempt(_ end: FrameGenerator.ChunkEnd, spoken: Double) -> FrameGenerator.ChunkResult {
        FrameGenerator.ChunkResult(index: 0, count: 1, text: "", frames: [], promptRows: 0, end: end, frameBudget: 0, prefillSeconds: 0,
                                   decodeSeconds: 0, gpuSeconds: 0, spokenSeconds: spoken, attempts: 1, samples: [], codecSeconds: 0)
    }

    func testAnAttemptThatReachedEOSBeatsALongerOneThatDidNot() {
        XCTAssertTrue(FrameGenerator.better(attempt(.eos, spoken: 1), than: attempt(.frameCap, spoken: 5)))
        XCTAssertFalse(FrameGenerator.better(attempt(.loopGuard, spoken: 5), than: attempt(.eos, spoken: 1)))
    }

    func testBetweenTwoEOSAttemptsTheLongerWins() {
        XCTAssertTrue(FrameGenerator.better(attempt(.eos, spoken: 2.5), than: attempt(.eos, spoken: 2)))
        XCTAssertFalse(FrameGenerator.better(attempt(.eos, spoken: 2), than: attempt(.eos, spoken: 2.5)))
    }
}
