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
}
