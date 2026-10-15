import XCTest
@testable import DramaEngine

final class FrameBudgetTests: XCTestCase {
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
