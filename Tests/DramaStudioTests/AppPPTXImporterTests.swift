import Foundation
import XCTest
@testable import DramaStudio

final class AppPPTXImporterTests: XCTestCase {
    func testNotesInPresentationOrderWithFallbackForSlideWithoutNotes() throws {
        for method in [ZipArchive.Method.stored, .deflated] {
            let archive = try ZipArchive(data: PPTXFixture.package(PPTXFixture.standard, method: method))
            let document = try PPTXImporter.importPackage(archive, fallbackTitle: "deck")

            XCTAssertEqual(document.title, "Quarterly Review")
            XCTAssertEqual(document.source, .pptx)
            XCTAssertEqual(document.segments.map(\.label), ["Slide 1: Welcome", "Slide 2: Quarterly results", "Slide 3: Questions (no notes)"])
            XCTAssertEqual(document.segments.map(\.index), [1, 2, 3])
            XCTAssertEqual(document.segments.map(\.text), [
                "Good morning, everyone.\nToday we cover the numbers.",
                "Revenue grew twelve percent.",
                "Questions\n\nAsk anything\nEmail us",
            ])
            XCTAssertEqual(document.segments.map(\.skip), [false, false, false])
            XCTAssertEqual(document.segments.map(\.speakerTag), [nil, nil, nil])
        }
    }

    func testSlideNumberFooterAndThumbnailPlaceholdersNeverReachTheText() throws {
        let document = try PPTXImporter.importPackage(ZipArchive(data: PPTXFixture.package(PPTXFixture.standard)), fallbackTitle: "deck")
        for segment in document.segments {
            XCTAssertFalse(segment.text.contains("Confidential"), segment.label)
            XCTAssertNil(segment.text.rangeOfCharacter(from: .decimalDigits), "slide-number field leaked into \(segment.label): \(segment.text)")
        }
    }

    func testBlankNotesFallBackAndEmptySlidesAreSkipped() throws {
        let slides = [
            PPTXFixture.Slide(file: 1, title: "Agenda", body: ["Intro", "Plan"], notes: [["   "], [""]]),
            PPTXFixture.Slide(file: 2, title: nil, body: [], notes: nil),
            PPTXFixture.Slide(file: 3, title: nil, body: ["Just a body"], notes: nil),
        ]
        let document = try PPTXImporter.importPackage(ZipArchive(data: PPTXFixture.package(slides)), fallbackTitle: "deck")
        XCTAssertEqual(document.segments.map(\.label), ["Slide 1: Agenda (no notes)", "Slide 2 (no notes)", "Slide 3 (no notes)"])
        XCTAssertEqual(document.segments.map(\.text), ["Agenda\n\nIntro\nPlan", "", "Just a body"])
        XCTAssertEqual(document.segments.map(\.skip), [false, true, false])
    }

    func testTitleFallsBackToFileNameAndMissingPresentationThrows() throws {
        let dir = try temporaryDirectory()
        let url = dir.appending(path: "Board Update.pptx")
        try PPTXFixture.package(PPTXFixture.standard, title: nil).write(to: url)
        XCTAssertEqual(try PPTXImporter.importFile(at: url).title, "Board Update")

        var zip = ZipWriter()
        zip.add("hello.txt", "not a deck")
        XCTAssertThrowsError(try PPTXImporter.importPackage(ZipArchive(data: zip.finish()), fallbackTitle: "x")) {
            XCTAssertEqual($0 as? DocumentImportError, .malformed("no ppt/presentation.xml"))
        }
    }

    /// Set DRAMA_PPTX_SAMPLE to a real .pptx to check the importer against PowerPoint output.
    func testRealDeckFromEnvironment() throws {
        guard let path = ProcessInfo.processInfo.environment["DRAMA_PPTX_SAMPLE"] else { throw XCTSkip("DRAMA_PPTX_SAMPLE is not set") }
        let document = try PPTXImporter.importFile(at: URL(fileURLWithPath: path))
        print("PPTX sample \"\(document.title)\": \(document.segments.count) segments")
        for segment in document.segments {
            print("[\(segment.label)] skip=\(segment.skip)\n\(segment.text)\n---")
        }
        XCTAssertFalse(document.segments.isEmpty)
        XCTAssertTrue(document.segments.contains { !$0.skip && !$0.text.isEmpty })
    }
}
