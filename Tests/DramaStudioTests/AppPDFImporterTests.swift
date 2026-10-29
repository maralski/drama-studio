// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import PDFKit
import XCTest
@testable import DramaStudio

final class AppPDFImporterTests: XCTestCase {
    func testOnePagePerSegmentWithShortPagesSkipped() throws {
        let url = try temporaryDirectory().appending(path: "Annual Report.pdf")
        PDFFixture.write(pages: [
            "The first page tells a long story about the company and its exam-\nple products.\nIt keeps going on a new line.",
            "12",
            "Second chapter.\n\nA new paragraph starts here after a blank line.",
        ], title: "Annual Report 2026", to: url)

        let document = try PDFImporter.importFile(at: url)
        XCTAssertEqual(document.title, "Annual Report 2026")
        XCTAssertEqual(document.source, .pdf)
        XCTAssertEqual(document.segments.map(\.label), ["Page 1", "Page 2", "Page 3"])
        XCTAssertEqual(document.segments.map(\.index), [1, 2, 3])
        XCTAssertEqual(document.segments.map(\.skip), [false, true, false])
        XCTAssertEqual(document.segments[0].text, "The first page tells a long story about the company and its example products. It keeps going on a new line.")
        XCTAssertEqual(document.segments[2].text, "Second chapter.\n\nA new paragraph starts here after a blank line.")
    }

    func testTitleFallsBackToFileNameAndUnreadableFileThrows() throws {
        let dir = try temporaryDirectory()
        let url = dir.appending(path: "Untitled Scan.pdf")
        PDFFixture.write(pages: ["Enough words on this page to be narrated aloud."], title: nil, to: url)
        XCTAssertEqual(try PDFImporter.importFile(at: url).title, "Untitled Scan")

        let bogus = dir.appending(path: "bogus.pdf")
        try Data("not a pdf".utf8).write(to: bogus)
        XCTAssertThrowsError(try PDFImporter.importFile(at: bogus))
    }

    func testCleanPageTextJoinsHyphenationAndUnwrapsParagraphs() {
        XCTAssertEqual(PDFImporter.cleanPageText("an exam-\nple of\nwrapped text"), "an example of wrapped text")
        XCTAssertEqual(PDFImporter.cleanPageText("First para\nline two.\n\n  Second para\r\nline two."), "First para line two.\n\nSecond para line two.")
        XCTAssertEqual(PDFImporter.cleanPageText("Jean-\nPaul arrived"), "Jean-Paul arrived", "a capital after the break keeps the hyphen")
        XCTAssertEqual(PDFImporter.cleanPageText("soft\u{00AD}\nhyphen"), "softhyphen")
        XCTAssertEqual(PDFImporter.cleanPageText(" \n\n \n"), "")
    }

    func testLineGapsBecomeParagraphBreaks() {
        let line = { (text: String, top: CGFloat) in PDFImporter.Line(text: text, bounds: CGRect(x: 72, y: top - 16, width: 300, height: 16)) }
        let lines = [line("Heading", 700), line("First line of body", 670), line("second line.", 652), line("Next paragraph.", 620)]
        XCTAssertEqual(PDFImporter.joinLines(lines), "Heading\n\nFirst line of body\nsecond line.\n\nNext paragraph.")
        XCTAssertEqual(PDFImporter.joinLines([]), "")
    }
}
