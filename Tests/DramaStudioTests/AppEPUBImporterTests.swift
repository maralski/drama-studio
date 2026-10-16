import Foundation
import XCTest
@testable import DramaStudio

final class AppEPUBImporterTests: XCTestCase {
    func testChaptersInSpineOrderWithCoverAndNonLinearItemsLeftOut() throws {
        for toc in [EPUBFixture.TableOfContents.nav, .ncx] {
            let document = try EPUBImporter.importPackage(ZipArchive(data: EPUBFixture.package(toc: toc)), fallbackTitle: "book")
            XCTAssertEqual(document.title, "The Long Walk")
            XCTAssertEqual(document.source, .epub)
            XCTAssertEqual(document.segments.map(\.label), ["Chapter 1: The Beginning", "Chapter 2: Into the Woods"], "\(toc)")
            XCTAssertEqual(document.segments.map(\.index), [1, 2])
            XCTAssertEqual(document.segments[0].text, "The Beginning\n\n\(EPUBFixture.chapterOneBody)\n\nShe said nothing.\n\nThen \u{201C}Hello.\u{201D}")
            XCTAssertEqual(document.segments[1].text, EPUBFixture.chapterTwoBody)
        }
    }

    func testMissingContainerThrowsAndFileNameIsFallbackTitle() throws {
        var zip = ZipWriter()
        zip.add("mimetype", "application/epub+zip")
        XCTAssertThrowsError(try EPUBImporter.importPackage(ZipArchive(data: zip.finish()), fallbackTitle: "x")) {
            XCTAssertEqual($0 as? ZipError, .missingEntry("META-INF/container.xml"))
        }

        let url = try temporaryDirectory().appending(path: "Walk.epub")
        try EPUBFixture.package().write(to: url)
        XCTAssertEqual(try EPUBImporter.importFile(at: url).segments.count, 2)
    }

    func testXHTMLTextSkipsHeadAndScriptsAndTakesFirstHeading() {
        let page = EPUBFixture.page(title: "Ignored", body: "<div><p>Intro &amp; more</p><h2>Part <em>One</em></h2><h1>Later</h1><ul><li>a</li><li>b</li></ul><style>x{}</style></div>")
        let result = XHTMLText.extract(Data(page.utf8))
        XCTAssertEqual(result.heading, "Part One")
        XCTAssertEqual(result.text, "Intro & more\n\nPart One\n\nLater\n\na\n\nb")
    }

    func testMalformedXHTMLFallsBackToTagStripping() {
        let result = XHTMLText.extract(Data("<html><body><p>Unclosed paragraph<p>Second one</body>".utf8))
        XCTAssertEqual(result.text, "Unclosed paragraph\n\nSecond one")
        XCTAssertNil(result.heading)
    }
}
