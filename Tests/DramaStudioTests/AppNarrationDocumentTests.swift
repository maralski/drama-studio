import DramaEngine
import Foundation
import XCTest
@testable import DramaStudio

final class AppNarrationDocumentTests: XCTestCase {
    func testImportDispatchesOnExtension() throws {
        let dir = try temporaryDirectory()
        let deck = dir.appending(path: "deck.PPTX")
        try PPTXFixture.package(PPTXFixture.standard).write(to: deck)
        let book = dir.appending(path: "book.epub")
        try EPUBFixture.package().write(to: book)
        let notes = dir.appending(path: "Read me.txt")
        try "First line.\r\n\r\n\r\nSecond line.".write(to: notes, atomically: true, encoding: .utf8)

        XCTAssertEqual(try NarrationDocument.importFile(at: deck).source, .pptx)
        XCTAssertEqual(try NarrationDocument.importFile(at: book).source, .epub)
        let text = try NarrationDocument.importFile(at: notes)
        XCTAssertEqual(text.source, .text)
        XCTAssertEqual(text.segments.map(\.label), ["Read me"])
        XCTAssertEqual(text.segments.map(\.text), ["First line.\n\nSecond line."])
        XCTAssertThrowsError(try NarrationDocument.importFile(at: dir.appending(path: "song.mp3"))) {
            XCTAssertEqual($0 as? DocumentImportError, .unsupported("song.mp3"))
        }
    }

    func testTakeDerivesFramesAndSecondsAndRoundTripsThroughJSON() throws {
        let take = NarrationTake(segmentID: UUID(), sampleRange: 44_100..<132_300, sentences: [.init(text: "One.", frames: 20), .init(text: "Two.", frames: 23)])
        XCTAssertEqual(take.frames, 43)
        XCTAssertEqual(take.seconds, 2.0)

        let document = NarrationDocument(title: "T", source: .pdf, segments: [.init(index: 1, label: "Page 1", text: "Hi", speakerTag: 2, skip: true)])
        XCTAssertEqual(try JSONDecoder().decode(NarrationTake.self, from: JSONEncoder().encode(take)), take)
        XCTAssertEqual(try JSONDecoder().decode(NarrationDocument.self, from: JSONEncoder().encode(document)), document)
    }

    func testTakeCarriesItsCodesAndDecodesWithoutThem() throws {
        let id = UUID()
        let codes = Take(text: "One. Two.", codes: Array(repeating: [1, 2, 3], count: 10))
        let take = NarrationTake(segmentID: id, sampleRange: 0..<6144, sentences: [.init(text: "One.", frames: 3)], take: codes)
        XCTAssertEqual(try JSONDecoder().decode(NarrationTake.self, from: JSONEncoder().encode(take)).take, codes)

        let older = """
        {"segmentID":"\(id.uuidString)","sampleRange":[0,6144],"sentences":[{"text":"One.","frames":3}]}
        """
        let decoded = try JSONDecoder().decode(NarrationTake.self, from: Data(older.utf8))
        XCTAssertEqual(decoded, NarrationTake(segmentID: id, sampleRange: 0..<6144, sentences: [.init(text: "One.", frames: 3)]))
        XCTAssertNil(decoded.take)
    }
}
