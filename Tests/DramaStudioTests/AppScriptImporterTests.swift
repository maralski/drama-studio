import AppKit
import XCTest
@testable import DramaStudio

final class AppScriptImporterTests: XCTestCase {
    func testMarkdownBecomesPlainProse() {
        let markdown = """
        # Chapter One

        It was a **dark** and _stormy_ night, said [the narrator](https://example.com).

        > She paused. [sighs]

        - First item
        2. Second item with `code`
        ***
        | Name | Line |
        |------|------|
        | Ann | Hi |

        ```
        let x = 1
        ```
        Keep snake_case and <whisper>quiet</whisper> <span>but drop span</span>.
        """
        XCTAssertEqual(ScriptImporter.plainText(fromMarkdown: markdown), """
        Chapter One

        It was a dark and stormy night, said the narrator.

        She paused. [sighs]

        First item
        Second item with code
        Name, Line
        Ann, Hi

        let x = 1
        Keep snake_case and <whisper>quiet</whisper> but drop span.
        """)
    }

    func testSubRipDropsIndicesAndTimestampsAndJoinsCues() {
        let srt = "1\r\n00:00:01,000 --> 00:00:02,500\r\nHello there.\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\n<i>General</i> Kenobi,\r\nyou are bold.\r\n\r\n3\r\n00:00:05,000 --> 00:00:06,000\r\n1984\r\n"
        XCTAssertEqual(ScriptImporter.plainText(fromSubtitles: srt), "Hello there. General Kenobi, you are bold. 1984")
    }

    func testWebVTTDropsHeaderNotesIdentifiersAndVoiceSpans() {
        let vtt = """
        WEBVTT - Example

        NOTE This is a comment

        intro
        00:01.000 --> 00:02.000 align:start
        <v Roger>Welcome back.

        00:02.000 --> 00:03.000
        <v Roger>Welcome back.

        00:03.500 --> 00:05.000
        - Ready when you are.
        """
        XCTAssertEqual(ScriptImporter.plainText(fromSubtitles: vtt), "Welcome back. Ready when you are.")
    }

    @MainActor
    func testRichFormatsImportThroughTheirDocumentTypes() throws {
        let source = NSAttributedString(string: "Rich line one.\nRich line two.")
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        for format in [ImportFormat.rtf, .html, .docx] {
            let url = dir.appending(path: "script.\(format.extensions[0])")
            let data = try source.data(from: NSRange(location: 0, length: source.length), documentAttributes: [.documentType: format.documentType!])
            try data.write(to: url)
            XCTAssertEqual(try ScriptImporter.importFile(at: url), "Rich line one.\nRich line two.", "\(format)")
        }
        let rtfd = dir.appending(path: "script.rtfd")
        try source.fileWrapper(from: NSRange(location: 0, length: source.length), documentAttributes: [.documentType: NSAttributedString.DocumentType.rtfd])
            .write(to: rtfd, options: .atomic, originalContentsURL: nil)
        XCTAssertEqual(try ScriptImporter.importFile(at: rtfd), "Rich line one.\nRich line two.")
    }

    func testDocxUsesOfficeOpenXML() {
        XCTAssertEqual(ImportFormat(url: URL(filePath: "/x/Script.DOCX"))?.documentType, .officeOpenXML)
    }

    @MainActor
    func testUnsupportedFileIsAnInlineError() {
        XCTAssertThrowsError(try ScriptImporter.importFile(at: URL(filePath: "/tmp/clip.mp3"))) { error in
            XCTAssertEqual((error as? LocalizedError)?.errorDescription?.hasPrefix("Drama Studio can't import clip.mp3"), true)
        }
    }
}
