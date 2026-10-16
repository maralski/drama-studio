import Foundation
import PDFKit

enum PDFImporter {
    /// Pages with less text than this are cover art, page numbers or blank.
    static let minimumCharacters = 20

    static func importFile(at url: URL) throws -> NarrationDocument {
        guard let pdf = PDFDocument(url: url) else { throw DocumentImportError.malformed("\(url.lastPathComponent) is not a readable PDF") }
        if pdf.isLocked { throw DocumentImportError.malformed("\(url.lastPathComponent) is password protected") }
        return importDocument(pdf, fallbackTitle: url.deletingPathExtension().lastPathComponent)
    }

    static func importDocument(_ pdf: PDFDocument, fallbackTitle: String) -> NarrationDocument {
        let segments = (0..<pdf.pageCount).map { page in
            let text = cleanPageText(pdf.page(at: page).map(lines(of:)) ?? "")
            return NarrationDocument.Segment(index: page + 1, label: "Page \(page + 1)", text: text, skip: text.count < minimumCharacters)
        }
        let title = (pdf.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return NarrationDocument(title: title?.isEmpty == false ? title! : fallbackTitle, source: .pdf, segments: segments)
    }

    struct Line: Equatable {
        var text: String
        var bounds: CGRect
    }

    /// `PDFPage.string` drops blank lines, so paragraph breaks come back from
    /// the vertical gaps between lines instead.
    static func lines(of page: PDFPage) -> String {
        guard let selections = page.selection(for: page.bounds(for: .mediaBox))?.selectionsByLine(), !selections.isEmpty else { return page.string ?? "" }
        return joinLines(selections.map { Line(text: $0.string ?? "", bounds: $0.bounds(for: page)) })
    }

    static func joinLines(_ lines: [Line]) -> String {
        let heights = lines.map(\.bounds.height).filter { $0 > 0 }.sorted()
        let typical = heights.isEmpty ? 0 : heights[heights.count / 2]
        var out = ""
        for (offset, line) in lines.enumerated() {
            if offset > 0 {
                let gap = lines[offset - 1].bounds.minY - line.bounds.maxY
                out += typical > 0 && gap > typical * 0.6 ? "\n\n" : "\n"
            }
            out += line.text.trimmingCharacters(in: .newlines)
        }
        return out
    }

    /// Rejoins words hyphenated across lines and unwraps lines inside a
    /// paragraph; blank lines stay paragraph breaks.
    static func cleanPageText(_ raw: String) -> String {
        let unix = raw.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\u{00AD}\n", with: "")
            .replacingOccurrences(of: #"(\p{L})-[ \t]*\n[ \t]*(\p{Ll})"#, with: "$1$2", options: .regularExpression)
            .replacingOccurrences(of: #"(\p{L})-[ \t]*\n[ \t]*(\p{Lu})"#, with: "$1-$2", options: .regularExpression)
        return unix.split(separator: #/\n[ \t]*\n\s*/#)
            .map { paragraph in
                paragraph.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression).trimmingCharacters(in: .whitespaces)
            }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
