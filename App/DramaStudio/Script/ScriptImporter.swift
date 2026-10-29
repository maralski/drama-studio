// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import UniformTypeIdentifiers

/// Every file the app opens, by extension. The script formats become plain
/// script text; the document formats open as a `NarrationDocument` read
/// section by section.
enum ImportFormat: CaseIterable {
    case plainText, markdown, rtf, rtfd, html, docx, subRip, webVTT
    case pptx, pdf, epub

    static let subRipType = UTType(importedAs: "io.github.maralski.DramaStudio.subrip", conformingTo: .plainText)

    init?(url: URL) {
        let ext = url.pathExtension.lowercased()
        guard let format = Self.allCases.first(where: { $0.extensions.contains(ext) }) else { return nil }
        self = format
    }

    var extensions: [String] {
        switch self {
        case .plainText: ["txt", "text"]
        case .markdown: ["md", "markdown"]
        case .rtf: ["rtf"]
        case .rtfd: ["rtfd"]
        case .html: ["html", "htm"]
        case .docx: ["docx"]
        case .subRip: ["srt"]
        case .webVTT: ["vtt"]
        case .pptx: ["pptx"]
        case .pdf: ["pdf"]
        case .epub: ["epub"]
        }
    }

    var contentType: UTType {
        switch self {
        case .plainText: .plainText
        case .markdown: UTType("net.daringfireball.markdown") ?? .plainText
        case .rtf: .rtf
        case .rtfd: .rtfd
        case .html: .html
        case .docx: UTType("org.openxmlformats.wordprocessingml.document") ?? .data
        case .subRip: Self.subRipType
        case .webVTT: UTType("org.w3.webvtt") ?? .plainText
        case .pptx: UTType("org.openxmlformats.presentationml.presentation") ?? .presentation
        case .pdf: .pdf
        case .epub: .epub
        }
    }

    /// Whether the file opens as a document of sections rather than as script text.
    var opensAsDocument: Bool {
        switch self {
        case .pptx, .pdf, .epub: true
        case .plainText, .markdown, .rtf, .rtfd, .html, .docx, .subRip, .webVTT: false
        }
    }

    var documentType: NSAttributedString.DocumentType? {
        switch self {
        case .rtf: .rtf
        case .rtfd: .rtfd
        case .html: .html
        case .docx: .officeOpenXML
        case .plainText, .markdown, .subRip, .webVTT, .pptx, .pdf, .epub: nil
        }
    }
}

enum ImportError: LocalizedError {
    case unsupported(String)
    case unreadable(String, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .unsupported(let name): "Drama Studio can't import \(name). Try a text, Markdown, RTF, HTML, Word, SRT, or WebVTT script, or a PowerPoint, PDF, or EPUB document."
        case .unreadable(let name, let error): "Couldn't read \(name): \(error.localizedDescription)"
        }
    }
}

enum ScriptImporter {
    @MainActor
    static func importFile(at url: URL) throws -> String {
        guard let format = ImportFormat(url: url), !format.opensAsDocument else { throw ImportError.unsupported(url.lastPathComponent) }
        do {
            return normalize(try text(of: url, as: format))
        } catch {
            throw ImportError.unreadable(url.lastPathComponent, underlying: error)
        }
    }

    @MainActor
    private static func text(of url: URL, as format: ImportFormat) throws -> String {
        if let documentType = format.documentType {
            return try NSAttributedString(url: url, options: [.documentType: documentType], documentAttributes: nil).string
        }
        let raw = try String(contentsOf: url, encoding: .utf8)
        switch format {
        case .markdown: return plainText(fromMarkdown: raw)
        case .subRip, .webVTT: return plainText(fromSubtitles: raw)
        default: return raw
        }
    }

    static func normalize(_ text: String) -> String {
        let unixLines = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        let trimmed = unixLines.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.replacingOccurrences(of: "\u{2028}", with: " ").trimmingCharacters(in: .whitespaces) }
            .joined(separator: "\n")
        return trimmed.replacingOccurrences(of: "\n{3,}", with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func plainText(fromMarkdown markdown: String) -> String {
        var inFence = false
        var lines: [String] = []
        for rawLine in markdown.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(rawLine)
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") || line.trimmingCharacters(in: .whitespaces).hasPrefix("~~~") {
                inFence.toggle()
                continue
            }
            if inFence { lines.append(line); continue }
            if line.range(of: #"^\s*([-*_]\s*){3,}$"#, options: .regularExpression) != nil { continue }
            if line.range(of: #"^\s*\|?\s*:?-{3,}"#, options: .regularExpression) != nil { continue }
            for pattern in [#"^\s{0,3}#{1,6}\s+"#, #"^\s*(>\s?)+"#, #"^\s*([-*+]|\d+[.)])\s+(\[[ xX]\]\s+)?"#] {
                line = line.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
            }
            line = line.replacingOccurrences(of: #"\s+#+\s*$"#, with: "", options: .regularExpression)
            if line.contains("|") {
                line = line.trimmingCharacters(in: CharacterSet(charactersIn: "| "))
                    .replacingOccurrences(of: #"\s*\|\s*"#, with: ", ", options: .regularExpression)
            }
            lines.append(inlinePlainText(line))
        }
        return normalize(lines.joined(separator: "\n"))
    }

    private static func inlinePlainText(_ line: String) -> String {
        let withoutImages = line.replacingOccurrences(of: #"!\[([^\]]*)\]\([^)]*\)"#, with: "$1", options: .regularExpression)
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        guard let parsed = try? AttributedString(markdown: withoutImages, options: options) else { return withoutImages }
        let keep = PerformanceCue.pairedTagNames.joined(separator: "|")
        return String(parsed.characters)
            .replacingOccurrences(of: "</?(?!(?:\(keep))>)[A-Za-z][^>]*>", with: "", options: .regularExpression)
    }

    static func plainText(fromSubtitles subtitles: String) -> String {
        let blocks = subtitles.replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n\n")
        var cues: [String] = []
        for block in blocks {
            let lines = block.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            guard let first = lines.first, !["WEBVTT", "NOTE", "STYLE", "REGION"].contains(where: { first.hasPrefix($0) }) else { continue }
            let isTiming = { (i: Int) in i < lines.count && lines[i].contains("-->") }
            let text = lines.indices
                .filter { !isTiming($0) && !isTiming($0 + 1) }
                .map { lines[$0] }
                .map {
                    $0.replacingOccurrences(of: #"<[^>]*>|\{\\[^}]*\}"#, with: "", options: .regularExpression)
                        .replacingOccurrences(of: #"^-\s*"#, with: "", options: .regularExpression)
                }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            guard !text.isEmpty, text != cues.last else { continue }
            cues.append(text)
        }
        return cues.joined(separator: " ")
    }
}
