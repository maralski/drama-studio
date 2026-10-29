// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

enum EPUBImporter {
    static let minimumCharacters = 200

    static func importFile(at url: URL) throws -> NarrationDocument {
        try importPackage(ZipArchive(url: url), fallbackTitle: url.deletingPathExtension().lastPathComponent)
    }

    static func importPackage(_ archive: ZipArchive, fallbackTitle: String) throws -> NarrationDocument {
        guard let opfPath = try archive.xml("META-INF/container.xml").descendants(named: "rootfile").first?.value(of: "full-path") else {
            throw DocumentImportError.malformed("META-INF/container.xml names no package document")
        }
        let opf = try archive.xml(opfPath)
        let items = Dictionary(
            (opf.child(named: "manifest")?.children(named: "item") ?? []).compactMap { item -> (String, ManifestItem)? in
                guard let id = item.value(of: "id"), let href = item.value(of: "href") else { return nil }
                return (id, ManifestItem(path: ZipArchive.resolve(href, relativeTo: opfPath), mediaType: item.value(of: "media-type") ?? "", properties: item.value(of: "properties") ?? ""))
            },
            uniquingKeysWith: { first, _ in first }
        )
        let spine = opf.child(named: "spine")
        let navTitles = try tableOfContents(items: items, ncxID: spine?.value(of: "toc"), in: archive)

        var segments: [NarrationDocument.Segment] = []
        for itemref in spine?.children(named: "itemref") ?? [] where itemref.value(of: "linear") != "no" {
            guard let item = itemref.value(of: "idref").flatMap({ items[$0] }), archive.contains(item.path) else { continue }
            let chapter = XHTMLText.extract(try archive.data(for: item.path))
            guard chapter.text.count >= minimumCharacters else { continue }
            let number = segments.count + 1
            let title = chapter.heading ?? navTitles[item.path]
            segments.append(.init(index: number, label: title.map { "Chapter \(number): \($0)" } ?? "Chapter \(number)", text: chapter.text))
        }
        guard !segments.isEmpty else { throw DocumentImportError.empty(fallbackTitle) }

        let title = opf.child(named: "metadata")?.child(named: "title")?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
        return NarrationDocument(title: title?.isEmpty == false ? title! : fallbackTitle, source: .epub, segments: segments)
    }

    private struct ManifestItem {
        var path: String
        var mediaType: String
        var properties: String
    }

    private static func tableOfContents(items: [String: ManifestItem], ncxID: String?, in archive: ZipArchive) throws -> [String: String] {
        var titles: [String: String] = [:]
        func record(_ title: String?, _ href: String?, base: String) {
            guard let title = title?.trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty, let href else { return }
            let path = ZipArchive.resolve(href, relativeTo: base)
            if titles[path] == nil { titles[path] = title.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression) }
        }
        if let nav = items.values.first(where: { $0.properties.split(separator: " ").contains("nav") }), archive.contains(nav.path) {
            let root = try archive.xml(nav.path)
            let toc = root.descendants(named: "nav").first { $0.prefixedValue(of: "type") == "toc" } ?? root
            for link in toc.descendants(named: "a") { record(link.stringValue, link.value(of: "href"), base: nav.path) }
        }
        let ncx = ncxID.flatMap { items[$0] } ?? items.values.first { $0.mediaType == "application/x-dtbncx+xml" }
        if let ncx, archive.contains(ncx.path) {
            for point in try archive.xml(ncx.path).descendants(named: "navPoint") {
                record(point.child(named: "navLabel")?.child(named: "text")?.stringValue, point.child(named: "content")?.value(of: "src"), base: ncx.path)
            }
        }
        return titles
    }
}

/// XMLParser rather than NSAttributedString(html:), which needs WebKit on the main thread.
final class XHTMLText: NSObject, XMLParserDelegate {
    struct Result: Equatable {
        var text: String
        var heading: String?
    }

    private static let blocks: Set<String> = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "blockquote", "section", "article", "tr", "dt", "dd", "pre", "figcaption", "hr", "aside", "header", "footer"]
    private static let silent: Set<String> = ["head", "script", "style", "svg", "math"]
    private static let xmlEntities: Set<String> = ["amp", "lt", "gt", "quot", "apos"]
    private static let htmlEntities: [String: String] = [
        "nbsp": "&#160;", "mdash": "&#8212;", "ndash": "&#8211;", "hellip": "&#8230;", "lsquo": "&#8216;", "rsquo": "&#8217;",
        "ldquo": "&#8220;", "rdquo": "&#8221;", "copy": "&#169;", "eacute": "&#233;", "shy": "", "thinsp": "&#8201;",
    ]

    private var paragraphs: [String] = []
    private var current = ""
    private var heading: String?
    private var headingBuffer: String?
    private var silentDepth = 0

    static func extract(_ data: Data) -> Result {
        let source = String(decoding: data, as: UTF8.self)
        let parser = XMLParser(data: Data(declaringHTMLEntities(in: source).utf8))
        parser.shouldResolveExternalEntities = false
        let walker = XHTMLText()
        parser.delegate = walker
        guard parser.parse() else { return fallback(source) }
        walker.flush()
        return Result(text: ScriptImporter.normalize(walker.paragraphs.joined(separator: "\n\n")), heading: walker.heading)
    }

    private static func declaringHTMLEntities(in source: String) -> String {
        let pattern = #/&([A-Za-z][A-Za-z0-9]*);/#
        return source.replacing(pattern) { match in
            let name = String(match.1)
            if xmlEntities.contains(name) { return String(match.0) }
            return htmlEntities[name] ?? ""
        }
    }

    private static func fallback(_ source: String) -> Result {
        let stripped = source
            .replacingOccurrences(of: #"(?is)<(head|script|style)\b.*?</\1>"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"(?i)</?(p|div|h[1-6]|li|br|blockquote|section|tr)\b[^>]*>"#, with: "\n\n", options: .regularExpression)
            .replacingOccurrences(of: #"<[^>]*>"#, with: "", options: .regularExpression)
        let text = stripped.split(separator: "\n", omittingEmptySubsequences: false).map { $0.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression) }.joined(separator: "\n")
        return Result(text: ScriptImporter.normalize(text), heading: nil)
    }

    private func flush() {
        let paragraph = current.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression).trimmingCharacters(in: .whitespaces)
        if !paragraph.isEmpty { paragraphs.append(paragraph) }
        current = ""
    }

    func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?, qualifiedName: String?, attributes: [String: String] = [:]) {
        let local = name.split(separator: ":").last.map { $0.lowercased() } ?? name
        if Self.silent.contains(local) { silentDepth += 1 }
        if Self.blocks.contains(local) { flush() }
        if heading == nil, headingBuffer == nil, local == "h1" || local == "h2" { headingBuffer = "" }
    }

    func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
        let local = name.split(separator: ":").last.map { $0.lowercased() } ?? name
        if Self.silent.contains(local) { silentDepth -= 1 }
        if Self.blocks.contains(local) { flush() }
        if let buffer = headingBuffer, local == "h1" || local == "h2" {
            let title = buffer.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression).trimmingCharacters(in: .whitespaces)
            heading = title.isEmpty ? nil : title
            headingBuffer = nil
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard silentDepth == 0 else { return }
        let text = string.replacingOccurrences(of: "\u{00A0}", with: " ")
        current += text
        headingBuffer? += text
    }

    func parser(_ parser: XMLParser, foundCDATA CDATABlock: Data) {
        self.parser(parser, foundCharacters: String(decoding: CDATABlock, as: UTF8.self))
    }
}
