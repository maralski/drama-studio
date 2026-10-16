import Foundation

/// Speaker notes per slide, in presentation order, with the slide's own text
/// standing in when a slide has no notes.
enum PPTXImporter {
    static func importFile(at url: URL) throws -> NarrationDocument {
        let fallbackTitle = url.deletingPathExtension().lastPathComponent
        return try importPackage(ZipArchive(url: url), fallbackTitle: fallbackTitle)
    }

    static func importPackage(_ archive: ZipArchive, fallbackTitle: String) throws -> NarrationDocument {
        let presentationPath = try officeDocumentPath(in: archive)
        let presentation = try archive.xml(presentationPath)
        let relationships = try archive.relationships(of: presentationPath)
        let slidePaths = presentation.descendants(named: "sldId").compactMap { $0.prefixedValue(of: "id").flatMap { relationships[$0]?.target } }
        guard !slidePaths.isEmpty else { throw DocumentImportError.empty(fallbackTitle) }

        let segments = try slidePaths.enumerated().map { offset, path in
            try segment(number: offset + 1, slidePath: path, in: archive)
        }
        return NarrationDocument(title: try documentTitle(in: archive) ?? fallbackTitle, source: .pptx, segments: segments)
    }

    private static func segment(number: Int, slidePath: String, in archive: ZipArchive) throws -> NarrationDocument.Segment {
        let slide = try archive.xml(slidePath)
        let slideText = ShapeText(slide)
        let notesPath = try archive.relationships(of: slidePath).values.first { $0.type.hasSuffix("/notesSlide") }?.target
        let notes = try notesPath.map { ShapeText(try archive.xml($0)).body } ?? ""
        let heading = slideText.title.isEmpty ? "Slide \(number)" : "Slide \(number): \(slideText.title.replacingOccurrences(of: "\n", with: " "))"
        let hidden = slide.value(of: "show") == "0"

        if !notes.isEmpty {
            return .init(index: number, label: heading, text: notes, skip: hidden)
        }
        let fallback = ScriptImporter.normalize([slideText.title, slideText.body].filter { !$0.isEmpty }.joined(separator: "\n\n"))
        return .init(index: number, label: heading + " (no notes)", text: fallback, skip: hidden || fallback.isEmpty)
    }

    private static func officeDocumentPath(in archive: ZipArchive) throws -> String {
        let root = try archive.relationships(of: "")
        if let main = root.values.first(where: { $0.type.hasSuffix("/officeDocument") }) { return main.target }
        if archive.contains("ppt/presentation.xml") { return "ppt/presentation.xml" }
        throw DocumentImportError.malformed("no ppt/presentation.xml")
    }

    private static func documentTitle(in archive: ZipArchive) throws -> String? {
        guard archive.contains("docProps/core.xml") else { return nil }
        let title = try archive.xml("docProps/core.xml").child(named: "title")?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
        return title?.isEmpty == false ? title : nil
    }
}

/// The text of a slide or notes slide, split into the title placeholder and everything else.
struct ShapeText {
    private static let ignoredPlaceholders: Set<String> = ["sldNum", "hdr", "ftr", "dt", "sldImg"]
    private static let titlePlaceholders: Set<String> = ["title", "ctrTitle"]

    private(set) var title = ""
    private(set) var body = ""

    init(_ root: XMLElement) {
        var titles: [String] = []
        var bodies: [String] = []
        func walk(_ element: XMLElement) {
            switch element.localName {
            case "sp":
                let type = element.child(named: "nvSpPr")?.child(named: "nvPr")?.child(named: "ph")?.value(of: "type")
                if let type, Self.ignoredPlaceholders.contains(type) { return }
                let text = element.descendants(named: "txBody").map(Self.text(of:)).joined(separator: "\n")
                if let type, Self.titlePlaceholders.contains(type) { titles.append(text) } else { bodies.append(text) }
            case "txBody":
                bodies.append(Self.text(of: element))
            default:
                for child in (element.children ?? []).compactMap({ $0 as? XMLElement }) { walk(child) }
            }
        }
        walk(root)
        title = ScriptImporter.normalize(titles.joined(separator: "\n"))
        body = ScriptImporter.normalize(bodies.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.joined(separator: "\n\n"))
    }

    private static func text(of body: XMLElement) -> String {
        body.children(named: "p").map(runs(in:)).joined(separator: "\n")
    }

    private static func runs(in element: XMLElement) -> String {
        (element.children ?? []).compactMap { $0 as? XMLElement }.map { child in
            switch child.localName {
            case "t": child.stringValue ?? ""
            case "br": "\n"
            case "r", "fld": runs(in: child)
            default: ""
            }
        }.joined()
    }
}
