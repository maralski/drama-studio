import Foundation

enum DocumentImportError: LocalizedError, Equatable {
    case unsupported(String)
    case malformed(String)
    case empty(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let name): "Drama Studio can't narrate \(name). Try a PowerPoint, PDF, EPUB, or text file."
        case .malformed(let detail): "The document is damaged: \(detail)."
        case .empty(let name): "\(name) has no text to narrate."
        }
    }
}

/// XML parts of a ZIP package (OOXML, EPUB), matched by local name so either
/// namespace flavour (transitional or strict OOXML, EPUB 2 or 3) reads the same.
extension ZipArchive {
    func xml(_ path: String) throws -> XMLElement {
        let data = try data(for: path)
        guard let root = try? XMLDocument(data: data, options: [.nodeLoadExternalEntitiesNever]).rootElement() else {
            throw DocumentImportError.malformed("\(path) is not valid XML")
        }
        return root
    }

    struct Relationship: Equatable {
        var id: String
        var type: String
        var target: String
    }

    func relationships(of part: String) throws -> [String: Relationship] {
        let name = part.split(separator: "/").last.map(String.init) ?? part
        let folder = part.split(separator: "/").dropLast().joined(separator: "/")
        let path = (folder.isEmpty ? "" : folder + "/") + "_rels/\(name).rels"
        guard contains(path) else { return [:] }
        var out: [String: Relationship] = [:]
        for element in try xml(path).children(named: "Relationship") {
            guard let id = element.value(of: "Id"), let target = element.value(of: "Target"), element.value(of: "TargetMode") != "External" else { continue }
            out[id] = Relationship(id: id, type: element.value(of: "Type") ?? "", target: ZipArchive.resolve(target, relativeTo: part))
        }
        return out
    }
}

extension XMLElement {
    func children(named localName: String) -> [XMLElement] {
        (children ?? []).compactMap { $0 as? XMLElement }.filter { $0.localName == localName }
    }

    func child(named localName: String) -> XMLElement? { children(named: localName).first }

    func descendants(named localName: String) -> [XMLElement] {
        (children ?? []).compactMap { $0 as? XMLElement }.flatMap { ($0.localName == localName ? [$0] : []) + $0.descendants(named: localName) }
    }

    func value(of attributeLocalName: String) -> String? {
        attributes?.first { $0.localName == attributeLocalName && ($0.prefix ?? "").isEmpty }?.stringValue
    }

    /// A namespaced attribute such as `r:id`, which can sit beside an unprefixed `id`.
    func prefixedValue(of attributeLocalName: String) -> String? {
        attributes?.first { $0.localName == attributeLocalName && !($0.prefix ?? "").isEmpty }?.stringValue
    }
}
