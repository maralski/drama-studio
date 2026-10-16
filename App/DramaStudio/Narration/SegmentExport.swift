import Foundation

/// One audio file per narrated segment plus a `manifest.csv`, e.g. `01-slide-3.wav`.
enum SegmentExport {
    struct Item: Equatable, Sendable {
        var index: Int
        var label: String
        var seconds: Double
        var file: String
    }

    static func plan(takes: [NarrationTake], document: NarrationDocument, format: ExportFormat = .wav16) -> [Item] {
        let labels = Dictionary(document.segments.map { ($0.id, $0.label) }, uniquingKeysWith: { first, _ in first })
        let width = max(2, String(takes.count).count)
        return takes.enumerated().map { offset, take in
            let label = labels[take.segmentID] ?? "Segment \(offset + 1)"
            let number = String(format: "%0\(width)ld", offset + 1)
            return Item(index: offset + 1, label: label, seconds: take.seconds, file: "\(number)-\(slug(label)).\(format.fileExtension)")
        }
    }

    /// The label's leading part ("Slide 3" of "Slide 3: Quarterly results (no notes)"), lowercased and hyphenated.
    static func slug(_ label: String) -> String {
        let head = label.split(separator: ":", maxSplits: 1).first.map(String.init) ?? label
        let trimmed = head.components(separatedBy: " (").first ?? head
        let slug = trimmed.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil).lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "segment" : slug
    }

    static func manifestCSV(_ items: [Item]) -> String {
        let rows = items.map { item in
            [String(item.index), item.label, String(format: "%.3f", item.seconds), item.file].map(csvField).joined(separator: ",")
        }
        return (["index,label,seconds,file"] + rows).joined(separator: "\n") + "\n"
    }

    @discardableResult
    static func export(takes: [NarrationTake], document: NarrationDocument, samples: [Float], to folder: URL, format: ExportFormat = .wav16) throws -> [Item] {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let items = plan(takes: takes, document: document, format: format)
        for (item, take) in zip(items, takes) {
            try AudioExporter.write(Array(samples[take.sampleRange.clamped(to: 0..<samples.count)]), to: folder.appending(path: item.file), as: format)
        }
        try Data(manifestCSV(items).utf8).write(to: folder.appending(path: "manifest.csv"), options: .atomic)
        return items
    }

    private static func csvField(_ value: String) -> String {
        guard value.contains(where: { $0 == "," || $0 == "\"" || $0.isNewline }) else { return value }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
