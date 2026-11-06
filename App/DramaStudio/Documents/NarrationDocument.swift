// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

struct NarrationDocument: Codable, Sendable, Equatable {
    enum Source: String, Codable, Sendable, CaseIterable {
        case pptx, pdf, epub, text
    }

    struct Segment: Identifiable, Codable, Sendable, Equatable {
        var id: UUID
        var index: Int
        var label: String
        var text: String
        var speakerTag: Int?
        var skip: Bool

        init(id: UUID = UUID(), index: Int, label: String, text: String, speakerTag: Int? = nil, skip: Bool = false) {
            self.id = id
            self.index = index
            self.label = label
            self.text = text
            self.speakerTag = speakerTag
            self.skip = skip
        }
    }

    var title: String
    var source: Source
    var segments: [Segment]

    /// The narrated segments as one plain script, blank lines between them.
    /// Speaker tags appear only when the narrated segments use more than one speaker.
    func flattened() -> String {
        let narrated = segments.filter { !$0.skip && !SectionRequests.spokenText(of: $0).isEmpty }
        let tagged = Set(narrated.map { $0.speakerTag ?? 0 }).count > 1
        return narrated.map { segment in
            let text = SectionRequests.spokenText(of: segment)
            return tagged ? "<|speaker:\(segment.speakerTag ?? 0)|>\(text)" : text
        }.joined(separator: "\n\n")
    }

    static func importFile(at url: URL) throws -> NarrationDocument {
        switch url.pathExtension.lowercased() {
        case "pptx": return try PPTXImporter.importFile(at: url)
        case "pdf": return try PDFImporter.importFile(at: url)
        case "epub": return try EPUBImporter.importFile(at: url)
        case "txt", "text":
            let text = ScriptImporter.normalize(try String(contentsOf: url, encoding: .utf8))
            guard !text.isEmpty else { throw DocumentImportError.empty(url.lastPathComponent) }
            let title = url.deletingPathExtension().lastPathComponent
            return NarrationDocument(title: title, source: .text, segments: [Segment(index: 1, label: title, text: text)])
        default:
            throw DocumentImportError.unsupported(url.lastPathComponent)
        }
    }
}

struct NarrationTake: Codable, Sendable, Equatable {
    struct Sentence: Codable, Sendable, Equatable {
        var text: String
        /// The sentence's audio length, in 44.1 kHz samples.
        var samples: Int

        init(text: String, samples: Int) {
            self.text = text
            self.samples = samples
        }

        private enum CodingKeys: String, CodingKey { case text, samples, frames }

        /// Projects before format 5 time sentences in Drama 3 frames of 2048 samples.
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            text = try container.decode(String.self, forKey: .text)
            samples = try container.decodeIfPresent(Int.self, forKey: .samples)
                ?? container.decode(Int.self, forKey: .frames) * ModelConstants.samplesPerFrame
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(text, forKey: .text)
            try container.encode(samples, forKey: .samples)
        }
    }

    var segmentID: NarrationDocument.Segment.ID
    var sampleRange: Range<Int>
    var sentences: [Sentence]
    /// The turn text and engine context, so a reopened project still has history. Absent in older projects.
    var take: Take?

    init(segmentID: NarrationDocument.Segment.ID, sampleRange: Range<Int>, sentences: [Sentence], take: Take? = nil) {
        self.segmentID = segmentID
        self.sampleRange = sampleRange
        self.sentences = sentences
        self.take = take
    }

    var seconds: Double { Double(sampleRange.count) / Double(ModelConstants.sampleRate) }
}
