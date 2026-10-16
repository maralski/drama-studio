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

    var narratedSegments: [Segment] { segments.filter { !$0.skip } }
}

/// One generated segment, placed in the combined audio of the whole document.
struct NarrationTake: Codable, Sendable, Equatable {
    struct Sentence: Codable, Sendable, Equatable {
        var text: String
        var frames: Int
    }

    var segmentID: NarrationDocument.Segment.ID
    var sampleRange: Range<Int>
    var sentences: [Sentence]

    var frames: Int { sentences.reduce(0) { $0 + $1.frames } }
    var seconds: Double { Double(sampleRange.count) / Double(ModelConstants.sampleRate) }
}
