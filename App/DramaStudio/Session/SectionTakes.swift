// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// One document section's takes, oldest first and never empty, and the one
/// the narration uses.
struct SectionTakes: Equatable {
    private(set) var takes: [SessionTake]
    var chosenID: SessionTake.ID

    init(first: SessionTake) {
        takes = [first]
        chosenID = first.id
    }

    var chosen: SessionTake { takes.first { $0.id == chosenID } ?? takes[takes.count - 1] }
    var nextNumber: Int { (takes.last?.number ?? 0) + 1 }

    /// A new take becomes the chosen one.
    mutating func add(_ take: SessionTake) {
        takes.append(take)
        chosenID = take.id
    }
}

/// Where one spoken sentence sits in the transport's audio.
struct SentenceSpan: Equatable {
    var segmentID: NarrationDocument.Segment.ID?
    var text: String
    var range: Range<Int>

    /// Consecutive spans for a take's sentences, cut off at the take's end.
    static func spans(of sentences: [NarrationTake.Sentence], in range: Range<Int>, segmentID: NarrationDocument.Segment.ID?) -> [SentenceSpan] {
        var cursor = range.lowerBound
        return sentences.map { sentence in
            let end = min(cursor + sentence.frames * ModelConstants.samplesPerFrame, range.upperBound)
            defer { cursor = end }
            return SentenceSpan(segmentID: segmentID, text: sentence.text, range: cursor..<end)
        }
    }
}

/// The sentence being spoken: which one of its take's sentences, with the
/// texts of all of them so the editor can find it in order.
struct SpokenHighlight: Equatable {
    var segmentID: NarrationDocument.Segment.ID?
    var index: Int
    var texts: [String]
}
