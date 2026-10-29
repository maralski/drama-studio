// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Maps the player node's clock to a position in the script audio. Chunks
/// arrive while playback runs, so the node may starve and resume later; each
/// scheduled buffer is a segment placed on the node clock.
struct PlaybackTimeline: Equatable {
    struct Segment: Equatable {
        var nodeStart: Int64
        var audioStart: Int
        var length: Int
    }

    let origin: Int
    private(set) var segments: [Segment] = []

    init(origin: Int) { self.origin = origin }

    var audioEnd: Int { segments.last.map { $0.audioStart + $0.length } ?? origin }
    var nodeEnd: Int64 { segments.last.map { $0.nodeStart + Int64($0.length) } ?? 0 }

    mutating func append(length: Int, earliest: Int64) -> Int64 {
        let start = max(nodeEnd, earliest)
        segments.append(Segment(nodeStart: start, audioStart: audioEnd, length: length))
        return start
    }

    func audioPosition(atNodeSample time: Int64) -> Int {
        guard let segment = segments.last(where: { $0.nodeStart <= time }) else { return origin }
        return segment.audioStart + min(Int(time - segment.nodeStart), segment.length)
    }
}
