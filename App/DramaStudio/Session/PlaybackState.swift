// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The session lifecycle. Playback may run while chunks are still being
/// generated, and across the sections of a document while the next one
/// starts, so `loading` and `generating` carry whether audio is playing.
enum PlaybackState: Equatable {
    case idle
    case loading(progress: Double, detail: String, playing: Bool = false)
    case generating(chunk: Int, of: Int, playing: Bool)
    case ready
    case playing
    case failed(String)

    var isGenerating: Bool {
        switch self {
        case .loading, .generating: true
        default: false
        }
    }

    var isPlaying: Bool {
        switch self {
        case .playing, .generating(_, _, true), .loading(_, _, true): true
        default: false
        }
    }

    func withPlayback(_ playing: Bool) -> PlaybackState {
        switch self {
        case .generating(let chunk, let count, _): .generating(chunk: chunk, of: count, playing: playing)
        case .loading(let progress, let detail, _): .loading(progress: progress, detail: detail, playing: playing)
        default: playing ? .playing : .ready
        }
    }

    var afterGeneration: PlaybackState { isPlaying ? .playing : .ready }
}

struct GenerationTelemetry: Equatable {
    var frame = 0
    var maxFrames = 0
    var framesPerSecond: Double = 0
    var chunkStart: ContinuousClock.Instant?
    var retries = 0
}
