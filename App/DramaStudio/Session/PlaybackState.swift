import Foundation

/// The session lifecycle. Playback may run while chunks are still being
/// generated, so `generating` carries whether audio is playing.
enum PlaybackState: Equatable {
    case idle
    case loading(progress: Double, detail: String)
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
        case .playing, .generating(_, _, true): true
        default: false
        }
    }

    func withPlayback(_ playing: Bool) -> PlaybackState {
        switch self {
        case .generating(let chunk, let count, _): .generating(chunk: chunk, of: count, playing: playing)
        case .loading: self
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
