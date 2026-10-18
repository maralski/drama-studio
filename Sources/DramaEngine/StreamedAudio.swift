/// Joins a run's audio events into one signal: partials as they arrive, and
/// of each chunk's final audio only what its partials did not already carry.
///
/// Pass every event through ``newSamples(from:)`` and play or append what it
/// returns. The result is the same signal under either ``StreamingMode``.
public struct StreamedAudio: Sendable, Equatable {
    private var streamed: [Int: Int] = [:]

    /// An empty signal.
    public init() {}

    /// The samples `event` adds to the signal; empty for events without audio.
    public mutating func newSamples(from event: SpeechEvent) -> [Float] {
        switch event {
        case .partialAudio(let samples, let chunk, _):
            streamed[chunk, default: 0] += samples.count
            return samples
        case .audio(let samples, let chunk):
            return Array(samples.dropFirst(streamed[chunk] ?? 0))
        default:
            return []
        }
    }
}
