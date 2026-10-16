/// Joins a run's audio events into one signal: partials as they arrive, and
/// of each chunk's final audio only what its partials did not already carry.
public struct StreamedAudio: Sendable, Equatable {
    private var streamed: [Int: Int] = [:]

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
