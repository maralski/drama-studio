import Foundation

/// Time left in a generation, from the events it streams. Times are seconds
/// since the request started.
///
/// remaining = (expected frames of unfinished chunks - frames of the current
/// chunk) / frames per second + prefill seconds * chunks not yet prefilled,
/// counted down from the last event.
public struct GenerationEstimate: Sendable, Equatable {
    public static let audioFramesPerSecond = Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)
    /// What the engine typically generates, until the run measures its own.
    public static let typicalFramesPerSecond = 35.0
    /// A 324-row prompt prefills in 0.72 s end to end.
    public static let typicalPrefillSeconds = 0.7
    /// The time constant of the moving average of frames per second.
    static let smoothingSeconds = 3.0
    static let sampleSeconds = 0.5

    public enum Phase: Sendable, Equatable {
        case estimating
        /// The last chunk is past its expected length, in its silence tail.
        case finishing
        /// A retry regenerates the chunk, so the frame counts no longer predict the end.
        case retrying
    }

    public let expectedFrames: [Int]
    package private(set) var chunk = 0
    package private(set) var chunkPrefilled = false
    package private(set) var framesInChunk = 0
    public private(set) var framesDone = 0
    public private(set) var framesPerSecond = typicalFramesPerSecond
    package private(set) var prefillSeconds = typicalPrefillSeconds
    package private(set) var retrying = false
    package private(set) var lastEventSeconds = 0.0
    private var measuredRate = false
    private var chunkStartSeconds = 0.0
    private var sampleSeconds: Double?
    private var sampleFrames = 0

    /// `lines` are the script text of each chunk, as `GenerationPlan` cuts them.
    public init(lines: [String]) {
        expectedFrames = lines.map { Int((FrameGenerator.expectedSeconds(for: $0) * Self.audioFramesPerSecond).rounded()) }
    }

    package init(plan: GenerationPlan) {
        self.init(lines: plan.chunks.map(\.line))
    }

    public var framesExpected: Int { expectedFrames.reduce(0, +) }

    public var phase: Phase {
        if retrying { return .retrying }
        let last = expectedFrames.count - 1
        if chunk > last || (chunk == last && chunkPrefilled && framesInChunk >= expectedFrames[last]) { return .finishing }
        return .estimating
    }

    public mutating func apply(_ event: SpeechEvent, at seconds: Double) {
        lastEventSeconds = seconds
        switch event {
        case .chunkStarted(let index, _, _):
            chunk = index
            chunkPrefilled = false
            framesInChunk = 0
            retrying = false
            chunkStartSeconds = seconds
            sampleSeconds = nil
        case .prefill:
            chunkPrefilled = true
            prefillSeconds = seconds - chunkStartSeconds
        case .frame(let index, _):
            chunkPrefilled = true
            framesDone += index + 1 - framesInChunk
            framesInChunk = index + 1
            measure(at: seconds)
        case .retry:
            retrying = true
            framesInChunk = 0
            sampleSeconds = nil
        case .audio(_, let index):
            if index == expectedFrames.count - 1 { chunk = expectedFrames.count }
        case .loadingModel, .partialAudio, .finished:
            break
        }
    }

    private mutating func measure(at seconds: Double) {
        guard let last = sampleSeconds else {
            sampleSeconds = seconds
            sampleFrames = framesInChunk
            return
        }
        let dt = seconds - last
        guard dt >= Self.sampleSeconds else { return }
        let rate = Double(framesInChunk - sampleFrames) / dt
        if measuredRate {
            framesPerSecond += (1 - exp(-dt / Self.smoothingSeconds)) * (rate - framesPerSecond)
        } else {
            framesPerSecond = rate
            measuredRate = true
        }
        sampleSeconds = seconds
        sampleFrames = framesInChunk
    }

    /// Nil once the estimate stops meaning anything: finishing or retrying.
    public func remainingSeconds(at seconds: Double) -> Double? {
        guard phase == .estimating else { return nil }
        let unfinished = expectedFrames.indices.filter { $0 >= chunk }.map { expectedFrames[$0] }.reduce(0, +)
        let frames = max(0, unfinished - min(framesInChunk, expectedFrames[chunk]))
        let prefills = expectedFrames.count - chunk - (chunkPrefilled ? 1 : 0)
        let atLastEvent = Double(frames) / max(framesPerSecond, 1) + prefillSeconds * Double(prefills)
        return max(0, atLastEvent - max(0, seconds - lastEventSeconds))
    }

    /// "about 12 s left", "finishing…", or "retrying".
    public func label(at seconds: Double) -> String {
        switch phase {
        case .finishing: return "finishing\u{2026}"
        case .retrying: return "retrying"
        case .estimating: return Self.describe(remaining: remainingSeconds(at: seconds) ?? 0)
        }
    }

    static func describe(remaining: Double) -> String {
        let rounded = remaining > 20 ? (remaining / 5).rounded() * 5 : remaining.rounded()
        return "about \(max(1, Int(rounded))) s left"
    }
}
