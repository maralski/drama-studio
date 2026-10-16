import Foundation

/// One codec model call: frames `start ..< end` go in, and the audio of
/// frames `keep ..< end` is kept. The frames before `keep` are left context.
public struct DecodeWindow: Equatable, Sendable {
    public let start: Int
    public let keep: Int
    public let end: Int

    public init(start: Int, keep: Int, end: Int) {
        self.start = start
        self.keep = keep
        self.end = end
    }

    public var kept: Range<Int> { keep ..< end }

    /// The calls of a full decode of `frames` frames.
    public static func windows(frames: Int) -> [DecodeWindow] {
        var out: [DecodeWindow] = []
        var keep = 0
        while keep < frames {
            let start = keep == 0 ? 0 : keep - DramaCodec.contextFrames
            let end = min(frames, start + DramaCodec.windowFrames)
            out.append(DecodeWindow(start: start, keep: keep, end: end))
            keep = end
        }
        return out
    }

    /// The calls that produce frames `range` with the same left context a full
    /// decode of the chunk gives them. A call costs the same however few of
    /// its frames are real, so this is no dearer than a bare 256-frame context.
    public static func covering(_ range: Range<Int>) -> [DecodeWindow] {
        windows(frames: range.upperBound).compactMap { w in
            w.end > range.lowerBound ? DecodeWindow(start: w.start, keep: max(w.keep, range.lowerBound), end: w.end) : nil
        }
    }
}

/// One chunk's audio, decoded as its frames land. The silence probe and the
/// streamed partials read this one decode, and the pieces join into exactly
/// the full decode of the chunk.
final class ChunkDecoder: @unchecked Sendable {
    typealias Decode = (_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float]

    struct Piece: Equatable {
        var samples: [Float]
        var frames: Range<Int>
    }

    private let decode: Decode
    private let lock = NSLock()
    private var decoded: [Float] = []
    private var undelivered: [Piece] = []
    private var seconds = 0.0

    init(_ decode: @escaping Decode) { self.decode = decode }

    var samples: [Float] { lock.withLock { decoded } }
    var codecSeconds: Double { lock.withLock { seconds } }

    /// Decodes the frames not yet decoded and returns the seconds of silence
    /// that end the audio so far. Calls must not overlap.
    func advance(to frames: [Frame]) throws -> Double {
        let done = lock.withLock { decoded.count } / ModelConstants.samplesPerFrame
        if frames.count > done {
            let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
            let t0 = Date()
            var piece: [Float] = []
            for w in DecodeWindow.covering(done ..< frames.count) { piece += try decode(codes, w) }
            let elapsed = Date().timeIntervalSince(t0)
            lock.withLock {
                decoded += piece
                undelivered.append(Piece(samples: piece, frames: done ..< frames.count))
                seconds += elapsed
            }
        }
        return lock.withLock { DramaSynthesizer.trailingSilenceSeconds(decoded) }
    }

    /// The pieces decoded since the last call, oldest first.
    func takePieces() -> [Piece] {
        lock.withLock {
            defer { undelivered = [] }
            return undelivered
        }
    }
}

/// Runs codec work off the generation thread, one job at a time and in the
/// order given, so the CPU codec overlaps the GPU and a chunk's audio never
/// lands after the next chunk's.
final class DecodeRunner: @unchecked Sendable {
    private let queue = DispatchQueue(label: "DramaEngine.decode", qos: .userInitiated)
    private let lock = NSLock()
    private var queued = 0

    /// Starts `work` unless earlier work is still queued or running.
    @discardableResult
    func offer(_ work: @escaping () -> Void) -> Bool {
        guard lock.withLock({ queued == 0 }) else { return false }
        enqueue(work)
        return true
    }

    func enqueue(_ work: @escaping () -> Void) {
        lock.withLock { queued += 1 }
        queue.async { [self] in
            work()
            lock.withLock { queued -= 1 }
        }
    }

    func drain() { queue.sync {} }
}

/// One attempt's readings of the silence that ends its audio.
final class SilenceProbe: @unchecked Sendable {
    private let reading: ([Frame]) throws -> Double
    private let lock = NSLock()
    private var last = 0.0
    private var finalReading: Result<Double, Error>?
    private let finalTaken = DispatchSemaphore(value: 0)

    init(_ reading: @escaping ([Frame]) throws -> Double) { self.reading = reading }

    var latest: Double { lock.withLock { last } }

    /// A reading taken while frames still land; a failure reads as no silence.
    func measure(_ frames: [Frame]) {
        let silence = (try? reading(frames)) ?? 0
        lock.withLock { last = silence }
    }

    func measureNow(_ frames: [Frame]) throws -> Double { try reading(frames) }

    /// The reading over every frame of the attempt, kept for `final()`.
    func measureFinal(_ frames: [Frame]) {
        let silence = Result { try reading(frames) }
        lock.withLock { finalReading = silence }
        finalTaken.signal()
    }

    /// Nil until `measureFinal` has run.
    func final() throws -> Double? { try lock.withLock { finalReading }?.get() }

    func waitForFinal() {
        finalTaken.wait()
        finalTaken.signal()
    }
}
