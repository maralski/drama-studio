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
