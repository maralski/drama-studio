// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One codec model call: frames `start ..< end` go in, and the audio of
/// frames `keep ..< end` is kept. The frames before `keep` are left context.
package struct DecodeWindow: Equatable, Sendable {
    package let start: Int
    package let keep: Int
    package let end: Int

    package init(start: Int, keep: Int, end: Int) {
        self.start = start
        self.keep = keep
        self.end = end
    }

    package var kept: Range<Int> { keep ..< end }

    /// The calls of a full decode of `frames` frames.
    package static func windows(frames: Int) -> [DecodeWindow] {
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
    package static func covering(_ range: Range<Int>) -> [DecodeWindow] {
        windows(frames: range.upperBound).compactMap { w in
            w.end > range.lowerBound ? DecodeWindow(start: w.start, keep: max(w.keep, range.lowerBound), end: w.end) : nil
        }
    }

    /// The probe calls that classify frames `pending`: windows of at most
    /// `shape` frames, each keeping the frames after up to `context` frames
    /// of left context, fewer only at the start of the chunk.
    package static func probing(_ pending: Range<Int>, shape: Int = DramaCodec.probeFrames, context: Int = DramaCodec.probeContext) -> [DecodeWindow] {
        precondition(context < shape, "a probe window must keep at least one frame")
        var out: [DecodeWindow] = []
        var keep = pending.lowerBound
        while keep < pending.upperBound {
            let start = max(0, keep - context)
            let end = min(pending.upperBound, start + shape)
            out.append(DecodeWindow(start: start, keep: keep, end: end))
            keep = end
        }
        return out
    }
}

/// One chunk's audio, decoded as its frames land in the windows a full decode
/// would run. The streamed partials read this one decode, and the pieces join
/// into exactly the full decode of the chunk.
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
    private var finalReading: Result<Double, Error>?
    private let finalTaken = DispatchSemaphore(value: 0)

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

    /// Decodes every frame of the attempt and keeps the trailing silence of
    /// the whole for `finalSilence()`.
    func measureFinal(_ frames: [Frame]) {
        let silence = Result { try advance(to: frames) }
        lock.withLock { finalReading = silence }
        finalTaken.signal()
    }

    /// Nil until `measureFinal` has run.
    func finalSilence() throws -> Double? { try lock.withLock { finalReading }?.get() }

    func waitForFinal() {
        finalTaken.wait()
        finalTaken.signal()
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

/// When the silence stop reads a chunk's newest frames: every `everyFrames`
/// frames, and not before `firstFrame`.
package struct SilenceProbeSchedule: Equatable, Sendable {
    package var everyFrames: Int
    package var firstFrame: Int
    /// A reading that fell due while the previous one still ran: the frame
    /// count it classifies up to.
    package private(set) var waiting: Int?

    package init(everyFrames: Int, firstFrame: Int) {
        self.everyFrames = everyFrames
        self.firstFrame = firstFrame
    }

    /// Whether a reading falls due once `frames` frames have landed. A
    /// cadence of 0 never reads.
    package func reads(at frames: Int) -> Bool {
        everyFrames > 0 && frames >= firstFrame && frames.isMultiple(of: everyFrames)
    }

    /// The frame count to classify up to when a reading should start now,
    /// once `frames` frames have landed: the newest one that fell due. A
    /// reading that cannot start yet waits for the probe's queue rather than
    /// being dropped, and covers only the frames up to its due point, so one
    /// slow reading costs one late reading instead of leaving every later
    /// one too long to finish within the cadence. Call `started()` when it
    /// starts.
    package mutating func due(at frames: Int) -> Int? {
        if reads(at: frames) { waiting = frames }
        return waiting
    }

    package mutating func started() { waiting = nil }
}

/// How the silence stop paces its readings. With the probe shape of 64 frames
/// a reading of 16 new frames costs about 190 ms, under the 0.4 s those frames
/// take to generate, so a reading every 16 frames fires the stop within about
/// a reading of the audio going silent. Each reading shares memory bandwidth
/// with the GPU, so the shipped policy reads only once the chunk could have
/// spoken its text: that gave back about 2 ms of every frame on a paragraph,
/// with the cue, prose and long-sentence harness inside its own run-to-run
/// spread (`eval/reliability/REPORT.md`, "Silence probe cadence").
package struct SilenceProbePolicy: Equatable, Sendable {
    /// Frames between readings; 0 turns the probe off.
    package var everyFrames: Int
    /// Read only from the chunk's expected end less the silence the stop
    /// needs, the first frame at which the stop could end a chunk that spoke
    /// its text; before it no probe work shares the GPU's memory bandwidth.
    package var afterExpectedEnd: Bool

    package init(everyFrames: Int, afterExpectedEnd: Bool) {
        self.everyFrames = everyFrames
        self.afterExpectedEnd = afterExpectedEnd
    }

    package static let shipped = SilenceProbePolicy(everyFrames: 16, afterExpectedEnd: true)

    /// The readings for a chunk expected to speak `expectedSeconds`, whose
    /// stop needs `silenceFrames` silent frames. No reading lands before the
    /// audio could hold that silence.
    package func schedule(expectedSeconds: Double, silenceFrames: Int) -> SilenceProbeSchedule {
        var first = silenceFrames + 1
        if afterExpectedEnd {
            let expected = Int((expectedSeconds * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded())
            first = max(first, expected - silenceFrames)
        }
        return SilenceProbeSchedule(everyFrames: everyFrames, firstFrame: first)
    }
}

/// The silence that ends an attempt's audio, read from short decodes of its
/// newest frames. Each call decodes the frames not yet classified in probe
/// windows (`DecodeWindow.probing`, the small decoder shape) and extends the
/// run of silent frames at the tail. The audio is dropped: the chunk's audio
/// comes from `ChunkDecoder`, which decodes each window once at the end.
package final class SilenceProbe: @unchecked Sendable {
    package typealias Decode = (_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float]

    private let decode: Decode
    private let shape: Int
    private let context: Int
    private let silenceFrames: Int
    private let lock = NSLock()
    private var classified = 0
    private var trailingSilentFrames = 0
    private var seconds = 0.0

    /// `silenceFrames` is the silent run the stop waits for. A reading that
    /// has fallen more than that many frames behind classifies only the
    /// newest `silenceFrames`: either they are all silent, and the stop fires
    /// whatever came before, or one is not, and the older frames could not
    /// have extended the tail. The reading then saturates at `silenceFrames`.
    package init(shape: Int = DramaCodec.probeFrames, context: Int = DramaCodec.probeContext, silenceFrames: Int = .max, _ decode: @escaping Decode) {
        self.decode = decode
        self.shape = shape
        self.context = context
        self.silenceFrames = silenceFrames
    }

    /// Seconds of silence at the end of the frames classified so far.
    package var latest: Double {
        lock.withLock { Double(trailingSilentFrames * ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }
    }
    /// How many frames of the attempt have been classified.
    package var classifiedFrames: Int { lock.withLock { classified } }
    /// Codec time spent in probe calls.
    package var codecSeconds: Double { lock.withLock { seconds } }

    /// Classifies the frames not yet classified. A failed decode leaves the
    /// reading where it was. Calls must not overlap.
    package func measure(_ frames: [Frame]) {
        var (done, run) = lock.withLock { (classified, trailingSilentFrames) }
        guard frames.count > done else { return }
        if frames.count - done > silenceFrames {
            done = frames.count - silenceFrames
            run = 0
        }
        let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
        let t0 = Date()
        var classifiedTo = done
        for w in DecodeWindow.probing(done ..< frames.count, shape: shape, context: context) {
            guard let audio = try? decode(codes, w) else { break }
            let n = ModelConstants.samplesPerFrame
            for f in 0 ..< w.kept.count {
                run = DramaSynthesizer.isSilent(audio[f * n ..< (f + 1) * n]) ? run + 1 : 0
            }
            classifiedTo = w.end
        }
        let elapsed = Date().timeIntervalSince(t0)
        lock.withLock {
            seconds += elapsed
            guard classifiedTo > done else { return }
            classified = classifiedTo
            trailingSilentFrames = run
        }
    }
}
