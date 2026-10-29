// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import DramaEngine

/// The GPU sampler kernels must pick exactly what the CPU `Sampler` and
/// `MainTokenSampler` pick from the same logits and the same seeded state,
/// and leave the generator in the same state.
final class SamplerParityTests: XCTestCase {
    /// The CPU reference sorts 4096 logits per sample, so a debug build takes
    /// about a minute per 2,000; DRAMA_SAMPLER_PARITY_SAMPLES=20000 is the long
    /// run (160,000 picks over the four tests, all identical when last run).
    static let samplesPerSet = ProcessInfo.processInfo.environment["DRAMA_SAMPLER_PARITY_SAMPLES"].flatMap(Int.init) ?? 2_000
    static let settings = [SamplingParams(temperature: 0.7, topK: 30, topP: 0.7), SamplingParams(temperature: 1.0, topK: 30, topP: 0.95)]
    /// Keeps more candidates than a sampler threadgroup has threads.
    static let wide = SamplingParams(temperature: 1.0, topK: 600, topP: 1.0)
    static let codebookSize = ModelConstants.codebookSize
    static let candidates = ModelConstants.codebookSize + 1

    /// Drives the sampler kernels without a model, one dispatch per row, the
    /// RNG chained through slot 1 after the first row starts from slot 0.
    final class Harness {
        let ctx: MetalContext
        let k: Kernels
        let state: SamplerState
        let frames: Buf
        let code: Buf
        let semantic: Buf
        let logits: Buf
        let n: Int
        static let batch = 500

        init(n: Int, frameRows: Int) throws {
            ctx = try MetalContext()
            k = Kernels(ctx: ctx, precision: .exact)
            state = SamplerState(device: ctx.device)
            self.n = n
            frames = Buf(ctx.device, floats: frameRows * 11, label: "frames")
            frames.zero()
            code = Buf(ctx.device, floats: 1, label: "code")
            semantic = Buf(ctx.device, floats: 1, label: "semantic")
            logits = Buf(ctx.device, floats: Self.batch * n, label: "logits")
        }

        private func params(_ p: SamplingParams, rngIn: Int, position: Int, head: Int) -> SamplerKernelParams {
            SamplerKernelParams(p, n: n, rngIn: rngIn, rngOut: 1, position: position, head: head,
                                eos: ModelConstants.eos, semanticStart: ModelConstants.semanticStart, codebookSize: ModelConstants.codebookSize)
        }

        private func upload(_ rows: ArraySlice<[Float]>) {
            for (i, row) in rows.enumerated() {
                precondition(row.count == n)
                row.withUnsafeBufferPointer { memcpy(logits.floats + i * n, $0.baseAddress!, n * 4) }
            }
        }

        private func run(_ encode: (MTLComputeCommandEncoder) throws -> Void) throws {
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            try encode(enc)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("\(e)") }
        }

        /// Tower-head picks for every row, as `Sampler.sample` indices.
        func towerPicks(_ rows: [[Float]], start: SIMD4<UInt64>, settings p: SamplingParams) throws -> (picks: [Int], end: SIMD4<UInt64>) {
            state.setState(start, slot: 0)
            var picks: [Int] = []
            var first = true
            for b in stride(from: 0, to: rows.count, by: Self.batch) {
                let slice = rows[b..<min(b + Self.batch, rows.count)]
                upload(slice)
                try run { enc in
                    for r in 0..<slice.count {
                        try k.sampleTower(enc, logits: logits, logitsOffset: r * n, rng: state.rng, frames: frames, codeOut: code, status: state.status,
                                          params: params(p, rngIn: first ? 0 : 1, position: r, head: 0))
                        first = false
                    }
                }
                for r in 0..<slice.count { picks.append(Int(frames.ints[(r + 1) * 11 + 2])) }
            }
            XCTAssertEqual(state.statusWord(slot: 1), 0)
            return (picks, state.state(slot: 1))
        }

        /// Main tokens for every row, continuing `history` (which fills frame rows 0..<count).
        func mainPicks(_ rows: [[Float]], history: [Int32], start: SIMD4<UInt64>, settings p: SamplingParams) throws -> (picks: [Int32], end: SIMD4<UInt64>) {
            state.setState(start, slot: 0)
            for (r, m) in history.enumerated() { frames.ints[r * 11] = m }
            var picks: [Int32] = []
            var position = history.count - 1
            var first = true
            for b in stride(from: 0, to: rows.count, by: Self.batch) {
                let slice = rows[b..<min(b + Self.batch, rows.count)]
                upload(slice)
                let p0 = position
                try run { enc in
                    for r in 0..<slice.count {
                        try k.sampleMain(enc, logits: Buf(view: logits, byteOffset: r * n * 4), rng: state.rng, frames: frames, code0: code, semantic: semantic,
                                         status: state.status, params: params(p, rngIn: first ? 0 : 1, position: position, head: 0))
                        first = false
                        position += 1
                    }
                }
                for r in 0..<slice.count { picks.append(frames.ints[(p0 + r + 1) * 11]) }
            }
            XCTAssertEqual(state.statusWord(slot: 1), 0)
            return (picks, state.state(slot: 1))
        }
    }

    static func uniformRows(count: Int, n: Int, seed: UInt64, peak: Bool) -> [[Float]] {
        var g = SeededGenerator(seed: seed)
        return (0..<count).map { _ in
            var row = (0..<n).map { _ in Float(g.next() >> 40) / Float(1 << 24) * 20 - 10 }
            if peak { row[Int(g.next() % UInt64(n))] += 12 }
            return row
        }
    }

    static func dumpRows() throws -> (cand: [[Float]], tower: [[Float]])? {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("eval/out/perf/dump_before.npz")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let d = try NPY.readNPZ(url: url)
        guard let cand = d["cand_logits"], let tower = d["tower_logits"], cand.shape.count == 2, tower.shape.count == 3 else { return nil }
        let cv = cand.float32Values, tv = tower.float32Values
        let cn = cand.shape[1], tn = tower.shape[2]
        let candRows = (0..<cand.shape[0]).map { Array(cv[($0 * cn)..<(($0 + 1) * cn)]) }
        let towerRows = (0..<(tower.shape[0] * tower.shape[1])).map { Array(tv[($0 * tn)..<(($0 + 1) * tn)]) }
        return (candRows, towerRows)
    }

    static func cycled(_ rows: [[Float]], count: Int) -> [[Float]] { (0..<count).map { rows[$0 % rows.count] } }

    func checkTower(_ rows: [[Float]], seed: UInt64, label: String, settings: [SamplingParams] = settings) throws {
        let harness = try Harness(n: rows[0].count, frameRows: Harness.batch + 1)
        for p in settings {
            let start = Sampler(seed: seed).state
            let gpu = try harness.towerPicks(rows, start: start, settings: p)
            var cpu = Sampler(state: start)
            var mismatches = 0
            var firstMismatch = -1
            for (i, row) in rows.enumerated() where cpu.sample(row, temperature: p.temperature, topK: p.topK, topP: p.topP) != gpu.picks[i] {
                mismatches += 1
                if firstMismatch < 0 { firstMismatch = i }
            }
            XCTAssertEqual(mismatches, 0, "\(label) \(p): picks differ, first at row \(firstMismatch)")
            XCTAssertEqual(cpu.state, gpu.end, "\(label) \(p): generator state differs")
        }
    }

    func testUniformLogits() throws {
        try checkTower(Self.uniformRows(count: Self.samplesPerSet, n: Self.codebookSize, seed: 1, peak: false), seed: 11, label: "uniform", settings: Self.settings + [Self.wide])
    }

    func testPeakedLogits() throws {
        try checkTower(Self.uniformRows(count: Self.samplesPerSet, n: Self.codebookSize, seed: 2, peak: true), seed: 12, label: "peaked", settings: Self.settings + [Self.wide])
    }

    func testRealTowerLogits() throws {
        guard let dump = try Self.dumpRows() else { throw XCTSkip("eval/out/perf/dump_before.npz not present") }
        try checkTower(Self.cycled(dump.tower, count: Self.samplesPerSet), seed: 13, label: "real tower", settings: Self.settings + [Self.wide])
    }

    /// The main-token path: candidate mapping, the retry draw and the
    /// repetition window against `MainTokenSampler` on a synthetic history.
    func testMainTokenPath() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        let reference = MainTokenSampler(config: config)
        var sets: [(String, [[Float]])] = [("peaked", Self.uniformRows(count: Self.samplesPerSet / 2, n: Self.candidates, seed: 3, peak: true))]
        if let dump = try Self.dumpRows() { sets.append(("real", Self.cycled(dump.cand, count: Self.samplesPerSet / 2))) }
        var g = SeededGenerator(seed: 5)
        let history = (0..<MainTokenSampler.windowFrames).map { _ in config.semanticStart + Int32(g.next() % UInt64(Self.codebookSize)) }
        for (label, rows) in sets {
            let harness = try Harness(n: Self.candidates, frameRows: history.count + rows.count + 1)
            for p in Self.settings {
                let start = Sampler(seed: 21).state
                let gpu = try harness.mainPicks(rows, history: history, start: start, settings: p)
                var cpu = Sampler(state: start)
                var mains = history
                var mismatches = 0, retries = 0
                for (i, row) in rows.enumerated() {
                    var probe = cpu
                    let first = reference.candidates[probe.sample(row, temperature: p.temperature, topK: p.topK, topP: p.topP)]
                    let main = reference.sample(candidateLogits: row, history: mains[...], temperature: p.temperature, topK: p.topK, topP: p.topP, sampler: &cpu)
                    if main != first { retries += 1 }
                    if main != gpu.picks[i] { mismatches += 1 }
                    mains.append(main)
                }
                XCTAssertEqual(mismatches, 0, "\(label) \(p): main tokens differ")
                XCTAssertEqual(cpu.state, gpu.end, "\(label) \(p): generator state differs")
                XCTAssertGreaterThan(retries, 0, "\(label) \(p): the repetition retry never triggered")
            }
        }
    }
}
