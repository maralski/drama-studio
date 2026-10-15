import Foundation
import Metal

/// Per-device kernel autotune. Each entry is one kernel family's real
/// workload on the resident weights; every candidate of the family's domain
/// is validated against the default on seeded inputs, the valid ones are
/// timed interleaved round by round, and the fastest one that beats the
/// default by `minGainPercent` wins. The result is one `KernelSchedule`,
/// cached per `TuningKey`.
public enum Autotune {
    public struct Options: Sendable {
        public var rounds = 5
        public var budgetSeconds = 60.0
        /// Largest relative L2 against the default a candidate may show.
        public var tolerance = 1e-5
        public var minGainPercent = 2.0
        public var progress: (@Sendable (String) -> Void)? = nil
        public init() {}
    }

    struct Candidate {
        let label: String
        /// Sets this candidate's value of the entry's field; the default's is a no-op.
        let apply: (inout KernelSchedule) -> Void
        var schedule: KernelSchedule { var s = KernelSchedule.default; apply(&s); return s }

        /// One candidate per value of `values`, the default's value first.
        static func over<V: Equatable>(_ path: WritableKeyPath<KernelSchedule, V>, _ values: [V], label: (V) -> String) -> [Candidate] {
            let current = KernelSchedule.default[keyPath: path]
            let ordered = [current] + values.filter { $0 != current }
            return ordered.map { v in Candidate(label: label(v)) { $0[keyPath: path] = v } }
        }
    }

    struct Entry {
        let name: String
        /// `candidates[0]` is the default.
        let candidates: [Candidate]
        /// Distinct layers the workload rotates over; sample i runs layer i.
        let layers: Int
        /// Encodes the entry's kernels for one layer with the candidate's kernels.
        let encode: (MTLComputeCommandEncoder, Kernels, _ layer: Int) throws -> Void
        /// Every output the workload writes for layer 0.
        let outputs: () -> [Float]
        /// Writes the seeded inputs, including any the workload mutates in place.
        let seedInputs: () -> Void
    }

    /// Relative L2 of `b` against `a`: sqrt(sum (a-b)^2 / sum a^2).
    public static func relL2(_ a: [Float], _ b: [Float]) -> Double {
        var num = 0.0, den = 0.0
        for (x, y) in zip(a, b) { num += Double(x - y) * Double(x - y); den += Double(x) * Double(x) }
        return (num / max(den, 1e-30)).squareRoot()
    }

    typealias CommandBufferMaker = () -> (MTLCommandBuffer, MTLComputeCommandEncoder)

    /// Encodes `layers` samples of `entry` in one command buffer and returns
    /// its GPU seconds. Throws when a kernel is missing or the GPU faults.
    private static func sample(_ entry: Entry, _ kernels: Kernels, layers: Int, _ makeCommandBuffer: CommandBufferMaker) throws -> Double {
        let (cb, enc) = makeCommandBuffer()
        do {
            for i in 0..<layers { try entry.encode(enc, kernels, i) }
        } catch {
            enc.endEncoding()
            throw error
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("GPU error: \(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    struct Accepted {
        let candidate: Candidate
        let kernels: Kernels
        let relL2: Double
    }

    /// Runs the default once on the seeded inputs, then every other candidate,
    /// and splits them by whether their outputs are finite and within
    /// `tolerance` of the default's. Nothing is timed.
    static func validate(_ entry: Entry, ctx: MetalContext, precision: Precision, makeCommandBuffer: CommandBufferMaker, tolerance: Double) throws
        -> (accepted: [Accepted], rejected: [TuningEntryResult.Rejected])
    {
        let defaultKernels = Kernels(ctx: ctx, precision: precision, schedule: entry.candidates[0].schedule)
        entry.seedInputs()
        _ = try sample(entry, defaultKernels, layers: 1, makeCommandBuffer)
        let reference = entry.outputs()
        guard reference.allSatisfy({ $0.isFinite }) else { throw EngineError.badSchedule("\(entry.name): the default schedule produced non-finite outputs") }
        var accepted = [Accepted(candidate: entry.candidates[0], kernels: defaultKernels, relL2: 0)]
        var rejected: [TuningEntryResult.Rejected] = []
        for candidate in entry.candidates.dropFirst() {
            let kernels = Kernels(ctx: ctx, precision: precision, schedule: candidate.schedule)
            entry.seedInputs()
            do {
                _ = try sample(entry, kernels, layers: 1, makeCommandBuffer)
            } catch {
                rejected.append(.init(label: candidate.label, reason: "did not run: \(error)", relL2: nil))
                continue
            }
            let got = entry.outputs()
            guard got.count == reference.count, got.allSatisfy({ $0.isFinite }) else {
                rejected.append(.init(label: candidate.label, reason: "non-finite output", relL2: nil))
                continue
            }
            let err = relL2(reference, got)
            if err <= tolerance {
                accepted.append(Accepted(candidate: candidate, kernels: kernels, relL2: err))
            } else {
                rejected.append(.init(label: candidate.label, reason: String(format: "relative L2 %.2e above %.0e", err, tolerance), relL2: err))
            }
        }
        return (accepted, rejected)
    }

    /// Validates and times every entry in order, within the wall-clock budget,
    /// and returns the schedule built from the choices.
    static func run(entries: [Entry], ctx: MetalContext, precision: Precision, key: TuningKey,
                    makeCommandBuffer: CommandBufferMaker, options: Options) throws -> TuningReport {
        let start = Date()
        var schedule = KernelSchedule.default
        var results: [TuningEntryResult] = []
        for entry in entries {
            let defaultLabel = entry.candidates[0].label
            func kept(_ rounds: Int, rejected: [TuningEntryResult.Rejected], defaultMs: Double) -> TuningEntryResult {
                TuningEntryResult(name: entry.name, defaultLabel: defaultLabel, chosenLabel: defaultLabel, defaultMs: defaultMs, chosenMs: defaultMs,
                                  gainPercent: 0, relL2: 0, rounds: rounds, rejected: rejected)
            }
            if Date().timeIntervalSince(start) >= options.budgetSeconds {
                options.progress?("\(entry.name): budget spent, not measured, kept \(defaultLabel)")
                results.append(kept(0, rejected: [], defaultMs: 0))
                continue
            }
            options.progress?("\(entry.name): validating \(entry.candidates.count) candidates")
            let (accepted, rejected) = try validate(entry, ctx: ctx, precision: precision, makeCommandBuffer: makeCommandBuffer, tolerance: options.tolerance)
            for r in rejected { options.progress?("\(entry.name): rejected \(r.label), \(r.reason)") }
            _ = try sample(entry, accepted[0].kernels, layers: entry.layers, makeCommandBuffer)
            var seconds = [[Double]](repeating: [], count: accepted.count)
            var rounds = 0
            while rounds < options.rounds {
                for (i, a) in accepted.enumerated() {
                    seconds[i].append(try sample(entry, a.kernels, layers: entry.layers, makeCommandBuffer))
                }
                rounds += 1
                if rounds >= min(2, options.rounds), Date().timeIntervalSince(start) >= options.budgetSeconds { break }
            }
            let perSampleMs = seconds.map { median($0) / Double(entry.layers) * 1000 }
            let defaultMs = perSampleMs[0]
            var best = 0
            for i in 1..<accepted.count where perSampleMs[i] < perSampleMs[best] { best = i }
            let gain = (defaultMs - perSampleMs[best]) / defaultMs * 100
            if best != 0, gain >= options.minGainPercent {
                let chosen = accepted[best]
                chosen.candidate.apply(&schedule)
                results.append(TuningEntryResult(name: entry.name, defaultLabel: defaultLabel, chosenLabel: chosen.candidate.label, defaultMs: defaultMs,
                                                 chosenMs: perSampleMs[best], gainPercent: gain, relL2: chosen.relL2, rounds: rounds, rejected: rejected))
                options.progress?(String(format: "%@: %@ %.3f ms -> %@ %.3f ms (%.1f%% faster) over %d rounds", entry.name, defaultLabel, defaultMs, chosen.candidate.label, perSampleMs[best], gain, rounds))
            } else {
                results.append(kept(rounds, rejected: rejected, defaultMs: defaultMs))
                options.progress?(String(format: "%@: kept %@ at %.3f ms (best other %@ %.3f ms, %.1f%%) over %d rounds", entry.name, defaultLabel, defaultMs,
                                         accepted[best].candidate.label, perSampleMs[best], gain, rounds))
            }
        }
        try schedule.validate()
        return TuningReport(key: key, created: Date(), schedule: schedule, entries: results, wallSeconds: Date().timeIntervalSince(start))
    }

    static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// Tunes `model`, applies the schedule and saves it under `cacheDirectory`.
    /// Run it before any generation session: the workloads overwrite the
    /// scratch buffers, KV caches and recurrent states.
    @discardableResult
    public static func tune(model: DramaModel, options: Options = Options(), cacheDirectory: URL = TuningCache.defaultDirectory) throws -> TuningReport {
        let entries = try census(model: model)
        let report = try run(entries: entries, ctx: model.ctx, precision: model.precision, key: model.tuningKey,
                             makeCommandBuffer: model.commandBuffer, options: options)
        try model.apply(report.schedule)
        let url = try TuningCache.save(report, directory: cacheDirectory)
        model.scheduleSource = .cache(url)
        return report
    }
}

/// Deterministic input noise: one seeded pool, copied into each buffer from
/// a moving offset, so filling a 100 MB cache is a memcpy.
struct Noise {
    private let pool: [Float]
    private var cursor = 0

    init(seed: UInt64, poolCount: Int = 1 << 20) {
        var rng = SeededGenerator(seed: seed)
        pool = (0..<poolCount).map { _ in Float(rng.uniformOpenZero()) * 2 - 1 }
    }

    /// Writes `count` values scaled by `scale` to `buf`, and returns the
    /// closure that writes the same values again.
    mutating func fill(_ buf: Buf, count: Int, scale: Float = 1) -> () -> Void {
        let offset = cursor
        cursor = (cursor + count) % pool.count
        let pool = self.pool
        let write = {
            var i = 0
            while i < count {
                let p = (offset + i) % pool.count
                let n = min(pool.count - p, count - i)
                pool.withUnsafeBufferPointer { src in
                    if scale == 1 {
                        memcpy(buf.floats + i, src.baseAddress! + p, n * 4)
                    } else {
                        for j in 0..<n { buf.floats[i + j] = src[p + j] * scale }
                    }
                }
                i += n
            }
        }
        write()
        return write
    }
}

extension Autotune {
    /// The nine workloads, on the model's resident weights and scratch buffers.
    static func census(model: DramaModel) throws -> [Entry] {
        let c = model.config, bb = model.backbone, tw = model.tower, weights = model.weights
        let H = c.hiddenSize
        let linearLayers = c.layerTypes.indices.filter { c.layerTypes[$0] == .linear }.prefix(10)
        let fullLayers = c.layerTypes.indices.filter { c.layerTypes[$0] == .full }.prefix(10)
        guard linearLayers.count == 10, fullLayers.count == 10, bb.maxRows >= 128 else { throw EngineError.badModel("the autotune census expects 10 linear and 10 full layers and 128-row chunks") }
        func linear(_ i: Int) -> LinearAttentionWeights {
            guard case .linear(let w) = weights.backbone.layers[linearLayers[i]].mixer else { preconditionFailure() }
            return w
        }
        func full(_ i: Int) -> FullAttentionWeights {
            guard case .full(let w) = weights.backbone.layers[fullLayers[i]].mixer else { preconditionFailure() }
            return w
        }
        var noise = Noise(seed: 2026)
        let capacityPairs = bb.maxRows * c.topK
        var entries: [Entry] = []

        func seeds(_ fills: [() -> Void]) -> () -> Void { { for f in fills { f() } } }

        // gemv.decode: every M = 1 projection of a linear layer plus the candidate logits.
        do {
            let seed = seeds([noise.fill(bb.h, count: H), noise.fill(bb.normed, count: 4096), noise.fill(bb.sharedAct, count: 512), noise.fill(bb.hiddenOut, count: H)])
            entries.append(Entry(
                name: "gemv.decode",
                candidates: Candidate.over(\.gemvDecode, KernelSchedule.gemvShapes, label: \.label),
                layers: 10,
                encode: { enc, k, i in
                    let lw = linear(i), mw = weights.backbone.layers[linearLayers[i]].moe
                    try k.gemv(enc, lw.inProjQKV, x: bb.h, y: bb.big, n: 8192, k: H, m: 1)
                    try k.gemv(enc, lw.inProjZ, x: bb.h, y: bb.z, n: 4096, k: H, m: 1)
                    try k.gemv(enc, lw.inProjA, x: bb.h, y: bb.a, n: 32, k: H, m: 1)
                    try k.gemv(enc, lw.inProjB, x: bb.h, y: bb.b, n: 32, k: H, m: 1)
                    try k.gemv(enc, lw.outProj, x: bb.normed, y: bb.mixOut, n: H, k: 4096, m: 1)
                    try k.gemv(enc, mw.router, x: bb.h, y: bb.routerLogits, n: c.numExperts, k: H, m: 1)
                    try k.gemv(enc, mw.sharedGate, x: bb.h, y: bb.sharedG, n: 512, k: H, m: 1)
                    try k.gemv(enc, mw.sharedUp, x: bb.h, y: bb.sharedU, n: 512, k: H, m: 1)
                    try k.gemv(enc, mw.sharedDown, x: bb.sharedAct, y: bb.sharedOut, n: H, k: 512, m: 1)
                    try k.gemv(enc, mw.sharedExpertGate, x: bb.h, y: bb.sharedGate, n: 1, k: H, m: 1)
                    try k.gemv(enc, weights.backbone.lmHead, x: bb.hiddenOut, y: bb.cand, n: 1, k: H, m: 1, rowStart: Int(c.eos))
                    try k.gemv(enc, weights.backbone.lmHead, x: bb.hiddenOut, y: bb.cand, yOffset: 1, n: c.codebookSize, k: H, m: 1, rowStart: Int(c.semanticStart))
                },
                outputs: {
                    bb.big.read(8192) + bb.z.read(4096) + bb.a.read(32) + bb.b.read(32) + bb.mixOut.read(H) + bb.routerLogits.read(c.numExperts)
                        + bb.sharedG.read(512) + bb.sharedU.read(512) + bb.sharedOut.read(H) + bb.sharedGate.read(1) + bb.cand.read(c.codebookSize + 1)
                },
                seedInputs: seed))
        }

        // gemv.narrow: the N <= 64 projections at 128 and 32 rows.
        do {
            let seed = seeds([noise.fill(bb.h, count: 128 * H)])
            entries.append(Entry(
                name: "gemv.narrow",
                candidates: Candidate.over(\.gemvNarrow, KernelSchedule.gemvShapes, label: \.label),
                layers: 10,
                encode: { enc, k, i in
                    let lw = linear(i), mw = weights.backbone.layers[linearLayers[i]].moe
                    for m in [128, 32] {
                        try k.gemv(enc, lw.inProjA, x: bb.h, y: bb.a, n: 32, k: H, m: m)
                        try k.gemv(enc, lw.inProjB, x: bb.h, y: bb.b, n: 32, k: H, m: m)
                        try k.gemv(enc, mw.sharedExpertGate, x: bb.h, y: bb.sharedGate, n: 1, k: H, m: m)
                    }
                },
                outputs: { bb.a.read(128 * 32) + bb.b.read(128 * 32) + bb.sharedGate.read(128) },
                seedInputs: seed))
        }

        // gemm.prefill: the wide projections of a linear and a full layer at 128 and 32 rows.
        do {
            let seed = seeds([noise.fill(bb.h, count: 128 * H), noise.fill(bb.normed, count: 128 * 4096)])
            entries.append(Entry(
                name: "gemm.prefill",
                candidates: Candidate.over(\.gemmTile, KernelSchedule.gemmTiles, label: \.label),
                layers: 10,
                encode: { enc, k, i in
                    let lw = linear(i), fw = full(i)
                    for m in [128, 32] {
                        try k.gemv(enc, lw.inProjQKV, x: bb.h, y: bb.big, n: 8192, k: H, m: m)
                        try k.gemv(enc, lw.inProjZ, x: bb.h, y: bb.z, n: 4096, k: H, m: m)
                        try k.gemv(enc, lw.outProj, x: bb.normed, y: bb.mixOut, n: H, k: 4096, m: m)
                        try k.gemv(enc, fw.qProj, x: bb.h, y: bb.convOut, n: c.numHeads * 512, k: H, m: m)
                        try k.gemv(enc, fw.oProj, x: bb.normed, y: bb.x, n: H, k: c.numHeads * 256, m: m)
                    }
                },
                outputs: { bb.big.read(128 * 8192) + bb.z.read(128 * 4096) + bb.mixOut.read(128 * H) + bb.convOut.read(128 * 8192) + bb.x.read(128 * H) },
                seedInputs: seed))
        }

        // moe.tiled and moe.rows: the routed experts, routed by the real router.
        func moeEntry(name: String, rowCounts: [Int], candidates: [Candidate]) -> Entry {
            let rows = rowCounts.max()!
            let seed = seeds([noise.fill(bb.h, count: rows * H)])
            return Entry(
                name: name,
                candidates: candidates,
                layers: 10,
                encode: { enc, k, i in
                    let mw = weights.backbone.layers[i].moe
                    for m in rowCounts {
                        let maxActive = min(m * c.topK, c.numExperts)
                        try k.gemv(enc, mw.router, x: bb.h, y: bb.routerLogits, n: c.numExperts, k: H, m: m)
                        try k.moeRouter(enc, ids: bb.ids, weights: bb.rw, logits: bb.routerLogits, rows: m, config: c)
                        try k.zero(enc, bb.counts)
                        try k.moeBucket(enc, counts: bb.counts, lists: bb.lists, ids: bb.ids, active: bb.active, pairs: m * c.topK, capacity: capacityPairs)
                        try k.moeGateUp(enc, act: bb.act, x: bb.h, table: mw.table, counts: bb.counts, lists: bb.lists, active: bb.active, maxActive: maxActive, capacity: capacityPairs, rows: m, config: c)
                        try k.moeDown(enc, slotOut: bb.slotOut, act: bb.act, weights: bb.rw, table: mw.table, counts: bb.counts, lists: bb.lists, active: bb.active, maxActive: maxActive, capacity: capacityPairs, rows: m, config: c)
                    }
                },
                outputs: { bb.act.read(rows * c.topK * c.moeIntermediate) + bb.slotOut.read(rows * c.topK * H) },
                seedInputs: seed)
        }
        entries.append(moeEntry(name: "moe.tiled", rowCounts: [128, 32], candidates: Candidate.over(\.moeTile, KernelSchedule.moeTiles, label: \.label)))
        entries.append(moeEntry(name: "moe.rows", rowCounts: [1], candidates: Candidate.over(\.moeRowSimdgroups, KernelSchedule.moeRowSimdgroups, label: { "\($0)sg" })))

        // attention.decode and attention.prefill over the backbone's full
        // layers and the tower's caches, 16 heads over 2 KV heads in both.
        func attentionEntry(name: String, rows: Int, pos0: Int, path: WritableKeyPath<KernelSchedule, Int>) -> Entry {
            let positions = pos0 + rows
            var fills = [noise.fill(bb.qn, count: rows * c.numHeads * 256), noise.fill(tw.tqn, count: rows * c.towerHeads * 256)]
            for i in 0..<10 {
                fills.append(noise.fill(bb.kcache[fullLayers[i]]!, count: positions * c.numKVHeads * 256))
                fills.append(noise.fill(bb.vcache[fullLayers[i]]!, count: positions * c.numKVHeads * 256))
            }
            for i in 0..<tw.kcache.count {
                fills.append(noise.fill(tw.kcache[i], count: positions * c.towerKVHeads * 256))
                fills.append(noise.fill(tw.vcache[i], count: positions * c.towerKVHeads * 256))
            }
            return Entry(
                name: name,
                candidates: Candidate.over(path, KernelSchedule.attentionSplits, label: { "split\($0)" }),
                layers: 10 + tw.kcache.count,
                encode: { enc, k, i in
                    if i < 10 {
                        let li = fullLayers[i]
                        try k.attention(enc, out: bb.attnOut, q: bb.qn, kcache: bb.kcache[li]!, vcache: bb.vcache[li]!, heads: c.numHeads, kvHeads: c.numKVHeads, rows: rows, pos0: pos0)
                    } else {
                        let t = i - 10
                        try k.attention(enc, out: tw.attnOut, q: tw.tqn, kcache: tw.kcache[t], vcache: tw.vcache[t], heads: c.towerHeads, kvHeads: c.towerKVHeads, rows: rows, pos0: pos0)
                    }
                },
                outputs: { bb.attnOut.read(rows * c.numHeads * 256) },
                seedInputs: seeds(fills))
        }
        let decodePos = min(1024, model.capacity - 2)
        entries.append(attentionEntry(name: "attention.decode", rows: 1, pos0: decodePos, path: \.attentionSplitDecode))
        entries.append(attentionEntry(name: "attention.prefill", rows: 128, pos0: min(256, model.capacity - 130), path: \.attentionSplitPrefill))

        // gdn.decode and gdn.prefill: the recurrence over random states.
        func gdnEntry(name: String, rows: Int, path: WritableKeyPath<KernelSchedule, Int>) -> Entry {
            let heads = c.linearValueHeads
            var fills = [noise.fill(bb.convOut, count: rows * 8192), noise.fill(bb.a, count: rows * heads), noise.fill(bb.b, count: rows * heads)]
            for i in 0..<10 { fills.append(noise.fill(bb.recState[linearLayers[i]]!, count: heads * 128 * 128, scale: 0.1)) }
            let state0 = bb.recState[linearLayers[0]]!
            return Entry(
                name: name,
                candidates: Candidate.over(path, KernelSchedule.gdnParts, label: { "p\($0)" }),
                layers: 10,
                encode: { enc, k, i in
                    let lw = linear(i)
                    try k.gdnRecurrent(enc, core: bb.core, convOut: bb.convOut, a: bb.a, b: bb.b, state: bb.recState[linearLayers[i]]!, aLog: lw.aLog, dtBias: lw.dtBias, rows: rows, config: c)
                },
                outputs: { bb.core.read(rows * heads * 128) + state0.read(state0.count) },
                seedInputs: seeds(fills))
        }
        entries.append(gdnEntry(name: "gdn.decode", rows: 1, path: \.gdnPartsDecode))
        entries.append(gdnEntry(name: "gdn.prefill", rows: 128, path: \.gdnPartsPrefill))
        return entries
    }
}
