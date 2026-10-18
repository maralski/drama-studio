import Foundation
import Metal

/// Loaded weights plus the two transformer stacks. One per process.
package final class DramaModel: @unchecked Sendable {
    package let ctx: MetalContext
    package let config: ModelConfig
    package let location: ModelLocation
    package let precision: Precision
    package let capacity: Int
    /// The kernel schedule both stacks run with; `apply` replaces it.
    package private(set) var schedule: KernelSchedule
    package enum ScheduleSource: Equatable, Sendable {
        /// Loaded from, or saved to, this tuning cache file.
        case cache(URL)
        /// No cache for this key; `Autotune.tune` would fill it.
        case `default`
        /// Passed in by the caller.
        case given
    }
    package internal(set) var scheduleSource: ScheduleSource
    package let tuningKey: TuningKey
    let store: SafeTensorsStore
    let weights: ModelWeights
    let backbone: Backbone
    let tower: Tower
    let sampler: SamplerState
    /// The frame command buffer a pipelined session committed last. A new
    /// session waits for it before it touches the shared buffers.
    private(set) var inFlight: MTLCommandBuffer?

    package static let defaultCapacity = 8192 + 2048
    /// Rows per prefill command buffer. Every chunk walks all 30 GB of expert
    /// weights once, so a 10 s reference clip (about 300 rows) should fit in one.
    package static let prefillChunkRows = 512

    /// States after the shared system prefix of the last request, so the next
    /// chunk with the same prefix skips that part of the prefill. The KV caches
    /// for those positions are still in place; only the GDN states move.
    struct PrefixCache {
        var rows: [Frame]
        var states: Backbone.StateSnapshot
    }
    var prefixCache: PrefixCache?
    package func clearPrefixCache() { prefixCache = nil }

    /// `schedule` nil means the tuning cache's schedule for this device and
    /// model when there is one, else the default; a value is used as given.
    package init(location: ModelLocation, precision: Precision = .exact, capacity: Int = DramaModel.defaultCapacity,
                schedule: KernelSchedule? = nil, tuningDirectory: URL = TuningCache.defaultDirectory) throws {
        ctx = try MetalContext()
        self.location = location
        self.precision = precision
        self.capacity = capacity
        config = try ModelConfig(url: location.directory.appendingPathComponent("config.json"))
        tuningKey = try TuningKey(ctx: ctx, location: location)
        store = try SafeTensorsStore(directory: location.directory, device: ctx.device)
        weights = try ModelWeights(store: store, config: config, device: ctx.device)
        let chosen: KernelSchedule
        if let schedule {
            try schedule.validate()
            chosen = schedule
            scheduleSource = .given
        } else if let hit = TuningCache.load(key: tuningKey, directory: tuningDirectory) {
            chosen = hit.report.schedule
            scheduleSource = .cache(hit.url)
        } else {
            chosen = .default
            scheduleSource = .default
        }
        self.schedule = chosen
        backbone = Backbone(ctx: ctx, config: config, weights: weights.backbone, codebooks: weights.tower.codebookEmbeddings, precision: precision, schedule: chosen, capacity: capacity, maxRows: DramaModel.prefillChunkRows)
        tower = Tower(ctx: ctx, config: config, weights: weights.tower, precision: precision, schedule: chosen, capacity: capacity, maxRows: DramaModel.prefillChunkRows)
        sampler = SamplerState(device: ctx.device)
    }

    /// Switches both stacks to `schedule` once the GPU is idle.
    package func apply(_ schedule: KernelSchedule) throws {
        try schedule.validate()
        drain()
        self.schedule = schedule
        scheduleSource = .given
        backbone.k = Kernels(ctx: ctx, precision: precision, schedule: schedule)
        tower.k = Kernels(ctx: ctx, precision: precision, schedule: schedule)
    }

    deinit { drain() }

    func track(_ cb: MTLCommandBuffer) { inFlight = cb }

    func drain() {
        inFlight?.waitUntilCompleted()
        inFlight = nil
    }

    /// GPU seconds for one backbone forward of `rows` rows at position 0.
    package func benchBackbone(rows: Int, skip: Set<String>) throws -> Double {
        backbone.skipStages = skip
        defer { backbone.skipStages = [] }
        for r in 0..<rows { let p = backbone.frames.ints + r * 11; p[0] = 1000 + Int32(r); for i in 1..<11 { p[i] = 0 } }
        let (cb, enc) = commandBuffer()
        try backbone.forward(enc, rows: rows, pos0: 0)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    /// One backbone forward of `rows` synthetic frames from position 0 on
    /// cleared states, with the current schedule. Even rows are text tokens,
    /// odd rows audio frames.
    private func probeForward(rows: Int) throws {
        backbone.reset()
        for r in 0..<rows {
            let p = backbone.frames.ints + r * 11
            if r % 2 == 0 {
                p[0] = 1000 + Int32(r)
                for i in 1..<11 { p[i] = 0 }
            } else {
                p[0] = config.semanticStart + Int32(r % config.codebookSize)
                for i in 1..<11 { p[i] = Int32((i * 13 + r) % 1024) }
            }
        }
        let (cb, enc) = commandBuffer()
        try backbone.forward(enc, rows: rows, pos0: 0)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
    }

    /// The final-norm output of the probe forward, for comparing schedules.
    package func probeHidden(rows: Int) throws -> [Float] {
        try probeForward(rows: rows)
        return backbone.hiddenOut.read(rows * config.hiddenSize)
    }

    /// The last layer's MoE input after the probe forward: its post-attention
    /// norm output, `rows` x `hiddenSize`, which the real router routes unevenly.
    package func probeMoEInput(rows: Int) throws -> [Float] {
        try probeForward(rows: rows)
        return backbone.h.read(rows * config.hiddenSize)
    }

    /// GPU seconds for one tower prefill of `rows` rows.
    package func benchTower(rows: Int) throws -> Double {
        var fake: [Frame] = []
        for r in 0..<rows {
            var f: Frame = [config.semanticStart + Int32(r)]
            for c in 0..<10 { f.append(Int32((c * 7 + r) % 1024)) }
            fake.append(f)
        }
        tower.setTargets(nextFrames: fake)
        let (cb, enc) = commandBuffer()
        try tower.prefill(enc, hidden: backbone.hiddenOut, frames: backbone.framesView(0), rows: rows, pos0: 0, codebooks: weights.tower.codebookEmbeddings)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    /// Per-kernel-group GPU milliseconds for one backbone forward of `rows` rows.
    package func profileBackbone(rows: Int) throws -> [(String, Double)] {
        _ = try benchBackbone(rows: rows, skip: [])
        return try backbone.profile(rows: rows, makeCommandBuffer: commandBuffer)
    }

    package func benchWarmUp() -> Double {
        let t0 = Date(); warmUp(); return Date().timeIntervalSince(t0)
    }

    func commandBuffer() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.useResources(store.shards, usage: .read)
        return (cb, enc)
    }

    /// Touches every weight page once, builds every pipeline state, and runs
    /// one command buffer that declares the shards, so the first request pays
    /// for none of them (the GPU wires the 41 GB on its first use). Shards are
    /// walked on parallel threads. Returns bytes walked.
    @discardableResult
    package func warmUp() -> Int {
        ctx.warmPipelines()
        let shards = store.shards
        defer {
            let (cb, enc) = commandBuffer()
            try? backbone.k.zero(enc, backbone.counts)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
        }
        var total = 0
        DispatchQueue.concurrentPerform(iterations: shards.count) { i in
            let shard = shards[i]
            madvise(shard.contents(), shard.length, MADV_WILLNEED)
            let p = shard.contents().assumingMemoryBound(to: UInt64.self)
            var sink: UInt64 = 0
            var j = 0
            let stride = 16384 / 8
            while j * 8 < shard.length { sink &+= p[j]; j += stride }
            withExtendedLifetime(sink) {}
            Self.sinkLock.lock(); Self.sinkSum &+= sink; total += shard.length; Self.sinkLock.unlock()
        }
        return total
    }
    private static var sinkSum: UInt64 = 0
    private static let sinkLock = NSLock()

}

/// Per-step logits the evaluation suite compares. Both arrays are empty in the
/// pipelined path (`trace == false`), where the logits never leave the GPU.
package struct StepTrace {
    package var candidateLogits: [Float]
    package var towerLogits: [[Float]]   // 9 x 4096
}

/// One sequence: a prompt, then frames appended one at a time. Owns the
/// position counter; the caches live in the model and are rewritten per session.
///
/// A frame is one command buffer: the main-token sampler, the tower with a
/// sampler after each head, then the backbone step for the new row. With
/// `trace == false` the next frame's command buffer is committed before the
/// current one is waited for, so consecutive `step` calls must pass the same
/// settings and the sampler the previous call returned.
package final class GenerationSession {
    package let model: DramaModel
    package private(set) var frames: [Frame] = []
    package let promptRows: Int
    private var lastHiddenRow = 0
    package var trace = false
    package private(set) var layerOutStep0: [[Float]] = []
    package private(set) var towerHiddenStep0: [[Float]] = []
    /// GPU time of the completed frame command buffers.
    package private(set) var gpuSeconds: Double = 0

    /// A committed frame command buffer. It samples the frame for `position`,
    /// writes it to frame row `position + 1`, and leaves the sampler's
    /// post-state in RNG slot `rngSlot`.
    private struct FrameJob {
        var commandBuffer: MTLCommandBuffer
        var position: Int
        var rngSlot: Int
        var params: SamplingParams
        /// The sampler state the next `step` must pass; nil until the previous
        /// frame's post-state is known.
        var startState: SIMD4<UInt64>?
    }
    private var inFlight: FrameJob?
    private var encodedFrames = 0
    private var finished = false

    var config: ModelConfig { model.config }
    var position: Int { frames.count - 1 }

    /// `prefixRows` leading rows are the shared system prefix: their states are
    /// reused when the previous session had the same prefix, and saved otherwise.
    package init(model: DramaModel, prompt: [Frame], prefixRows: Int = 0, trace: Bool = false) throws {
        self.model = model
        self.promptRows = prompt.count
        self.trace = trace
        guard prompt.count >= 1, prompt.count + 1 < model.capacity else { throw EngineError.badModel("prompt of \(prompt.count) rows does not fit") }
        model.drain()
        let usable = prefixRows > 0 && prefixRows < prompt.count ? prefixRows : 0
        if usable > 0, let cache = model.prefixCache, cache.rows.count == usable, cache.rows.elementsEqual(prompt.prefix(usable), by: ==) {
            model.backbone.restore(cache.states)
            try prefill(prompt, from: usable, snapshotAt: nil)
        } else {
            model.backbone.reset()
            try prefill(prompt, from: 0, snapshotAt: usable > 0 ? usable : nil)
        }
    }

    private func writeFrame(_ f: Frame, at row: Int) {
        precondition(f.count == 11)
        let p = model.backbone.frames.ints + row * 11
        for i in 0..<11 { p[i] = f[i] }
    }

    private func prefill(_ prompt: [Frame], from: Int, snapshotAt: Int?) throws {
        frames = prompt
        for (i, f) in prompt.enumerated() { writeFrame(f, at: i) }
        let R = prompt.count
        let chunk = model.backbone.maxRows
        var start = from
        while start < R {
            var end = min(start + chunk, R)
            // A chunk ends exactly at the prefix boundary so the states can be saved there.
            if let s = snapshotAt, start < s, end > s { end = s }
            let M = end - start
            let pairedEnd = min(end, R - 1)
            let (cb, enc) = model.commandBuffer()
            let isLast = end == R
            model.backbone.captureLayers = trace && isLast
            model.backbone.captureRow = M - 1
            try model.backbone.forward(enc, rows: M, pos0: start)
            if pairedEnd > start {
                model.tower.setTargets(nextFrames: Array(prompt[(start + 1)...pairedEnd]))
                try model.tower.prefill(enc, hidden: model.backbone.hiddenOut, frames: model.backbone.framesView(start), rows: pairedEnd - start, pos0: start, codebooks: model.weights.tower.codebookEmbeddings)
            }
            if isLast { try model.backbone.candidateLogits(enc, row0: M - 1, rows: 1) }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("GPU error during prefill: \(e)") }
            lastHiddenRow = M - 1
            if end == snapshotAt { model.prefixCache = DramaModel.PrefixCache(rows: Array(prompt.prefix(end)), states: model.backbone.snapshot()) }
            start = end
        }
        if trace { layerOutStep0 = model.backbone.layerCapture.map { $0.read(config.hiddenSize) } }
        model.backbone.captureLayers = false
    }

    /// Samples the next frame. The frame is nil when the model emits eos, after
    /// which every call returns nil without touching the GPU. In trace mode the
    /// trace holds the candidate logits, and the head logits when a frame was
    /// produced.
    package func step(temperature: Double, topK: Int, topP: Double, sampler: inout Sampler) throws -> (frame: Frame?, trace: StepTrace) {
        guard !finished else { return (nil, StepTrace(candidateLogits: [], towerLogits: [])) }
        let params = SamplingParams(temperature: temperature, topK: topK, topP: topP)
        var cand: [Float] = []
        let job: FrameJob
        if let pending = inFlight {
            guard pending.params == params, pending.startState == sampler.state else {
                throw EngineError.badModel("sampling settings or sampler state changed while a frame was in flight")
            }
            job = pending
        } else {
            if trace { cand = model.backbone.readCandidateLogits() }
            guard let encoded = try encodeFrame(params: params, state: sampler.state) else { throw EngineError.badModel("sequence capacity exceeded") }
            job = encoded
        }
        inFlight = trace ? nil : try encodeFrame(params: params, state: nil)
        job.commandBuffer.waitUntilCompleted()
        if let e = job.commandBuffer.error { throw EngineError.badModel("GPU error in decode: \(e)") }
        gpuSeconds += job.commandBuffer.gpuEndTime - job.commandBuffer.gpuStartTime
        sampler.state = model.sampler.state(slot: job.rngSlot)
        inFlight?.startState = sampler.state
        guard model.sampler.statusWord(slot: job.rngSlot) == 0 else { throw EngineError.badModel("non-finite logits at position \(job.position)") }
        let frame = model.backbone.frames.readInts(11, offset: (job.position + 1) * 11)
        if frame[0] == config.eos {
            finished = true
            return (nil, StepTrace(candidateLogits: cand, towerLogits: []))
        }
        frames.append(frame)
        lastHiddenRow = 0
        var tower: [[Float]] = []
        if trace {
            tower = (0..<(config.numCodebooks - 1)).map { model.tower.readHeadLogits(head: $0) }
            if job.position == promptRows - 1 { towerHiddenStep0 = model.tower.layerCapture.map { $0.read(config.hiddenSize) } }
        }
        return (frame, StepTrace(candidateLogits: cand, towerLogits: tower))
    }

    /// Encodes and commits the command buffer for the next frame the GPU has
    /// not started: samplers, tower and backbone step. `state` seeds the RNG
    /// slot it starts from; nil continues the chain left by the previous job.
    /// Returns nil when the frame would not fit the capacity.
    private func encodeFrame(params: SamplingParams, state: SIMD4<UInt64>?) throws -> FrameJob? {
        let k = encodedFrames
        let p = promptRows - 1 + k
        guard p + 2 < model.capacity else { return nil }
        let slotIn = k % 2, slotOut = (k + 1) % 2
        if let state { model.sampler.setState(state, slot: slotIn) }
        let n = config.codebookSize + 1
        let kernels = model.tower.k
        let (cb, enc) = model.commandBuffer()
        try kernels.sampleMain(enc, logits: model.backbone.cand, rng: model.sampler.rng, frames: model.backbone.frames,
                               code0: model.tower.codes[0], semantic: model.tower.semantic, status: model.sampler.status,
                               params: kernelParams(params, n: n, rngIn: slotIn, rngOut: slotOut, position: p, head: 0))
        try encodeTower(enc, position: p, hiddenRow: k == 0 ? lastHiddenRow : 0) { head in
            try kernels.sampleTower(enc, logits: model.tower.headLogits, logitsOffset: head * config.codebookSize, rng: model.sampler.rng,
                                    frames: model.backbone.frames, codeOut: model.tower.codes[head + 1], status: model.sampler.status,
                                    params: kernelParams(params, n: config.codebookSize, rngIn: slotOut, rngOut: slotOut, position: p, head: head))
        }
        try model.backbone.forward(enc, rows: 1, pos0: p + 1)
        try model.backbone.candidateLogits(enc, row0: 0, rows: 1)
        enc.endEncoding()
        cb.commit()
        model.track(cb)
        encodedFrames += 1
        return FrameJob(commandBuffer: cb, position: p, rngSlot: slotOut, params: params, startState: state)
    }

    private func kernelParams(_ params: SamplingParams, n: Int, rngIn: Int, rngOut: Int, position: Int, head: Int) -> SamplerKernelParams {
        SamplerKernelParams(params, n: n, rngIn: rngIn, rngOut: rngOut, position: position, head: head,
                            eos: config.eos, semanticStart: config.semanticStart, codebookSize: config.codebookSize)
    }

    /// The tower for position `p`: the shared layers, then residual layer i and
    /// head i for each of the nine heads, with `afterHead(i)` encoded between heads.
    private func encodeTower(_ enc: MTLComputeCommandEncoder, position p: Int, hiddenRow: Int, afterHead: (Int) throws -> Void) throws {
        model.tower.captureLayers = trace && p == promptRows - 1
        defer { model.tower.captureLayers = false }
        let frames = model.backbone.framesView(p)
        let codebooks = model.weights.tower.codebookEmbeddings
        for i in 0..<(config.numCodebooks - 1) {
            if i == 0 {
                try model.tower.decodeShared(enc, hidden: model.backbone.hiddenOut, hiddenRow: hiddenRow, frames: frames, pos: p, codebooks: codebooks)
            } else {
                try model.tower.residualStep(enc, i, frames: frames, pos: p, codebooks: codebooks)
            }
            try afterHead(i)
        }
    }

    /// Teacher forcing: feeds `frame` as the next frame and returns the logits
    /// the model produced for it. Not available while a pipelined frame is in flight.
    package func force(_ frame: Frame) throws -> StepTrace {
        guard inFlight == nil else { throw EngineError.badModel("force after a pipelined step") }
        let p = position
        let cand = model.backbone.readCandidateLogits()
        model.tower.setMain(row: 0, main: frame[0])
        for i in 1..<(config.numCodebooks - 1) { model.tower.setCode(row: 0, index: i, value: frame[1 + i]) }
        let (cb, enc) = model.commandBuffer()
        try encodeTower(enc, position: p, hiddenRow: lastHiddenRow) { _ in }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("GPU error in tower: \(e)") }
        let tower = (0..<(config.numCodebooks - 1)).map { model.tower.readHeadLogits(head: $0) }
        if trace && p == promptRows - 1 { towerHiddenStep0 = model.tower.layerCapture.map { $0.read(config.hiddenSize) } }
        try advance(frame)
        return StepTrace(candidateLogits: cand, towerLogits: tower)
    }

    /// The backbone's final-norm output and the candidate logits for the GPU's
    /// current position: the CPU's position, or one frame ahead while a
    /// pipelined frame is in flight.
    package var lastHidden: [Float] {
        model.drain()
        return model.backbone.hiddenOut.read(config.hiddenSize, offset: lastHiddenRow * config.hiddenSize)
    }

    package var currentCandidateLogits: [Float] {
        model.drain()
        return model.backbone.readCandidateLogits()
    }

    /// Appends the frame and runs the backbone on it, leaving the next
    /// candidate logits ready.
    private func advance(_ frame: Frame) throws {
        let next = frames.count
        guard next + 1 < model.capacity else { throw EngineError.badModel("sequence capacity exceeded") }
        frames.append(frame)
        writeFrame(frame, at: next)
        let (cb, enc) = model.commandBuffer()
        try model.backbone.forward(enc, rows: 1, pos0: next)
        try model.backbone.candidateLogits(enc, row0: 0, rows: 1)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("GPU error in backbone: \(e)") }
        lastHiddenRow = 0
    }
}
