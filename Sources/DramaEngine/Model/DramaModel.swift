import Foundation
import Metal

/// Loaded weights plus the two transformer stacks. One per process.
public final class DramaModel: @unchecked Sendable {
    public let ctx: MetalContext
    public let config: ModelConfig
    public let location: ModelLocation
    public let precision: Precision
    public let capacity: Int
    let store: SafeTensorsStore
    let weights: ModelWeights
    let backbone: Backbone
    let tower: Tower

    public static let defaultCapacity = 8192 + 2048

    public init(location: ModelLocation, precision: Precision = .exact, capacity: Int = DramaModel.defaultCapacity) throws {
        ctx = try MetalContext()
        self.location = location
        self.precision = precision
        self.capacity = capacity
        config = try ModelConfig(url: location.directory.appendingPathComponent("config.json"))
        store = try SafeTensorsStore(directory: location.directory, device: ctx.device)
        weights = try ModelWeights(store: store, config: config, device: ctx.device)
        backbone = Backbone(ctx: ctx, config: config, weights: weights.backbone, codebooks: weights.tower.codebookEmbeddings, precision: precision, capacity: capacity)
        tower = Tower(ctx: ctx, config: config, weights: weights.tower, precision: precision, capacity: capacity)
    }

    func commandBuffer() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.useResources(store.shards, usage: .read)
        return (cb, enc)
    }

    /// Touches every weight page once so the first request does not pay for
    /// page faults. Returns the number of bytes walked.
    @discardableResult
    public func warmUp() -> Int {
        var total = 0
        for shard in store.shards {
            let p = shard.contents().assumingMemoryBound(to: UInt8.self)
            let stride = 16384
            var sink: UInt8 = 0
            var i = 0
            while i < shard.length { sink &+= p[i]; i += stride }
            total += shard.length
            _ = sink
        }
        return total
    }
}

/// Per-step logits the evaluation suite compares.
public struct StepTrace {
    public var candidateLogits: [Float]
    public var towerLogits: [[Float]]   // 9 x 4096
}

/// One sequence: a prompt, then frames appended one at a time. Owns the
/// position counter; the caches live in the model and are rewritten per session.
public final class GenerationSession {
    public let model: DramaModel
    public private(set) var frames: [Frame] = []
    public let promptRows: Int
    private var lastHiddenRow = 0
    private var mainSampler: MainTokenSampler
    public var trace = false
    public private(set) var layerOutStep0: [[Float]] = []
    public private(set) var towerHiddenStep0: [[Float]] = []

    var config: ModelConfig { model.config }
    var position: Int { frames.count - 1 }

    public init(model: DramaModel, prompt: [Frame], trace: Bool = false) throws {
        self.model = model
        self.promptRows = prompt.count
        self.trace = trace
        mainSampler = MainTokenSampler(config: model.config)
        guard prompt.count >= 1, prompt.count + 1 < model.capacity else { throw EngineError.badModel("prompt of \(prompt.count) rows does not fit") }
        model.backbone.reset()
        try prefill(prompt)
    }

    private func writeFrame(_ f: Frame, at row: Int) {
        precondition(f.count == 11)
        let p = model.backbone.frames.ints + row * 11
        for i in 0..<11 { p[i] = f[i] }
    }

    private func prefill(_ prompt: [Frame]) throws {
        frames = prompt
        for (i, f) in prompt.enumerated() { writeFrame(f, at: i) }
        let R = prompt.count
        let chunk = model.backbone.maxRows
        var start = 0
        while start < R {
            let end = min(start + chunk, R)
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
            start = end
        }
        if trace { layerOutStep0 = model.backbone.layerCapture.map { $0.read(config.hiddenSize) } }
        model.backbone.captureLayers = false
    }

    /// Samples the next frame. The frame is nil when the model emits eos; the
    /// trace always holds the candidate logits, and the head logits when a
    /// frame was produced.
    public func step(temperature: Double, topK: Int, topP: Double, sampler: inout Sampler) throws -> (frame: Frame?, trace: StepTrace) {
        let p = position
        let cand = model.backbone.readCandidateLogits(row: 0)
        guard cand.allSatisfy({ $0.isFinite }) else { throw EngineError.badModel("non-finite logits at position \(p)") }
        let history = frames[...p].map { $0[0] }
        let main = mainSampler.sample(candidateLogits: cand, history: history[...], temperature: temperature, topK: topK, topP: topP, sampler: &sampler)
        if main == config.eos { return (nil, StepTrace(candidateLogits: cand, towerLogits: [])) }
        var s = sampler
        let (frame, tower) = try runTower(main: main) { _, logits in Int32(s.sample(logits, temperature: temperature, topK: topK, topP: topP)) }
        sampler = s
        try advance(frame)
        return (frame, StepTrace(candidateLogits: cand, towerLogits: tower))
    }

    /// Teacher forcing: feeds `frame` as the next frame and returns the logits
    /// the model produced for it.
    public func force(_ frame: Frame) throws -> StepTrace {
        let cand = model.backbone.readCandidateLogits(row: 0)
        let (_, tower) = try runTower(main: frame[0]) { i, _ in frame[2 + i] }
        try advance(frame)
        return StepTrace(candidateLogits: cand, towerLogits: tower)
    }

    /// Runs the tower for the current position with `main` fixed; `choose`
    /// picks code i + 1 from head i's logits.
    private func runTower(main: Int32, choose: (Int, [Float]) throws -> Int32) throws -> (Frame, [[Float]]) {
        let p = position
        var frame: Frame = [main, max(0, min(Int32(config.codebookSize - 1), main - config.semanticStart))]
        model.tower.setMain(row: 0, main: main)
        model.tower.captureLayers = trace && p == promptRows - 1
        var tower: [[Float]] = []
        for i in 0..<(config.numCodebooks - 1) {
            let logits = try towerLogits(i, pos: p)
            tower.append(logits)
            let code = try choose(i, logits)
            frame.append(code)
            if i + 1 < config.numCodebooks - 1 { model.tower.setCode(row: 0, index: i + 1, value: code) }
        }
        if model.tower.captureLayers { towerHiddenStep0 = model.tower.layerCapture.map { $0.read(config.hiddenSize) } }
        model.tower.captureLayers = false
        return (frame, tower)
    }

    public var lastHidden: [Float] { model.backbone.hiddenOut.read(config.hiddenSize, offset: lastHiddenRow * config.hiddenSize) }

    public var currentCandidateLogits: [Float] { model.backbone.readCandidateLogits(row: 0) }

    private func towerLogits(_ i: Int, pos p: Int) throws -> [Float] {
        let (cb, enc) = model.commandBuffer()
        if i == 0 {
            try model.tower.decodeShared(enc, hidden: model.backbone.hiddenOut, hiddenRow: lastHiddenRow, frames: model.backbone.framesView(p), pos: p, codebooks: model.weights.tower.codebookEmbeddings)
        } else {
            try model.tower.residualStep(enc, i, frames: model.backbone.framesView(p), pos: p, codebooks: model.weights.tower.codebookEmbeddings)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("GPU error in tower: \(e)") }
        return model.tower.readHeadLogits()
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
