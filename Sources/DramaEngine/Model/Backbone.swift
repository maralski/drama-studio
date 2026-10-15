import Foundation
import Metal

/// The 40-layer hybrid MoE backbone with its decode state. `forward` encodes
/// one chunk of up to `maxRows` consecutive positions into a command encoder.
final class Backbone {
    let config: ModelConfig
    let weights: BackboneWeights
    let codebooks: TensorRef
    let k: Kernels
    let maxRows: Int
    let capacity: Int

    // State
    private(set) var kcache: [Int: Buf] = [:]
    private(set) var vcache: [Int: Buf] = [:]
    private(set) var convState: [Int: Buf] = [:]
    private(set) var recState: [Int: Buf] = [:]

    // Work buffers, sized for maxRows
    let frames: Buf          // [capacity, 11] int32, the whole sequence
    let x: Buf, h: Buf, big: Buf, z: Buf, a: Buf, b: Buf, convOut: Buf, core: Buf, normed: Buf
    let qn: Buf, kbuf: Buf, vbuf: Buf, kn: Buf, attnOut: Buf, mixOut: Buf
    let routerLogits: Buf, ids: Buf, rw: Buf, counts: Buf, lists: Buf, active: Buf, act: Buf, slotOut: Buf
    let sharedG: Buf, sharedU: Buf, sharedAct: Buf, sharedOut: Buf, sharedGate: Buf
    let hiddenOut: Buf       // [maxRows, 2048] final-norm output
    let candEos: Buf, candBand: Buf
    /// Residual stream after each layer for the first row (dump support).
    var captureLayers: Bool = false
    var captureRow: Int = 0
    /// Benchmark aid: stages named here are not encoded ("linear", "full", "moe").
    var skipStages: Set<String> = []
    private(set) var layerCapture: [Buf] = []

    init(ctx: MetalContext, config: ModelConfig, weights: BackboneWeights, codebooks: TensorRef, precision: Precision, capacity: Int, maxRows: Int = 128) {
        self.config = config
        self.weights = weights
        self.codebooks = codebooks
        self.k = Kernels(ctx: ctx, precision: precision)
        self.maxRows = maxRows
        self.capacity = capacity
        let d = ctx.device
        let H = config.hiddenSize
        let R = maxRows
        frames = Buf(d, floats: capacity * 11, label: "frames")
        x = Buf(d, floats: R * H, label: "x"); h = Buf(d, floats: R * H, label: "h")
        big = Buf(d, floats: R * 8192, label: "big"); z = Buf(d, floats: R * 4096, label: "z")
        a = Buf(d, floats: R * 32, label: "a"); b = Buf(d, floats: R * 32, label: "b")
        convOut = Buf(d, floats: R * 8192, label: "convOut"); core = Buf(d, floats: R * 4096, label: "core")
        normed = Buf(d, floats: R * 4096, label: "normed")
        qn = Buf(d, floats: R * 4096, label: "qn"); kbuf = Buf(d, floats: R * 512, label: "kbuf"); vbuf = Buf(d, floats: R * 512, label: "vbuf")
        kn = Buf(d, floats: R * 512, label: "kn"); attnOut = Buf(d, floats: R * 4096, label: "attnOut"); mixOut = Buf(d, floats: R * H, label: "mixOut")
        routerLogits = Buf(d, floats: R * config.numExperts, label: "routerLogits")
        ids = Buf(d, floats: R * config.topK, label: "ids"); rw = Buf(d, floats: R * config.topK, label: "rw")
        counts = Buf(d, floats: config.numExperts + 1, label: "counts"); lists = Buf(d, floats: config.numExperts * R * config.topK, label: "lists")
        active = Buf(d, floats: config.numExperts, label: "active")
        act = Buf(d, floats: R * config.topK * config.moeIntermediate, label: "act")
        slotOut = Buf(d, floats: R * config.topK * H, label: "slotOut")
        sharedG = Buf(d, floats: R * 512, label: "sharedG"); sharedU = Buf(d, floats: R * 512, label: "sharedU")
        sharedAct = Buf(d, floats: R * 512, label: "sharedAct"); sharedOut = Buf(d, floats: R * H, label: "sharedOut")
        sharedGate = Buf(d, floats: R, label: "sharedGate")
        hiddenOut = Buf(d, floats: R * H, label: "hiddenOut")
        candEos = Buf(d, floats: R, label: "candEos"); candBand = Buf(d, floats: R * config.codebookSize, label: "candBand")
        for (i, type) in config.layerTypes.enumerated() {
            switch type {
            case .full:
                kcache[i] = Buf(d, floats: capacity * config.numKVHeads * 256, label: "kcache.\(i)")
                vcache[i] = Buf(d, floats: capacity * config.numKVHeads * 256, label: "vcache.\(i)")
            case .linear:
                convState[i] = Buf(d, floats: 8192 * 3, label: "convState.\(i)")
                recState[i] = Buf(d, floats: config.linearValueHeads * 128 * 128, label: "recState.\(i)")
            }
        }
        layerCapture = (0..<config.numLayers).map { Buf(d, floats: H, label: "capture.\($0)") }
    }

    /// Clears the recurrent and conv states. KV caches are masked by position.
    func reset() {
        for s in convState.values { s.zero() }
        for s in recState.values { s.zero() }
    }

    struct StateSnapshot {
        var conv: [Int: [Float]]
        var rec: [Int: [Float]]
    }

    func snapshot() -> StateSnapshot {
        StateSnapshot(conv: convState.mapValues { $0.read($0.count) }, rec: recState.mapValues { $0.read($0.count) })
    }

    func restore(_ snap: StateSnapshot) {
        for (i, v) in snap.conv { v.withUnsafeBufferPointer { memcpy(convState[i]!.floats, $0.baseAddress!, v.count * 4) } }
        for (i, v) in snap.rec { v.withUnsafeBufferPointer { memcpy(recState[i]!.floats, $0.baseAddress!, v.count * 4) } }
    }

    func useWeights(_ enc: MTLComputeCommandEncoder, shards: [MTLBuffer]) {
        enc.useResources(shards, usage: .read)
    }

    /// Runs rows [pos0, pos0 + rows) whose frames are already in `frames`.
    /// Leaves the final-norm output in `hiddenOut`.
    func forward(_ enc: MTLComputeCommandEncoder, rows M: Int, pos0: Int) throws {
        precondition(M >= 1 && M <= maxRows)
        let H = config.hiddenSize
        let c = config
        if captureLayers { for buf in layerCapture { try k.zero(enc, buf) } }
        try k.embedMix(enc, out: x, frames: framesView(pos0), embed: weights.embed, codebooks: codebooks, rows: M, config: c)
        try k.roundIfMimic(enc, x: x, count: M * H)
        for (i, layer) in weights.layers.enumerated() {
            try k.rmsnorm(enc, out: h, outStride: H, x: x, view: .contiguous(H), rows: M, dim: H, w: layer.inputNorm, eps: c.rmsNormEps)
            try k.roundIfMimic(enc, x: h, count: M * H)
            switch layer.mixer {
            case .linear(let w): if !skipStages.contains("linear") { try linearAttention(enc, w, layer: i, rows: M) }
            case .full(let w): if !skipStages.contains("full") { try fullAttention(enc, w, layer: i, rows: M, pos0: pos0) }
            }
            try k.residualAdd(enc, x: x, y: mixOut, count: M * H)
            try k.roundIfMimic(enc, x: x, count: M * H)
            try k.rmsnorm(enc, out: h, outStride: H, x: x, view: .contiguous(H), rows: M, dim: H, w: layer.postNorm, eps: c.rmsNormEps)
            try k.roundIfMimic(enc, x: h, count: M * H)
            if !skipStages.contains("moe") { try moe(enc, layer.moe, rows: M) }
            if captureLayers {
                try k.residualAdd(enc, x: layerCapture[i], y: Buf(view: x, byteOffset: captureRow * H * 4), count: H)
            }
        }
        try k.rmsnorm(enc, out: hiddenOut, outStride: H, x: x, view: .contiguous(H), rows: M, dim: H, w: weights.finalNorm, eps: c.rmsNormEps)
        try k.roundIfMimic(enc, x: hiddenOut, count: M * H)
    }

    /// Logits of the 4097 candidate ids (eos, then the semantic band) for
    /// `hiddenOut` rows [row0, row0 + rows).
    func candidateLogits(_ enc: MTLComputeCommandEncoder, row0: Int, rows M: Int) throws {
        let H = config.hiddenSize
        try k.gemv(enc, weights.lmHead, x: hiddenOut, xOffset: row0 * H, y: candEos, n: 1, k: H, m: M, rowStart: Int(config.eos))
        try k.gemv(enc, weights.lmHead, x: hiddenOut, xOffset: row0 * H, y: candBand, n: config.codebookSize, k: H, m: M, rowStart: Int(config.semanticStart))
    }

    func readCandidateLogits(row: Int) -> [Float] {
        var out = [Float](repeating: 0, count: config.codebookSize + 1)
        out[0] = candEos.floats[row]
        let band = candBand.floats + row * config.codebookSize
        for i in 0..<config.codebookSize { out[i + 1] = band[i] }
        return out
    }

    func framesView(_ pos0: Int) -> Buf { Buf(view: frames, byteOffset: pos0 * 11 * 4) }

    private func linearAttention(_ enc: MTLComputeCommandEncoder, _ w: LinearAttentionWeights, layer: Int, rows M: Int) throws {
        let H = config.hiddenSize
        try k.gemv(enc, w.inProjQKV, x: h, y: big, n: 8192, k: H, m: M)
        try k.gemv(enc, w.inProjZ, x: h, y: z, n: 4096, k: H, m: M)
        try k.gemv(enc, w.inProjA, x: h, y: a, n: 32, k: H, m: M)
        try k.gemv(enc, w.inProjB, x: h, y: b, n: 32, k: H, m: M)
        try k.gdnConv(enc, out: convOut, mixed: big, state: convState[layer]!, w: w.conv, channels: 8192, rows: M)
        try k.roundIfMimic(enc, x: convOut, count: M * 8192)
        try k.gdnRecurrent(enc, core: core, convOut: convOut, a: a, b: b, state: recState[layer]!, aLog: w.aLog, dtBias: w.dtBias, rows: M, config: config)
        try k.roundIfMimic(enc, x: core, count: M * 4096)
        try k.gatedRMSNorm(enc, out: normed, x: core, z: z, w: w.norm, rows: M * config.linearValueHeads, eps: config.rmsNormEps)
        try k.roundIfMimic(enc, x: normed, count: M * 4096)
        try k.gemv(enc, w.outProj, x: normed, y: mixOut, n: H, k: 4096, m: M)
    }

    private func fullAttention(_ enc: MTLComputeCommandEncoder, _ w: FullAttentionWeights, layer: Int, rows M: Int, pos0: Int) throws {
        let H = config.hiddenSize
        let heads = config.numHeads, kvHeads = config.numKVHeads
        try k.gemv(enc, w.qProj, x: h, y: big, n: heads * 512, k: H, m: M)
        try k.gemv(enc, w.kProj, x: h, y: kbuf, n: kvHeads * 256, k: H, m: M)
        try k.gemv(enc, w.vProj, x: h, y: vbuf, n: kvHeads * 256, k: H, m: M)
        try k.rmsnorm(enc, out: qn, outStride: 256, x: big, view: HeadView(heads: UInt32(heads), groupStride: UInt32(heads * 512), headStride: 512),
                      rows: M * heads, dim: 256, w: w.qNorm, eps: config.rmsNormEps)
        try k.rmsnorm(enc, out: kn, outStride: 256, x: kbuf, view: HeadView(heads: UInt32(kvHeads), groupStride: UInt32(kvHeads * 256), headStride: 256),
                      rows: M * kvHeads, dim: 256, w: w.kNorm, eps: config.rmsNormEps)
        try k.roundIfMimic(enc, x: qn, count: M * heads * 256)
        try k.roundIfMimic(enc, x: kn, count: M * kvHeads * 256)
        try k.rope(enc, interleaved: false, x: qn, heads: heads, rows: M * heads, pos0: pos0, rotaryDim: config.rotaryDim, theta: config.ropeTheta)
        try k.rope(enc, interleaved: false, x: kn, heads: kvHeads, rows: M * kvHeads, pos0: pos0, rotaryDim: config.rotaryDim, theta: config.ropeTheta)
        try k.roundIfMimic(enc, x: qn, count: M * heads * 256)
        try k.roundIfMimic(enc, x: kn, count: M * kvHeads * 256)
        let kvView = HeadView(heads: UInt32(kvHeads), groupStride: UInt32(kvHeads * 256), headStride: 256)
        try k.kvWrite(enc, cache: kcache[layer]!, src: kn, srcOffset: 0, view: kvView, kvHeads: kvHeads, rows: M, pos0: pos0)
        try k.kvWrite(enc, cache: vcache[layer]!, src: vbuf, srcOffset: 0, view: kvView, kvHeads: kvHeads, rows: M, pos0: pos0)
        try k.attention(enc, out: attnOut, q: qn, kcache: kcache[layer]!, vcache: vcache[layer]!, heads: heads, kvHeads: kvHeads, rows: M, pos0: pos0)
        try k.roundIfMimic(enc, x: attnOut, count: M * heads * 256)
        try k.sigmoidGateMul(enc, out: normed, attn: attnOut, gate: big, gateOffset: 256,
                             view: HeadView(heads: UInt32(heads), groupStride: UInt32(heads * 512), headStride: 512), rows: M * heads)
        try k.roundIfMimic(enc, x: normed, count: M * heads * 256)
        try k.gemv(enc, w.oProj, x: normed, y: mixOut, n: H, k: heads * 256, m: M)
    }

    private func moe(_ enc: MTLComputeCommandEncoder, _ w: MoEWeights, rows M: Int) throws {
        let H = config.hiddenSize
        let capacityPairs = maxRows * config.topK
        try k.gemv(enc, w.router, x: h, y: routerLogits, n: config.numExperts, k: H, m: M)
        try k.moeRouter(enc, ids: ids, weights: rw, logits: routerLogits, rows: M, config: config)
        try k.zero(enc, counts)
        let maxActive = min(M * config.topK, config.numExperts)
        try k.moeBucket(enc, counts: counts, lists: lists, ids: ids, active: active, pairs: M * config.topK, capacity: capacityPairs)
        try k.moeGateUp(enc, act: act, x: h, table: w.table, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, config: config)
        try k.moeDown(enc, slotOut: slotOut, act: act, weights: rw, table: w.table, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, config: config)
        try k.gemv(enc, w.sharedGate, x: h, y: sharedG, n: 512, k: H, m: M)
        try k.gemv(enc, w.sharedUp, x: h, y: sharedU, n: 512, k: H, m: M)
        try k.siluMul(enc, out: sharedAct, a: sharedG, b: sharedU, count: M * 512)
        try k.roundIfMimic(enc, x: sharedAct, count: M * 512)
        try k.gemv(enc, w.sharedDown, x: sharedAct, y: sharedOut, n: H, k: 512, m: M)
        try k.gemv(enc, w.sharedExpertGate, x: h, y: sharedGate, n: 1, k: H, m: M)
        try k.moeCombine(enc, x: x, slotOut: slotOut, sharedOut: sharedOut, sharedGate: sharedGate, rows: M, config: config)
    }
}
