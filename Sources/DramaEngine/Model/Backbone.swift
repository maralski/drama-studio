import Foundation
import Metal

/// The 40-layer hybrid MoE backbone with its decode state. `forward` encodes
/// one chunk of up to `maxRows` consecutive positions into a command encoder.
final class Backbone {
    let config: ModelConfig
    let weights: BackboneWeights
    let codebooks: TensorRef
    var k: Kernels
    let maxRows: Int
    let capacity: Int

    private(set) var kcache: [Int: Buf] = [:]
    private(set) var vcache: [Int: Buf] = [:]
    private(set) var convState: [Int: Buf] = [:]
    private(set) var recState: [Int: Buf] = [:]

    let frames: Buf          // [capacity, 11] int32, the whole sequence
    let x: Buf, h: Buf, big: Buf, z: Buf, a: Buf, b: Buf, convOut: Buf, core: Buf, normed: Buf
    let qn: Buf, kbuf: Buf, vbuf: Buf, kn: Buf, attnOut: Buf, mixOut: Buf
    let routerLogits: Buf, ids: Buf, rw: Buf, counts: Buf, lists: Buf, active: Buf, act: Buf, slotOut: Buf
    let sharedG: Buf, sharedU: Buf, sharedAct: Buf, sharedOut: Buf, sharedGate: Buf
    let hiddenOut: Buf       // [maxRows, 2048] final-norm output
    let cand: Buf            // [codebookSize + 1] candidate logits: eos, then the semantic band
    /// Residual stream after each layer for the first row (dump support).
    var captureLayers: Bool = false
    var captureRow: Int = 0
    /// Benchmark aid: stages named here are not encoded: "linear", "full", "moe",
    /// or within them "gdn" (conv and recurrence), "attn" (the attention core),
    /// "experts" (the routed FP8 experts), "shared" (the shared expert).
    var skipStages: Set<String> = []
    private(set) var layerCapture: [Buf] = []

    init(ctx: MetalContext, config: ModelConfig, weights: BackboneWeights, codebooks: TensorRef, precision: Precision, schedule: KernelSchedule = .default, capacity: Int, maxRows: Int = 128) {
        self.config = config
        self.weights = weights
        self.codebooks = codebooks
        self.k = Kernels(ctx: ctx, precision: precision, schedule: schedule)
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
        cand = Buf(d, floats: config.codebookSize + 1, label: "cand")
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

    /// GPU milliseconds per forward of `rows` rows for each kernel group, each
    /// timed on its own after a real forward left its inputs in place, so the
    /// numbers are not confounded by the routing changing. A group's time is
    /// its per-layer time times the number of layers of that type.
    func profile(rows M: Int, makeCommandBuffer: () -> (MTLCommandBuffer, MTLComputeCommandEncoder)) throws -> [(String, Double)] {
        let H = config.hiddenSize
        let c = config
        let iterations = 5
        func time(_ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
            let (cb, enc) = makeCommandBuffer()
            for _ in 0..<iterations { try body(enc) }
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(iterations) * 1000
        }
        var out: [(String, Double)] = []
        func group(_ name: String, layers: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            out.append((name, try time(body) * Double(layers)))
        }
        guard let li = c.layerTypes.firstIndex(of: .linear), let fi = c.layerTypes.firstIndex(of: .full),
              case .linear(let lw) = weights.layers[li].mixer, case .full(let fw) = weights.layers[fi].mixer else { return out }
        let linearCount = c.layerTypes.filter { $0 == .linear }.count, fullCount = c.layerTypes.count - linearCount
        let layer = weights.layers[li]
        let mw = layer.moe
        let capacityPairs = maxRows * c.topK
        let maxActive = min(M * c.topK, c.numExperts)
        let heads = c.numHeads, kvHeads = c.numKVHeads
        let kvView = HeadView(heads: UInt32(kvHeads), groupStride: UInt32(kvHeads * 256), headStride: 256)
        try group("norms+residual", layers: c.numLayers) { enc in
            try k.rmsnorm(enc, out: h, outStride: H, x: x, view: .contiguous(H), rows: M, dim: H, w: layer.inputNorm, eps: c.rmsNormEps)
            try k.residualAdd(enc, x: x, y: mixOut, count: M * H)
            try k.rmsnorm(enc, out: h, outStride: H, x: x, view: .contiguous(H), rows: M, dim: H, w: layer.postNorm, eps: c.rmsNormEps)
        }
        try group("linear.gemm.qkv", layers: linearCount) { enc in try k.gemv(enc, lw.inProjQKV, x: h, y: big, m: M) }
        try group("linear.gemm.z", layers: linearCount) { enc in try k.gemv(enc, lw.inProjZ, x: h, y: z, m: M) }
        try group("linear.gemm.ab", layers: linearCount) { enc in
            try k.gemv(enc, lw.inProjA, x: h, y: a, m: M)
            try k.gemv(enc, lw.inProjB, x: h, y: b, m: M)
        }
        try group("linear.conv", layers: linearCount) { enc in try k.gdnConv(enc, out: convOut, mixed: big, state: convState[li]!, w: lw.conv, channels: 8192, rows: M) }
        try group("linear.recurrent", layers: linearCount) { enc in
            try k.gdnRecurrent(enc, core: core, convOut: convOut, a: a, b: b, state: recState[li]!, aLog: lw.aLog, dtBias: lw.dtBias, rows: M, config: c)
        }
        try group("linear.gatednorm", layers: linearCount) { enc in try k.gatedRMSNorm(enc, out: normed, x: core, z: z, w: lw.norm, rows: M * c.linearValueHeads, eps: c.rmsNormEps) }
        try group("linear.gemm.out", layers: linearCount) { enc in try k.gemv(enc, lw.outProj, x: normed, y: mixOut, m: M) }
        try group("full.gemm.qkv", layers: fullCount) { enc in
            try k.gemv(enc, fw.qProj, x: h, y: big, m: M)
            try k.gemv(enc, fw.kProj, x: h, y: kbuf, m: M)
            try k.gemv(enc, fw.vProj, x: h, y: vbuf, m: M)
        }
        try group("full.norm.rope.kv", layers: fullCount) { enc in
            try k.rmsnorm(enc, out: qn, outStride: 256, x: big, view: HeadView(heads: UInt32(heads), groupStride: UInt32(heads * 512), headStride: 512), rows: M * heads, dim: 256, w: fw.qNorm, eps: c.rmsNormEps)
            try k.rmsnorm(enc, out: kn, outStride: 256, x: kbuf, view: kvView, rows: M * kvHeads, dim: 256, w: fw.kNorm, eps: c.rmsNormEps)
            try k.rope(enc, interleaved: false, x: qn, heads: heads, rows: M * heads, pos0: 0, rotaryDim: c.rotaryDim, theta: c.ropeTheta)
            try k.rope(enc, interleaved: false, x: kn, heads: kvHeads, rows: M * kvHeads, pos0: 0, rotaryDim: c.rotaryDim, theta: c.ropeTheta)
            try k.kvWrite(enc, cache: kcache[fi]!, src: kn, srcOffset: 0, view: kvView, kvHeads: kvHeads, rows: M, pos0: 0)
            try k.kvWrite(enc, cache: vcache[fi]!, src: vbuf, srcOffset: 0, view: kvView, kvHeads: kvHeads, rows: M, pos0: 0)
        }
        try group("full.attention", layers: fullCount) { enc in
            try k.attention(enc, out: attnOut, q: qn, kcache: kcache[fi]!, vcache: vcache[fi]!, heads: heads, kvHeads: kvHeads, rows: M, pos0: 0)
        }
        try group("full.gate+gemm.out", layers: fullCount) { enc in
            try k.sigmoidGateMul(enc, out: normed, attn: attnOut, gate: big, gateOffset: 256, view: HeadView(heads: UInt32(heads), groupStride: UInt32(heads * 512), headStride: 512), rows: M * heads)
            try k.gemv(enc, fw.oProj, x: normed, y: mixOut, m: M)
        }
        try group("moe.router+bucket", layers: c.numLayers) { enc in
            try k.gemv(enc, mw.router, x: h, y: routerLogits, m: M)
            try k.moeRouter(enc, ids: ids, weights: rw, logits: routerLogits, rows: M, config: c)
            try k.zero(enc, counts)
            try k.moeBucket(enc, counts: counts, lists: lists, ids: ids, active: active, pairs: M * c.topK, capacity: capacityPairs)
        }
        try group("moe.gateup", layers: c.numLayers) { enc in
            try k.moeGateUp(enc, act: act, x: h, experts: mw.experts, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, rows: M, config: c)
        }
        try group("moe.down", layers: c.numLayers) { enc in
            try k.moeDown(enc, slotOut: slotOut, act: act, weights: rw, experts: mw.experts, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, rows: M, config: c)
        }
        try group("moe.shared", layers: c.numLayers) { enc in
            try k.gemv(enc, mw.sharedGate, x: h, y: sharedG, m: M)
            try k.gemv(enc, mw.sharedUp, x: h, y: sharedU, m: M)
            try k.siluMul(enc, out: sharedAct, a: sharedG, b: sharedU, count: M * 512)
            try k.gemv(enc, mw.sharedDown, x: sharedAct, y: sharedOut, m: M)
            try k.gemv(enc, mw.sharedExpertGate, x: h, y: sharedGate, m: M)
        }
        try group("moe.combine", layers: c.numLayers) { enc in
            try k.moeCombine(enc, x: x, slotOut: slotOut, sharedOut: sharedOut, sharedGate: sharedGate, rows: M, config: c)
        }
        return out
    }

    /// Logits of the 4097 candidate ids (eos, then the semantic band) for
    /// `hiddenOut` row `row0`, into `cand`.
    func candidateLogits(_ enc: MTLComputeCommandEncoder, row0: Int, rows M: Int) throws {
        precondition(M == 1)
        let H = config.hiddenSize
        try k.gemv(enc, weights.lmHeadEos, x: hiddenOut, xOffset: row0 * H, y: cand, m: 1)
        try k.gemv(enc, weights.lmHeadSemantic, x: hiddenOut, xOffset: row0 * H, y: cand, yOffset: 1, m: 1)
    }

    func readCandidateLogits() -> [Float] { cand.read(config.codebookSize + 1) }

    func framesView(_ pos0: Int) -> Buf { Buf(view: frames, byteOffset: pos0 * 11 * 4) }

    private func linearAttention(_ enc: MTLComputeCommandEncoder, _ w: LinearAttentionWeights, layer: Int, rows M: Int) throws {
        try k.gemv(enc, w.inProjQKV, x: h, y: big, m: M)
        try k.gemv(enc, w.inProjZ, x: h, y: z, m: M)
        try k.gemv(enc, w.inProjA, x: h, y: a, m: M)
        try k.gemv(enc, w.inProjB, x: h, y: b, m: M)
        if !skipStages.contains("gdn") {
            try k.gdnConv(enc, out: convOut, mixed: big, state: convState[layer]!, w: w.conv, channels: 8192, rows: M)
            try k.roundIfMimic(enc, x: convOut, count: M * 8192)
            try k.gdnRecurrent(enc, core: core, convOut: convOut, a: a, b: b, state: recState[layer]!, aLog: w.aLog, dtBias: w.dtBias, rows: M, config: config)
            try k.roundIfMimic(enc, x: core, count: M * 4096)
        }
        try k.gatedRMSNorm(enc, out: normed, x: core, z: z, w: w.norm, rows: M * config.linearValueHeads, eps: config.rmsNormEps)
        try k.roundIfMimic(enc, x: normed, count: M * 4096)
        try k.gemv(enc, w.outProj, x: normed, y: mixOut, m: M)
    }

    private func fullAttention(_ enc: MTLComputeCommandEncoder, _ w: FullAttentionWeights, layer: Int, rows M: Int, pos0: Int) throws {
        let heads = config.numHeads, kvHeads = config.numKVHeads
        try k.gemv(enc, w.qProj, x: h, y: big, m: M)
        try k.gemv(enc, w.kProj, x: h, y: kbuf, m: M)
        try k.gemv(enc, w.vProj, x: h, y: vbuf, m: M)
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
        if !skipStages.contains("attn") {
            try k.attention(enc, out: attnOut, q: qn, kcache: kcache[layer]!, vcache: vcache[layer]!, heads: heads, kvHeads: kvHeads, rows: M, pos0: pos0)
        }
        try k.roundIfMimic(enc, x: attnOut, count: M * heads * 256)
        try k.sigmoidGateMul(enc, out: normed, attn: attnOut, gate: big, gateOffset: 256,
                             view: HeadView(heads: UInt32(heads), groupStride: UInt32(heads * 512), headStride: 512), rows: M * heads)
        try k.roundIfMimic(enc, x: normed, count: M * heads * 256)
        try k.gemv(enc, w.oProj, x: normed, y: mixOut, m: M)
    }

    private func moe(_ enc: MTLComputeCommandEncoder, _ w: MoEWeights, rows M: Int) throws {
        let capacityPairs = maxRows * config.topK
        try k.gemv(enc, w.router, x: h, y: routerLogits, m: M)
        try k.moeRouter(enc, ids: ids, weights: rw, logits: routerLogits, rows: M, config: config)
        try k.zero(enc, counts)
        let maxActive = min(M * config.topK, config.numExperts)
        try k.moeBucket(enc, counts: counts, lists: lists, ids: ids, active: active, pairs: M * config.topK, capacity: capacityPairs)
        if !skipStages.contains("experts") {
            try k.moeGateUp(enc, act: act, x: h, experts: w.experts, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, rows: M, config: config)
            try k.moeDown(enc, slotOut: slotOut, act: act, weights: rw, experts: w.experts, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacityPairs, rows: M, config: config)
        }
        if !skipStages.contains("shared") {
            try k.gemv(enc, w.sharedGate, x: h, y: sharedG, m: M)
            try k.gemv(enc, w.sharedUp, x: h, y: sharedU, m: M)
            try k.siluMul(enc, out: sharedAct, a: sharedG, b: sharedU, count: M * 512)
            try k.roundIfMimic(enc, x: sharedAct, count: M * 512)
            try k.gemv(enc, w.sharedDown, x: sharedAct, y: sharedOut, m: M)
            try k.gemv(enc, w.sharedExpertGate, x: h, y: sharedGate, m: M)
        }
        try k.moeCombine(enc, x: x, slotOut: slotOut, sharedOut: sharedOut, sharedGate: sharedGate, rows: M, config: config)
    }
}
