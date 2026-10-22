import Foundation
import Metal

/// The 13-layer codebook tower: 4 shared layers, then 9 residual layers each
/// followed by a head for the next codebook.
final class Tower {
    let config: ModelConfig
    let weights: TowerWeights
    var k: Kernels
    let maxRows: Int
    let capacity: Int

    private(set) var kcache: [Buf] = []
    private(set) var vcache: [Buf] = []

    let e: Buf, cat: Buf, mixed: Buf, th: Buf, tqkv: Buf, tqn: Buf, tkn: Buf, attnOut: Buf, gated: Buf, proj: Buf
    let w1o: Buf, w3o: Buf, ffAct: Buf, ffOut: Buf
    let hiddenA: Buf, hiddenB: Buf
    let attnPartials: Buf
    /// codes[i] holds code i of the frame being predicted for each row.
    let codes: [Buf]
    let semantic: Buf
    /// [numCodebooks - 1][codebookSize]: head i's logits for the decode row.
    let headLogits: Buf
    var captureLayers: Bool = false
    private(set) var layerCapture: [Buf] = []

    init(ctx: MetalContext, config: ModelConfig, weights: TowerWeights, precision: Precision, schedule: KernelSchedule = .default, capacity: Int, maxRows: Int = 128) {
        self.config = config
        self.weights = weights
        self.k = Kernels(ctx: ctx, precision: precision, schedule: schedule)
        self.maxRows = maxRows
        self.capacity = capacity
        let d = ctx.device
        let H = config.hiddenSize, R = maxRows, I = config.towerIntermediate
        e = Buf(d, floats: R * H, label: "t.e"); cat = Buf(d, floats: R * 2 * H, label: "t.cat")
        mixed = Buf(d, floats: R * H, label: "t.mixed"); th = Buf(d, floats: R * H, label: "t.h")
        tqkv = Buf(d, floats: R * 9216, label: "t.qkv"); tqn = Buf(d, floats: R * 4096, label: "t.qn"); tkn = Buf(d, floats: R * 512, label: "t.kn")
        attnOut = Buf(d, floats: R * 4096, label: "t.attn"); gated = Buf(d, floats: R * 4096, label: "t.gated"); proj = Buf(d, floats: R * H, label: "t.proj")
        w1o = Buf(d, floats: R * I, label: "t.w1"); w3o = Buf(d, floats: R * I, label: "t.w3"); ffAct = Buf(d, floats: R * I, label: "t.ffAct"); ffOut = Buf(d, floats: R * H, label: "t.ffOut")
        hiddenA = Buf(d, floats: R * H, label: "t.hiddenA"); hiddenB = Buf(d, floats: R * H, label: "t.hiddenB")
        attnPartials = Buf(d, floats: Kernels.attentionPartialsCount(heads: config.towerHeads), label: "t.attnPartials")
        codes = (0..<config.numCodebooks).map { Buf(d, floats: R, label: "t.codes.\($0)") }
        semantic = Buf(d, floats: R, label: "t.semantic")
        headLogits = Buf(d, floats: (config.numCodebooks - 1) * config.codebookSize, label: "t.headLogits")
        let layerCount = config.towerLayers + config.numCodebooks - 1
        kcache = (0..<layerCount).map { Buf(d, floats: capacity * config.towerKVHeads * 256, label: "t.kcache.\($0)") }
        vcache = (0..<layerCount).map { Buf(d, floats: capacity * config.towerKVHeads * 256, label: "t.vcache.\($0)") }
        layerCapture = (0..<layerCount).map { Buf(d, floats: H, label: "t.capture.\($0)") }
    }

    var allLayers: [TowerLayerWeights] { weights.shared + weights.residual }

    /// Fill the per-row code and semantic inputs from the frames being predicted.
    func setTargets(nextFrames: [Frame]) {
        for (m, f) in nextFrames.enumerated() {
            semantic.ints[m] = (f[0] >= config.semanticStart && f[0] <= config.semanticEnd) ? 1 : 0
            for c in 0..<config.numCodebooks { codes[c].ints[m] = f[1 + c] }
        }
    }

    func setMain(row: Int, main: Int32) {
        semantic.ints[row] = (main >= config.semanticStart && main <= config.semanticEnd) ? 1 : 0
        codes[0].ints[row] = max(0, min(Int32(config.codebookSize - 1), main - config.semanticStart))
    }

    func setCode(row: Int, index: Int, value: Int32) { codes[index].ints[row] = value }

    /// One tower layer. `hidden` is the layer input (backbone hidden for the
    /// first shared layer); the output lands in the returned buffer.
    @discardableResult
    private func layer(_ enc: MTLComputeCommandEncoder, _ index: Int, hidden: Buf, hiddenOffset: Int, codes codeBuf: Buf, frames: Buf, rows M: Int, pos0: Int,
                       codebooks: TensorRef, addUnseen: Bool) throws -> Buf {
        let w = allLayers[index]
        let H = config.hiddenSize, I = config.towerIntermediate
        let heads = config.towerHeads, kvHeads = config.towerKVHeads
        let out = index % 2 == 0 ? hiddenA : hiddenB
        if M == 1 {
            try k.towerCatNorm(enc, out: cat, hidden: hidden, hiddenOffset: hiddenOffset * H, enorm: w.enorm, hnorm: w.hnorm, embedding: w.embedding,
                               codes: codeBuf, semanticNext: semantic, frames: frames, codebooks: codebooks, addUnseen: addUnseen, config: config)
        } else {
            try k.parallel(enc) {
                try k.towerEmbed(enc, out: e, codes: codeBuf, semanticNext: semantic, table: w.embedding, frames: frames, framesOffset: 0,
                                 codebooks: codebooks, addUnseen: addUnseen, rows: M, config: config)
                try k.rmsnorm(enc, out: cat, outOffset: H, outStride: 2 * H, x: hidden, xOffset: hiddenOffset * H, view: .contiguous(H), rows: M, dim: H, w: w.hnorm, eps: config.towerNormEps)
            }
            try k.roundIfMimic(enc, x: e, count: M * H)
            try k.rmsnorm(enc, out: cat, outStride: 2 * H, x: e, view: .contiguous(H), rows: M, dim: H, w: w.enorm, eps: config.towerNormEps)
            try k.roundIfMimic(enc, x: cat, count: M * 2 * H)
        }
        try k.gemv(enc, w.ehProj, x: cat, y: mixed, m: M)
        try k.rmsnorm(enc, out: th, outStride: H, x: mixed, view: .contiguous(H), rows: M, dim: H, w: w.attentionNorm, eps: config.towerNormEps)
        try k.roundIfMimic(enc, x: th, count: M * H)
        try k.gemv(enc, w.wqkv, x: th, y: tqkv, m: M)
        try normRopeKV(enc, w, index: index, rows: M, pos0: pos0)
        if M == 1 {
            try k.attentionSplitGated(enc, out: gated, q: tqn, kcache: kcache[index], vcache: vcache[index], partials: attnPartials, gate: tqkv, gateOffset: heads * 256,
                                      heads: heads, kvHeads: kvHeads, pos0: pos0)
            try k.gemvResidual(enc, w.wo, x: gated, y: mixed)
        } else {
            let qView = HeadView(heads: UInt32(heads), groupStride: 9216, headStride: 256)
            try k.attention(enc, out: attnOut, q: tqn, kcache: kcache[index], vcache: vcache[index], heads: heads, kvHeads: kvHeads, rows: M, pos0: pos0)
            try k.roundIfMimic(enc, x: attnOut, count: M * heads * 256)
            try k.sigmoidGateMul(enc, out: gated, attn: attnOut, gate: tqkv, gateOffset: heads * 256, view: qView, rows: M * heads)
            try k.roundIfMimic(enc, x: gated, count: M * heads * 256)
            try k.gemv(enc, w.wo, x: gated, y: proj, m: M)
            try k.residualAdd(enc, x: mixed, y: proj, count: M * H)
        }
        try k.roundIfMimic(enc, x: mixed, count: M * H)
        try k.rmsnorm(enc, out: th, outStride: H, x: mixed, view: .contiguous(H), rows: M, dim: H, w: w.ffnNorm, eps: config.towerNormEps)
        try k.roundIfMimic(enc, x: th, count: M * H)
        if M == 1 {
            try k.swigluDecode(enc, w1: w.w1, w3: w.w3, x: th, y: ffAct)
            try k.gemvResidual(enc, w.w2, x: ffAct, y: mixed)
        } else {
            try k.parallel(enc) {
                try k.gemv(enc, w.w1, x: th, y: w1o, m: M)
                try k.gemv(enc, w.w3, x: th, y: w3o, m: M)
            }
            try k.siluMul(enc, out: ffAct, a: w1o, b: w3o, count: M * I)
            try k.roundIfMimic(enc, x: ffAct, count: M * I)
            try k.gemv(enc, w.w2, x: ffAct, y: ffOut, m: M)
            try k.residualAdd(enc, x: mixed, y: ffOut, count: M * H)
        }
        try k.roundIfMimic(enc, x: mixed, count: M * H)
        try k.rmsnorm(enc, out: out, outStride: H, x: mixed, view: .contiguous(H), rows: M, dim: H, w: w.finalNorm, eps: config.towerNormEps)
        try k.roundIfMimic(enc, x: out, count: M * H)
        if captureLayers {
            try k.zero(enc, layerCapture[index])
            try k.residualAdd(enc, x: layerCapture[index], y: out, count: H)
        }
        return out
    }

    /// The q and k norms, both ropes and both cache writes of layer `index`:
    /// one dispatch in exact precision, the separate dispatches when the
    /// reference rounds between the norm and the rope.
    func normRopeKV(_ enc: MTLComputeCommandEncoder, _ w: TowerLayerWeights, index: Int, rows M: Int, pos0: Int) throws {
        let heads = config.towerHeads, kvHeads = config.towerKVHeads, eps = config.towerNormEps
        let qView = HeadView(heads: UInt32(heads), groupStride: 9216, headStride: 256)
        let kvView = HeadView(heads: UInt32(kvHeads), groupStride: 9216, headStride: 256)
        let kOffset = 2 * heads * 256, vOffset = 2 * heads * 256 + kvHeads * 256
        if !k.precision.mimicBF16 {
            try k.normRopeKV(enc, .init(q: tqkv, qOffset: 0, qView: qView, qNorm: w.qNorm, k: tqkv, kOffset: kOffset, kView: kvView, kNorm: w.kNorm,
                                        v: tqkv, vOffset: vOffset, vView: kvView, qOut: tqn, kcache: kcache[index], vcache: vcache[index],
                                        heads: heads, kvHeads: kvHeads, rotaryDim: config.towerRotaryDim, theta: config.towerRopeBase, eps: eps, interleaved: true),
                             rows: M, pos0: pos0)
            return
        }
        try k.parallel(enc) {
            try k.rmsnorm(enc, out: tqn, outStride: 256, x: tqkv, view: qView, rows: M * heads, dim: 256, w: w.qNorm, eps: eps)
            try k.rmsnorm(enc, out: tkn, outStride: 256, x: tqkv, xOffset: kOffset, view: kvView, rows: M * kvHeads, dim: 256, w: w.kNorm, eps: eps)
            try k.kvWrite(enc, cache: vcache[index], src: tqkv, srcOffset: vOffset, view: kvView, kvHeads: kvHeads, rows: M, pos0: pos0)
        }
        try k.roundIfMimic(enc, x: tqn, count: M * heads * 256)
        try k.roundIfMimic(enc, x: tkn, count: M * kvHeads * 256)
        try k.parallel(enc) {
            try k.rope(enc, interleaved: true, x: tqn, heads: heads, rows: M * heads, pos0: pos0, rotaryDim: config.towerRotaryDim, theta: config.towerRopeBase)
            try k.rope(enc, interleaved: true, x: tkn, heads: kvHeads, rows: M * kvHeads, pos0: pos0, rotaryDim: config.towerRotaryDim, theta: config.towerRopeBase)
        }
        try k.roundIfMimic(enc, x: tqn, count: M * heads * 256)
        try k.roundIfMimic(enc, x: tkn, count: M * kvHeads * 256)
        try k.kvWrite(enc, cache: kcache[index], src: tkn, srcOffset: 0, view: HeadView(heads: UInt32(kvHeads), groupStride: UInt32(kvHeads * 256), headStride: 256), kvHeads: kvHeads, rows: M, pos0: pos0)
    }

    /// Teacher-forces rows [pos0, pos0 + M): row p is labelled with frame p + 1.
    /// `setTargets` must hold frames p + 1; `frames` points at row pos0.
    func prefill(_ enc: MTLComputeCommandEncoder, hidden: Buf, frames: Buf, rows M: Int, pos0: Int, codebooks: TensorRef) throws {
        var h = hidden
        var off = 0
        for i in 0..<config.towerLayers {
            h = try layer(enc, i, hidden: h, hiddenOffset: off, codes: codes[0], frames: frames, rows: M, pos0: pos0, codebooks: codebooks, addUnseen: i == 0)
            off = 0
        }
        for i in 0..<(config.numCodebooks - 1) {
            h = try layer(enc, config.towerLayers + i, hidden: h, hiddenOffset: 0, codes: codes[i], frames: frames, rows: M, pos0: pos0, codebooks: codebooks, addUnseen: false)
        }
    }

    /// Decode: the shared layers for one row.
    func decodeShared(_ enc: MTLComputeCommandEncoder, hidden: Buf, hiddenRow: Int, frames: Buf, pos: Int, codebooks: TensorRef) throws {
        var h = hidden
        var off = hiddenRow
        for i in 0..<config.towerLayers {
            h = try layer(enc, i, hidden: h, hiddenOffset: off, codes: codes[0], frames: frames, rows: 1, pos0: pos, codebooks: codebooks, addUnseen: i == 0)
            off = 0
        }
    }

    /// Residual layer `i` for one row (input code i); returns its output.
    func residualStep(_ enc: MTLComputeCommandEncoder, _ i: Int, frames: Buf, pos: Int, codebooks: TensorRef) throws -> Buf {
        let index = config.towerLayers + i
        let input = (index - 1) % 2 == 0 ? hiddenA : hiddenB
        return try layer(enc, index, hidden: input, hiddenOffset: 0, codes: codes[i], frames: frames, rows: 1, pos0: pos, codebooks: codebooks, addUnseen: false)
    }

    /// Head `i` on residual layer i's output: the logits for code i + 1.
    func head(_ enc: MTLComputeCommandEncoder, _ i: Int, input: Buf) throws {
        try k.gemv(enc, weights.heads[i], x: input, y: headLogits, yOffset: i * config.codebookSize, m: 1)
    }

    func readHeadLogits(head: Int) -> [Float] { headLogits.read(config.codebookSize, offset: head * config.codebookSize) }
}
