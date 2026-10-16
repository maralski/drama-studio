import Foundation
import Metal

/// Numerics switches the evaluation suite drives. Off means fp32 everywhere
/// after the bf16 weights are widened.
public struct Precision: Sendable, Equatable {
    /// Round linear outputs and the residual stream to bf16 where the reference does.
    public var mimicBF16: Bool = false
    /// Round dequantized FP8 expert weights to bf16 (the reference's CPU path does).
    public var roundExpertWeights: Bool = false
    public init() {}
    public static let exact = Precision()
    public static var reference: Precision { var p = Precision(); p.mimicBF16 = true; p.roundExpertWeights = true; return p }
}

struct HeadView {
    var heads: UInt32
    var groupStride: UInt32
    var headStride: UInt32
    static func contiguous(_ width: Int) -> HeadView { HeadView(heads: 1, groupStride: UInt32(width), headStride: 0) }
}

/// A float32 or int32 GPU buffer the engine owns, or a view at a byte offset
/// into one.
final class Buf {
    let buffer: MTLBuffer
    let byteOffset: Int
    let count: Int
    init(_ device: MTLDevice, floats count: Int, label: String) {
        self.count = count
        self.byteOffset = 0
        buffer = device.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)!
        buffer.label = label
    }
    init(view: Buf, byteOffset: Int) {
        buffer = view.buffer
        self.byteOffset = view.byteOffset + byteOffset
        count = view.count - byteOffset / 4
    }
    var floats: UnsafeMutablePointer<Float> { (buffer.contents() + byteOffset).assumingMemoryBound(to: Float.self) }
    var ints: UnsafeMutablePointer<Int32> { (buffer.contents() + byteOffset).assumingMemoryBound(to: Int32.self) }
    func zero() { memset(buffer.contents() + byteOffset, 0, count * 4) }
    func read(_ n: Int, offset: Int = 0) -> [Float] { Array(UnsafeBufferPointer(start: floats + offset, count: n)) }
    func readInts(_ n: Int, offset: Int = 0) -> [Int32] { Array(UnsafeBufferPointer(start: ints + offset, count: n)) }
}

/// Thin encoders around each kernel. Every call appends one dispatch. The
/// schedule chooses each family's instantiation and dispatch shape; path
/// choice between families depends on M == 1 against M > 1 and on N only.
struct Kernels {
    let ctx: MetalContext
    let precision: Precision
    let schedule: KernelSchedule
    /// Dev benches only: run the GEMV for every M, with the narrow shape.
    var forceGemv = false
    /// Dev benches only: run the one-row expert kernels for every M.
    var forceMoeRows = false

    init(ctx: MetalContext, precision: Precision, schedule: KernelSchedule = .default) {
        self.ctx = ctx
        self.precision = precision
        self.schedule = schedule
    }

    func set<T>(_ enc: MTLComputeCommandEncoder, _ value: T, index: Int) {
        var v = value
        enc.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ name: String, grid: MTLSize, group: MTLSize) throws {
        enc.setComputePipelineState(try ctx.pipeline(name))
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
    }

    func dispatchGroups(_ enc: MTLComputeCommandEncoder, _ name: String, groups: MTLSize, group: MTLSize) throws {
        enc.setComputePipelineState(try ctx.pipeline(name))
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: group)
    }

    // y[m*N + n] = W[n] . x[m] for the matrix's N rows and K columns.
    //
    // One GEMM tile for every M above 1 so a row's result is the same
    // whichever chunk it lands in (the prefix cache relies on that); below 16
    // rows the GEMV or a 16-row tile is 1.5x to 4x faster on narrow shapes
    // (eval/out/perf/kbench_shapes.txt), a cost accepted for the chunk
    // invariance. Narrow outputs (N up to 64) keep the GEMV at every M: a
    // GEMM would have one or two column tiles walking K alone, while the
    // GEMV spreads the rows over threadgroups.
    func gemv(_ enc: MTLComputeCommandEncoder, _ w: WeightMatrix, x: Buf, xOffset: Int = 0, y: Buf, yOffset: Int = 0, m: Int) throws {
        struct P { var N: UInt32; var K: UInt32; var M: UInt32; var round: UInt32 }
        let n = w.rows, k = w.cols
        precondition(k % 256 == 0)
        enc.setBuffer(w.buffer, offset: w.offset, index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset + (xOffset * 4), index: 1)
        enc.setBuffer(y.buffer, offset: y.byteOffset + (yOffset * 4), index: 2)
        set(enc, P(N: UInt32(n), K: UInt32(k), M: UInt32(m), round: precision.mimicBF16 ? 1 : 0), index: 3)
        if m > 1 && n > 64 && !forceGemv {
            let tile = schedule.gemmTile
            try dispatchGroups(enc, "gemm_\(w.format.kernelTag)_\(tile.label)",
                               groups: MTLSize(width: (n + tile.bn - 1) / tile.bn, height: (m + tile.bm - 1) / tile.bm, depth: 1),
                               group: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            let shape = m == 1 ? schedule.gemvDecode : schedule.gemvNarrow
            try dispatchGroups(enc, "gemv_\(w.format.kernelTag)_t\(shape.tokenTile)",
                               groups: MTLSize(width: (n + shape.simdgroups - 1) / shape.simdgroups, height: (m + shape.tokenTile - 1) / shape.tokenTile, depth: 1),
                               group: MTLSize(width: 32 * shape.simdgroups, height: 1, depth: 1))
        }
    }

    func rmsnorm(_ enc: MTLComputeCommandEncoder, out: Buf, outOffset: Int = 0, outStride: Int, x: Buf, xOffset: Int = 0, view: HeadView,
                 rows: Int, dim: Int, w: TensorRef, eps: Float) throws {
        struct P { var dim: UInt32; var eps: Float; var view: HeadView; var outStride: UInt32 }
        enc.setBuffer(out.buffer, offset: out.byteOffset + (outOffset * 4), index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset + (xOffset * 4), index: 1)
        enc.setBuffer(w.buffer, offset: w.offset, index: 2)
        set(enc, P(dim: UInt32(dim), eps: eps, view: view, outStride: UInt32(outStride)), index: 3)
        try dispatchGroups(enc, "rmsnorm_zc", groups: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func gatedRMSNorm(_ enc: MTLComputeCommandEncoder, out: Buf, x: Buf, z: Buf, w: TensorRef, rows: Int, eps: Float) throws {
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 1)
        enc.setBuffer(z.buffer, offset: z.byteOffset + (0), index: 2)
        enc.setBuffer(w.buffer, offset: w.offset, index: 3)
        set(enc, eps, index: 4)
        try dispatch(enc, "gated_rmsnorm", grid: MTLSize(width: rows * 32, height: 1, depth: 1), group: MTLSize(width: 128, height: 1, depth: 1))
    }

    func zero(_ enc: MTLComputeCommandEncoder, _ x: Buf) throws {
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 0)
        try dispatch(enc, "zero_u32", grid: MTLSize(width: x.count, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func residualAdd(_ enc: MTLComputeCommandEncoder, x: Buf, y: Buf, count: Int) throws {
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 0)
        enc.setBuffer(y.buffer, offset: y.byteOffset + (0), index: 1)
        try dispatch(enc, "residual_add", grid: MTLSize(width: count, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func roundIfMimic(_ enc: MTLComputeCommandEncoder, x: Buf, count: Int) throws {
        guard precision.mimicBF16 else { return }
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 0)
        try dispatch(enc, "round_bf16_inplace", grid: MTLSize(width: count, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func siluMul(_ enc: MTLComputeCommandEncoder, out: Buf, a: Buf, b: Buf, count: Int) throws {
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(a.buffer, offset: a.byteOffset + (0), index: 1)
        enc.setBuffer(b.buffer, offset: b.byteOffset + (0), index: 2)
        try dispatch(enc, "silu_mul", grid: MTLSize(width: count, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func embedMix(_ enc: MTLComputeCommandEncoder, out: Buf, frames: Buf, embed: TensorRef, codebooks: TensorRef, rows: Int, config: ModelConfig) throws {
        struct P { var dim: UInt32; var codebookSize: UInt32; var semanticStart: Int32; var semanticEnd: Int32 }
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(frames.buffer, offset: frames.byteOffset + (0), index: 1)
        enc.setBuffer(embed.buffer, offset: embed.offset, index: 2)
        enc.setBuffer(codebooks.buffer, offset: codebooks.offset, index: 3)
        set(enc, P(dim: UInt32(config.hiddenSize), codebookSize: UInt32(config.codebookSize), semanticStart: config.semanticStart, semanticEnd: config.semanticEnd), index: 4)
        try dispatch(enc, "embed_mix", grid: MTLSize(width: config.hiddenSize, height: rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func towerEmbed(_ enc: MTLComputeCommandEncoder, out: Buf, codes: Buf, semanticNext: Buf, table: TensorRef, frames: Buf, framesOffset: Int,
                    codebooks: TensorRef, addUnseen: Bool, rows: Int, config: ModelConfig) throws {
        struct P { var dim: UInt32; var codebookSize: UInt32; var numCodebooks: UInt32; var addUnseen: UInt32; var semanticStart: Int32; var semanticEnd: Int32 }
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(codes.buffer, offset: codes.byteOffset + (0), index: 1)
        enc.setBuffer(semanticNext.buffer, offset: semanticNext.byteOffset + (0), index: 2)
        enc.setBuffer(table.buffer, offset: table.offset, index: 3)
        enc.setBuffer(frames.buffer, offset: frames.byteOffset + (framesOffset * 4), index: 4)
        enc.setBuffer(codebooks.buffer, offset: codebooks.offset, index: 5)
        set(enc, P(dim: UInt32(config.hiddenSize), codebookSize: UInt32(config.codebookSize), numCodebooks: UInt32(config.numCodebooks),
                   addUnseen: addUnseen ? 1 : 0, semanticStart: config.semanticStart, semanticEnd: config.semanticEnd), index: 6)
        try dispatch(enc, "tower_embed", grid: MTLSize(width: config.hiddenSize, height: rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func sigmoidGateMul(_ enc: MTLComputeCommandEncoder, out: Buf, attn: Buf, gate: Buf, gateOffset: Int, view: HeadView, rows: Int) throws {
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(attn.buffer, offset: attn.byteOffset + (0), index: 1)
        enc.setBuffer(gate.buffer, offset: gate.byteOffset + (gateOffset * 4), index: 2)
        set(enc, view, index: 3)
        try dispatch(enc, "sigmoid_gate_mul", grid: MTLSize(width: 256, height: rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func rope(_ enc: MTLComputeCommandEncoder, interleaved: Bool, x: Buf, heads: Int, rows: Int, pos0: Int, rotaryDim: Int, theta: Double) throws {
        struct P { var heads: UInt32; var pos0: UInt32; var rotaryDim: UInt32; var theta: Float }
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 0)
        set(enc, P(heads: UInt32(heads), pos0: UInt32(pos0), rotaryDim: UInt32(rotaryDim), theta: Float(theta)), index: 1)
        try dispatch(enc, interleaved ? "rope_interleaved" : "rope_neox", grid: MTLSize(width: rotaryDim / 2, height: rows, depth: 1), group: MTLSize(width: 32, height: 1, depth: 1))
    }

    func kvWrite(_ enc: MTLComputeCommandEncoder, cache: Buf, src: Buf, srcOffset: Int, view: HeadView, kvHeads: Int, rows: Int, pos0: Int) throws {
        struct P { var kvHeads: UInt32; var pos0: UInt32; var view: HeadView }
        enc.setBuffer(cache.buffer, offset: cache.byteOffset + (0), index: 0)
        enc.setBuffer(src.buffer, offset: src.byteOffset + (srcOffset * 4), index: 1)
        set(enc, P(kvHeads: UInt32(kvHeads), pos0: UInt32(pos0), view: view), index: 2)
        try dispatch(enc, "kv_write", grid: MTLSize(width: 256, height: rows * kvHeads, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func attention(_ enc: MTLComputeCommandEncoder, out: Buf, q: Buf, kcache: Buf, vcache: Buf, heads: Int, kvHeads: Int, rows: Int, pos0: Int) throws {
        struct P { var heads: UInt32; var kvHeads: UInt32; var pos0: UInt32; var scale: Float }
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(q.buffer, offset: q.byteOffset + (0), index: 1)
        enc.setBuffer(kcache.buffer, offset: kcache.byteOffset + (0), index: 2)
        enc.setBuffer(vcache.buffer, offset: vcache.byteOffset + (0), index: 3)
        set(enc, P(heads: UInt32(heads), kvHeads: UInt32(kvHeads), pos0: UInt32(pos0), scale: 1.0 / 16.0), index: 4)
        let split = rows == 1 ? schedule.attentionSplitDecode : schedule.attentionSplitPrefill
        enc.setThreadgroupMemoryLength(split * 256 * 4, index: 0)
        try dispatchGroups(enc, "attention", groups: MTLSize(width: heads, height: rows, depth: 1), group: MTLSize(width: 32 * split, height: 1, depth: 1))
    }

    func gdnConv(_ enc: MTLComputeCommandEncoder, out: Buf, mixed: Buf, state: Buf, w: TensorRef, channels: Int, rows: Int) throws {
        struct P { var C: UInt32; var M: UInt32 }
        enc.setBuffer(out.buffer, offset: out.byteOffset + (0), index: 0)
        enc.setBuffer(mixed.buffer, offset: mixed.byteOffset + (0), index: 1)
        enc.setBuffer(state.buffer, offset: state.byteOffset + (0), index: 2)
        enc.setBuffer(w.buffer, offset: w.offset, index: 3)
        set(enc, P(C: UInt32(channels), M: UInt32(rows)), index: 4)
        try dispatch(enc, "gdn_conv_silu", grid: MTLSize(width: channels, height: rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        enc.setBuffer(state.buffer, offset: state.byteOffset + (0), index: 0)
        enc.setBuffer(mixed.buffer, offset: mixed.byteOffset + (0), index: 1)
        set(enc, P(C: UInt32(channels), M: UInt32(rows)), index: 2)
        try dispatch(enc, "gdn_conv_state_update", grid: MTLSize(width: channels, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func gdnRecurrent(_ enc: MTLComputeCommandEncoder, core: Buf, convOut: Buf, a: Buf, b: Buf, state: Buf, aLog: TensorRef, dtBias: TensorRef, rows: Int, config: ModelConfig) throws {
        struct P { var M: UInt32; var rowStride: UInt32; var numVHeads: UInt32; var numKHeads: UInt32; var eps: Float }
        enc.setBuffer(core.buffer, offset: core.byteOffset + (0), index: 0)
        enc.setBuffer(convOut.buffer, offset: convOut.byteOffset + (0), index: 1)
        enc.setBuffer(a.buffer, offset: a.byteOffset + (0), index: 2)
        enc.setBuffer(b.buffer, offset: b.byteOffset + (0), index: 3)
        enc.setBuffer(state.buffer, offset: state.byteOffset + (0), index: 4)
        enc.setBuffer(aLog.buffer, offset: aLog.offset, index: 5)
        enc.setBuffer(dtBias.buffer, offset: dtBias.offset, index: 6)
        set(enc, P(M: UInt32(rows), rowStride: 8192, numVHeads: UInt32(config.linearValueHeads), numKHeads: UInt32(config.linearKeyHeads), eps: 1e-6), index: 7)
        let parts = rows == 1 ? schedule.gdnPartsDecode : schedule.gdnPartsPrefill
        try dispatchGroups(enc, "gdn_recurrent_p\(parts)", groups: MTLSize(width: config.linearValueHeads, height: 1, depth: 1), group: MTLSize(width: 128 * parts, height: 1, depth: 1))
    }

    func moeRouter(_ enc: MTLComputeCommandEncoder, ids: Buf, weights: Buf, logits: Buf, rows: Int, config: ModelConfig) throws {
        struct P { var numExperts: UInt32; var topK: UInt32; var roundLogits: UInt32 }
        enc.setBuffer(ids.buffer, offset: ids.byteOffset + (0), index: 0)
        enc.setBuffer(weights.buffer, offset: weights.byteOffset + (0), index: 1)
        enc.setBuffer(logits.buffer, offset: logits.byteOffset + (0), index: 2)
        set(enc, P(numExperts: UInt32(config.numExperts), topK: UInt32(config.topK), roundLogits: precision.mimicBF16 ? 1 : 0), index: 3)
        try dispatchGroups(enc, "moe_router", groups: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    func moeBucket(_ enc: MTLComputeCommandEncoder, counts: Buf, lists: Buf, ids: Buf, active: Buf, pairs: Int, capacity: Int) throws {
        enc.setBuffer(counts.buffer, offset: counts.byteOffset + (0), index: 0)
        enc.setBuffer(lists.buffer, offset: lists.byteOffset + (0), index: 1)
        enc.setBuffer(ids.buffer, offset: ids.byteOffset + (0), index: 2)
        enc.setBuffer(active.buffer, offset: active.byteOffset + (0), index: 3)
        set(enc, UInt32(capacity), index: 4)
        try dispatch(enc, "moe_bucket", grid: MTLSize(width: pairs, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// The tiled expert kernels serve every M above 1 (one tile, so a routed
    /// row's result is the same in any chunk); one row runs the row kernels.
    private func moeTiled(rows: Int) -> Bool { rows > 1 && !forceMoeRows }

    private func moeParams(capacity: Int, config: ModelConfig) -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
        (UInt32(config.hiddenSize), UInt32(config.moeIntermediate), UInt32(config.topK), UInt32(capacity),
         precision.roundExpertWeights ? 1 : 0, precision.mimicBF16 ? 1 : 0)
    }

    func moeGateUp(_ enc: MTLComputeCommandEncoder, act: Buf, x: Buf, experts: ExpertTable, counts: Buf, lists: Buf, active: Buf, maxActive: Int, capacity: Int, rows: Int, config: ModelConfig) throws {
        struct P { var hidden: UInt32; var intermediate: UInt32; var topK: UInt32; var capacity: UInt32; var roundWeights: UInt32; var roundOutput: UInt32 }
        let p = moeParams(capacity: capacity, config: config)
        enc.setBuffer(act.buffer, offset: act.byteOffset + (0), index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 1)
        enc.setBuffer(experts.table, offset: 0, index: 2)
        enc.setBuffer(counts.buffer, offset: counts.byteOffset + (0), index: 3)
        enc.setBuffer(lists.buffer, offset: lists.byteOffset + (0), index: 4)
        set(enc, P(hidden: p.0, intermediate: p.1, topK: p.2, capacity: p.3, roundWeights: p.4, roundOutput: p.5), index: 5)
        enc.setBuffer(active.buffer, offset: active.byteOffset, index: 6)
        let tag = experts.format.kernelTag
        if moeTiled(rows: rows) {
            let t = schedule.moeTile
            try dispatchGroups(enc, "moe_gate_up_\(tag)_tiled_\(t.label)", groups: MTLSize(width: maxActive, height: config.moeIntermediate / t.bn, depth: 1), group: MTLSize(width: 4 * t.bn, height: 1, depth: 1))
        } else {
            let sg = schedule.moeRowSimdgroups
            try dispatchGroups(enc, "moe_gate_up_\(tag)", groups: MTLSize(width: maxActive, height: config.moeIntermediate / sg, depth: 1), group: MTLSize(width: 32 * sg, height: 1, depth: 1))
        }
    }

    func moeDown(_ enc: MTLComputeCommandEncoder, slotOut: Buf, act: Buf, weights: Buf, experts: ExpertTable, counts: Buf, lists: Buf, active: Buf, maxActive: Int, capacity: Int, rows: Int, config: ModelConfig) throws {
        struct P { var hidden: UInt32; var intermediate: UInt32; var topK: UInt32; var capacity: UInt32; var roundWeights: UInt32; var roundOutput: UInt32 }
        let p = moeParams(capacity: capacity, config: config)
        enc.setBuffer(slotOut.buffer, offset: slotOut.byteOffset + (0), index: 0)
        enc.setBuffer(act.buffer, offset: act.byteOffset + (0), index: 1)
        enc.setBuffer(weights.buffer, offset: weights.byteOffset + (0), index: 2)
        enc.setBuffer(experts.table, offset: 0, index: 3)
        enc.setBuffer(counts.buffer, offset: counts.byteOffset + (0), index: 4)
        enc.setBuffer(lists.buffer, offset: lists.byteOffset + (0), index: 5)
        set(enc, P(hidden: p.0, intermediate: p.1, topK: p.2, capacity: p.3, roundWeights: p.4, roundOutput: p.5), index: 6)
        enc.setBuffer(active.buffer, offset: active.byteOffset, index: 7)
        let tag = experts.format.kernelTag
        if moeTiled(rows: rows) {
            let t = schedule.moeTile
            try dispatchGroups(enc, "moe_down_\(tag)_tiled_\(t.label)", groups: MTLSize(width: maxActive, height: config.hiddenSize / t.bn, depth: 1), group: MTLSize(width: 4 * t.bn, height: 1, depth: 1))
        } else {
            let sg = schedule.moeRowSimdgroups
            try dispatchGroups(enc, "moe_down_\(tag)", groups: MTLSize(width: maxActive, height: config.hiddenSize / sg, depth: 1), group: MTLSize(width: 32 * sg, height: 1, depth: 1))
        }
    }

    func moeCombine(_ enc: MTLComputeCommandEncoder, x: Buf, slotOut: Buf, sharedOut: Buf, sharedGate: Buf, rows: Int, config: ModelConfig) throws {
        struct P { var hidden: UInt32; var topK: UInt32; var roundOutput: UInt32 }
        enc.setBuffer(x.buffer, offset: x.byteOffset + (0), index: 0)
        enc.setBuffer(slotOut.buffer, offset: slotOut.byteOffset + (0), index: 1)
        enc.setBuffer(sharedOut.buffer, offset: sharedOut.byteOffset + (0), index: 2)
        enc.setBuffer(sharedGate.buffer, offset: sharedGate.byteOffset + (0), index: 3)
        set(enc, P(hidden: UInt32(config.hiddenSize), topK: UInt32(config.topK), roundOutput: precision.mimicBF16 ? 1 : 0), index: 4)
        try dispatch(enc, "moe_combine", grid: MTLSize(width: config.hiddenSize, height: rows, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }
}
