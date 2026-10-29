// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Decode blocks with the model's shapes on synthetic data, each encodable as
/// the engine's separate dispatches (in their parallel groups) or, where one
/// exists, as the fused dispatch. FusionTests compares the two bit for bit on
/// serial and concurrent encoders; `drama dbench --blocks` times them.
package final class FusionReplica {
    package struct Block {
        package let name: String
        /// Instances of the block in one decode frame.
        package let perFrame: Int
        let inputs: [(Buf, [Float])]
        let outputs: [Buf]
        let separate: (MTLComputeCommandEncoder) throws -> Void
        let fused: ((MTLComputeCommandEncoder) throws -> Void)?

        package func reset() {
            for (b, v) in inputs { v.withUnsafeBufferPointer { _ = memcpy(b.floats, $0.baseAddress!, v.count * 4) } }
        }

        package func encode(_ enc: MTLComputeCommandEncoder, fused f: Bool) throws { try (f ? fused ?? separate : separate)(enc) }

        package func outputBits() -> [[UInt32]] { outputs.map { b in (0..<b.count).map { b.floats[$0].bitPattern } } }
    }

    static let H = 2048, heads = 16, kvHeads = 2, rotaryDim = 64, eps: Float = 1e-6

    let ctx: MetalContext
    let k: Kernels
    var rng = SeededGenerator(seed: 11)
    /// Random bf16 bits per element count, reused so that many copies of a
    /// block (distinct buffers for timing) are cheap to build.
    var templates: [Int: [UInt16]] = [:]

    package init(ctx: MetalContext, schedule: KernelSchedule = .default) {
        self.ctx = ctx
        k = Kernels(ctx: ctx, precision: .exact, schedule: schedule)
    }

    func uniform() -> Float { Float(rng.uniformOpenZero()) * 2 - 1 }

    func floats(_ n: Int, _ label: String, scale: Float = 1) -> Buf {
        let b = Buf(ctx.device, floats: n, label: label)
        for i in 0..<n { b.floats[i] = uniform() * scale }
        return b
    }

    func bf16(_ rows: Int, _ cols: Int, scale: Float = 1) -> TensorRef {
        let n = rows * cols
        if templates[n] == nil { templates[n] = (0..<n).map { _ in UInt16(truncatingIfNeeded: (uniform() * scale).bitPattern >> 16) } }
        let buf = templates[n]!.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: n * 2, options: .storageModeShared)! }
        return TensorRef(name: "w", dtype: .bf16, shape: [rows, cols], buffer: buf, offset: 0)
    }

    func block(_ name: String, perFrame: Int, inputs: [Buf], outputs: [Buf],
               separate: @escaping (MTLComputeCommandEncoder) throws -> Void, fused: ((MTLComputeCommandEncoder) throws -> Void)? = nil) -> Block {
        Block(name: name, perFrame: perFrame, inputs: inputs.map { ($0, $0.read($0.count)) }, outputs: outputs, separate: separate, fused: fused)
    }

    /// Gate and up projections in one group, silu(g) * u, down projection:
    /// the tower's FFN (width 6144).
    package func gatedFFN(width: Int, perFrame: Int) -> Block {
        let H = Self.H, k = k
        let x = floats(H, "x"), wg = bf16(width, H, scale: 0.05), wu = bf16(width, H, scale: 0.05), wd = bf16(H, width, scale: 0.05)
        let g = Buf(ctx.device, floats: width, label: "g"), u = Buf(ctx.device, floats: width, label: "u")
        let act = Buf(ctx.device, floats: width, label: "act"), out = Buf(ctx.device, floats: H, label: "out")
        return block("ffn \(width)", perFrame: perFrame, inputs: [], outputs: [out], separate: { enc in
            try k.parallel(enc) {
                try k.gemv(enc, wg, x: x, y: g, m: 1)
                try k.gemv(enc, wu, x: x, y: u, m: 1)
            }
            try k.siluMul(enc, out: act, a: g, b: u, count: width)
            try k.gemv(enc, wd, x: act, y: out, m: 1)
        })
    }

    /// Per-head q and k norms, both ropes and both cache writes.
    package func normRopeKV(tower: Bool) -> Block {
        let heads = Self.heads, kv = Self.kvHeads, k = k, pos0 = 37, capacity = 64
        let theta = tower ? 10_000.0 : 10_000_000.0
        let qkv = floats(tower ? 9216 : heads * 512, "q"), kb = floats(kv * 256, "k"), vb = floats(kv * 256, "v")
        let qNorm = bf16(1, 256, scale: 0.1), kNorm = bf16(1, 256, scale: 0.1)
        let qn = Buf(ctx.device, floats: heads * 256, label: "qn"), kn = Buf(ctx.device, floats: kv * 256, label: "kn")
        let kcache = Buf(ctx.device, floats: capacity * kv * 256, label: "kcache"), vcache = Buf(ctx.device, floats: capacity * kv * 256, label: "vcache")
        let qView = HeadView(heads: UInt32(heads), groupStride: UInt32(tower ? 9216 : heads * 512), headStride: UInt32(tower ? 256 : 512))
        let kvView = HeadView(heads: UInt32(kv), groupStride: UInt32(tower ? 9216 : kv * 256), headStride: 256)
        let packed = HeadView(heads: UInt32(kv), groupStride: UInt32(kv * 256), headStride: 256)
        let kSrc = tower ? qkv : kb, vSrc = tower ? qkv : vb
        let kOff = tower ? 2 * heads * 256 : 0, vOff = tower ? 2 * heads * 256 + kv * 256 : 0
        let interleaved = tower
        return block(tower ? "tower qk-norm+rope+kv" : "backbone qk-norm+rope+kv", perFrame: tower ? 13 : 10, inputs: [kcache, vcache], outputs: [qn, kcache, vcache], separate: { enc in
            try k.parallel(enc) {
                try k.rmsnorm(enc, out: qn, outStride: 256, x: qkv, view: qView, rows: heads, dim: 256, w: qNorm, eps: Self.eps)
                try k.rmsnorm(enc, out: kn, outStride: 256, x: kSrc, xOffset: kOff, view: kvView, rows: kv, dim: 256, w: kNorm, eps: Self.eps)
                try k.kvWrite(enc, cache: vcache, src: vSrc, srcOffset: vOff, view: kvView, kvHeads: kv, rows: 1, pos0: pos0)
            }
            try k.parallel(enc) {
                try k.rope(enc, interleaved: interleaved, x: qn, heads: heads, rows: heads, pos0: pos0, rotaryDim: Self.rotaryDim, theta: theta)
                try k.rope(enc, interleaved: interleaved, x: kn, heads: kv, rows: kv, pos0: pos0, rotaryDim: Self.rotaryDim, theta: theta)
            }
            try k.kvWrite(enc, cache: kcache, src: kn, srcOffset: 0, view: packed, kvHeads: kv, rows: 1, pos0: pos0)
        }, fused: { enc in
            try k.normRopeKV(enc, .init(q: qkv, qOffset: 0, qView: qView, qNorm: qNorm, k: kSrc, kOffset: kOff, kView: kvView, kNorm: kNorm,
                                        v: vSrc, vOffset: vOff, vView: kvView, qOut: qn, kcache: kcache, vcache: vcache,
                                        heads: heads, kvHeads: kv, rotaryDim: Self.rotaryDim, theta: theta, eps: Self.eps, interleaved: interleaved), rows: 1, pos0: pos0)
        })
    }

    /// The input norm and a linear-attention layer's four input projections in one group.
    package func normProjections() -> Block {
        let H = Self.H, k = k
        let x = floats(H, "x"), w = bf16(1, H, scale: 0.1), h = Buf(ctx.device, floats: H, label: "h")
        let shapes = [8192, 4096, 32, 32]
        let ws = shapes.map { bf16($0, H, scale: 0.05) }
        let ys = shapes.map { Buf(ctx.device, floats: $0, label: "y\($0)") }
        return block("rmsnorm+4 projections", perFrame: 30, inputs: [], outputs: ys, separate: { enc in
            try k.rmsnorm(enc, out: h, outStride: H, x: x, view: .contiguous(H), rows: 1, dim: H, w: w, eps: Self.eps)
            try k.parallel(enc) { for (wi, y) in zip(ws, ys) { try k.gemv(enc, wi, x: h, y: y, m: 1) } }
        })
    }

    package func allBlocks() -> [Block] {
        [gatedFFN(width: 6144, perFrame: 13), normRopeKV(tower: false), normRopeKV(tower: true), normProjections()]
    }
}
