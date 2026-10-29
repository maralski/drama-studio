// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import DramaEngine

/// The tower's one-row kernels against the dispatches they replace, bit for
/// bit, on random data in both precisions.
final class TowerDecodeTests: XCTestCase {
    struct Fixture {
        let ctx: MetalContext
        var rng = SeededGenerator(seed: 19)

        init() throws { ctx = try MetalContext() }

        func kernels(_ precision: Precision, split: Int = 8) -> Kernels {
            var s = KernelSchedule.default
            s.attentionSplitDecode = split
            return Kernels(ctx: ctx, precision: precision, schedule: s)
        }

        mutating func uniform() -> Float { Float(rng.uniformOpenZero()) * 2 - 1 }

        mutating func bf16(_ shape: [Int], scale: Float = 1) -> TensorRef {
            let t = zeroBF16(shape)
            let p = t.buffer.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<shape.reduce(1, *) { p[i] = UInt16(truncatingIfNeeded: (uniform() * scale).bitPattern >> 16) }
            return t
        }

        func zeroBF16(_ shape: [Int]) -> TensorRef {
            let buf = ctx.device.makeBuffer(length: shape.reduce(1, *) * 2, options: .storageModeShared)!
            return TensorRef(name: "w", dtype: .bf16, shape: shape, buffer: buf, offset: 0)
        }

        mutating func fillRow(_ t: TensorRef, row: Int) {
            let cols = t.shape[1]
            let p = t.buffer.contents().assumingMemoryBound(to: UInt16.self) + row * cols
            for i in 0..<cols { p[i] = UInt16(truncatingIfNeeded: uniform().bitPattern >> 16) }
        }

        mutating func floats(_ count: Int, scale: Float = 1) -> Buf {
            let b = Buf(ctx.device, floats: count, label: "x")
            for i in 0..<count { b.floats[i] = uniform() * scale }
            return b
        }

        func copy(_ b: Buf, _ count: Int) -> Buf {
            let c = Buf(ctx.device, floats: count, label: "copy")
            memcpy(c.floats, b.floats, count * 4)
            return c
        }

        func run(_ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            try body(enc)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
        }
    }

    static func same(_ a: Buf, _ b: Buf, _ count: Int, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        let diff = (0..<count).first { a.floats[$0].bitPattern != b.floats[$0].bitPattern }
        XCTAssertNil(diff, "\(what): first difference at \(diff ?? -1)", file: file, line: line)
    }

    static let precisions: [(String, Precision)] = [("exact", .exact), ("reference", .reference)]

    func testGemvResidualMatchesGemvAndAdd() throws {
        var f = try Fixture()
        for (pname, precision) in Self.precisions {
            let k = f.kernels(precision)
            for (n, kd) in [(2048, 4096), (1000, 2048), (2048, 6144)] {
                let w = f.bf16([n, kd])
                let x = f.floats(kd)
                let base = f.floats(n)
                let proj = Buf(f.ctx.device, floats: n, label: "proj")
                let ref = f.copy(base, n), fused = f.copy(base, n)
                try f.run { enc in
                    try k.gemv(enc, w, x: x, y: proj, m: 1)
                    try k.residualAdd(enc, x: ref, y: proj, count: n)
                    try k.gemvResidual(enc, w, x: x, y: fused)
                }
                XCTAssertNotEqual(ref.floats[0], base.floats[0], "\(pname) \(n)x\(kd): the reference residual stayed at its base")
                Self.same(ref, fused, n, "\(pname) \(n)x\(kd) gemv_residual")
            }
        }
    }

    func testSwigluMatchesTwoGemvsAndSilu() throws {
        var f = try Fixture()
        let n = 1536, kd = 2048
        for (pname, precision) in Self.precisions {
            let k = f.kernels(precision)
            let w1 = f.bf16([n, kd]), w3 = f.bf16([n, kd])
            let th = f.floats(kd, scale: 3)
            let a = Buf(f.ctx.device, floats: n, label: "a"), b = Buf(f.ctx.device, floats: n, label: "b")
            let ref = Buf(f.ctx.device, floats: n, label: "ref"), act = Buf(f.ctx.device, floats: n, label: "act")
            try f.run { enc in
                try k.gemv(enc, w1, x: th, y: a, m: 1)
                try k.gemv(enc, w3, x: th, y: b, m: 1)
                try k.siluMul(enc, out: ref, a: a, b: b, count: n)
                try k.roundIfMimic(enc, x: ref, count: n)
                try k.swigluDecode(enc, w1: w1, w3: w3, x: th, y: act)
            }
            Self.same(ref, act, n, "\(pname) swiglu_dec")
        }
    }

    /// Needs the model's config.json for the embedding geometry; the tables are random.
    func testCatNormMatchesEmbedAndNorms() throws {
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("config.json of drama-3 needed for the tower geometry") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        var f = try Fixture()
        let H = config.hiddenSize, hiddenOffset = 2
        let table = f.bf16([config.codebookSize, H])
        let codebooks = f.zeroBF16([config.codebookSize * config.numCodebooks, H])
        let frames = Buf(f.ctx.device, floats: 11, label: "frames")
        for i in 1..<11 { frames.ints[i] = Int32((i * 37) % config.codebookSize) }
        for k in 2..<config.numCodebooks { f.fillRow(codebooks, row: k * config.codebookSize + Int(frames.ints[1 + k])) }
        let codes = Buf(f.ctx.device, floats: 1, label: "codes"), semanticNext = Buf(f.ctx.device, floats: 1, label: "semantic")
        codes.ints[0] = 17
        let cases: [(String, mainId: Int32, semantic: Int32, addUnseen: Bool)] = [
            ("semantic code", config.semanticStart + 5, 1, false),
            ("semantic code with unseen codebooks", config.semanticStart + 5, 1, true),
            ("non-semantic main", config.semanticStart - 1, 0, true),
        ]
        for (pname, precision) in Self.precisions {
            let k = f.kernels(precision)
            let enorm = f.bf16([H], scale: 0.5), hnorm = f.bf16([H], scale: 0.5)
            let hidden = f.floats(3 * H, scale: 2)
            for c in cases {
                frames.ints[0] = c.mainId
                semanticNext.ints[0] = c.semantic
                let e = Buf(f.ctx.device, floats: H, label: "e")
                let ref = Buf(f.ctx.device, floats: 2 * H, label: "ref"), cat = Buf(f.ctx.device, floats: 2 * H, label: "cat")
                try f.run { enc in
                    try k.towerEmbed(enc, out: e, codes: codes, semanticNext: semanticNext, table: table, frames: frames, framesOffset: 0,
                                     codebooks: codebooks, addUnseen: c.addUnseen, rows: 1, config: config)
                    try k.roundIfMimic(enc, x: e, count: H)
                    try k.rmsnorm(enc, out: ref, outStride: 2 * H, x: e, view: .contiguous(H), rows: 1, dim: H, w: enorm, eps: config.towerNormEps)
                    try k.rmsnorm(enc, out: ref, outOffset: H, outStride: 2 * H, x: hidden, xOffset: hiddenOffset * H, view: .contiguous(H), rows: 1, dim: H, w: hnorm, eps: config.towerNormEps)
                    try k.roundIfMimic(enc, x: ref, count: 2 * H)
                    try k.towerCatNorm(enc, out: cat, hidden: hidden, hiddenOffset: hiddenOffset * H, enorm: enorm, hnorm: hnorm, embedding: table,
                                       codes: codes, semanticNext: semanticNext, frames: frames, codebooks: codebooks, addUnseen: c.addUnseen, config: config)
                }
                if c.semantic != 0 { XCTAssertTrue((0..<H).contains { ref.floats[$0] != 0 }, "\(pname) \(c.0): the embedding half stayed zero") }
                Self.same(ref, cat, 2 * H, "\(pname) \(c.0) tower_cat_norm")
            }
        }
    }

    func testAttentionSplitGatedMatchesAttentionAndGate() throws {
        var f = try Fixture()
        let heads = 16, kvHeads = 2, capacity = 700
        let kcache = f.floats(capacity * kvHeads * 256, scale: 2)
        let vcache = f.floats(capacity * kvHeads * 256)
        let q = f.floats(heads * 256, scale: 2)
        let qkv = f.floats(9216, scale: 3)
        let qView = HeadView(heads: UInt32(heads), groupStride: 9216, headStride: 256)
        let count = heads * 256
        let partials = Buf(f.ctx.device, floats: Kernels.attentionPartialsCount(heads: heads), label: "partials")
        for (pname, precision) in Self.precisions {
            for split in KernelSchedule.attentionSplits {
                let k = f.kernels(precision, split: split)
                for pos in [0, 3, 17, 130, 650] {
                    let attn = Buf(f.ctx.device, floats: count, label: "attn")
                    let ref = Buf(f.ctx.device, floats: count, label: "ref"), gated = Buf(f.ctx.device, floats: count, label: "gated")
                    try f.run { enc in
                        try k.attention(enc, out: attn, q: q, kcache: kcache, vcache: vcache, heads: heads, kvHeads: kvHeads, rows: 1, pos0: pos)
                        try k.roundIfMimic(enc, x: attn, count: count)
                        try k.sigmoidGateMul(enc, out: ref, attn: attn, gate: qkv, gateOffset: heads * 256, view: qView, rows: heads)
                        try k.roundIfMimic(enc, x: ref, count: count)
                        try k.attentionSplitGated(enc, out: gated, q: q, kcache: kcache, vcache: vcache, partials: partials, gate: qkv, gateOffset: heads * 256,
                                                  heads: heads, kvHeads: kvHeads, pos0: pos)
                    }
                    Self.same(ref, gated, count, "\(pname) split \(split) pos \(pos) attention_split + combine_gate")
                }
            }
        }
    }
}
