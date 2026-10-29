// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Synthetic decode-time MoE layers of the real shapes (one row, top 8 of 256
/// FP8 experts, bf16 shared expert and router), encoded either as the twelve
/// dispatches of the M > 1 path or as the three decode dispatches, for
/// `drama kbench --moe-decode` and MoEDecodeParityTests. No model needed.
package final class MoEDecodeRig {
    static let hidden = 2048, inter = 512, shared = 512, experts = 256, topK = 8
    static let expertBytes = 3 * inter * hidden + 3 * (inter / 128) * (hidden / 128) * 4
    static let layerBytes = topK * expertBytes + (2 * shared * hidden + hidden * shared + experts * hidden + hidden) * 2

    /// One layer's weights. The table maps the 8 experts the router picks to
    /// 8 distinct physical experts and every other entry to the first.
    struct Layer {
        let pool: MTLBuffer
        let weights: MoEWeights
        let ids: Buf, rw: Buf, counts: Buf, lists: Buf, active: Buf
    }

    let ctx: MetalContext
    let k: Kernels
    let config: ModelConfig
    let layers: [Layer]
    let h: Buf, x: Buf, logits: Buf, act: Buf, slotOut: Buf
    let sharedG: Buf, sharedU: Buf, sharedAct: Buf, sharedOut: Buf, sharedGateOut: Buf
    let pre: Buf, ids: Buf, rw: Buf

    init(ctx: MetalContext, config: ModelConfig, precision: Precision, copies: Int, seed: UInt64 = 11) throws {
        precondition(config.hiddenSize == Self.hidden && config.moeIntermediate == Self.inter && config.sharedIntermediate == Self.shared
                     && config.numExperts == Self.experts && config.topK == Self.topK, "MoEDecode.metal is compiled for the drama-3 shapes")
        self.ctx = ctx
        self.config = config
        k = Kernels(ctx: ctx, precision: precision)
        let d = ctx.device
        var rng = SeededGenerator(seed: seed)
        func uniform() -> Float { Float(rng.uniformOpenZero()) * 2 - 1 }
        func bf16(_ name: String, _ rows: Int, _ cols: Int, scale: Float) -> TensorRef {
            let buf = d.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
            let p = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(rows * cols) { p[i] = UInt16(truncatingIfNeeded: (uniform() * scale).bitPattern >> 16) }
            return TensorRef(name: name, dtype: .bf16, shape: [rows, cols], buffer: buf, offset: 0)
        }
        h = Buf(d, floats: Self.hidden, label: "h")
        for i in 0..<Self.hidden { h.floats[i] = uniform() }
        x = Buf(d, floats: Self.hidden, label: "x")
        for i in 0..<Self.hidden { x.floats[i] = uniform() }
        logits = Buf(d, floats: Self.experts, label: "logits")
        act = Buf(d, floats: Self.topK * Self.inter, label: "act")
        slotOut = Buf(d, floats: Self.topK * Self.hidden, label: "slotOut")
        sharedG = Buf(d, floats: Self.shared, label: "sharedG")
        sharedU = Buf(d, floats: Self.shared, label: "sharedU")
        sharedAct = Buf(d, floats: Self.shared, label: "sharedAct")
        sharedOut = Buf(d, floats: Self.hidden, label: "sharedOut")
        sharedGateOut = Buf(d, floats: 1, label: "sharedGateOut")
        pre = Buf(d, floats: Kernels.moePreRows, label: "pre")
        ids = Buf(d, floats: Self.topK, label: "ids")
        rw = Buf(d, floats: Self.topK, label: "rw")

        let rb = Self.inter / 128, cb = Self.hidden / 128
        var built: [Layer] = []
        for _ in 0..<copies {
            let pool = d.makeBuffer(length: Self.topK * Self.expertBytes, options: .storageModeShared)!
            let bytes = pool.contents()
            var cursor = 0
            func fp8(_ count: Int) -> UInt64 {
                let p = (bytes + cursor).assumingMemoryBound(to: UInt64.self)
                for i in 0..<(count / 8) {
                    var v = rng.next()
                    let nan = (v & 0x7f7f7f7f7f7f7f7f) ^ 0x7f7f7f7f7f7f7f7f
                    for j in 0..<8 where (nan >> (8 * UInt64(j))) & 0xff == 0 { v &= ~(1 << (8 * UInt64(j))) }
                    p[i] = v
                }
                defer { cursor += count }
                return pool.gpuAddress + UInt64(cursor)
            }
            func scales(_ count: Int) -> UInt64 {
                let p = (bytes + cursor).assumingMemoryBound(to: Float.self)
                for i in 0..<count { p[i] = Float(0.002 + 0.004 * rng.uniformOpenZero()) }
                defer { cursor += count * 4 }
                return pool.gpuAddress + UInt64(cursor)
            }
            var physical: [[UInt64]] = []
            for _ in 0..<Self.topK {
                physical.append([fp8(Self.inter * Self.hidden), scales(rb * cb), fp8(Self.inter * Self.hidden), scales(rb * cb),
                                 fp8(Self.hidden * Self.inter), scales(cb * rb)])
            }
            let router = bf16("router", Self.experts, Self.hidden, scale: 0.05)
            let ids = Buf(d, floats: Self.topK, label: "ids"), rw = Buf(d, floats: Self.topK, label: "rw")
            let counts = Buf(d, floats: Self.experts + 1, label: "counts")
            let lists = Buf(d, floats: Self.experts * Self.topK, label: "lists")
            let active = Buf(d, floats: Self.experts, label: "active")
            let k = self.k, h = self.h, logits = self.logits
            try Self.run(ctx) { enc in
                try k.gemv(enc, router, x: h, y: logits, m: 1)
                try k.moeRouter(enc, ids: ids, weights: rw, logits: logits, rows: 1, config: config)
                try k.zero(enc, counts)
                try k.moeBucket(enc, counts: counts, lists: lists, ids: ids, active: active, pairs: Self.topK, capacity: Self.topK)
            }
            var table = [UInt64](repeating: 0, count: Self.experts * 6)
            for e in 0..<Self.experts { for j in 0..<6 { table[e * 6 + j] = physical[0][j] } }
            for s in 0..<Self.topK { for j in 0..<6 { table[Int(ids.ints[s]) * 6 + j] = physical[s][j] } }
            let weights = MoEWeights(router: router,
                                     sharedGate: bf16("sharedGate", Self.shared, Self.hidden, scale: 0.05),
                                     sharedUp: bf16("sharedUp", Self.shared, Self.hidden, scale: 0.05),
                                     sharedDown: bf16("sharedDown", Self.hidden, Self.shared, scale: 0.05),
                                     sharedExpertGate: bf16("expertGate", 1, Self.hidden, scale: 0.05),
                                     experts: d.makeBuffer(bytes: table, length: table.count * 8, options: .storageModeShared)!)
            built.append(Layer(pool: pool, weights: weights, ids: ids, rw: rw, counts: counts, lists: lists, active: active))
        }
        layers = built
    }

    static func run(_ ctx: MetalContext, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
    }

    /// Backbone.moe's M > 1 path run at one row: all twelve dispatches.
    func shippedLayer(_ enc: MTLComputeCommandEncoder, _ l: Layer) throws {
        enc.useResource(l.pool, usage: .read)
        let w = l.weights
        try k.gemv(enc, w.router, x: h, y: logits, m: 1)
        try k.moeRouter(enc, ids: l.ids, weights: l.rw, logits: logits, rows: 1, config: config)
        try k.zero(enc, l.counts)
        try k.moeBucket(enc, counts: l.counts, lists: l.lists, ids: l.ids, active: l.active, pairs: Self.topK, capacity: Self.topK)
        try k.moeGateUp(enc, act: act, x: h, experts: w.experts, counts: l.counts, lists: l.lists, active: l.active, maxActive: Self.topK, capacity: Self.topK, rows: 1, config: config)
        try k.moeDown(enc, slotOut: slotOut, act: act, weights: l.rw, experts: w.experts, counts: l.counts, lists: l.lists, active: l.active, maxActive: Self.topK, capacity: Self.topK, rows: 1, config: config)
        try k.gemv(enc, w.sharedGate, x: h, y: sharedG, m: 1)
        try k.gemv(enc, w.sharedUp, x: h, y: sharedU, m: 1)
        try k.siluMul(enc, out: sharedAct, a: sharedG, b: sharedU, count: Self.shared)
        try k.roundIfMimic(enc, x: sharedAct, count: Self.shared)
        try k.gemv(enc, w.sharedDown, x: sharedAct, y: sharedOut, m: 1)
        try k.gemv(enc, w.sharedExpertGate, x: h, y: sharedGateOut, m: 1)
        try k.moeCombine(enc, x: x, slotOut: slotOut, sharedOut: sharedOut, sharedGate: sharedGateOut, rows: 1, config: config)
    }

    /// The decode layer: moe_pre_bf16, moe_mid_fp8, moe_post_fp8, with `kernels` in place of the rig's own when given.
    func decodeLayer(_ enc: MTLComputeCommandEncoder, _ l: Layer, kernels: Kernels? = nil) throws {
        let k = kernels ?? self.k
        enc.useResource(l.pool, usage: .read)
        let w = l.weights
        try k.moePre(enc, pre: pre, h: h, w: w)
        try k.moeMid(enc, act: act, h: h, experts: w.experts, pre: pre, ids: ids, rw: rw, config: config)
        try k.moePost(enc, x: x, act: act, experts: w.experts, ids: ids, rw: rw, pre: pre, sharedDown: w.sharedDown, config: config)
    }
}

extension KernelBench {
    struct Timing {
        let samples: [Double]
        var median: Double { samples.sorted()[samples.count / 2] }
        var min: Double { samples.min()! }
        /// Median over rounds of base / self, each round's pair run back to back.
        func pairedSpeedup(over base: Timing) -> Double {
            let r = zip(base.samples, samples).map { $0 / $1 }.sorted()
            return r[r.count / 2]
        }
    }

    typealias DecodeVariant = (name: String, body: (MTLComputeCommandEncoder, MoEDecodeRig.Layer) throws -> Void)

    /// Seconds per layer of each variant. Every sample is one command buffer
    /// of `layers` layers, layer i on weight copy i % copies. The variants take
    /// turns within each round so they share clocks and load, and up to 16
    /// command buffers stay queued so the GPU never idles between samples (an
    /// idle gap drops its clock, as between frames it does not). 5 warm-up
    /// rounds, then `repeats` timed ones.
    static func perLayer(_ rig: MoEDecodeRig, _ variants: [DecodeVariant], repeats: Int, layers: Int) throws -> [Timing] {
        var samples = [[Double]](repeating: [], count: variants.count)
        var pending: [(MTLCommandBuffer, Int, Bool)] = []
        func retire() throws {
            let (cb, vi, timed) = pending.removeFirst()
            cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("\(variants[vi].name): \(e)") }
            if timed { samples[vi].append((cb.gpuEndTime - cb.gpuStartTime) / Double(layers)) }
        }
        for r in 0..<(5 + repeats) {
            for (vi, v) in variants.enumerated() {
                let cb = rig.ctx.queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                for i in 0..<layers { try v.body(enc, rig.layers[i % rig.layers.count]) }
                enc.endEncoding()
                cb.commit()
                pending.append((cb, vi, r >= 5))
                if pending.count > 16 { try retire() }
            }
        }
        while !pending.isEmpty { try retire() }
        return samples.map { Timing(samples: $0) }
    }

    /// `drama kbench --moe-decode`: the twelve-dispatch and the three-dispatch
    /// decode MoE layer on 8 rotating copies of a synthetic layer (192 MB of
    /// experts, past the SLC). GB/s from the minimum, since other processes on
    /// the GPU only ever add time.
    package static func moeDecode(layers: Int = 40, repeats: Int = 20) throws {
        let ctx = try MetalContext()
        guard let loc = ModelLocation.huggingFaceCache() else { throw EngineError.badModel("config.json needed for the MoE shapes") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        let rig = try MoEDecodeRig(ctx: ctx, config: config, precision: .exact, copies: 8)
        var variants: [DecodeVariant] = [("twelve dispatches", { enc, l in try rig.shippedLayer(enc, l) }),
                                         ("three dispatches", { enc, l in try rig.decodeLayer(enc, l) })]
        for sg in [32, 16, 8, 4] where sg != rig.k.moePostSimdgroups {
            var k = rig.k
            k.moePostSimdgroups = sg
            variants.append(("post x\(sg)sg, \(2048 / (2 * sg)) tg", { enc, l in try rig.decodeLayer(enc, l, kernels: k) }))
        }
        _ = try perLayer(rig, [variants[0]], repeats: 1, layers: layers)
        let ts = try perLayer(rig, variants, repeats: repeats, layers: layers)
        print("decode MoE layer, \(layers) layers per command buffer, \(repeats) repeats (per layer; GB/s at the min; speedup = median paired ratio to the first line)")
        for (v, t) in zip(variants, ts) {
            print("  " + v.name.padding(toLength: 20, withPad: " ", startingAt: 0)
                  + String(format: "min %6.1f  med %6.1f us  %4.0f GB/s  %5.3fx", t.min * 1e6, t.median * 1e6, Double(MoEDecodeRig.layerBytes) / t.min / 1e9, t.pairedSpeedup(over: ts[0])))
        }
    }
}
