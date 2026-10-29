// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// One-row (decode) tower kernels. Each replaces a run of the dispatches the
/// prefill path still encodes, bit for bit (TowerDecodeTests).
extension Kernels {
    private struct GemvDecodeParams { var N: UInt32; var K: UInt32; var roundOutput: UInt32 }
    private static let towerGemvSimdgroups = 4

    private func gemvDecodeParams(_ w: TensorRef) -> GemvDecodeParams {
        precondition(w.shape[1] % 256 == 0)
        return GemvDecodeParams(N: UInt32(w.shape[0]), K: UInt32(w.shape[1]), roundOutput: precision.mimicBF16 ? 1 : 0)
    }

    private func dispatchRows(_ enc: MTLComputeCommandEncoder, _ name: String, rows n: Int) throws {
        let g = Self.towerGemvSimdgroups
        try dispatchGroups(enc, name, groups: MTLSize(width: (n + g - 1) / g, height: 1, depth: 1), group: MTLSize(width: 32 * g, height: 1, depth: 1))
    }

    /// y += W x for one row: the GEMV and its residual add in one dispatch.
    func gemvResidual(_ enc: MTLComputeCommandEncoder, _ w: TensorRef, x: Buf, y: Buf) throws {
        enc.setBuffer(w.buffer, offset: w.offset, index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        enc.setBuffer(y.buffer, offset: y.byteOffset, index: 2)
        set(enc, gemvDecodeParams(w), index: 3)
        try dispatchRows(enc, "gemv_residual", rows: w.shape[0])
    }

    /// y = silu(W1 x) * (W3 x) for one row: both GEMVs and silu_mul in one dispatch.
    func swigluDecode(_ enc: MTLComputeCommandEncoder, w1: TensorRef, w3: TensorRef, x: Buf, y: Buf) throws {
        precondition(w3.shape == w1.shape)
        enc.setBuffer(w1.buffer, offset: w1.offset, index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        enc.setBuffer(y.buffer, offset: y.byteOffset, index: 2)
        set(enc, gemvDecodeParams(w1), index: 3)
        enc.setBuffer(w3.buffer, offset: w3.offset, index: 4)
        try dispatchRows(enc, "swiglu_dec", rows: w1.shape[0])
    }

    /// The tower layer's concat input [enorm(embedding) | hnorm(hidden)] for one
    /// row: tower_embed, both norms and the reference roundings in one dispatch.
    func towerCatNorm(_ enc: MTLComputeCommandEncoder, out: Buf, hidden: Buf, hiddenOffset: Int, enorm: TensorRef, hnorm: TensorRef, embedding: TensorRef,
                      codes: Buf, semanticNext: Buf, frames: Buf, codebooks: TensorRef, addUnseen: Bool, config: ModelConfig) throws {
        struct P { var dim: UInt32; var eps: Float; var roundOutput: UInt32; var codebookSize: UInt32; var numCodebooks: UInt32
                   var addUnseen: UInt32; var semanticStart: Int32; var semanticEnd: Int32 }
        precondition(config.hiddenSize <= 4096)
        enc.setBuffer(out.buffer, offset: out.byteOffset, index: 0)
        enc.setBuffer(hidden.buffer, offset: hidden.byteOffset + hiddenOffset * 4, index: 1)
        enc.setBuffer(enorm.buffer, offset: enorm.offset, index: 2)
        enc.setBuffer(hnorm.buffer, offset: hnorm.offset, index: 3)
        enc.setBuffer(codes.buffer, offset: codes.byteOffset, index: 4)
        enc.setBuffer(semanticNext.buffer, offset: semanticNext.byteOffset, index: 5)
        enc.setBuffer(embedding.buffer, offset: embedding.offset, index: 6)
        enc.setBuffer(frames.buffer, offset: frames.byteOffset, index: 7)
        enc.setBuffer(codebooks.buffer, offset: codebooks.offset, index: 8)
        set(enc, P(dim: UInt32(config.hiddenSize), eps: config.towerNormEps, roundOutput: precision.mimicBF16 ? 1 : 0,
                   codebookSize: UInt32(config.codebookSize), numCodebooks: UInt32(config.numCodebooks), addUnseen: addUnseen ? 1 : 0,
                   semanticStart: config.semanticStart, semanticEnd: config.semanticEnd), index: 9)
        try dispatchGroups(enc, "tower_cat_norm", groups: MTLSize(width: 2, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Floats `partials` must hold for `attentionSplitGated`.
    static func attentionPartialsCount(heads: Int) -> Int { heads * KernelSchedule.attentionSplits.max()! * 258 }

    /// Gated attention for one row: each head's key residues over up to 8
    /// threadgroups, combined in residue order, then the sigmoid output gate;
    /// bit-identical to attention, sigmoid_gate_mul and the reference roundings.
    func attentionSplitGated(_ enc: MTLComputeCommandEncoder, out: Buf, q: Buf, kcache: Buf, vcache: Buf, partials: Buf, gate: Buf, gateOffset: Int,
                             heads: Int, kvHeads: Int, pos0: Int) throws {
        struct P { var heads: UInt32; var kvHeads: UInt32; var pos0: UInt32; var scale: Float; var roundOutput: UInt32 }
        let split = schedule.attentionSplitDecode
        let blocks = min(8, split)
        precondition(split % blocks == 0 && partials.count >= heads * split * 258)
        let params = P(heads: UInt32(heads), kvHeads: UInt32(kvHeads), pos0: UInt32(pos0), scale: 1.0 / 16.0, roundOutput: precision.mimicBF16 ? 1 : 0)
        enc.setBuffer(partials.buffer, offset: partials.byteOffset, index: 0)
        enc.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        enc.setBuffer(kcache.buffer, offset: kcache.byteOffset, index: 2)
        enc.setBuffer(vcache.buffer, offset: vcache.byteOffset, index: 3)
        set(enc, params, index: 4)
        set(enc, UInt32(split), index: 5)
        try dispatchGroups(enc, "attention_split_u2", groups: MTLSize(width: heads, height: blocks, depth: 1),
                           group: MTLSize(width: 32 * split / blocks, height: 1, depth: 1))
        enc.setBuffer(out.buffer, offset: out.byteOffset, index: 0)
        enc.setBuffer(partials.buffer, offset: partials.byteOffset, index: 1)
        set(enc, params, index: 4)
        set(enc, UInt32(split), index: 5)
        enc.setBuffer(gate.buffer, offset: gate.byteOffset + gateOffset * 4, index: 6)
        try dispatchGroups(enc, "attention_combine_gate", groups: MTLSize(width: heads, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
    }
}

extension DramaModel {
    private static func median(_ v: [Double]) -> Double { let s = v.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }

    private func towerSamplerParams(pos: Int, head: Int, main: Bool) -> SamplerKernelParams {
        SamplerKernelParams(SamplingParams(temperature: 0.7, topK: 30, topP: 0.7), n: main ? config.codebookSize + 1 : config.codebookSize,
                            rngIn: main ? 0 : 1, rngOut: 1, position: pos, head: head, eos: config.eos, semanticStart: config.semanticStart,
                            codebookSize: config.codebookSize)
    }

    /// The tower part of a decode frame at position `pos` as `encodeFrame`
    /// encodes it: the main sampler, then the shared layers, residual layer i,
    /// head i and its sampler for each head.
    private func encodeTowerStep(_ enc: MTLComputeCommandEncoder, pos: Int, samplers: Bool) throws {
        let k = tower.k
        let frames = backbone.framesView(pos)
        let codebooks = weights.tower.codebookEmbeddings
        if samplers {
            try k.sampleMain(enc, logits: backbone.cand, rng: sampler.rng, frames: backbone.frames, code0: tower.codes[0], semantic: tower.semantic,
                             status: sampler.status, params: towerSamplerParams(pos: pos, head: 0, main: true))
        }
        try tower.decodeShared(enc, hidden: backbone.hiddenOut, hiddenRow: 0, frames: frames, pos: pos, codebooks: codebooks)
        for i in 0..<(config.numCodebooks - 1) {
            let out = try tower.residualStep(enc, i, frames: frames, pos: pos, codebooks: codebooks)
            try tower.head(enc, i, input: out)
            if samplers {
                try k.sampleTower(enc, logits: tower.headLogits, logitsOffset: i * config.codebookSize, rng: sampler.rng, frames: backbone.frames,
                                  codeOut: tower.codes[i + 1], status: sampler.status, params: towerSamplerParams(pos: pos, head: i, main: false))
            }
        }
    }

    /// The tower decode step taken apart: GPU time of each kernel group of one
    /// residual layer and its head, rotating over the nine residual layers so
    /// every call streams cold weights, scaled to a frame; and the whole step
    /// as the real frame encodes it, with and without the samplers. Returns
    /// report lines.
    package func profileTower(pos: Int, rounds: Int) throws -> [String] {
        precondition(pos + 2 < capacity)
        let c = config
        let H = c.hiddenSize
        var rng = SeededGenerator(seed: 21)
        for i in 0..<H { backbone.hiddenOut.floats[i] = Float(rng.uniformOpenZero()) * 2 - 1 }
        backbone.cand.floats[0] = -1e4
        for i in 1...c.codebookSize { backbone.cand.floats[i] = Float(rng.uniformOpenZero()) * 4 }
        let fr = backbone.frames.ints + pos * 11
        fr[0] = c.semanticStart + 17
        for i in 1..<11 { fr[i] = Int32((i * 37) % c.codebookSize) }
        tower.setMain(row: 0, main: c.semanticStart + 17)
        for i in 1..<c.numCodebooks { tower.setCode(row: 0, index: i, value: Int32((i * 101) % c.codebookSize)) }
        sampler.setState(SIMD4(1, 2, 3, 4), slot: 0)
        sampler.setState(SIMD4(5, 6, 7, 8), slot: 1)
        let k = tower.k
        let iterations = 18
        func time(_ body: (MTLComputeCommandEncoder, Int) throws -> Void) throws -> Double {
            let (cb, enc) = commandBuffer()
            for it in 0..<iterations { try body(enc, it % 9) }
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(iterations) * 1000
        }
        let heads = c.towerHeads, kvHeads = c.towerKVHeads
        let frames = backbone.framesView(pos)
        let codebooks = weights.tower.codebookEmbeddings
        func lw(_ j: Int) -> TowerLayerWeights { tower.weights.residual[j] }
        func cacheIndex(_ j: Int) -> Int { c.towerLayers + j }
        struct Group { var name: String; var perFrame: Int; var dispatches: Int; var bytes: (Int) -> Int; var body: (MTLComputeCommandEncoder, Int) throws -> Void }
        let L = c.towerLayers + c.numCodebooks - 1, nh = c.numCodebooks - 1
        let t = tower
        let groups: [Group] = [
            Group(name: "embed+norm.cat", perFrame: L, dispatches: 1, bytes: { _ in 0 }) { enc, j in
                try k.towerCatNorm(enc, out: t.cat, hidden: self.backbone.hiddenOut, hiddenOffset: 0, enorm: lw(j).enorm, hnorm: lw(j).hnorm, embedding: lw(j).embedding,
                                   codes: t.codes[j], semanticNext: t.semantic, frames: frames, codebooks: codebooks, addUnseen: false, config: c) },
            Group(name: "gemv.eh", perFrame: L, dispatches: 1, bytes: { lw($0).ehProj.byteCount }) { enc, j in try k.gemv(enc, lw(j).ehProj, x: t.cat, y: t.mixed, m: 1) },
            Group(name: "norm.attn", perFrame: L, dispatches: 1, bytes: { _ in 0 }) { enc, j in
                try k.rmsnorm(enc, out: t.th, outStride: H, x: t.mixed, view: .contiguous(H), rows: 1, dim: H, w: lw(j).attentionNorm, eps: c.towerNormEps) },
            Group(name: "gemv.wqkv", perFrame: L, dispatches: 1, bytes: { lw($0).wqkv.byteCount }) { enc, j in try k.gemv(enc, lw(j).wqkv, x: t.th, y: t.tqkv, m: 1) },
            Group(name: "norm_rope_kv", perFrame: L, dispatches: 1, bytes: { _ in 0 }) { enc, j in try t.normRopeKV(enc, lw(j), index: cacheIndex(j), rows: 1, pos0: pos) },
            Group(name: "attention split+gate", perFrame: L, dispatches: 2, bytes: { _ in 2 * (pos + 1) * kvHeads * 256 * 4 }) { enc, j in
                try k.attentionSplitGated(enc, out: t.gated, q: t.tqn, kcache: t.kcache[cacheIndex(j)], vcache: t.vcache[cacheIndex(j)], partials: t.attnPartials,
                                          gate: t.tqkv, gateOffset: heads * 256, heads: heads, kvHeads: kvHeads, pos0: pos) },
            Group(name: "gemv.wo+residual", perFrame: L, dispatches: 1, bytes: { lw($0).wo.byteCount }) { enc, j in try k.gemvResidual(enc, lw(j).wo, x: t.gated, y: t.mixed) },
            Group(name: "norm.ffn", perFrame: L, dispatches: 1, bytes: { _ in 0 }) { enc, j in
                try k.rmsnorm(enc, out: t.th, outStride: H, x: t.mixed, view: .contiguous(H), rows: 1, dim: H, w: lw(j).ffnNorm, eps: c.towerNormEps) },
            Group(name: "swiglu (w1+w3+silu)", perFrame: L, dispatches: 1, bytes: { lw($0).w1.byteCount + lw($0).w3.byteCount }) { enc, j in
                try k.swigluDecode(enc, w1: lw(j).w1, w3: lw(j).w3, x: t.th, y: t.ffAct) },
            Group(name: "gemv.w2+residual", perFrame: L, dispatches: 1, bytes: { lw($0).w2.byteCount }) { enc, j in try k.gemvResidual(enc, lw(j).w2, x: t.ffAct, y: t.mixed) },
            Group(name: "norm.final", perFrame: L, dispatches: 1, bytes: { _ in 0 }) { enc, j in
                try k.rmsnorm(enc, out: t.hiddenA, outStride: H, x: t.mixed, view: .contiguous(H), rows: 1, dim: H, w: lw(j).finalNorm, eps: c.towerNormEps) },
            Group(name: "gemv.head", perFrame: nh, dispatches: 1, bytes: { t.weights.heads[$0].byteCount }) { enc, j in
                try k.gemv(enc, t.weights.heads[j], x: t.hiddenA, y: t.headLogits, yOffset: j * c.codebookSize, m: 1) },
            Group(name: "sample.tower", perFrame: nh, dispatches: 1, bytes: { _ in 0 }) { enc, j in
                try k.sampleTower(enc, logits: t.headLogits, logitsOffset: j * c.codebookSize, rng: self.sampler.rng, frames: self.backbone.frames,
                                  codeOut: t.codes[j + 1], status: self.sampler.status, params: self.towerSamplerParams(pos: pos, head: j, main: false)) },
            Group(name: "sample.main", perFrame: 1, dispatches: 1, bytes: { _ in 0 }) { enc, _ in
                try k.sampleMain(enc, logits: self.backbone.cand, rng: self.sampler.rng, frames: self.backbone.frames, code0: t.codes[0], semantic: t.semantic,
                                 status: self.sampler.status, params: self.towerSamplerParams(pos: pos, head: 0, main: true)) },
        ]
        func chain(samplers: Bool) throws -> Double {
            let (cb, enc) = commandBuffer()
            try encodeTowerStep(enc, pos: pos, samplers: samplers)
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        var groupMs = [[Double]](repeating: [], count: groups.count)
        var chainMs: [Double] = [], chainNoSampleMs: [Double] = []
        _ = try chain(samplers: true)
        for g in groups { _ = try time(g.body) }
        for _ in 0..<rounds {
            for (i, g) in groups.enumerated() { groupMs[i].append(try time(g.body)) }
            chainMs.append(try chain(samplers: true))
            chainNoSampleMs.append(try chain(samplers: false))
        }
        func fmt(_ v: [Double], scale: Double = 1) -> String {
            String(format: "%7.3f [%6.3f..%6.3f]", Self.median(v) * scale, v.min()! * scale, v.max()! * scale)
        }
        var out: [String] = []
        out.append("tower decode step at position \(pos), \(rounds) rounds; group times are per call over \(iterations) calls rotating the 9 residual layers")
        out.append("group".padding(toLength: 28, withPad: " ", startingAt: 0) + "disp  ms/call median [min..max]      x/frame  ms/frame [min..max]           GB/s")
        var sum = 0.0, sumMin = 0.0, sumMax = 0.0
        for (i, g) in groups.enumerated() {
            let v = groupMs[i]
            let bytes = (0..<9).map(g.bytes).reduce(0, +) / 9
            let gbs = bytes > 0 ? String(format: "%5.0f [%3.0f..%3.0f]", Double(bytes) / (Self.median(v) / 1000) / 1e9, Double(bytes) / (v.max()! / 1000) / 1e9, Double(bytes) / (v.min()! / 1000) / 1e9) : ""
            sum += Self.median(v) * Double(g.perFrame); sumMin += v.min()! * Double(g.perFrame); sumMax += v.max()! * Double(g.perFrame)
            out.append(g.name.padding(toLength: 28, withPad: " ", startingAt: 0) + String(format: "%3d  ", g.dispatches) + fmt(v) + String(format: "  x%-3d ", g.perFrame)
                       + fmt(v, scale: Double(g.perFrame)) + "  " + gbs)
        }
        out.append(String(format: "sum of groups per frame: %.3f ms [%.3f..%.3f]", sum, sumMin, sumMax))
        out.append("chain with samplers:    " + fmt(chainMs) + " ms")
        out.append("chain without samplers: " + fmt(chainNoSampleMs) + " ms (layers and heads)")
        let samplerGroups = groups.enumerated().filter { $0.element.name.hasPrefix("sample.") }.map { Self.median(groupMs[$0.offset]) * Double($0.element.perFrame) }.reduce(0, +)
        out.append(String(format: "chain minus sum of groups: with samplers %+.3f ms, without samplers %+.3f ms (medians)", Self.median(chainMs) - sum, Self.median(chainNoSampleMs) - (sum - samplerGroups)))
        return out
    }
}
