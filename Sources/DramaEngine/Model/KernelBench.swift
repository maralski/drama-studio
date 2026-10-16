import Foundation
import Metal

/// GPU time of single kernels on synthetic data, for choosing tile shapes.
/// No model needed.
public enum KernelBench {
    static func gpuSeconds(_ ctx: MetalContext, iterations: Int, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for _ in 0..<iterations { try body(enc) }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
        return (cb.gpuEndTime - cb.gpuStartTime) / Double(iterations)
    }

    static func bf16Weights(_ device: MTLDevice, rows: Int, cols: Int) -> WeightMatrix {
        let buf = device.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: UInt16.self)
        var rng = SeededGenerator(seed: 3)
        for i in 0..<(rows * cols) { p[i] = UInt16(truncatingIfNeeded: Float(rng.uniformOpenZero() * 2 - 1).bitPattern >> 16) }
        return WeightMatrix(name: "w", format: .bf16, rows: rows, cols: cols, buffer: buf, offset: 0, paramOffset: 0)
    }

    /// Random FP8 experts with block scales, laid out like the model's, plus a
    /// routing of `rows` tokens over them.
    struct SyntheticMoE {
        let config: ModelConfig
        let table: ExpertTable
        let buffers: [MTLBuffer]
        let x: Buf, ids: Buf, rw: Buf, counts: Buf, lists: Buf, active: Buf, act: Buf, slotOut: Buf
        let rows: Int, experts: Int, capacity: Int

        init(device d: MTLDevice, config: ModelConfig, rows: Int, experts: Int, maxRows: Int, seed: UInt64 = 5) {
            self.config = config
            self.rows = rows
            self.experts = experts
            let hidden = config.hiddenSize, inter = config.moeIntermediate, topK = config.topK
            capacity = maxRows * topK
            let rb = inter / 128, cb = hidden / 128
            var rng = SeededGenerator(seed: seed)
            let perExpert = 3 * inter * hidden + (2 * rb * cb + cb * rb) * 4
            let pool = d.makeBuffer(length: experts * perExpert, options: .storageModeShared)!
            let bytes = pool.contents()
            var cursor = 0
            func fp8(_ count: Int) -> Int {
                let p = (bytes + cursor).assumingMemoryBound(to: UInt64.self)
                for i in 0..<(count / 8) {
                    var v = rng.next()
                    // Keep every byte finite: clear bit 0 of a 0x7f or 0xff pattern.
                    let nan = (v & 0x7f7f7f7f7f7f7f7f) ^ 0x7f7f7f7f7f7f7f7f
                    var mask: UInt64 = 0
                    for j in 0..<8 where (nan >> (8 * UInt64(j))) & 0xff == 0 { mask |= 1 << (8 * UInt64(j)) }
                    v &= ~mask
                    p[i] = v
                }
                defer { cursor += count }
                return cursor
            }
            func scales(_ count: Int) -> Int {
                let p = (bytes + cursor).assumingMemoryBound(to: Float.self)
                for i in 0..<count { p[i] = 0.5 + Float(rng.uniformOpenZero()) }
                defer { cursor += count * 4 }
                return cursor
            }
            var addresses = [UInt64](repeating: 0, count: config.numExperts * 6)
            for e in 0..<experts {
                let offsets = [fp8(inter * hidden), scales(rb * cb), fp8(inter * hidden), scales(rb * cb), fp8(hidden * inter), scales(cb * rb)]
                for (j, o) in offsets.enumerated() { addresses[e * 6 + j] = pool.gpuAddress + UInt64(o) }
            }
            buffers = [pool]
            table = ExpertTable(table: d.makeBuffer(bytes: addresses, length: addresses.count * 8, options: .storageModeShared)!, format: .fp8BlockScaled)
            x = Buf(d, floats: rows * hidden, label: "x")
            for i in 0..<(rows * hidden) { x.floats[i] = Float(rng.uniformOpenZero()) * 2 - 1 }
            ids = Buf(d, floats: rows * topK, label: "ids"); rw = Buf(d, floats: rows * topK, label: "rw")
            for m in 0..<rows {
                var picked = Set<Int32>()
                while picked.count < topK { picked.insert(Int32(rng.next() % UInt64(experts))) }
                for (s, e) in picked.sorted().enumerated() { ids.ints[m * topK + s] = e; rw.floats[m * topK + s] = Float(rng.uniformOpenZero()) }
            }
            counts = Buf(d, floats: config.numExperts + 1, label: "counts")
            lists = Buf(d, floats: config.numExperts * capacity, label: "lists")
            active = Buf(d, floats: config.numExperts, label: "active")
            act = Buf(d, floats: rows * topK * inter, label: "act")
            slotOut = Buf(d, floats: rows * topK * config.hiddenSize, label: "slotOut")
        }

        /// The first `rows` rows of `other`, sharing its experts: the same
        /// tokens routed the same way, in a smaller chunk.
        init(prefixOf other: SyntheticMoE, rows: Int, device d: MTLDevice) {
            config = other.config; self.rows = rows; experts = other.experts; capacity = other.capacity
            table = other.table; buffers = other.buffers
            let hidden = config.hiddenSize, topK = config.topK
            x = Buf(d, floats: rows * hidden, label: "x"); memcpy(x.floats, other.x.floats, rows * hidden * 4)
            ids = Buf(d, floats: rows * topK, label: "ids"); memcpy(ids.floats, other.ids.floats, rows * topK * 4)
            rw = Buf(d, floats: rows * topK, label: "rw"); memcpy(rw.floats, other.rw.floats, rows * topK * 4)
            counts = Buf(d, floats: config.numExperts + 1, label: "counts")
            lists = Buf(d, floats: config.numExperts * capacity, label: "lists")
            active = Buf(d, floats: config.numExperts, label: "active")
            act = Buf(d, floats: rows * topK * config.moeIntermediate, label: "act")
            slotOut = Buf(d, floats: rows * topK * hidden, label: "slotOut")
        }

        private init(_ other: SyntheticMoE, table: ExpertTable, buffers: [MTLBuffer]) {
            config = other.config; rows = other.rows; experts = other.experts; capacity = other.capacity
            x = other.x; ids = other.ids; rw = other.rw; counts = other.counts; lists = other.lists; active = other.active
            act = other.act; slotOut = other.slotOut
            self.table = table
            self.buffers = buffers
        }

        /// The same tokens, routing and experts with the experts quantized to
        /// int4 by the engine's quantizer, expert-major per matrix kind as the
        /// cache file lays them out.
        func quantizedInt4(_ ctx: MetalContext) throws -> SyntheticMoE {
            precondition(table.format == .fp8BlockScaled)
            func aligned(_ n: Int) -> Int { (n + 255) / 256 * 256 }
            var slots: [(kind: ExpertMatrix, weights: Int, params: Int)] = []
            var total = 0
            for kind in ExpertMatrix.allCases {
                let (rows, cols) = kind.shape(config)
                let weights = total
                total += aligned(experts * WeightFormat.int4.weightBytes(rows: rows, cols: cols))
                let params = total
                total += aligned(experts * WeightFormat.int4.paramBytes(rows: rows, cols: cols))
                slots.append((kind, weights, params))
            }
            let pool = ctx.device.makeBuffer(length: total, options: .storageModeShared)!
            var addresses = [UInt64](repeating: 0, count: config.numExperts * 6)
            let k = Kernels(ctx: ctx, precision: .exact)
            _ = try gpuSeconds(ctx, iterations: 1) { enc in
                enc.useResources(buffers, usage: .read)
                for slot in slots {
                    let (rows, cols) = slot.kind.shape(config)
                    try k.quantizeExpertsInt4(enc, experts: table, kind: slot.kind, count: experts, rows: rows, cols: cols,
                                              into: QuantizedSlot(buffer: pool, offset: slot.weights, paramOffset: slot.params))
                    for e in 0..<experts {
                        addresses[e * 6 + slot.kind.rawValue * 2] = pool.gpuAddress + UInt64(slot.weights + e * WeightFormat.int4.weightBytes(rows: rows, cols: cols))
                        addresses[e * 6 + slot.kind.rawValue * 2 + 1] = pool.gpuAddress + UInt64(slot.params + e * WeightFormat.int4.paramBytes(rows: rows, cols: cols))
                    }
                }
            }
            let int4Table = ExpertTable(table: ctx.device.makeBuffer(bytes: addresses, length: addresses.count * 8, options: .storageModeShared)!, format: .int4)
            return SyntheticMoE(self, table: int4Table, buffers: [pool])
        }

        var maxActive: Int { min(rows * config.topK, config.numExperts) }

        func bucket(_ enc: MTLComputeCommandEncoder, _ k: Kernels) throws {
            enc.useResources(buffers, usage: .read)
            try k.zero(enc, counts)
            try k.moeBucket(enc, counts: counts, lists: lists, ids: ids, active: active, pairs: rows * config.topK, capacity: capacity)
        }
        func gateUp(_ enc: MTLComputeCommandEncoder, _ k: Kernels) throws {
            enc.useResources(buffers, usage: .read)
            try k.moeGateUp(enc, act: act, x: x, experts: table, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacity, rows: rows, config: config)
        }
        func down(_ enc: MTLComputeCommandEncoder, _ k: Kernels) throws {
            enc.useResources(buffers, usage: .read)
            try k.moeDown(enc, slotOut: slotOut, act: act, weights: rw, experts: table, counts: counts, lists: lists, active: active, maxActive: maxActive, capacity: capacity, rows: rows, config: config)
        }
    }

    /// Prints GPU ms per call of the row kernels and every tiled kernel over
    /// all 256 synthetic experts (768 MB of FP8 weights, as in one real layer).
    public static func moe(rowCounts: [Int], iterations: Int = 5) throws {
        let ctx = try MetalContext()
        guard let loc = ModelLocation.huggingFaceCache() else { throw EngineError.badModel("config.json needed for the MoE shapes") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        var rowKernels = Kernels(ctx: ctx, precision: .exact)
        rowKernels.forceMoeRows = true
        var variants: [(String, Kernels)] = [("rows", rowKernels)]
        for tile in KernelSchedule.moeTiles {
            var s = KernelSchedule.default
            s.moeTile = tile
            variants.append((tile.label, Kernels(ctx: ctx, precision: .exact, schedule: s)))
        }
        for rows in rowCounts {
            let moe = SyntheticMoE(device: ctx.device, config: config, rows: rows, experts: config.numExperts, maxRows: max(rows, 32))
            _ = try gpuSeconds(ctx, iterations: 1) { enc in try moe.bucket(enc, rowKernels) }
            let activeCount = Int(moe.counts.ints[config.numExperts])
            var line = String(format: "rows %4d (%3d experts active):", rows, activeCount)
            for (label, kernels) in variants {
                let g = try gpuSeconds(ctx, iterations: iterations) { enc in try moe.gateUp(enc, kernels) }
                let dn = try gpuSeconds(ctx, iterations: iterations) { enc in try moe.down(enc, kernels) }
                line += String(format: "  %@ %.2f+%.2f", label, g * 1000, dn * 1000)
            }
            print(line + "  (gateup+down ms per layer)")
        }
    }

    /// Prints GPU ms per call of the GEMV and every GEMM tile for y = x W^T.
    public static func denseMatmul(n: Int, k: Int, ms: [Int], iterations: Int = 5) throws {
        let ctx = try MetalContext()
        let w = bf16Weights(ctx.device, rows: n, cols: k)
        var gemv = Kernels(ctx: ctx, precision: .exact)
        gemv.forceGemv = true
        var variants: [(String, Kernels)] = [("gemv", gemv)]
        for tile in KernelSchedule.gemmTiles {
            var s = KernelSchedule.default
            s.gemmTile = tile
            variants.append((tile.label, Kernels(ctx: ctx, precision: .exact, schedule: s)))
        }
        print("N=\(n) K=\(k): ms per call; GFLOP/s in brackets")
        for m in ms {
            let x = Buf(ctx.device, floats: m * k, label: "x")
            for i in 0..<(m * k) { x.floats[i] = Float(i % 97) / 97 - 0.5 }
            let y = Buf(ctx.device, floats: m * n, label: "y")
            var line = String(format: "M=%4d", m)
            for (name, kernels) in variants {
                let t = try gpuSeconds(ctx, iterations: iterations) { enc in try kernels.gemv(enc, w, x: x, y: y, m: m) }
                line += String(format: "  %@=%.3f[%.0f]", name, t * 1000, 2.0 * Double(m * n * k) / t / 1e9)
            }
            print(line)
        }
    }
}
