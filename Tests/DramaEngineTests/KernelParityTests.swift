import Metal
import XCTest
@testable import DramaEngine

/// Kernel-level parity: the tiled prefill kernels against the one-row kernels
/// on random data. No model needed.
final class KernelParityTests: XCTestCase {
    struct Fixture {
        let ctx: MetalContext
        let k: Kernels
        var rng = SeededGenerator(seed: 7)

        init() throws {
            ctx = try MetalContext()
            k = Kernels(ctx: ctx, precision: .exact)
        }

        mutating func uniform() -> Float { Float(rng.uniformOpenZero()) * 2 - 1 }

        mutating func bf16Tensor(_ name: String, rows: Int, cols: Int) -> TensorRef {
            let buf = ctx.device.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
            let p = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(rows * cols) { p[i] = UInt16(truncatingIfNeeded: uniform().bitPattern >> 16) }
            return TensorRef(name: name, dtype: .bf16, shape: [rows, cols], buffer: buf, offset: 0)
        }

        mutating func bf16Matrix(rows: Int, cols: Int) -> WeightMatrix {
            let t = bf16Tensor("w", rows: rows, cols: cols)
            return WeightMatrix(name: t.name, format: .bf16, rows: rows, cols: cols, buffer: t.buffer, offset: 0, paramOffset: 0)
        }

        /// A random bf16 matrix in `format`, quantized by the engine's own
        /// kernels into one buffer of weights then params.
        mutating func matrix(_ format: WeightFormat, rows: Int, cols: Int) throws -> WeightMatrix {
            let w = bf16Matrix(rows: rows, cols: cols)
            if format == .bf16 { return w }
            let src = TensorRef(name: w.name, dtype: .bf16, shape: [rows, cols], buffer: w.buffer, offset: 0)
            let paramOffset = (format.weightBytes(rows: rows, cols: cols) + 255) / 256 * 256
            let buf = ctx.device.makeBuffer(length: paramOffset + format.paramBytes(rows: rows, cols: cols), options: .storageModeShared)!
            let slot = QuantizedSlot(buffer: buf, offset: 0, paramOffset: paramOffset)
            let k = self.k
            try run { enc in
                switch format {
                case .bf16: break
                case .int8: try k.quantizeInt8(enc, src: src, rowOffset: 0, rows: rows, cols: cols, into: slot)
                case .int4: try k.quantizeInt4(enc, src: src, rowOffset: 0, rows: rows, cols: cols, into: slot)
                }
            }
            return WeightMatrix(name: w.name, format: format, rows: rows, cols: cols, buffer: buf, offset: 0, paramOffset: paramOffset)
        }

        mutating func floats(_ count: Int, label: String) -> Buf {
            let b = Buf(ctx.device, floats: count, label: label)
            for i in 0..<count { b.floats[i] = uniform() }
            return b
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

    static func relL2(_ a: [Float], _ b: [Float]) -> Double {
        var num = 0.0, den = 0.0
        for (x, y) in zip(a, b) { num += Double(x - y) * Double(x - y); den += Double(x) * Double(x) }
        return (num / max(den, 1e-30)).squareRoot()
    }

    /// The matrix as fp32, read back from its bytes by the format definition.
    static func dequantized(_ w: WeightMatrix) -> [Float] {
        let base = w.buffer.contents()
        var out = [Float](repeating: 0, count: w.rows * w.cols)
        switch w.format {
        case .bf16:
            let p = (base + w.offset).assumingMemoryBound(to: UInt16.self)
            for i in out.indices { out[i] = Float(bitPattern: UInt32(p[i]) << 16) }
        case .int8:
            let q = (base + w.offset).assumingMemoryBound(to: Int8.self)
            let s = (base + w.paramOffset).assumingMemoryBound(to: Float16.self)
            for i in out.indices { out[i] = Float(q[i]) * Float(s[(i / w.cols) * (w.cols / 128) + (i % w.cols) / 128]) }
        case .int4:
            let q = (base + w.offset).assumingMemoryBound(to: UInt8.self)
            let p = (base + w.paramOffset).assumingMemoryBound(to: Float16.self)
            for i in out.indices {
                let g = (i / w.cols) * (w.cols / 64) + (i % w.cols) / 64
                let nibble = i % 2 == 0 ? q[i / 2] & 0xf : q[i / 2] >> 4
                out[i] = fma(Float(nibble), Float(p[2 * g]), Float(p[2 * g + 1]))
            }
        }
        return out
    }

    /// The GEMM must agree with the GEMV to fp32 reordering noise for every
    /// shape the model uses, including ragged M and tiny N, in every format.
    func testGemmMatchesGemv() throws {
        var f = try Fixture()
        for format in WeightFormat.allCases {
            for (n, kDim, m) in [(256, 2048, 128), (8192, 2048, 44), (2048, 4096, 5), (32, 2048, 100), (1, 2048, 128), (512, 2048, 64), (2048, 512, 7)] {
                let w = try f.matrix(format, rows: n, cols: kDim)
                let x = f.floats(m * kDim, label: "x")
                let yGemv = Buf(f.ctx.device, floats: m * n, label: "yGemv")
                let yGemm = Buf(f.ctx.device, floats: m * n, label: "yGemm")
                let k = f.k
                try f.run { enc in
                    for row in 0..<m {
                        try k.gemv(enc, w, x: x, xOffset: row * kDim, y: yGemv, yOffset: row * n, m: 1)
                    }
                    try k.gemv(enc, w, x: x, y: yGemm, m: m)
                }
                let a = yGemv.read(m * n), b = yGemm.read(m * n)
                let err = Self.relL2(a, b)
                XCTAssertLessThan(err, 2e-6, "\(format) n=\(n) k=\(kDim) m=\(m)")
                XCTAssertTrue(b.allSatisfy { $0.isFinite })
            }
        }
    }

    /// A row's GEMM result must not depend on how many rows share its chunk.
    func testGemmRowsIndependentOfChunk() throws {
        var f = try Fixture()
        let n = 512, kDim = 2048, m = 70
        for format in WeightFormat.allCases {
            let w = try f.matrix(format, rows: n, cols: kDim)
            let x = f.floats(m * kDim, label: "x")
            let whole = Buf(f.ctx.device, floats: m * n, label: "whole")
            let split = Buf(f.ctx.device, floats: m * n, label: "split")
            let k = f.k
            try f.run { enc in
                try k.gemv(enc, w, x: x, y: whole, m: m)
                try k.gemv(enc, w, x: x, y: split, m: 3)
                try k.gemv(enc, w, x: x, xOffset: 3 * kDim, y: split, yOffset: 3 * n, m: m - 3)
            }
            XCTAssertEqual(whole.read(m * n), split.read(m * n), "\(format)")
        }
    }

    /// The quantized kernels must compute the dot product of the matrix the
    /// bytes define: the GEMV at one row and the GEMM at five, against a CPU
    /// dot product over the dequantized weights.
    func testQuantizedGemvAndGemmMatchCpuDotProduct() throws {
        var f = try Fixture()
        for format in [WeightFormat.int8, .int4] {
            for (n, kDim) in [(512, 2048), (2048, 512), (96, 256)] {
                let w = try f.matrix(format, rows: n, cols: kDim)
                let dense = Self.dequantized(w)
                for m in [1, 5] {
                    let x = f.floats(m * kDim, label: "x")
                    let y = Buf(f.ctx.device, floats: m * n, label: "y")
                    let k = f.k
                    try f.run { enc in try k.gemv(enc, w, x: x, y: y, m: m) }
                    var expected = [Float](repeating: 0, count: m * n)
                    for row in 0..<m {
                        for i in 0..<n {
                            var acc = 0.0
                            for j in 0..<kDim { acc += Double(dense[i * kDim + j]) * Double(x.floats[row * kDim + j]) }
                            expected[row * n + i] = Float(acc)
                        }
                    }
                    let err = Self.relL2(expected, y.read(m * n))
                    XCTAssertLessThan(err, 1e-5, "\(format) n=\(n) k=\(kDim) m=\(m)")
                    XCTAssertGreaterThan(expected.map { abs($0) }.max()!, 0.1, "the inputs must exercise the kernel")
                }
            }
        }
    }
}

extension KernelParityTests {
    /// Synthetic FP8 experts: the tiled prefill kernels against the one-row
    /// kernels on a random routing of 50 rows over 16 experts.
    func testMoeTiledMatchesRowKernels() throws {
        let f = try Fixture()
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("config.json of drama-3 needed for the MoE shapes") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        let rows = 50
        let moe = KernelBench.SyntheticMoE(device: f.ctx.device, config: config, rows: rows, experts: 16, maxRows: 64)
        func run(_ k: Kernels, _ m: KernelBench.SyntheticMoE) throws -> ([Float], [Float]) {
            try f.run { enc in
                try m.bucket(enc, k)
                try m.gateUp(enc, k)
                try m.down(enc, k)
            }
            return (m.act.read(m.act.count), m.slotOut.read(m.slotOut.count))
        }
        var rowKernels = f.k
        rowKernels.forceMoeRows = true
        let byRows = try run(rowKernels, moe)
        let tiled = try run(f.k, moe)
        let actErr = Self.relL2(byRows.0, tiled.0)
        let outErr = Self.relL2(byRows.1, tiled.1)
        XCTAssertLessThan(actErr, 2e-6)
        XCTAssertLessThan(outErr, 2e-6)
        XCTAssertTrue(tiled.1.allSatisfy { $0.isFinite })
        print("moe tiled parity: act \(actErr) slotOut \(outErr)")

        // Every tile accumulates each output in the same 8-wide order over K,
        // so the whole domain must agree exactly.
        for tile in KernelSchedule.moeTiles {
            var s = KernelSchedule.default
            s.moeTile = tile
            let got = try run(Kernels(ctx: f.ctx, precision: .exact, schedule: s), moe)
            XCTAssertEqual(got.0, tiled.0, "tile \(tile.label)")
            XCTAssertEqual(got.1, tiled.1, "tile \(tile.label)")
        }

        // The one-row kernels only change which rows share a threadgroup.
        for sg in KernelSchedule.moeRowSimdgroups {
            var s = KernelSchedule.default
            s.moeRowSimdgroups = sg
            var k = Kernels(ctx: f.ctx, precision: .exact, schedule: s)
            k.forceMoeRows = true
            let got = try run(k, moe)
            XCTAssertEqual(got.0, byRows.0, "simdgroups \(sg)")
            XCTAssertEqual(got.1, byRows.1, "simdgroups \(sg)")
        }

        // A routed row's result must not depend on which other rows share its chunk.
        let part = KernelBench.SyntheticMoE(prefixOf: moe, rows: 11, device: f.ctx.device)
        let partial = try run(f.k, part)
        XCTAssertEqual(partial.0, Array(tiled.0.prefix(part.act.count)))
        XCTAssertEqual(partial.1, Array(tiled.1.prefix(part.slotOut.count)))
    }
}

extension KernelParityTests {
    /// Every split of the GDN recurrence must agree with the single-column
    /// version over a 96-token chunk, outputs and carried state alike.
    func testGdnRecurrentSplitsAgree() throws {
        var f = try Fixture()
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("config.json of drama-3 needed") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        let rows = 96, heads = config.linearValueHeads
        let convOut = f.floats(rows * 8192, label: "convOut")
        let a = f.floats(rows * heads, label: "a"), b = f.floats(rows * heads, label: "b")
        let aLog = f.bf16Tensor("aLog", rows: 1, cols: heads), dtBias = f.bf16Tensor("dtBias", rows: 1, cols: heads)
        let state0 = f.floats(heads * 128 * 128, label: "state0")
        for i in 0..<state0.count { state0.floats[i] *= 0.1 }
        var reference: ([Float], [Float])?
        for parts in KernelSchedule.gdnParts {
            var s = KernelSchedule.default
            s.gdnPartsPrefill = parts
            let k = Kernels(ctx: f.ctx, precision: .exact, schedule: s)
            let core = Buf(f.ctx.device, floats: rows * heads * 128, label: "core")
            let state = Buf(f.ctx.device, floats: state0.count, label: "state")
            memcpy(state.floats, state0.floats, state0.count * 4)
            try f.run { enc in
                try k.gdnRecurrent(enc, core: core, convOut: convOut, a: a, b: b, state: state, aLog: aLog, dtBias: dtBias, rows: rows, config: config)
            }
            let got = (core.read(core.count), state.read(state.count))
            XCTAssertTrue(got.0.allSatisfy { $0.isFinite })
            if let reference {
                let coreErr = Self.relL2(reference.0, got.0), stateErr = Self.relL2(reference.1, got.1)
                XCTAssertLessThan(coreErr, 1e-5, "parts \(parts)")
                XCTAssertLessThan(stateErr, 1e-5, "parts \(parts)")
                print("gdn parts \(parts): core \(coreErr) state \(stateErr)")
            } else { reference = got }
        }
    }
}
