import Metal
import XCTest
@testable import DramaEngine

/// The exponent-packed GEMV against the shipped bf16 GEMV: bit-identical
/// outputs on synthetic matrices whose exponents are distributed like the
/// model's (eval/out/weight_entropy.txt). No model needed.
final class PackedGemvTests: XCTestCase {
    /// Gaussian weights with the model's exponent mode (-4), a 1 percent band
    /// of rows near 2^-124 like gdn.qkv, rare 2^-20 outliers and a few exact
    /// zeros, so the natural tails, the outliers and the zeros all escape.
    static func realisticBF16(_ device: MTLDevice, rows: Int, cols: Int, seed: UInt64) -> TensorRef {
        let buf = device.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: UInt16.self)
        var rng = SeededGenerator(seed: seed)
        for r in 0..<rows {
            let rowScale: Float = rng.uniformOpenZero() < 0.01 ? Float(sign: .plus, exponent: -120, significand: 1) : 1
            var c = 0
            while c < cols {
                let u1 = rng.uniformOpenZero(), u2 = rng.uniformOpenZero()
                let radius = (-2 * log(u1)).squareRoot() * 0.09
                for g in [radius * cos(2 * .pi * u2), radius * sin(2 * .pi * u2)] where c < cols {
                    var v = Float(g) * rowScale
                    let roll = rng.uniformOpenZero()
                    if roll < 1e-5 { v = 0 } else if roll < 6e-5 { v *= Float(sign: .plus, exponent: -20, significand: 1) }
                    p[r * cols + c] = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
                    c += 1
                }
            }
        }
        return TensorRef(name: "w\(seed)", dtype: .bf16, shape: [rows, cols], buffer: buf, offset: 0)
    }

    struct Fixture {
        let ctx: MetalContext
        var rng = SeededGenerator(seed: 11)
        init() throws { ctx = try MetalContext() }

        mutating func floats(_ count: Int, label: String) -> Buf {
            let b = Buf(ctx.device, floats: count, label: label)
            for i in 0..<count { b.floats[i] = Float(rng.uniformOpenZero()) * 2 - 1 }
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

    static let shapes = [(8192, 2048), (2048, 4096), (9216, 2048), (2048, 6144), (512, 2048), (2048, 512), (256, 2048), (32, 2048), (1, 2048)]

    func testPackRoundTripsEveryWeight() throws {
        let f = try Fixture()
        let w = Self.realisticBF16(f.ctx.device, rows: 512, cols: 2048, seed: 1)
        let packed = PackedBF16(w, device: f.ctx.device)
        let escapeRate = Double(packed.escapedWeights) / Double(w.count)
        XCTAssertGreaterThan(escapeRate, 0.005)
        XCTAssertLessThan(escapeRate, 0.1)
        let bits = w.pointer.assumingMemoryBound(to: UInt16.self)
        for row in 0..<512 {
            for col in 0..<2048 where packed.unpack(row: row, col: col) != bits[row * 2048 + col] {
                XCTFail("row \(row) col \(col): \(packed.unpack(row: row, col: col)) != \(bits[row * 2048 + col])")
                return
            }
        }
    }

    /// Decode shape (M = 1) on every dense shape the model uses.
    func testPackedGemvMatchesBF16AtDecode() throws {
        var f = try Fixture()
        let k = Kernels(ctx: f.ctx, precision: .exact)
        for (n, kDim) in Self.shapes {
            let w = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: UInt64(n + kDim))
            let packed = PackedBF16(w, device: f.ctx.device)
            let x = f.floats(kDim, label: "x")
            let yRef = Buf(f.ctx.device, floats: n, label: "yRef"), yPacked = Buf(f.ctx.device, floats: n, label: "yPacked")
            try f.run { enc in
                try k.gemv(enc, w, x: x, y: yRef, m: 1)
                try k.gemvPacked(enc, packed, x: x, y: yPacked, m: 1)
            }
            let a = yRef.read(n).map(\.bitPattern), b = yPacked.read(n).map(\.bitPattern)
            XCTAssertEqual(a, b, "n=\(n) k=\(kDim)")
            XCTAssertTrue(yRef.read(n).allSatisfy { $0.isFinite })
            let bits = Double(packed.byteCount * 8) / Double(n * kDim)
            print(String(format: "packed %5d x %4d: %.3f bits/weight, %.2f%% escaped", n, kDim, bits, 100 * Double(packed.escapedWeights) / Double(n * kDim)))
        }
    }

    /// Several tokens through the GEMV path, with every GEMV shape in the domain.
    func testPackedGemvMatchesBF16ForTokenTiles() throws {
        var f = try Fixture()
        let n = 2048, kDim = 2048, m = 37
        let w = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 5)
        let packed = PackedBF16(w, device: f.ctx.device)
        let x = f.floats(m * kDim, label: "x")
        for shape in KernelSchedule.gemvShapes {
            var s = KernelSchedule.default
            s.gemvDecode = shape
            s.gemvNarrow = shape
            var k = Kernels(ctx: f.ctx, precision: .exact, schedule: s)
            k.forceGemv = true
            let yRef = Buf(f.ctx.device, floats: m * n, label: "yRef"), yPacked = Buf(f.ctx.device, floats: m * n, label: "yPacked")
            try f.run { enc in
                try k.gemv(enc, w, x: x, y: yRef, m: m)
                try k.gemvPacked(enc, packed, x: x, y: yPacked, m: m)
            }
            XCTAssertEqual(yRef.read(m * n).map(\.bitPattern), yPacked.read(m * n).map(\.bitPattern), shape.label)
        }
    }

    /// A packed copy with one mantissa byte flipped, so the two routes give
    /// different bits: one row of a `.coded` weight reads the packed form,
    /// every M > 1 reads the bf16 tensor.
    func testCodedWeightRoutesOneRowPackedAndMoreRowsBF16() throws {
        var f = try Fixture()
        let n = 2048, kDim = 2048, m = 37
        let w = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 8)
        let damaged = PackedBF16(w, device: f.ctx.device)
        (damaged.buffer.contents() + damaged.mantOffset).assumingMemoryBound(to: UInt8.self)[4097] ^= 0x01
        let coded = DenseWeight.coded(damaged, bf16: w)
        let k = Kernels(ctx: f.ctx, precision: .exact)
        let x = f.floats(m * kDim, label: "x")
        let yRef = Buf(f.ctx.device, floats: m * n, label: "yRef"), y = Buf(f.ctx.device, floats: m * n, label: "y")
        try f.run { enc in
            try k.gemv(enc, w, x: x, y: yRef, m: 1)
            try k.gemv(enc, coded, x: x, y: y, m: 1)
        }
        XCTAssertNotEqual(y.read(n).map(\.bitPattern), yRef.read(n).map(\.bitPattern), "one row must read the packed form")
        try f.run { enc in
            try k.gemv(enc, w, x: x, y: yRef, m: m)
            try k.gemv(enc, coded, x: x, y: y, m: m)
        }
        XCTAssertEqual(y.read(m * n).map(\.bitPattern), yRef.read(m * n).map(\.bitPattern), "M > 1 must read the bf16 tensor")
        try f.run { enc in
            try k.gemv(enc, w, x: x, y: yRef, m: 2)
            try k.gemv(enc, coded, x: x, y: y, m: 2)
        }
        XCTAssertEqual(y.read(2 * n).map(\.bitPattern), yRef.read(2 * n).map(\.bitPattern), "M = 2 must read the bf16 tensor")
    }

    /// The DenseWeight swiglu dispatcher: both matrices coded runs the packed
    /// kernel, one bf16 matrix sends both through swiglu_dec. A damaged
    /// packed W1 tells the routes apart.
    func testSwigluDispatcherRoutesMixedWeightsToBF16() throws {
        var f = try Fixture()
        let n = 512, kDim = 2048
        let w1 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 31), w3 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 32)
        let p1 = PackedBF16(w1, device: f.ctx.device), p3 = PackedBF16(w3, device: f.ctx.device)
        let damaged = PackedBF16(w1, device: f.ctx.device)
        (damaged.buffer.contents() + damaged.mantOffset).assumingMemoryBound(to: UInt8.self)[1000] ^= 0x02
        let k = Kernels(ctx: f.ctx, precision: .exact)
        let x = f.floats(kDim, label: "x")
        let yRef = Buf(f.ctx.device, floats: n, label: "yRef"), y = Buf(f.ctx.device, floats: n, label: "y")
        try f.run { enc in try k.swigluDecode(enc, w1: w1, w3: w3, x: x, y: yRef) }
        let reference = yRef.read(n).map(\.bitPattern)
        let routes: [(String, DenseWeight, DenseWeight, Bool)] = [
            ("both coded", .coded(p1, bf16: w1), .coded(p3, bf16: w3), true),
            ("damaged coded w1, coded w3", .coded(damaged, bf16: w1), .coded(p3, bf16: w3), false),
            ("damaged coded w1, bf16 w3", .coded(damaged, bf16: w1), .bf16(w3), true),
            ("bf16 w1, coded w3", .bf16(w1), .coded(p3, bf16: w3), true),
            ("both bf16", .bf16(w1), .bf16(w3), true),
        ]
        for (label, a, b, matches) in routes {
            y.zero()
            try f.run { enc in try k.swigluDecode(enc, w1: a, w3: b, x: x, y: y) }
            if matches {
                XCTAssertEqual(y.read(n).map(\.bitPattern), reference, label)
            } else {
                XCTAssertNotEqual(y.read(n).map(\.bitPattern), reference, label)
            }
        }
    }

    /// The packed swiglu against the fused bf16 one at every packed shape, in
    /// exact precision and with the reference's rounded epilogue; the two
    /// matrices swapped must differ, so a swapped binding would be caught.
    func testPackedSwigluMatchesFusedSwiglu() throws {
        var f = try Fixture()
        let n = 1003, kDim = 2048
        let w1 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 21), w3 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 22)
        let p1 = PackedBF16(w1, device: f.ctx.device), p3 = PackedBF16(w3, device: f.ctx.device)
        let x = f.floats(kDim, label: "x")
        let yRef = Buf(f.ctx.device, floats: n, label: "yRef"), y = Buf(f.ctx.device, floats: n, label: "y")
        for precision in [Precision.exact, .reference] {
            try f.run { enc in try Kernels(ctx: f.ctx, precision: precision).swigluDecode(enc, w1: w1, w3: w3, x: x, y: yRef) }
            let reference = yRef.read(n).map(\.bitPattern)
            XCTAssertTrue(yRef.read(n).allSatisfy(\.isFinite))
            for shape in KernelSchedule.gemvPackedShapes {
                var s = KernelSchedule.default
                s.gemvPackedDecode = shape
                let k = Kernels(ctx: f.ctx, precision: precision, schedule: s)
                y.zero()
                try f.run { enc in try k.swigluPackedDecode(enc, w1: p1, w3: p3, x: x, y: y) }
                XCTAssertEqual(y.read(n).map(\.bitPattern), reference, "\(shape.label) mimicBF16=\(precision.mimicBF16)")
            }
            try f.run { enc in try Kernels(ctx: f.ctx, precision: precision).swigluPackedDecode(enc, w1: p3, w3: p1, x: x, y: y) }
            XCTAssertNotEqual(y.read(n).map(\.bitPattern), reference, "swapped matrices")
        }
    }

    /// GPU time of the packed GEMV against the shipped GEMV at decode,
    /// rotating through 8 copies of each matrix so every dispatch streams from
    /// DRAM: 5 warm-ups, 20 timed command buffers, GPU timestamps, both
    /// kernels over the same GEMV shapes. Set DRAMA_PACKED_BENCH=1 and run
    /// under tools/with-gpu-lock.
    func testPackedGemvBench() throws {
        guard ProcessInfo.processInfo.environment["DRAMA_PACKED_BENCH"] == "1" else { throw XCTSkip("DRAMA_PACKED_BENCH=1 runs the bench") }
        var f = try Fixture()
        func kernels(_ shape: KernelSchedule.GemvShape) -> Kernels {
            var s = KernelSchedule.default
            s.gemvDecode = shape
            s.gemvPackedDecode = shape
            s.gemvNarrow = shape
            return Kernels(ctx: f.ctx, precision: .exact, schedule: s)
        }
        let shapes = [KernelSchedule.default.gemvDecode, KernelSchedule.GemvShape(tokenTile: 1, simdgroups: 4), KernelSchedule.GemvShape(tokenTile: 1, simdgroups: 8)]
        let copies = 8, warmups = 5, repeats = 20
        print("shape          kernel               ms/dispatch min/median/max   logical GB/s (median)   physical GB/s   bits/w")
        for (n, kDim) in [(8192, 2048), (4096, 2048), (2048, 4096), (9216, 2048), (6144, 2048), (2048, 6144), (512, 2048), (2048, 512), (256, 2048)] {
            let w0 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: UInt64(n * 7 + kDim))
            let p0 = PackedBF16(w0, device: f.ctx.device)
            var ws = [w0], ps = [p0]
            for _ in 1..<copies {
                let b = f.ctx.device.makeBuffer(length: w0.byteCount, options: .storageModeShared)!
                memcpy(b.contents(), w0.pointer, w0.byteCount)
                ws.append(TensorRef(name: w0.name, dtype: .bf16, shape: w0.shape, buffer: b, offset: 0))
                let pb = f.ctx.device.makeBuffer(length: p0.buffer.length, options: .storageModeShared)!
                memcpy(pb.contents(), p0.buffer.contents(), p0.buffer.length)
                ps.append(PackedBF16(n: n, k: kDim, buffer: pb, headerOffset: p0.headerOffset, mantOffset: p0.mantOffset, codeOffset: p0.codeOffset, escapeOffset: p0.escapeOffset, escapedWeights: p0.escapedWeights))
            }
            let x = f.floats(kDim, label: "x")
            let y = Buf(f.ctx.device, floats: n, label: "y")
            func time(_ body: (MTLComputeCommandEncoder, Int) throws -> Void) throws -> [Double] {
                var samples: [Double] = []
                for r in 0..<(warmups + repeats) {
                    let cb = f.ctx.queue.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    for c in 0..<copies { try body(enc, c) }
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    if r >= warmups { samples.append((cb.gpuEndTime - cb.gpuStartTime) / Double(copies) * 1000) }
                }
                return samples.sorted()
            }
            let logical = Double(n * kDim * 2)
            var runs: [(String, Double, (MTLComputeCommandEncoder, Int) throws -> Void)] = []
            for shape in shapes {
                runs.append(("bf16 " + shape.label, logical, { enc, c in try kernels(shape).gemv(enc, ws[c], x: x, y: y, m: 1) }))
                runs.append(("packed " + shape.label, Double(p0.byteCount), { enc, c in try kernels(shape).gemvPacked(enc, ps[c], x: x, y: y, m: 1) }))
            }
            for (label, bytes, run) in runs {
                let s = try time(run)
                let median = s[s.count / 2]
                print(String(format: "%5d x %4d   %-18@ %.4f/%.4f/%.4f ms   %6.1f   %6.1f   %.2f", n, kDim, label, s[0], median, s[s.count - 1],
                             logical / (median / 1000) / 1e9, bytes / (median / 1000) / 1e9, bytes * 8 / Double(n * kDim)))
            }
        }
        // The tower's swiglu: w1 and w3 of 6144 x 2048 in one dispatch, the
        // bf16 swiglu_dec (a fixed simdgroup count) against the packed one at
        // each packed shape, rotating copies as above.
        do {
            let n = 6144, kDim = 2048
            let w1 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 61), w3 = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 63)
            let p1 = PackedBF16(w1, device: f.ctx.device), p3 = PackedBF16(w3, device: f.ctx.device)
            func copy(_ w: TensorRef) -> TensorRef {
                let b = f.ctx.device.makeBuffer(length: w.byteCount, options: .storageModeShared)!
                memcpy(b.contents(), w.pointer, w.byteCount)
                return TensorRef(name: w.name, dtype: .bf16, shape: w.shape, buffer: b, offset: 0)
            }
            func copy(_ p: PackedBF16) -> PackedBF16 {
                let b = f.ctx.device.makeBuffer(length: p.buffer.length, options: .storageModeShared)!
                memcpy(b.contents(), p.buffer.contents(), p.buffer.length)
                return PackedBF16(n: p.n, k: p.k, buffer: b, headerOffset: p.headerOffset, mantOffset: p.mantOffset, codeOffset: p.codeOffset, escapeOffset: p.escapeOffset, escapedWeights: p.escapedWeights)
            }
            let w1s = [w1] + (1..<copies).map { _ in copy(w1) }, w3s = [w3] + (1..<copies).map { _ in copy(w3) }
            let p1s = [p1] + (1..<copies).map { _ in copy(p1) }, p3s = [p3] + (1..<copies).map { _ in copy(p3) }
            let x = f.floats(kDim, label: "x"), y = Buf(f.ctx.device, floats: n, label: "y")
            func time(_ body: (MTLComputeCommandEncoder, Int) throws -> Void) throws -> [Double] {
                var samples: [Double] = []
                for r in 0..<(warmups + repeats) {
                    let cb = f.ctx.queue.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    for c in 0..<copies { try body(enc, c) }
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    if r >= warmups { samples.append((cb.gpuEndTime - cb.gpuStartTime) / Double(copies) * 1000) }
                }
                return samples.sorted()
            }
            let logical = Double(2 * n * kDim * 2), packedBytes = Double(p1.byteCount + p3.byteCount)
            var runs: [(String, Double, (MTLComputeCommandEncoder, Int) throws -> Void)] = [
                ("bf16 swiglu_dec", logical, { enc, c in try kernels(KernelSchedule.default.gemvDecode).swigluDecode(enc, w1: w1s[c], w3: w3s[c], x: x, y: y) }),
            ]
            for shape in KernelSchedule.gemvPackedShapes where shape.simdgroups >= 4 {
                runs.append(("packed swiglu " + shape.label, packedBytes, { enc, c in try kernels(shape).swigluPackedDecode(enc, w1: p1s[c], w3: p3s[c], x: x, y: y) }))
            }
            for (label, bytes, run) in runs {
                let s = try time(run)
                let median = s[s.count / 2]
                print(String(format: "2x%d x %4d   %-20@ %.4f/%.4f/%.4f ms   %6.1f   %6.1f   %.2f", n, kDim, label, s[0], median, s[s.count - 1],
                             logical / (median / 1000) / 1e9, bytes / (median / 1000) / 1e9, bytes * 8 / Double(2 * n * kDim)))
            }
        }
        // Hot in cache: the unpack cost without DRAM in the way.
        let n = 2048, kDim = 2048
        let w = Self.realisticBF16(f.ctx.device, rows: n, cols: kDim, seed: 99)
        let p = PackedBF16(w, device: f.ctx.device)
        let x = f.floats(kDim, label: "x"), y = Buf(f.ctx.device, floats: n, label: "y")
        for shape in shapes {
            let k = kernels(shape)
            for (label, run) in [("bf16", { (enc: MTLComputeCommandEncoder) in try k.gemv(enc, w, x: x, y: y, m: 1) }),
                                 ("packed", { (enc: MTLComputeCommandEncoder) in try k.gemvPacked(enc, p, x: x, y: y, m: 1) })] {
                var samples: [Double] = []
                for r in 0..<(warmups + repeats) {
                    let t = try KernelBench.gpuSeconds(f.ctx, iterations: 100) { enc in try run(enc) }
                    if r >= warmups { samples.append(t * 1e6) }
                }
                samples.sort()
                print(String(format: "hot 2048 x 2048 %-7@ %-8@ %.2f/%.2f/%.2f us/dispatch  (%.0f GB/s logical at median)", label, shape.label, samples[0], samples[samples.count / 2], samples[samples.count - 1], Double(n * kDim * 2) / (samples[samples.count / 2] * 1e-6) / 1e9))
            }
        }
    }
}
