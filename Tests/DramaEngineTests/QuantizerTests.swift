import Metal
import XCTest
@testable import DramaEngine

/// The GPU quantizers against CPU quantizers written here from the format
/// definitions, byte for byte, and the round-trip error bounds those formats
/// promise. No model needed.
final class QuantizerTests: XCTestCase {
    /// Independent CPU int8 quantizer: per row, per 128 columns.
    static func int8Reference(_ w: [Float], rows: Int, cols: Int) -> (q: [Int8], scales: [Float16]) {
        var q = [Int8](repeating: 0, count: rows * cols)
        var scales: [Float16] = []
        for n in 0..<rows {
            for g in 0..<(cols / 128) {
                let group = w[(n * cols + g * 128)..<(n * cols + g * 128 + 128)]
                let absmax = group.map { abs($0) }.max()!
                let hs = Float16(absmax / 127)
                let s = Float(hs)
                scales.append(hs)
                for (i, v) in group.enumerated() {
                    q[n * cols + g * 128 + i] = s > 0 ? Int8(max(-127, min(127, (v / s).rounded(.toNearestOrEven)))) : 0
                }
            }
        }
        return (q, scales)
    }

    /// Independent CPU int4 quantizer: per row, per 64 columns, (scale, bias) pairs.
    static func int4Reference(_ w: [Float], rows: Int, cols: Int) -> (q: [UInt8], params: [Float16]) {
        var q = [UInt8](repeating: 0, count: rows * cols / 2)
        var params: [Float16] = []
        for n in 0..<rows {
            for g in 0..<(cols / 64) {
                let base = n * cols + g * 64
                let group = w[base..<(base + 64)]
                let lo = group.min()!, hi = group.max()!
                let hs = Float16((hi - lo) / 15), hb = Float16(lo)
                let s = Float(hs), b = Float(hb)
                params += [hs, hb]
                for i in 0..<64 {
                    let v = s > 0 ? UInt8(max(0, min(15, ((group[base + i] - b) / s).rounded(.toNearestOrEven)))) : 0
                    let byte = base / 2 + i / 2
                    q[byte] |= i % 2 == 0 ? v : v << 4
                }
            }
        }
        return (q, params)
    }

    static func fp8ToFloat(_ b: UInt8) -> Float {
        let s = b >> 7, e = (b >> 3) & 0xf, m = b & 7
        var v: Float
        if e == 0 { v = Float(m) * Float(pow(2.0, -9.0)) }
        else if e == 15 && m == 7 { v = .nan }
        else { v = (1 + Float(m) / 8) * Float(pow(2.0, Double(e) - 7)) }
        return s == 1 ? -v : v
    }

    struct Fixture {
        let ctx: MetalContext
        let k: Kernels
        var rng = SeededGenerator(seed: 21)
        init() throws { ctx = try MetalContext(); k = Kernels(ctx: ctx, precision: .exact) }

        /// Random bf16 weights in [-1, 1], with row 1 all zero and row 2 constant.
        mutating func bf16(rows: Int, cols: Int) -> (ref: TensorRef, floats: [Float]) {
            let buf = ctx.device.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
            let p = buf.contents().assumingMemoryBound(to: UInt16.self)
            var floats: [Float] = []
            for i in 0..<(rows * cols) {
                let row = i / cols
                let v: Float = row == 1 ? 0 : row == 2 ? 0.375 : Float(rng.uniformOpenZero()) * 2 - 1
                p[i] = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
                floats.append(Float(bitPattern: UInt32(p[i]) << 16))
            }
            return (TensorRef(name: "w", dtype: .bf16, shape: [rows, cols], buffer: buf, offset: 0), floats)
        }

        func slot(weightBytes: Int, paramBytes: Int) -> QuantizedSlot {
            let paramOffset = (weightBytes + 255) / 256 * 256
            let buf = ctx.device.makeBuffer(length: paramOffset + paramBytes, options: .storageModeShared)!
            memset(buf.contents(), 0x5a, buf.length)
            return QuantizedSlot(buffer: buf, offset: 0, paramOffset: paramOffset)
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

    static func bytes<T>(_ slot: QuantizedSlot, at offset: Int, count: Int, as: T.Type) -> [T] {
        Array(UnsafeBufferPointer(start: (slot.buffer.contents() + offset).assumingMemoryBound(to: T.self), count: count))
    }

    func testInt8MatchesCpuQuantizerAndRoundTrips() throws {
        var f = try Fixture()
        let rows = 96, cols = 512
        let (w, floats) = f.bf16(rows: rows, cols: cols)
        let slot = f.slot(weightBytes: rows * cols, paramBytes: rows * cols / 128 * 2)
        let k = f.k
        try f.run { enc in try k.quantizeInt8(enc, src: w, rowOffset: 0, rows: rows, cols: cols, into: slot) }
        let expected = Self.int8Reference(floats, rows: rows, cols: cols)
        let q = Self.bytes(slot, at: 0, count: rows * cols, as: Int8.self)
        let scales = Self.bytes(slot, at: slot.paramOffset, count: rows * cols / 128, as: Float16.self)
        XCTAssertEqual(q, expected.q)
        XCTAssertEqual(scales, expected.scales)
        XCTAssertEqual(scales[cols / 128], 0, "an all-zero group has a zero scale")
        XCTAssertTrue(q[cols..<(2 * cols)].allSatisfy { $0 == 0 })
        for i in 0..<(rows * cols) {
            let n = i / cols, g = (i % cols) / 128
            let s = Float(scales[n * (cols / 128) + g])
            let absmax = floats[(n * cols + g * 128)..<(n * cols + g * 128 + 128)].map { abs($0) }.max()!
            XCTAssertLessThanOrEqual(abs(floats[i] - Float(q[i]) * s), s / 2 + absmax * Float(pow(2.0, -11.0)) + 1e-6, "element \(i)")
        }
    }

    func testInt4FromBf16MatchesCpuQuantizerAndRoundTrips() throws {
        var f = try Fixture()
        let rows = 96, cols = 512
        let (w, floats) = f.bf16(rows: rows, cols: cols)
        let slot = f.slot(weightBytes: rows * cols / 2, paramBytes: rows * cols / 64 * 4)
        let k = f.k
        try f.run { enc in try k.quantizeInt4(enc, src: w, rowOffset: 0, rows: rows, cols: cols, into: slot) }
        let expected = Self.int4Reference(floats, rows: rows, cols: cols)
        let q = Self.bytes(slot, at: 0, count: rows * cols / 2, as: UInt8.self)
        let params = Self.bytes(slot, at: slot.paramOffset, count: rows * cols / 64 * 2, as: Float16.self)
        XCTAssertEqual(q, expected.q)
        XCTAssertEqual(params, expected.params)
        XCTAssertEqual(params[2 * (2 * cols / 64)], 0, "a constant group has a zero scale")
        XCTAssertEqual(params[2 * (2 * cols / 64) + 1], 0.375)
        XCTAssertEqual(q[0] & 0xf, UInt8(max(0, min(15, ((floats[0] - Float(params[1])) / Float(params[0])).rounded(.toNearestOrEven)))), "even element in the low nibble")
        try Self.checkInt4RoundTrip(floats, q: q, params: params, rows: rows, cols: cols)
    }

    static func checkInt4RoundTrip(_ floats: [Float], q: [UInt8], params: [Float16], rows: Int, cols: Int) throws {
        for i in 0..<(rows * cols) {
            let n = i / cols, g = (i % cols) / 64
            let s = Float(params[2 * (n * (cols / 64) + g)]), b = Float(params[2 * (n * (cols / 64) + g) + 1])
            let group = floats[(n * cols + g * 64)..<(n * cols + g * 64 + 64)]
            let nibble = i % 2 == 0 ? q[i / 2] & 0xf : q[i / 2] >> 4
            let bound = s / 2 + Float(pow(2.0, -11.0)) * (abs(b) + group.max()! - group.min()!) + 1e-6
            XCTAssertLessThanOrEqual(abs(floats[i] - fma(Float(nibble), s, b)), bound, "element \(i)")
        }
    }

    /// Two synthetic fp8 experts with block scales, one matrix kind at a time,
    /// into the expert-major staging the cache builder writes.
    func testInt4FromFp8ExpertsMatchesCpuQuantizer() throws {
        var f = try Fixture()
        let experts = 2
        let shapes = [(512, 256), (512, 256), (256, 512)]
        let pool = f.ctx.device.makeBuffer(length: experts * 3 * 512 * 256 + experts * 3 * 8 * 4, options: .storageModeShared)!
        var addresses: [UInt64] = []
        var cursor = 0
        var expected: [[[Float]]] = []
        for _ in 0..<experts {
            var perKind: [[Float]] = []
            for (rows, cols) in shapes {
                let w = (pool.contents() + cursor).assumingMemoryBound(to: UInt8.self)
                var bytes: [UInt8] = []
                for _ in 0..<(rows * cols) {
                    var b = UInt8(truncatingIfNeeded: f.rng.next())
                    if b & 0x7f == 0x7f { b &= 0xfe }
                    bytes.append(b)
                }
                for (i, b) in bytes.enumerated() { w[i] = b }
                addresses.append(pool.gpuAddress + UInt64(cursor))
                cursor += rows * cols
                let scaleCount = (rows / 128) * (cols / 128)
                let s = (pool.contents() + cursor).assumingMemoryBound(to: Float.self)
                var scales: [Float] = []
                for i in 0..<scaleCount { scales.append(0.5 + Float(f.rng.uniformOpenZero())); s[i] = scales[i] }
                addresses.append(pool.gpuAddress + UInt64(cursor))
                cursor += scaleCount * 4
                var floats: [Float] = []
                for i in 0..<(rows * cols) {
                    let n = i / cols, c = i % cols
                    floats.append(Self.fp8ToFloat(bytes[i]) * scales[(n / 128) * (cols / 128) + c / 128])
                }
                perKind.append(floats)
            }
            expected.append(perKind)
        }
        let table = ExpertTable(table: f.ctx.device.makeBuffer(bytes: addresses, length: addresses.count * 8, options: .storageModeShared)!, format: .fp8BlockScaled)
        for kind in ExpertMatrix.allCases {
            let (rows, cols) = shapes[kind.rawValue]
            let slot = f.slot(weightBytes: experts * rows * cols / 2, paramBytes: experts * rows * cols / 64 * 4)
            let k = f.k
            try f.run { enc in
                enc.useResource(pool, usage: .read)
                try k.quantizeExpertsInt4(enc, experts: table, kind: kind, count: experts, rows: rows, cols: cols, into: slot)
            }
            for e in 0..<experts {
                let floats = expected[e][kind.rawValue]
                let reference = Self.int4Reference(floats, rows: rows, cols: cols)
                let q = Self.bytes(slot, at: e * rows * cols / 2, count: rows * cols / 2, as: UInt8.self)
                let params = Self.bytes(slot, at: slot.paramOffset + e * rows * cols / 64 * 4, count: rows * cols / 64 * 2, as: Float16.self)
                XCTAssertEqual(q, reference.q, "\(kind) expert \(e)")
                XCTAssertEqual(params, reference.params, "\(kind) expert \(e)")
                try Self.checkInt4RoundTrip(floats, q: q, params: params, rows: rows, cols: cols)
            }
        }
    }
}
