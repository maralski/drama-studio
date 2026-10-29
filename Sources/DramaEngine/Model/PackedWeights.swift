// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// A bf16 [N, K] matrix in the exponent-coded layout gemv_packed_t reads
/// (Kernels/PackedLinear.metal): per 256-wide tile a header word, 256
/// sign+mantissa bytes and 96 bytes of 3-bit exponent codes, then one byte per
/// escaped exponent in an escape area the headers index. The sections sit in
/// one buffer, each 16-byte aligned. eval/pack_bf16.py writes the same layout.
struct PackedBF16 {
    let n: Int
    let k: Int
    let buffer: MTLBuffer
    let headerOffset: Int
    let mantOffset: Int
    let codeOffset: Int
    let escapeOffset: Int
    let escapedWeights: Int

    var tiles: Int { n * k / 256 }
    /// Bytes from the first header word to the last escape byte, wherever
    /// the sections sit in the buffer: what one decode row streams.
    var byteCount: Int { escapeOffset - headerOffset + escapedWeights }
    /// `byteCount` plus the escape pad: the bytes a buffer or file holds.
    var storedByteCount: Int { byteCount + Self.escapePad }

    static let tile = 256
    static let codesPerTile = 96
    static let window = 7
    /// Bytes after the last escape, so a matrix with no escapes still binds
    /// its escape section inside the buffer (Metal rejects offset == length).
    static let escapePad = 16

    static func sectionOffsets(tiles: Int) -> (header: Int, mant: Int, code: Int, escape: Int) {
        func align(_ x: Int) -> Int { (x + 15) / 16 * 16 }
        let header = 0
        let mant = align(header + tiles * 4)
        let code = align(mant + tiles * tile)
        let escape = align(code + tiles * codesPerTile)
        return (header, mant, code, escape)
    }

    /// The base exponent whose window [base, base + 6] holds the most of the
    /// tile's exponents; the first such base when several tie.
    static func bestBase(_ hist: UnsafeMutablePointer<Int32>) -> Int {
        var sum: Int32 = 0
        for e in 0..<window { sum += hist[e] }
        var best = sum, base = 0
        for b in 1...(256 - window) {
            sum += hist[b + window - 1] - hist[b - 1]
            if sum > best { best = sum; base = b }
        }
        return base
    }
}

extension PackedBF16 {
    init(bits: UnsafeBufferPointer<UInt16>, n: Int, k: Int, device: MTLDevice, label: String) {
        precondition(k % Self.tile == 0 && bits.count == n * k)
        self.n = n
        self.k = k
        let tiles = n * k / Self.tile
        let offsets = Self.sectionOffsets(tiles: tiles)
        var headers = [UInt32](repeating: 0, count: tiles)
        var mant = [UInt8](repeating: 0, count: tiles * Self.tile)
        var codes = [UInt8](repeating: 0, count: tiles * Self.codesPerTile)
        var escapes = [UInt8]()
        escapes.reserveCapacity(n * k / 32)
        let hist = UnsafeMutablePointer<Int32>.allocate(capacity: 256)
        defer { hist.deallocate() }
        for tile in 0..<tiles {
            let src = tile * Self.tile
            hist.initialize(repeating: 0, count: 256)
            for i in 0..<Self.tile { hist[Int(bits[src + i] >> 7) & 0xff] += 1 }
            let base = Self.bestBase(hist)
            precondition(escapes.count < 1 << 24, "\(label): escape area exceeds the 24-bit header offset")
            headers[tile] = UInt32(base) | UInt32(escapes.count) << 8
            for lane in 0..<32 {
                var q: UInt32 = 0
                for j in 0..<8 {
                    let i = lane * 8 + j
                    let b = bits[src + i]
                    mant[src + i] = UInt8(truncatingIfNeeded: (b >> 8) & 0x80 | (b & 0x7f))
                    let e = Int(b >> 7) & 0xff
                    let c: UInt32
                    if e >= base && e < base + Self.window {
                        c = UInt32(e - base)
                    } else {
                        c = 7
                        escapes.append(UInt8(e))
                    }
                    q |= c << (3 * UInt32(j))
                }
                let o = tile * Self.codesPerTile + lane * 3
                codes[o] = UInt8(q & 0xff)
                codes[o + 1] = UInt8((q >> 8) & 0xff)
                codes[o + 2] = UInt8(q >> 16)
            }
        }
        escapedWeights = escapes.count
        headerOffset = offsets.header
        mantOffset = offsets.mant
        codeOffset = offsets.code
        escapeOffset = offsets.escape
        buffer = device.makeBuffer(length: escapeOffset + escapes.count + Self.escapePad, options: .storageModeShared)!
        buffer.label = label
        let dst = buffer.contents()
        _ = headers.withUnsafeBytes { memcpy(dst + headerOffset, $0.baseAddress!, $0.count) }
        _ = mant.withUnsafeBytes { memcpy(dst + mantOffset, $0.baseAddress!, $0.count) }
        _ = codes.withUnsafeBytes { memcpy(dst + codeOffset, $0.baseAddress!, $0.count) }
        escapes.withUnsafeBytes { if $0.count > 0 { memcpy(dst + escapeOffset, $0.baseAddress!, $0.count) } }
        memset(dst + escapeOffset + escapes.count, 0, Self.escapePad)
    }

    /// The matrix of a bf16 TensorRef, packed.
    init(_ w: TensorRef, device: MTLDevice) {
        precondition(w.dtype == .bf16 && w.shape.count == 2)
        let p = w.pointer.assumingMemoryBound(to: UInt16.self)
        self.init(bits: UnsafeBufferPointer(start: p, count: w.count), n: w.shape[0], k: w.shape[1], device: device, label: w.name + ".packed")
    }

    /// Walks every tile in packing order and returns the flat index of the
    /// first weight whose unpacked pattern differs from `bits`, or nil.
    func firstMismatch(against bits: UnsafeBufferPointer<UInt16>) -> Int? {
        precondition(bits.count == n * k)
        let base = buffer.contents()
        let headers = (base + headerOffset).assumingMemoryBound(to: UInt32.self)
        let mant = (base + mantOffset).assumingMemoryBound(to: UInt8.self)
        let codes = (base + codeOffset).assumingMemoryBound(to: UInt8.self)
        let escapes = (base + escapeOffset).assumingMemoryBound(to: UInt8.self)
        for tile in 0..<tiles {
            let h = headers[tile]
            let exponentBase = UInt16(h & 0xff)
            var escaped = Int(h >> 8)
            let src = tile * Self.tile
            for lane in 0..<32 {
                let o = tile * Self.codesPerTile + lane * 3
                let q = UInt32(codes[o]) | UInt32(codes[o + 1]) << 8 | UInt32(codes[o + 2]) << 16
                for j in 0..<8 {
                    let i = src + lane * 8 + j
                    let c = (q >> (3 * UInt32(j))) & 7
                    let e: UInt16
                    if c == 7 {
                        e = UInt16(escapes[escaped])
                        escaped += 1
                    } else {
                        e = exponentBase + UInt16(c)
                    }
                    let m = UInt16(mant[i])
                    if (m & 0x80) << 8 | e << 7 | (m & 0x7f) != bits[i] { return i }
                }
            }
        }
        return nil
    }

    /// The bf16 pattern of weight (row, col), read back from the packed sections.
    func unpack(row: Int, col: Int) -> UInt16 {
        let base = buffer.contents()
        let tile = (row * k + col) / Self.tile, i = (row * k + col) % Self.tile
        let h = (base + headerOffset).assumingMemoryBound(to: UInt32.self)[tile]
        let m = UInt16((base + mantOffset).assumingMemoryBound(to: UInt8.self)[tile * Self.tile + i])
        let codes = (base + codeOffset).assumingMemoryBound(to: UInt8.self) + tile * Self.codesPerTile
        let escapes = (base + escapeOffset).assumingMemoryBound(to: UInt8.self) + Int(h >> 8)
        func code(_ index: Int) -> UInt32 {
            let o = index / 8 * 3
            let q = UInt32(codes[o]) | UInt32(codes[o + 1]) << 8 | UInt32(codes[o + 2]) << 16
            return (q >> (3 * UInt32(index % 8))) & 7
        }
        var escaped = 0
        for j in 0..<i where code(j) == 7 { escaped += 1 }
        let c = code(i)
        let e = c == 7 ? UInt16(escapes[escaped]) : UInt16(h & 0xff) + UInt16(c)
        return (m & 0x80) << 8 | e << 7 | (m & 0x7f)
    }
}

extension Kernels {
    /// The bf16 GEMV or GEMM of `gemv(_:_:x:y:m:)`, except one row of a coded
    /// matrix, which the packed GEMV serves with the same bits.
    func gemv(_ enc: MTLComputeCommandEncoder, _ w: DenseWeight, x: Buf, xOffset: Int = 0, y: Buf, yOffset: Int = 0, m: Int) throws {
        if m == 1, case .coded(let packed, _) = w {
            try gemvPacked(enc, packed, x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: 1)
        } else {
            try gemv(enc, w.bf16, x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m)
        }
    }

    /// y[m*N + n] = W[n] . x[m] over a packed matrix: the schedule's packed
    /// decode shape at one row, the narrow bf16 shape above (benches and
    /// parity tests only; the engine packs decode rows alone).
    func gemvPacked(_ enc: MTLComputeCommandEncoder, _ w: PackedBF16, x: Buf, xOffset: Int = 0, y: Buf, yOffset: Int = 0, m: Int) throws {
        enc.setBuffer(w.buffer, offset: w.headerOffset, index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset + (xOffset * 4), index: 1)
        enc.setBuffer(y.buffer, offset: y.byteOffset + (yOffset * 4), index: 2)
        set(enc, PackedGemvParams(N: UInt32(w.n), K: UInt32(w.k), M: UInt32(m), round: precision.mimicBF16 ? 1 : 0, escaped: UInt32(w.escapedWeights)), index: 3)
        setSections(enc, w, from: 4)
        let shape = m == 1 ? schedule.gemvPackedDecode : schedule.gemvNarrow
        try dispatchGroups(enc, "gemv_packed_t\(shape.tokenTile)",
                           groups: MTLSize(width: (w.n + shape.simdgroups - 1) / shape.simdgroups, height: (m + shape.tokenTile - 1) / shape.tokenTile, depth: 1),
                           group: MTLSize(width: 32 * shape.simdgroups, height: 1, depth: 1))
    }

    /// y = silu(W1 x) * (W3 x) for one row over two packed matrices, with the
    /// packed decode shape's simdgroups; swiglu_dec's output bit for bit.
    func swigluPackedDecode(_ enc: MTLComputeCommandEncoder, w1: PackedBF16, w3: PackedBF16, x: Buf, y: Buf) throws {
        precondition(w1.n == w3.n && w1.k == w3.k)
        enc.setBuffer(w1.buffer, offset: w1.headerOffset, index: 0)
        enc.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        enc.setBuffer(y.buffer, offset: y.byteOffset, index: 2)
        set(enc, PackedGemvParams(N: UInt32(w1.n), K: UInt32(w1.k), M: 1, round: precision.mimicBF16 ? 1 : 0, escaped: UInt32(w1.escapedWeights + w3.escapedWeights)), index: 3)
        setSections(enc, w1, from: 4)
        enc.setBuffer(w3.buffer, offset: w3.headerOffset, index: 7)
        setSections(enc, w3, from: 8)
        let sg = schedule.gemvPackedDecode.simdgroups
        try dispatchGroups(enc, "swiglu_packed_dec", groups: MTLSize(width: (w1.n + sg - 1) / sg, height: 1, depth: 1), group: MTLSize(width: 32 * sg, height: 1, depth: 1))
    }

    /// swiglu_dec on the bf16 tensors, or its packed form when both matrices are coded.
    func swigluDecode(_ enc: MTLComputeCommandEncoder, w1: DenseWeight, w3: DenseWeight, x: Buf, y: Buf) throws {
        if case .coded(let p1, _) = w1, case .coded(let p3, _) = w3 {
            try swigluPackedDecode(enc, w1: p1, w3: p3, x: x, y: y)
        } else {
            try swigluDecode(enc, w1: w1.bf16, w3: w3.bf16, x: x, y: y)
        }
    }

    /// The mantissa, code and escape sections at three consecutive indices.
    private func setSections(_ enc: MTLComputeCommandEncoder, _ w: PackedBF16, from index: Int) {
        enc.setBuffer(w.buffer, offset: w.mantOffset, index: index)
        enc.setBuffer(w.buffer, offset: w.codeOffset, index: index + 1)
        enc.setBuffer(w.buffer, offset: w.escapeOffset, index: index + 2)
    }
}

/// PackedGemvParams in PackedLinear.metal: GemvParams plus the escaped count
/// the dispatch trace reads.
private struct PackedGemvParams {
    var N: UInt32
    var K: UInt32
    var M: UInt32
    var round: UInt32
    var escaped: UInt32
}
