// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import zlib

/// Minimal NumPy `.npy` reading and writing, plus uncompressed `.npz`.
package enum NPY {
    package enum DType: String { case int64 = "<i8", int32 = "<i4", float32 = "<f4", float64 = "<f8", uint8 = "|u1" }

    package struct Array {
        package var dtype: DType
        package var shape: [Int]
        package var data: Data
        package init(dtype: DType, shape: [Int], data: Data) { self.dtype = dtype; self.shape = shape; self.data = data }

        package static func int64(_ values: [Int64], shape: [Int]) -> Array {
            Array(dtype: .int64, shape: shape, data: values.withUnsafeBufferPointer { Data(buffer: $0) })
        }
        package static func float32(_ values: [Float], shape: [Int]) -> Array {
            Array(dtype: .float32, shape: shape, data: values.withUnsafeBufferPointer { Data(buffer: $0) })
        }
        package var count: Int { shape.reduce(1, *) }
        package var int64Values: [Int64] {
            switch dtype {
            case .int64: return data.withUnsafeBytes { Swift.Array($0.bindMemory(to: Int64.self)) }
            case .int32: return data.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Int64.init) }
            default: return []
            }
        }
        package var float32Values: [Float] {
            switch dtype {
            case .float32: return data.withUnsafeBytes { Swift.Array($0.bindMemory(to: Float.self)) }
            case .float64: return data.withUnsafeBytes { $0.bindMemory(to: Double.self).map(Float.init) }
            default: return []
            }
        }
    }

    package static func read(url: URL) throws -> Array {
        try parse(Data(contentsOf: url))
    }

    static func parse(_ d: Data) throws -> Array {
        guard d.count > 10, d[0] == 0x93, String(data: d[1..<6], encoding: .ascii) == "NUMPY" else {
            throw EngineError.badModel("not an npy file")
        }
        let major = d[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(d[8]) | Int(d[9]) << 8
            headerStart = 10
        } else {
            headerLen = Int(d[8]) | Int(d[9]) << 8 | Int(d[10]) << 16 | Int(d[11]) << 24
            headerStart = 12
        }
        let header = String(data: d[headerStart..<headerStart + headerLen], encoding: .ascii) ?? ""
        guard let descrRange = header.range(of: "'descr': '"), let shapeRange = header.range(of: "'shape': (") else {
            throw EngineError.badModel("npy header: \(header)")
        }
        let descr = String(header[descrRange.upperBound...].prefix { $0 != "'" })
        guard let dtype = DType(rawValue: descr) else { throw EngineError.badModel("npy dtype \(descr)") }
        guard !header.contains("'fortran_order': True") else { throw EngineError.badModel("fortran order npy") }
        let shapeText = String(header[shapeRange.upperBound...].prefix { $0 != ")" })
        let shape = shapeText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let body = d[(headerStart + headerLen)...]
        return Array(dtype: dtype, shape: shape, data: Data(body))
    }

    /// Reads an uncompressed `.npz` (NumPy's default `savez`, and `writeNPZ`),
    /// keyed by member name without the `.npy` suffix.
    package static func readNPZ(url: URL) throws -> [String: Array] {
        let d = try Data(contentsOf: url)
        func u16(_ at: Int) -> Int { Int(d[at]) | Int(d[at + 1]) << 8 }
        func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
        var out: [String: Array] = [:]
        var i = 0
        while i + 30 <= d.count, u32(i) == 0x0403_4b50 {
            let method = u16(i + 8), size = u32(i + 22), nameLen = u16(i + 26), extraLen = u16(i + 28)
            guard method == 0 else { throw EngineError.badModel("npz member is compressed") }
            let name = String(data: d[(i + 30)..<(i + 30 + nameLen)], encoding: .utf8) ?? ""
            let start = i + 30 + nameLen + extraLen
            out[name.hasSuffix(".npy") ? String(name.dropLast(4)) : name] = try parse(Data(d[start..<(start + size)]))
            i = start + size
        }
        return out
    }

    static func serialize(_ a: Array) -> Data {
        let shapeText = a.shape.count == 1 ? "(\(a.shape[0]),)" : "(" + a.shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(a.dtype.rawValue)', 'fortran_order': False, 'shape': \(shapeText), }"
        let padding = 64 - ((10 + header.utf8.count + 1) % 64)
        header += String(repeating: " ", count: padding) + "\n"
        var out = Data([0x93]) + "NUMPY".data(using: .ascii)! + Data([1, 0])
        let len = UInt16(header.utf8.count)
        out.append(contentsOf: [UInt8(len & 0xff), UInt8(len >> 8)])
        out.append(header.data(using: .ascii)!)
        out.append(a.data)
        return out
    }

    package static func write(_ a: Array, to url: URL) throws {
        try serialize(a).write(to: url)
    }

    /// Writes an uncompressed zip with one `.npy` member per entry.
    package static func writeNPZ(_ entries: [(String, Array)], to url: URL) throws {
        var out = Data()
        var central = Data()
        var count: UInt16 = 0
        for (name, array) in entries {
            let fileName = (name + ".npy").data(using: .utf8)!
            let payload = serialize(array)
            let crc = payload.withUnsafeBytes { UInt32(crc32(0, $0.bindMemory(to: Bytef.self).baseAddress, uInt(payload.count))) }
            let offset = UInt32(out.count)
            var local = Data()
            local.append(le32(0x0403_4b50)); local.append(le16(20)); local.append(le16(0)); local.append(le16(0))
            local.append(le16(0)); local.append(le16(0)); local.append(le32(crc))
            local.append(le32(UInt32(payload.count))); local.append(le32(UInt32(payload.count)))
            local.append(le16(UInt16(fileName.count))); local.append(le16(0))
            local.append(fileName)
            out.append(local); out.append(payload)
            central.append(le32(0x0201_4b50)); central.append(le16(20)); central.append(le16(20)); central.append(le16(0)); central.append(le16(0))
            central.append(le16(0)); central.append(le16(0)); central.append(le32(crc))
            central.append(le32(UInt32(payload.count))); central.append(le32(UInt32(payload.count)))
            central.append(le16(UInt16(fileName.count))); central.append(le16(0)); central.append(le16(0)); central.append(le16(0)); central.append(le16(0))
            central.append(le32(0)); central.append(le32(offset)); central.append(fileName)
            count += 1
        }
        let centralOffset = UInt32(out.count)
        out.append(central)
        out.append(le32(0x0605_4b50)); out.append(le16(0)); out.append(le16(0)); out.append(le16(count)); out.append(le16(count))
        out.append(le32(UInt32(central.count))); out.append(le32(centralOffset)); out.append(le16(0))
        try out.write(to: url)
    }

    private static func le16(_ v: UInt16) -> Data { Data([UInt8(v & 0xff), UInt8(v >> 8)]) }
    private static func le32(_ v: UInt32) -> Data { Data([UInt8(v & 0xff), UInt8((v >> 8) & 0xff), UInt8((v >> 16) & 0xff), UInt8(v >> 24)]) }
}
