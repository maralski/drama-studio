import Compression
import Foundation

enum ZipError: LocalizedError, Equatable {
    case notAZip
    case zip64Unsupported
    case encrypted(String)
    case unsupportedMethod(UInt16, String)
    case missingEntry(String)
    case corrupt(String)

    var errorDescription: String? {
        switch self {
        case .notAZip: "The file is not a ZIP package."
        case .zip64Unsupported: "The file uses ZIP64, which Drama Studio does not read yet. Re-save it below 4 GB, or with fewer than 65,535 entries."
        case .encrypted(let path): "\(path) is encrypted."
        case .unsupportedMethod(let method, let path): "\(path) uses ZIP compression method \(method); only stored and deflate are supported."
        case .missingEntry(let path): "The package has no \(path)."
        case .corrupt(let detail): "The ZIP package is damaged: \(detail)."
        }
    }
}

/// Reads a ZIP archive held in memory: the central directory once, entries on demand.
struct ZipArchive: Sendable {
    enum Method: UInt16, Sendable {
        case stored = 0
        case deflated = 8
    }

    struct Entry: Sendable, Equatable {
        var path: String
        var method: UInt16
        var flags: UInt16
        var crc32: UInt32
        var compressedSize: Int
        var uncompressedSize: Int
        var localHeaderOffset: Int
    }

    let entries: [Entry]
    private let bytes: Data
    private let index: [String: Int]

    init(url: URL) throws {
        try self.init(data: Data(contentsOf: url, options: .mappedIfSafe))
    }

    init(data: Data) throws {
        let bytes = data.startIndex == 0 ? data : Data(data)
        let reader = ByteReader(bytes)
        let eocd = try Self.findEndOfCentralDirectory(reader)
        if eocd >= 20, try reader.u32(eocd - 20) == 0x0706_4B50 { throw ZipError.zip64Unsupported }
        let count = Int(try reader.u16(eocd + 10))
        let directorySize = try reader.u32(eocd + 12)
        let directoryOffset = try reader.u32(eocd + 16)
        if count == 0xFFFF || directorySize == 0xFFFF_FFFF || directoryOffset == 0xFFFF_FFFF { throw ZipError.zip64Unsupported }

        var entries: [Entry] = []
        var cursor = Int(directoryOffset)
        for _ in 0..<count {
            guard try reader.u32(cursor) == 0x0201_4B50 else { throw ZipError.corrupt("bad central directory header at \(cursor)") }
            let flags = try reader.u16(cursor + 8)
            let compressed = try reader.u32(cursor + 20)
            let uncompressed = try reader.u32(cursor + 24)
            let offset = try reader.u32(cursor + 42)
            if compressed == 0xFFFF_FFFF || uncompressed == 0xFFFF_FFFF || offset == 0xFFFF_FFFF { throw ZipError.zip64Unsupported }
            let nameLength = Int(try reader.u16(cursor + 28))
            let extraLength = Int(try reader.u16(cursor + 30))
            let commentLength = Int(try reader.u16(cursor + 32))
            let nameBytes = try reader.slice(cursor + 46, nameLength)
            let path = String(data: nameBytes, encoding: .utf8) ?? String(decoding: nameBytes, as: UTF8.self)
            entries.append(Entry(
                path: path,
                method: try reader.u16(cursor + 10),
                flags: flags,
                crc32: try reader.u32(cursor + 16),
                compressedSize: Int(compressed),
                uncompressedSize: Int(uncompressed),
                localHeaderOffset: Int(offset)
            ))
            cursor += 46 + nameLength + extraLength + commentLength
        }
        self.bytes = bytes
        self.entries = entries
        self.index = Dictionary(entries.enumerated().map { ($0.element.path, $0.offset) }, uniquingKeysWith: { first, _ in first })
    }

    func contains(_ path: String) -> Bool { index[path] != nil }

    func data(for path: String) throws -> Data {
        guard let position = index[path] else { throw ZipError.missingEntry(path) }
        let entry = entries[position]
        if entry.flags & 1 != 0 { throw ZipError.encrypted(path) }
        let reader = ByteReader(bytes)
        guard try reader.u32(entry.localHeaderOffset) == 0x0403_4B50 else { throw ZipError.corrupt("bad local header for \(path)") }
        let start = entry.localHeaderOffset + 30 + Int(try reader.u16(entry.localHeaderOffset + 26)) + Int(try reader.u16(entry.localHeaderOffset + 28))
        let payload = try reader.slice(start, entry.compressedSize)

        let output: Data
        switch Method(rawValue: entry.method) {
        case .stored: output = payload
        case .deflated: output = try Self.inflate(payload, expectedSize: entry.uncompressedSize, path: path)
        case nil: throw ZipError.unsupportedMethod(entry.method, path)
        }
        guard output.count == entry.uncompressedSize else { throw ZipError.corrupt("\(path) is \(output.count) bytes, expected \(entry.uncompressedSize)") }
        guard CRC32.checksum(output) == entry.crc32 else { throw ZipError.corrupt("\(path) fails its checksum") }
        return output
    }

    /// Resolves a relationship or manifest `href` against the entry that names it.
    static func resolve(_ href: String, relativeTo base: String) -> String {
        let target = (href.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? href)
        let decoded = target.removingPercentEncoding ?? target
        var parts: [Substring] = decoded.hasPrefix("/") ? [] : base.split(separator: "/").dropLast()
        for component in decoded.split(separator: "/") {
            switch component {
            case ".": continue
            case "..": if !parts.isEmpty { parts.removeLast() }
            default: parts.append(component)
            }
        }
        return parts.joined(separator: "/")
    }

    private static func findEndOfCentralDirectory(_ reader: ByteReader) throws -> Int {
        let lowest = max(0, reader.count - 22 - 0xFFFF)
        var position = reader.count - 22
        while position >= lowest {
            if try reader.u32(position) == 0x0605_4B50 { return position }
            position -= 1
        }
        throw ZipError.notAZip
    }

    private static func inflate(_ payload: Data, expectedSize: Int, path: String) throws -> Data {
        guard expectedSize > 0 else { return Data() }
        var output = Data(count: expectedSize)
        let written = output.withUnsafeMutableBytes { destination in
            payload.withUnsafeBytes { source in
                compression_decode_buffer(
                    destination.bindMemory(to: UInt8.self).baseAddress!, expectedSize,
                    source.bindMemory(to: UInt8.self).baseAddress!, payload.count,
                    nil, COMPRESSION_ZLIB
                )
            }
        }
        guard written > 0 else { throw ZipError.corrupt("\(path) does not inflate") }
        return output.prefix(written)
    }
}

/// Writes a ZIP archive in memory. Used to build OOXML and EPUB packages for tests.
struct ZipWriter {
    private var body = Data()
    private var directory = Data()
    private var count = 0

    mutating func add(_ path: String, _ text: String, method: ZipArchive.Method = .stored) {
        add(path, data: Data(text.utf8), method: method)
    }

    mutating func add(_ path: String, data: Data, method: ZipArchive.Method = .stored) {
        let payload = method == .deflated ? Self.deflate(data) : data
        let name = Data(path.utf8)
        let crc = CRC32.checksum(data)
        let offset = body.count

        var local = Data()
        local.append(le: UInt32(0x0403_4B50))
        local.append(le: UInt16(20))
        local.append(le: UInt16(0x0800))
        local.append(le: method.rawValue)
        local.append(le: UInt32(0))
        local.append(le: crc)
        local.append(le: UInt32(payload.count))
        local.append(le: UInt32(data.count))
        local.append(le: UInt16(name.count))
        local.append(le: UInt16(0))
        body.append(local)
        body.append(name)
        body.append(payload)

        directory.append(le: UInt32(0x0201_4B50))
        directory.append(le: UInt16(20))
        directory.append(local[4..<30])
        directory.append(le: UInt16(0))
        directory.append(le: UInt16(0))
        directory.append(le: UInt16(0))
        directory.append(le: UInt32(0))
        directory.append(le: UInt32(offset))
        directory.append(name)
        count += 1
    }

    func finish() -> Data {
        var out = body
        out.append(directory)
        out.append(le: UInt32(0x0605_4B50))
        out.append(le: UInt16(0))
        out.append(le: UInt16(0))
        out.append(le: UInt16(count))
        out.append(le: UInt16(count))
        out.append(le: UInt32(directory.count))
        out.append(le: UInt32(body.count))
        out.append(le: UInt16(0))
        return out
    }

    private static func deflate(_ data: Data) -> Data {
        let capacity = data.count + data.count / 2 + 1024
        var output = Data(count: capacity)
        let written = output.withUnsafeMutableBytes { destination in
            data.withUnsafeBytes { source in
                compression_encode_buffer(
                    destination.bindMemory(to: UInt8.self).baseAddress!, capacity,
                    source.bindMemory(to: UInt8.self).baseAddress!, data.count,
                    nil, COMPRESSION_ZLIB
                )
            }
        }
        return output.prefix(written)
    }
}

enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { n in
        (0..<8).reduce(UInt32(n)) { c, _ in c & 1 == 1 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
    }

    static func checksum(_ data: Data) -> UInt32 {
        ~data.reduce(~UInt32(0)) { table[Int(($0 ^ UInt32($1)) & 0xFF)] ^ ($0 >> 8) }
    }
}

private struct ByteReader {
    let data: Data
    init(_ data: Data) { self.data = data }
    var count: Int { data.count }

    func slice(_ offset: Int, _ length: Int) throws -> Data {
        guard offset >= 0, length >= 0, offset + length <= data.count else { throw ZipError.corrupt("read past the end at \(offset)") }
        return data.subdata(in: offset..<(offset + length))
    }

    func u16(_ offset: Int) throws -> UInt16 {
        let b = try slice(offset, 2)
        return UInt16(b[0]) | UInt16(b[1]) << 8
    }

    func u32(_ offset: Int) throws -> UInt32 {
        let b = try slice(offset, 4)
        return UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24
    }
}

private extension Data {
    mutating func append<T: FixedWidthInteger>(le value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
