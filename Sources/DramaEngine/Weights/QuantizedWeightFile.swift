import CryptoKit
import Foundation
import Metal

/// Where each item of a quantized mode lives in the cache file: every item
/// 256-byte aligned after a 4096-byte header, weights first and the scales
/// or (scale, bias) pairs after them. The recorder derives it from the
/// model's tensor list; the file header carries its digest.
struct QuantizedLayout: Equatable {
    enum Kind: String { case matrix, vector, experts }
    struct Key: Hashable {
        let name: String
        let rowOffset: Int
    }
    struct Item: Equatable {
        let name: String
        let kind: Kind
        let format: WeightFormat
        /// For experts, the rows of all experts of the layer, expert-major.
        let rows: Int
        let cols: Int
        let rowOffset: Int
        let offset: Int
        let byteCount: Int
        var key: Key { Key(name: name, rowOffset: rowOffset) }
        var paramOffset: Int { offset + QuantizedLayout.aligned(format.weightBytes(rows: rows, cols: cols)) }
        var row: String { [name, kind.rawValue, format.rawValue, String(rows), String(cols), String(rowOffset), String(offset), String(byteCount)].joined(separator: "\t") }
    }

    static let headerBytes = 4096
    static func aligned(_ n: Int) -> Int { (n + 255) / 256 * 256 }

    let items: [Item]
    let payloadBytes: Int
    let digest: Data
    private let index: [Key: Item]

    init(items: [Item]) {
        self.items = items
        payloadBytes = (items.last.map { $0.offset + $0.byteCount } ?? Self.headerBytes) - Self.headerBytes
        digest = Data(SHA256.hash(data: Data(items.map(\.row).joined(separator: "\n").utf8)))
        index = Dictionary(uniqueKeysWithValues: items.map { ($0.key, $0) })
    }

    subscript(_ key: Key) -> Item? { index[key] }
}

/// Where the bytes of a layout item come from when the file is built.
enum BuildSource {
    /// bf16 bytes copied straight from the mapping.
    case copy(TensorRef)
    /// A bf16 matrix slice to quantize to the item's format.
    case quantize(TensorRef)
    /// One matrix kind of every fp8 expert of a layer, with their block scales.
    case experts([(weights: TensorRef, scales: TensorRef)], ExpertMatrix)
}

/// A quantized mode's layout, how to build it, and which tensors stay in place.
struct QuantizationPlan {
    let layout: QuantizedLayout
    let sources: [BuildSource]
    let pinnedNames: [String]
}

/// What a `DramaModel` in a quantized mode reports about its cache file.
public struct QuantizedCacheInfo: Sendable, Equatable {
    public let url: URL
    public let bytes: Int
    /// Seconds spent building the file this load; nil on a cache hit.
    public let buildSeconds: Double?
}

/// A quantized mode's weights, memory-mapped read-only and wrapped as one
/// shared Metal buffer. The header is written last and the file renamed
/// into place, so a file left by a crash never verifies.
public final class QuantizedWeightFile {
    struct Header: Equatable {
        static let magic = Data("DRMQ".utf8)
        /// Bump when a quantizer changes the bytes it writes for the same
        /// layout: the header carries no checksum of the payload, so an old
        /// file would otherwise verify and be read.
        static let currentVersion: UInt32 = 1
        var version: UInt32 = currentVersion
        var mode: String
        var modelId: String
        var digest: Data
        var payloadBytes: UInt64
        var itemCount: UInt32

        func encoded() -> Data {
            var data = Self.magic
            data.append(le(version))
            data.append(padded(mode, 16))
            data.append(padded(modelId, 64))
            data.append(digest.prefix(32))
            data.append(le(payloadBytes))
            data.append(le(itemCount))
            data.append(Data(count: QuantizedLayout.headerBytes - data.count))
            return data
        }

        init(version: UInt32 = currentVersion, mode: String, modelId: String, digest: Data, payloadBytes: UInt64, itemCount: UInt32) {
            self.version = version; self.mode = mode; self.modelId = modelId; self.digest = digest; self.payloadBytes = payloadBytes; self.itemCount = itemCount
        }

        init?(data: Data) {
            guard data.count >= QuantizedLayout.headerBytes else { return nil }
            let d = Data(data.prefix(QuantizedLayout.headerBytes))
            guard d.prefix(4) == Self.magic else { return nil }
            func u32(_ at: Int) -> UInt32 { d[at..<(at + 4)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian }
            func u64(_ at: Int) -> UInt64 { d[at..<(at + 8)].withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian }
            func string(_ at: Int, _ n: Int) -> String { String(decoding: d[at..<(at + n)].prefix { $0 != 0 }, as: UTF8.self) }
            version = u32(4)
            mode = string(8, 16)
            modelId = string(24, 64)
            digest = Data(d[88..<120])
            payloadBytes = u64(120)
            itemCount = u32(128)
        }

        private func le<T: FixedWidthInteger>(_ v: T) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        private func padded(_ s: String, _ n: Int) -> Data { var d = Data(s.utf8.prefix(n)); d.append(Data(count: n - d.count)); return d }
    }

    let url: URL
    let buffer: MTLBuffer
    /// The file's length: header plus payload.
    let bytes: Int
    private let mapping: (UnsafeMutableRawPointer, Int)

    private init(url: URL, buffer: MTLBuffer, bytes: Int, mapping: (UnsafeMutableRawPointer, Int)) {
        self.url = url; self.buffer = buffer; self.bytes = bytes; self.mapping = mapping
    }

    deinit { munmap(mapping.0, mapping.1) }

    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Quantized", isDirectory: true)
    }

    /// `<directory>/<snapshot>-<mode>.bin`; `snapshot` is the model directory's last path component.
    public static func url(snapshot: String, mode: ModelPrecisionMode, directory: URL = defaultDirectory) -> URL {
        directory.appendingPathComponent("\(snapshot)-\(mode.rawValue).bin")
    }

    /// The file at `url` when its header names this layout, model and mode
    /// and its length matches; nil otherwise, and a file that is there but
    /// does not verify is removed so the next load rebuilds it.
    static func open(url: URL, layout: QuantizedLayout, modelId: String, mode: ModelPrecisionMode, device: MTLDevice) -> QuantizedWeightFile? {
        let fd = Darwin.open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { return nil }
        let fileSize = Int(st.st_size)
        var headerBytes = Data(count: QuantizedLayout.headerBytes)
        let read = headerBytes.withUnsafeMutableBytes { pread(fd, $0.baseAddress, QuantizedLayout.headerBytes, 0) }
        let expected = Header(mode: mode.rawValue, modelId: modelId, digest: layout.digest, payloadBytes: UInt64(layout.payloadBytes), itemCount: UInt32(layout.items.count))
        guard read == QuantizedLayout.headerBytes, Header(data: headerBytes) == expected, fileSize == QuantizedLayout.headerBytes + layout.payloadBytes else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        let pageSize = Int(getpagesize())
        let mappedLength = (fileSize + pageSize - 1) / pageSize * pageSize
        guard let base = mmap(nil, mappedLength, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else { return nil }
        guard let buffer = device.makeBuffer(bytesNoCopy: base, length: mappedLength, options: .storageModeShared, deallocator: nil) else {
            munmap(base, mappedLength)
            return nil
        }
        buffer.label = url.lastPathComponent
        return QuantizedWeightFile(url: url, buffer: buffer, bytes: fileSize, mapping: (base, mappedLength))
    }

    /// Quantizes every item of `plan` on the GPU into `<url>.partial`, item
    /// by item through a reusable staging buffer, then writes the header and
    /// renames the file into place. Each item's source tensors are pinned
    /// as runs for the duration of the item, so the build never binds a
    /// whole shard.
    static func build(plan: QuantizationPlan, store: SafeTensorsStore, mode: ModelPrecisionMode, modelId: String, ctx: MetalContext, to url: URL) throws -> (seconds: Double, bytes: Int) {
        let t0 = Date()
        let layout = plan.layout
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let partial = url.appendingPathExtension("partial")
        let total = QuantizedLayout.headerBytes + layout.payloadBytes
        let directory = url.deletingLastPathComponent()
        if let free = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage, free < Int64(total) {
            throw EngineError.badModel(String(format: "the %@ weight cache needs %.1f GB of disk at %@ and %.1f GB is free", mode.rawValue, Double(total) / 1e9, directory.path, Double(free) / 1e9))
        }
        let fd = Darwin.open(partial.path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else { throw EngineError.badModel("cannot create \(partial.path)") }
        var renamed = false
        defer {
            close(fd)
            if !renamed { try? FileManager.default.removeItem(at: partial) }
        }
        guard ftruncate(fd, off_t(total)) == 0 else { throw EngineError.badModel("cannot size \(partial.path)") }
        func write(_ pointer: UnsafeRawPointer, count: Int, at offset: Int) throws {
            var done = 0
            while done < count {
                let n = pwrite(fd, pointer + done, count - done, off_t(offset + done))
                guard n > 0 else { throw EngineError.badModel("write failed at \(offset + done) of \(partial.path): \(String(cString: strerror(errno)))") }
                done += n
            }
        }
        let largest = layout.items.map(\.byteCount).max() ?? 0
        guard let staging = ctx.device.makeBuffer(length: max(largest, 256), options: .storageModeShared) else { throw EngineError.badModel("cannot allocate the quantize staging buffer") }
        let k = Kernels(ctx: ctx, precision: .exact)
        for (item, source) in zip(layout.items, plan.sources) {
            if case .copy(let ref) = source {
                try write(ref.pointer, count: ref.byteCount, at: item.offset)
                continue
            }
            let slot = QuantizedSlot(buffer: staging, offset: 0, paramOffset: item.paramOffset - item.offset)
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            switch source {
            case .copy: break
            case .quantize(let ref):
                let (pinned, runs) = try store.pinned([ref], device: ctx.device)
                enc.useResources(runs, usage: .read)
                switch item.format {
                case .int8: try k.quantizeInt8(enc, src: pinned[0], rowOffset: 0, rows: item.rows, cols: item.cols, into: slot)
                case .int4: try k.quantizeInt4(enc, src: pinned[0], rowOffset: 0, rows: item.rows, cols: item.cols, into: slot)
                case .bf16: throw EngineError.badModel("\(item.name): bf16 items are copied, not quantized")
                }
            case .experts(let refs, let kind):
                let (pinned, runs) = try store.pinned(refs.flatMap { [$0.weights, $0.scales] }, device: ctx.device)
                enc.useResources(runs, usage: .read)
                var addresses = [UInt64](repeating: 0, count: refs.count * 6)
                for e in 0..<refs.count {
                    addresses[e * 6 + kind.rawValue * 2] = pinned[2 * e].gpuAddress
                    addresses[e * 6 + kind.rawValue * 2 + 1] = pinned[2 * e + 1].gpuAddress
                }
                let table = ExpertTable(table: try ExpertTable.buffer(addresses, layer: -1, device: ctx.device), format: .fp8BlockScaled)
                try k.quantizeExpertsInt4(enc, experts: table, kind: kind, count: refs.count, rows: item.rows / refs.count, cols: item.cols, into: slot)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw EngineError.badModel("GPU error quantizing \(item.name): \(e)") }
            try write(staging.contents(), count: item.byteCount, at: item.offset)
        }
        // F_FULLFSYNC asks the drive to flush as well; fsync alone does not on macOS.
        func flush() throws {
            guard fcntl(fd, F_FULLFSYNC) == 0 || fsync(fd) == 0 else { throw EngineError.badModel("fsync failed for \(partial.path)") }
        }
        try flush()
        let header = Header(mode: mode.rawValue, modelId: modelId, digest: layout.digest, payloadBytes: UInt64(layout.payloadBytes), itemCount: UInt32(layout.items.count)).encoded()
        try header.withUnsafeBytes { try write($0.baseAddress!, count: header.count, at: 0) }
        try flush()
        guard rename(partial.path, url.path) == 0 else { throw EngineError.badModel("cannot rename \(partial.path) into place") }
        renamed = true
        return (Date().timeIntervalSince(t0), total)
    }
}
