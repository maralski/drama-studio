import Foundation
import Metal

enum TensorDType: String, Sendable {
    case bf16 = "BF16"
    case f32 = "F32"
    case f8e4m3 = "F8_E4M3"

    var byteSize: Int {
        switch self {
        case .bf16: return 2
        case .f32: return 4
        case .f8e4m3: return 1
        }
    }
}

/// A tensor inside one memory-mapped shard. The GPU reads it in place.
/// Unchecked because `MTLBuffer` is not marked Sendable; the buffer wraps
/// read-only mapped weights that no one writes after the load.
struct TensorRef: @unchecked Sendable {
    let name: String
    let dtype: TensorDType
    let shape: [Int]
    let buffer: MTLBuffer
    let offset: Int
    var count: Int { shape.reduce(1, *) }
    var byteCount: Int { count * dtype.byteSize }
    /// GPU virtual address of the first element, for address tables.
    var gpuAddress: UInt64 { buffer.gpuAddress + UInt64(offset) }
    var pointer: UnsafeRawPointer { UnsafeRawPointer(buffer.contents()).advanced(by: offset) }

    func expect(_ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
        guard self.dtype == dtype, self.shape == shape else {
            throw EngineError.badModel("\(name): expected \(dtype) \(shape), found \(self.dtype) \(self.shape)")
        }
        return self
    }
}

/// What identifies one weight file's bytes without reading them: its name,
/// size, modification time in nanoseconds and inode. A re-export under the
/// same name changes the time and usually the inode.
struct ShardFingerprint: Codable, Equatable {
    let name: String
    let size: Int
    let modified: Int64
    let inode: UInt64
}

/// A read-only file mapped whole, page-rounded, and wrapped as a Metal buffer
/// without copying. The buffer owns the mapping: it is unmapped when the
/// last holder of the buffer lets go, so a TensorRef or PackedBF16 that
/// outlives this value still reads mapped bytes.
struct MappedBuffer {
    let buffer: MTLBuffer
    let fileSize: Int
    let fingerprint: ShardFingerprint

    init(url: URL, device: MTLDevice) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.badModel("cannot open \(url.lastPathComponent)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw EngineError.badModel("cannot stat \(url.lastPathComponent)") }
        fileSize = Int(st.st_size)
        fingerprint = ShardFingerprint(name: url.lastPathComponent, size: fileSize,
                                       modified: Int64(st.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(st.st_mtimespec.tv_nsec), inode: UInt64(st.st_ino))
        let pageSize = Int(getpagesize())
        let length = max(fileSize + pageSize - 1, pageSize) / pageSize * pageSize
        guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw EngineError.badModel("mmap failed for \(url.lastPathComponent)")
        }
        guard let buffer = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared, deallocator: { pointer, length in munmap(pointer, length) }) else {
            munmap(base, length)
            throw EngineError.badModel("Metal refused the mapping of \(url.lastPathComponent)")
        }
        buffer.label = url.lastPathComponent
        self.buffer = buffer
    }
}

/// All shards of a safetensors checkpoint, memory-mapped and wrapped as Metal
/// buffers without copying.
final class SafeTensorsStore: @unchecked Sendable {
    let directory: URL
    private(set) var shards: [MTLBuffer] = []
    private var tensors: [String: TensorRef] = [:]
    private var files: [MappedBuffer] = []

    init(directory: URL, device: MTLDevice) throws {
        self.directory = directory
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
        guard let weightMap = index?["weight_map"] as? [String: String] else {
            throw EngineError.badModel("model.safetensors.index.json has no weight_map")
        }
        let shardNames = Array(Set(weightMap.values)).sorted()
        for name in shardNames {
            try mapShard(directory.appendingPathComponent(name), device: device)
        }
    }

    func tensor(_ name: String) throws -> TensorRef {
        guard let t = tensors[name] else { throw EngineError.badModel("missing tensor \(name)") }
        return t
    }

    func has(_ name: String) -> Bool { tensors[name] != nil }
    var tensorNames: [String] { Array(tensors.keys) }
    /// Every shard's fingerprint, in shard name order.
    var fingerprints: [ShardFingerprint] { files.map(\.fingerprint) }

    private func mapShard(_ url: URL, device: MTLDevice) throws {
        let mapped = try MappedBuffer(url: url, device: device)
        files.append(mapped)
        let buffer = mapped.buffer
        shards.append(buffer)
        let base = buffer.contents()

        let headerLength = Int(base.load(as: UInt64.self).littleEndian)
        let headerData = Data(bytes: base.advanced(by: 8), count: headerLength)
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw EngineError.badModel("bad safetensors header in \(url.lastPathComponent)")
        }
        let dataBase = 8 + headerLength
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtypeName = entry["dtype"] as? String,
                  let dtype = TensorDType(rawValue: dtypeName),
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else {
                throw EngineError.badModel("unsupported tensor entry \(name) in \(url.lastPathComponent)")
            }
            tensors[name] = TensorRef(name: name, dtype: dtype, shape: shape, buffer: buffer, offset: dataBase + offsets[0])
        }
    }
}
