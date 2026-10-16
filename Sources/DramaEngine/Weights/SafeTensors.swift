import Foundation
import Metal

public enum TensorDType: String, Sendable {
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
public struct TensorRef: Sendable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let buffer: MTLBuffer
    public let offset: Int
    public var count: Int { shape.reduce(1, *) }
    public var byteCount: Int { count * dtype.byteSize }
    /// GPU virtual address of the first element, for address tables.
    public var gpuAddress: UInt64 { buffer.gpuAddress + UInt64(offset) }
    public var pointer: UnsafeRawPointer { UnsafeRawPointer(buffer.contents()).advanced(by: offset) }

    func expect(_ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
        guard self.dtype == dtype, self.shape == shape else {
            throw EngineError.badModel("\(name): expected \(dtype) \(shape), found \(self.dtype) \(self.shape)")
        }
        return self
    }
}

/// All shards of a safetensors checkpoint, memory-mapped and wrapped as Metal
/// buffers without copying.
public final class SafeTensorsStore: @unchecked Sendable {
    public let directory: URL
    public private(set) var shards: [MTLBuffer] = []
    private var tensors: [String: TensorRef] = [:]
    private var mappings: [(UnsafeMutableRawPointer, Int)] = []

    public init(directory: URL, device: MTLDevice) throws {
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

    deinit {
        for (pointer, length) in mappings { munmap(pointer, length) }
    }

    public func tensor(_ name: String) throws -> TensorRef {
        guard let t = tensors[name] else { throw EngineError.badModel("missing tensor \(name)") }
        return t
    }

    public func has(_ name: String) -> Bool { tensors[name] != nil }
    public var tensorNames: [String] { Array(tensors.keys) }

    /// Tensors closer than this share one pinned run.
    static let pinnedRunGap = 1 << 20

    /// Keeps `refs` readable in place without binding whole shards: coalesces
    /// them into page-aligned runs of their shard mappings, wraps each run in
    /// a Metal buffer of its own, and returns the refs re-pointed at the run
    /// buffers, in the given order, plus the runs. Metal wires a buffer in full
    /// the first time a command buffer binds it, so what a quantized mode keeps
    /// in place must be bound as runs and never as the 41 GB of shards.
    public func pinned(_ refs: [TensorRef], device: MTLDevice) throws -> (refs: [TensorRef], runs: [MTLBuffer]) {
        let pageSize = Int(getpagesize())
        var runs: [MTLBuffer] = []
        var repointed = [TensorRef?](repeating: nil, count: refs.count)
        for shard in shards {
            let members = refs.indices.filter { refs[$0].buffer === shard }.sorted { refs[$0].offset < refs[$1].offset }
            var i = 0
            while i < members.count {
                let start = refs[members[i]].offset / pageSize * pageSize
                var end = 0
                var j = i
                while j < members.count {
                    let ref = refs[members[j]]
                    let refStart = ref.offset / pageSize * pageSize
                    if j > i && refStart - end >= Self.pinnedRunGap { break }
                    end = max(end, (ref.offset + ref.byteCount + pageSize - 1) / pageSize * pageSize)
                    j += 1
                }
                guard let run = device.makeBuffer(bytesNoCopy: shard.contents() + start, length: end - start, options: .storageModeShared, deallocator: nil) else {
                    throw EngineError.badModel("Metal refused a \(end - start) byte run of \(shard.label ?? "a shard")")
                }
                run.label = "\(shard.label ?? "shard")+\(start)"
                for m in members[i..<j] {
                    let ref = refs[m]
                    repointed[m] = TensorRef(name: ref.name, dtype: ref.dtype, shape: ref.shape, buffer: run, offset: ref.offset - start)
                }
                runs.append(run)
                i = j
            }
        }
        let all = repointed.compactMap { $0 }
        guard all.count == refs.count else { throw EngineError.badModel("a tensor to pin is not in any shard") }
        return (all, runs)
    }

    private func mapShard(_ url: URL, device: MTLDevice) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.badModel("cannot open \(url.lastPathComponent)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw EngineError.badModel("cannot stat \(url.lastPathComponent)") }
        let fileSize = Int(st.st_size)
        let pageSize = Int(getpagesize())
        let mappedLength = (fileSize + pageSize - 1) / pageSize * pageSize
        guard let base = mmap(nil, mappedLength, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw EngineError.badModel("mmap failed for \(url.lastPathComponent)")
        }
        mappings.append((base, mappedLength))
        guard let buffer = device.makeBuffer(bytesNoCopy: base, length: mappedLength, options: .storageModeShared, deallocator: nil) else {
            throw EngineError.badModel("Metal refused the mapping of \(url.lastPathComponent)")
        }
        buffer.label = url.lastPathComponent
        shards.append(buffer)

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
