import Metal
import XCTest
@testable import DramaEngine

/// Pinned runs over synthetic shards: the tensors a quantized mode keeps in
/// place must come back readable from page-aligned run buffers that cover
/// them without binding the whole shard. No model needed.
final class SafeTensorsTests: XCTestCase {
    struct Shard {
        var tensors: [(name: String, count: Int)]
        var fill: (Int) -> UInt16
    }

    /// Writes `shards` as bf16 safetensors files plus the index, and returns the directory.
    func writeCheckpoint(_ shards: [Shard]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("drama-shards-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        var weightMap: [String: String] = [:]
        for (i, shard) in shards.enumerated() {
            let file = "model-0000\(i + 1)-of-0000\(shards.count).safetensors"
            var header: [String: Any] = [:]
            var data = Data()
            for (name, count) in shard.tensors {
                header[name] = ["dtype": "BF16", "shape": [count], "data_offsets": [data.count, data.count + count * 2]]
                var values = (0..<count).map { shard.fill(data.count / 2 + $0) }
                data.append(Data(bytes: &values, count: count * 2))
                weightMap[name] = file
            }
            let headerData = try JSONSerialization.data(withJSONObject: header)
            var length = UInt64(headerData.count).littleEndian
            var bytes = Data(bytes: &length, count: 8)
            bytes.append(headerData)
            bytes.append(data)
            try bytes.write(to: dir.appendingPathComponent(file))
        }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try index.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        return dir
    }

    func testPinnedRunsCoverTensorsAndCoalesceNearOnes() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let pageSize = Int(getpagesize())
        let near = 1 << 19, far = SafeTensorsStore.pinnedRunGap + 2 * pageSize
        let dir = try writeCheckpoint([
            Shard(tensors: [("a", 64), ("gap1", near / 2), ("b", 1024 * 1024), ("gap2", far / 2), ("c", 64)], fill: { UInt16(truncatingIfNeeded: $0 &* 2654435761) }),
            Shard(tensors: [("d", 256)], fill: { UInt16(truncatingIfNeeded: 7 &* $0 &+ 3) }),
        ])
        let store = try SafeTensorsStore(directory: dir, device: device)
        let originals = try ["a", "c", "d", "b"].map { try store.tensor($0) }
        let (refs, runs) = try store.pinned(originals, device: device)

        XCTAssertEqual(refs.map(\.name), ["a", "c", "d", "b"])
        XCTAssertEqual(runs.count, 3, "a and b are closer than the gap and share a run; c and d get their own")
        XCTAssertTrue(refs[0].buffer === refs[3].buffer)
        XCTAssertFalse(refs[1].buffer === refs[0].buffer)
        XCTAssertFalse(refs[2].buffer === refs[1].buffer)
        for run in runs {
            XCTAssertEqual(Int(bitPattern: run.contents()) % pageSize, 0)
            XCTAssertEqual(run.length % pageSize, 0)
            XCTAssertFalse(store.shards.contains { $0 === run })
        }
        for (ref, original) in zip(refs, originals) {
            XCTAssertEqual(ref.pointer, original.pointer, ref.name)
            XCTAssertEqual(ref.shape, original.shape)
            XCTAssertEqual(ref.dtype, original.dtype)
            XCTAssertEqual(memcmp(ref.pointer, original.pointer, ref.byteCount), 0, ref.name)
            XCTAssertLessThanOrEqual(ref.offset + ref.byteCount, ref.buffer.length, ref.name)
            XCTAssertGreaterThanOrEqual(ref.offset, 0, ref.name)
        }
        XCTAssertLessThan(runs.map(\.length).reduce(0, +), store.shards.map(\.length).reduce(0, +))
        let b = refs[3].pointer.assumingMemoryBound(to: UInt16.self)
        XCTAssertEqual(b[0], UInt16(truncatingIfNeeded: (64 + near / 2) &* 2654435761))
    }

    func testPinnedRejectsTensorFromAnotherStore() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let dir = try writeCheckpoint([Shard(tensors: [("a", 64)], fill: { UInt16($0) })])
        let store = try SafeTensorsStore(directory: dir, device: device)
        let other = try SafeTensorsStore(directory: dir, device: device)
        XCTAssertThrowsError(try store.pinned([try other.tensor("a")], device: device))
    }
}
