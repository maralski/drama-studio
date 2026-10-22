import Metal
import XCTest
@testable import DramaEngine

/// The packed weight cache on two small synthetic matrices: built once,
/// mapped back byte for byte, and rebuilt when a file is damaged or the
/// matrix list changes. No model needed.
final class PackedWeightCacheTests: XCTestCase {
    let device = MTLCreateSystemDefaultDevice()!
    let model = "snap-0123456789abcdef"
    let sources = [ShardFingerprint(name: "model-00001-of-00002.safetensors", size: 4_000_000_000, modified: 1_760_000_000_000_000_000, inode: 12345),
                   ShardFingerprint(name: "model-00002-of-00002.safetensors", size: 3_000_000_000, modified: 1_760_000_000_000_000_001, inode: 12346)]

    func tempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("drama-packed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func matrices() -> [(DenseMatrix, TensorRef)] {
        [(DenseMatrix(name: "model.layers.0.in_proj_qkv.weight", rowOffset: 0, rows: 64, cols: 512),
          PackedGemvTests.realisticBF16(device, rows: 64, cols: 512, seed: 3)),
         (DenseMatrix(name: "lm_head.weight", rowOffset: 16, rows: 48, cols: 256),
          PackedGemvTests.realisticBF16(device, rows: 48, cols: 256, seed: 4))]
    }

    func open(_ root: URL, _ ms: [(DenseMatrix, TensorRef)], sources: [ShardFingerprint]? = nil) throws -> PackedWeightCache {
        try PackedWeightCache(matrices: ms, model: model, sources: sources ?? self.sources, root: root, device: device, log: { _ in })
    }

    /// Every weight of every matrix, read back through `unpack(row:col:)`
    /// from the cache's buffers, against the source bits.
    func assertRoundTrip(_ cache: PackedWeightCache, _ ms: [(DenseMatrix, TensorRef)], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(cache.matrices.count, ms.count, file: file, line: line)
        for (m, source) in ms {
            guard let packed = cache.matrices[m] else { XCTFail("\(m.name) missing", file: file, line: line); continue }
            XCTAssertEqual(packed.n, m.rows, file: file, line: line)
            XCTAssertEqual(packed.k, m.cols, file: file, line: line)
            let bits = source.pointer.assumingMemoryBound(to: UInt16.self)
            var wrong = 0
            for row in 0..<m.rows {
                for col in 0..<m.cols where packed.unpack(row: row, col: col) != bits[row * m.cols + col] { wrong += 1 }
            }
            XCTAssertEqual(wrong, 0, "\(m.name): weights that do not round-trip", file: file, line: line)
        }
    }

    func testBuildThenMappedReloadRoundTripsEveryWeight() throws {
        let root = try tempRoot()
        let ms = matrices()
        let built = try open(root, ms)
        guard case .built = built.outcome else { return XCTFail("first open must build, got \(built.outcome)") }
        assertRoundTrip(built, ms)
        XCTAssertNil(built.verify(against: ms).map { "\($0)" })
        let dir = root.appendingPathComponent(model)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("manifest.json").path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".packed") }.count, 2)

        let loaded = try open(root, ms)
        XCTAssertEqual(loaded.outcome, .loaded)
        assertRoundTrip(loaded, ms)
        XCTAssertEqual(loaded.byteCount, built.byteCount)
        XCTAssertEqual(loaded.buffers.count, 2)
        for (m, _) in ms {
            let fresh = PackedBF16(ms.first { $0.0 == m }!.1, device: device)
            let mapped = loaded.matrices[m]!
            XCTAssertEqual(mapped.byteCount, fresh.byteCount)
            XCTAssertEqual(mapped.escapedWeights, fresh.escapedWeights)
            XCTAssertEqual(memcmp(mapped.buffer.contents() + mapped.headerOffset, fresh.buffer.contents() + fresh.headerOffset, fresh.storedByteCount), 0, m.name)
        }
        try assertGPUParity(loaded, ms)
    }

    /// The mapped matrices through the packed GEMV against the bf16 GEMV on
    /// their sources, bit for bit: the kernel decodes the files independently
    /// of the host decoders the build and `assertRoundTrip` trust.
    func assertGPUParity(_ cache: PackedWeightCache, _ ms: [(DenseMatrix, TensorRef)], file: StaticString = #filePath, line: UInt = #line) throws {
        var f = try PackedGemvTests.Fixture()
        let k = Kernels(ctx: f.ctx, precision: .exact)
        for (m, source) in ms {
            let packed = try XCTUnwrap(cache.matrices[m], file: file, line: line)
            let x = f.floats(m.cols, label: "x")
            let yRef = Buf(f.ctx.device, floats: m.rows, label: "yRef"), y = Buf(f.ctx.device, floats: m.rows, label: "y")
            try f.run { enc in
                try k.gemv(enc, source, x: x, y: yRef, m: 1)
                try k.gemvPacked(enc, packed, x: x, y: y, m: 1)
            }
            XCTAssertEqual(y.read(m.rows).map(\.bitPattern), yRef.read(m.rows).map(\.bitPattern), m.name, file: file, line: line)
        }
    }

    func testCorruptedPayloadIsRebuilt() throws {
        let root = try tempRoot()
        let ms = matrices()
        _ = try open(root, ms)
        let file = root.appendingPathComponent(model).appendingPathComponent(PackedWeightCache.fileName(ms[0].0))
        var data = try Data(contentsOf: file)
        data[PackedWeightCache.fileHeaderBytes + 300] ^= 0x40
        try data.write(to: file)
        let reopened = try open(root, ms)
        guard case .built = reopened.outcome else { return XCTFail("a corrupted payload must rebuild, got \(reopened.outcome)") }
        assertRoundTrip(reopened, ms)
        XCTAssertEqual(try open(root, ms).outcome, .loaded)
    }

    /// A manifest whose counts cannot be (a negative escape count, a payload
    /// size above any Int32) is a rebuild, not a trap.
    func testImpossibleManifestCountsAreRebuilt() throws {
        let root = try tempRoot()
        let ms = matrices()
        _ = try open(root, ms)
        let manifestURL = root.appendingPathComponent(model).appendingPathComponent(PackedWeightCache.manifestName)
        let good = try Data(contentsOf: manifestURL)
        var manifest = try JSONDecoder().decode(PackedWeightCache.Manifest.self, from: good)
        manifest.entries[0].escaped = -1
        try JSONEncoder().encode(manifest).write(to: manifestURL)
        guard case .built = try open(root, ms).outcome else { return XCTFail("a negative escape count must rebuild") }
        manifest = try JSONDecoder().decode(PackedWeightCache.Manifest.self, from: good)
        manifest.entries[1].bytes = Int.max - 7
        try JSONEncoder().encode(manifest).write(to: manifestURL)
        guard case .built = try open(root, ms).outcome else { return XCTFail("an impossible payload size must rebuild") }
        XCTAssertEqual(try open(root, ms).outcome, .loaded)
    }

    /// A matrix whose exponents all fit one window has no escapes; its
    /// escape section still binds and the GEMV still matches bf16.
    func testMatrixWithoutEscapesBindsAndMatches() throws {
        var f = try PackedGemvTests.Fixture()
        let n = 64, kDim = 256
        let buf = f.ctx.device.makeBuffer(length: n * kDim * 2, options: .storageModeShared)!
        let p = buf.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<(n * kDim) { p[i] = UInt16(0x3f80 + (i % 7) * 0x80) }
        let w = TensorRef(name: "flat", dtype: .bf16, shape: [n, kDim], buffer: buf, offset: 0)
        let packed = PackedBF16(w, device: f.ctx.device)
        XCTAssertEqual(packed.escapedWeights, 0)
        XCTAssertEqual(packed.buffer.length, packed.escapeOffset + PackedBF16.escapePad)
        let x = f.floats(kDim, label: "x")
        let yRef = Buf(f.ctx.device, floats: n, label: "yRef"), y = Buf(f.ctx.device, floats: n, label: "y")
        let k = Kernels(ctx: f.ctx, precision: .exact)
        try f.run { enc in
            try k.gemv(enc, w, x: x, y: yRef, m: 1)
            try k.gemvPacked(enc, packed, x: x, y: y, m: 1)
        }
        XCTAssertEqual(y.read(n).map(\.bitPattern), yRef.read(n).map(\.bitPattern))
        let root = try tempRoot()
        let ms = [(DenseMatrix(name: "flat.weight", rowOffset: 0, rows: n, cols: kDim), w)]
        _ = try open(root, ms)
        let loaded = try open(root, ms)
        XCTAssertEqual(loaded.outcome, .loaded)
        try assertGPUParity(loaded, ms)
    }

    func testTruncatedFileIsRebuilt() throws {
        let root = try tempRoot()
        let ms = matrices()
        _ = try open(root, ms)
        let file = root.appendingPathComponent(model).appendingPathComponent(PackedWeightCache.fileName(ms[1].0))
        let data = try Data(contentsOf: file)
        try data.prefix(data.count - 1).write(to: file)
        let reopened = try open(root, ms)
        guard case .built = reopened.outcome else { return XCTFail("a truncated file must rebuild, got \(reopened.outcome)") }
        assertRoundTrip(reopened, ms)
    }

    func testChangedMatrixListIsRebuilt() throws {
        let root = try tempRoot()
        var ms = matrices()
        _ = try open(root, ms)
        ms.append((DenseMatrix(name: "audio_decoder_mtp.heads.0.weight", rowOffset: 0, rows: 32, cols: 256),
                   PackedGemvTests.realisticBF16(device, rows: 32, cols: 256, seed: 5)))
        let reopened = try open(root, ms)
        guard case .built = reopened.outcome else { return XCTFail("a new matrix must rebuild, got \(reopened.outcome)") }
        assertRoundTrip(reopened, ms)
        XCTAssertEqual(try open(root, Array(ms.prefix(2))).matrices.count, 2)
    }

    /// Shards re-exported under the same names (a new modification time, or
    /// a new size) rebuild the cache; the same fingerprints load it.
    func testChangedShardFingerprintIsRebuilt() throws {
        let root = try tempRoot()
        let ms = matrices()
        _ = try open(root, ms)
        var touched = sources
        touched[1] = ShardFingerprint(name: touched[1].name, size: touched[1].size, modified: touched[1].modified + 1, inode: touched[1].inode)
        guard case .built = try open(root, ms, sources: touched).outcome else { return XCTFail("a re-exported shard must rebuild") }
        XCTAssertEqual(try open(root, ms, sources: touched).outcome, .loaded)
        var grown = touched
        grown[0] = ShardFingerprint(name: grown[0].name, size: grown[0].size + 1, modified: grown[0].modified, inode: grown[0].inode)
        guard case .built = try open(root, ms, sources: grown).outcome else { return XCTFail("a resized shard must rebuild") }
        guard case .built = try open(root, ms, sources: sources).outcome else { return XCTFail("the original shards must rebuild over the newer cache") }
    }

    /// A build directory of this process survives another open, one whose
    /// process is gone is removed, and a finished build leaves none behind.
    func testStaleBuildDirectoriesAreRemovedButOwnOnesKept() throws {
        let root = try tempRoot()
        let ms = matrices()
        let own = root.appendingPathComponent("\(model).building-\(getpid())-\(UUID().uuidString)")
        let dead = root.appendingPathComponent("\(model).building-\(pid_t.max)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: own, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dead, withIntermediateDirectories: true)
        _ = try open(root, ms)
        let left = try FileManager.default.contentsOfDirectory(atPath: root.path).sorted()
        XCTAssertEqual(left, [model, own.lastPathComponent].sorted())
    }

    /// A packed root that cannot hold a cache (a plain file in its place)
    /// leaves the model loaded with every family bf16 and the reason recorded.
    func testUnusablePackedRootLoadsTheModelBF16() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 weights needed") }
        guard PackedWeightCache.enabled else { throw XCTSkip("DRAMA_PACKED=0") }
        let root = try tempRoot().appendingPathComponent("not-a-directory")
        try Data().write(to: root)
        let model = try DramaModel(location: loc, schedule: .default, packedDirectory: root)
        guard case .unavailable(let reason) = model.packedSource else { return XCTFail("expected .unavailable, got \(model.packedSource)") }
        XCTAssertFalse(reason.isEmpty)
        XCTAssertNil(model.packed)
        var coded = 0
        for layer in model.weights.tower.shared + model.weights.tower.residual {
            for w in [layer.wqkv, layer.w1, layer.w3] { if case .coded = w { coded += 1 } }
        }
        for head in model.weights.tower.heads { if case .coded = head { coded += 1 } }
        if case .coded = model.weights.backbone.lmHeadSemantic { coded += 1 }
        for layer in model.weights.backbone.layers {
            switch layer.mixer {
            case .linear(let w): for d in [w.inProjQKV, w.inProjZ] { if case .coded = d { coded += 1 } }
            case .full(let w): if case .coded = w.qProj { coded += 1 }
            }
        }
        XCTAssertEqual(coded, 0)
        XCTAssertEqual(model.residentWeights.count, model.store.shards.count)
    }

    /// The buffer of a mapped file keeps its bytes after the MappedBuffer
    /// that made it is gone, as a PackedBF16 or TensorRef holding only the
    /// buffer needs.
    func testMappedBufferOutlivesItsMapper() throws {
        let url = try tempRoot().appendingPathComponent("weights.bin")
        let bytes = Data((0..<70_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        try bytes.write(to: url)
        func mapOnly() throws -> MTLBuffer {
            let mapped = try MappedBuffer(url: url, device: device)
            XCTAssertEqual(mapped.fileSize, bytes.count)
            XCTAssertEqual(mapped.fingerprint.name, "weights.bin")
            XCTAssertEqual(mapped.fingerprint.size, bytes.count)
            return mapped.buffer
        }
        let buffer = try mapOnly()
        XCTAssertEqual(bytes.withUnsafeBytes { memcmp(buffer.contents(), $0.baseAddress!, bytes.count) }, 0)
    }

    /// The tile verifier the build trusts: a flipped mantissa byte and a
    /// flipped exponent code are both found at their weight.
    func testTileVerifierFindsFlippedBytes() throws {
        let source = PackedGemvTests.realisticBF16(device, rows: 4, cols: 512, seed: 6)
        let bits = UnsafeBufferPointer(start: source.pointer.assumingMemoryBound(to: UInt16.self), count: source.count)
        let packed = PackedBF16(source, device: device)
        XCTAssertNil(packed.firstMismatch(against: bits))
        let mant = (packed.buffer.contents() + packed.mantOffset).assumingMemoryBound(to: UInt8.self)
        mant[777] ^= 0x01
        XCTAssertEqual(packed.firstMismatch(against: bits), 777)
        mant[777] ^= 0x01
        XCTAssertNil(packed.firstMismatch(against: bits))
        let codes = (packed.buffer.contents() + packed.codeOffset).assumingMemoryBound(to: UInt8.self)
        codes[PackedBF16.codesPerTile * 3] ^= 0x01
        let found = try XCTUnwrap(packed.firstMismatch(against: bits))
        XCTAssertTrue((768..<1024).contains(found), "code byte 0 of tile 3 lane 0 is weight 768 to 775, and its escapes shift the rest of tile 3; found \(found)")
    }
}
