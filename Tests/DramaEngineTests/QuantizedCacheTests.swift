import Metal
import XCTest
@testable import DramaEngine

/// The quantized weight cache: its file name, header, the checks that make
/// a stale or torn file a rebuild, the recorder's determinism and payload
/// sizes, and one balanced-mode load against mode full.
final class QuantizedCacheTests: XCTestCase {
    func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("drama-quantized-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    static let modelId = "snap-0123456789abcdef"

    /// One bf16 vector of 300 elements (600 bytes, padded to 768).
    static let layout = QuantizedLayout(items: [
        .init(name: "a", kind: .vector, format: .bf16, rows: 300, cols: 1, rowOffset: 0, offset: 4096, byteCount: 768),
    ])

    static func header(_ layout: QuantizedLayout = layout, mode: ModelPrecisionMode = .balanced, modelId: String = modelId) -> QuantizedWeightFile.Header {
        .init(mode: mode.rawValue, modelId: modelId, digest: layout.digest, payloadBytes: UInt64(layout.payloadBytes), itemCount: UInt32(layout.items.count))
    }

    func write(_ header: QuantizedWeightFile.Header, payload: Data, to url: URL) throws {
        var data = header.encoded()
        data.append(payload)
        try data.write(to: url)
    }

    func testFileNameFollowsSnapshotAndMode() {
        let dir = URL(fileURLWithPath: "/tmp/q")
        XCTAssertEqual(QuantizedWeightFile.url(snapshot: "e0a9218a", mode: .balanced, directory: dir).path, "/tmp/q/e0a9218a-balanced.bin")
        XCTAssertEqual(QuantizedWeightFile.url(snapshot: "e0a9218a", mode: .compact, directory: dir).lastPathComponent, "e0a9218a-compact.bin")
        XCTAssertEqual(Set(ModelPrecisionMode.allCases.map { QuantizedWeightFile.url(snapshot: "s", mode: $0, directory: dir) }).count, 3)
        XCTAssertNotEqual(QuantizedWeightFile.url(snapshot: "s1", mode: .balanced, directory: dir), QuantizedWeightFile.url(snapshot: "s2", mode: .balanced, directory: dir))
    }

    func testHeaderRoundTrips() throws {
        let h = Self.header()
        let data = h.encoded()
        XCTAssertEqual(data.count, QuantizedLayout.headerBytes)
        XCTAssertEqual(data.prefix(4), Data("DRMQ".utf8))
        XCTAssertEqual(QuantizedWeightFile.Header(data: data), h)
        XCTAssertEqual(QuantizedWeightFile.Header(data: data)?.digest.count, 32)
        XCTAssertNil(QuantizedWeightFile.Header(data: data.prefix(100)))
        var notMagic = data
        notMagic[0] = UInt8(ascii: "X")
        XCTAssertNil(QuantizedWeightFile.Header(data: notMagic))
    }

    func testLayoutPlacesItemsAlignedAndDigestsThem() {
        let a = QuantizedLayout(items: [
            .init(name: "m", kind: .matrix, format: .int8, rows: 96, cols: 512, rowOffset: 0, offset: 4096, byteCount: 96 * 512 + 96 * 4 * 2),
        ])
        XCTAssertEqual(a.items[0].paramOffset, 4096 + 96 * 512)
        XCTAssertEqual(a.payloadBytes, 96 * 512 + 768)
        XCTAssertEqual(a[.init(name: "m", rowOffset: 0)], a.items[0])
        XCTAssertNil(a[.init(name: "m", rowOffset: 1)])
        let b = QuantizedLayout(items: [
            .init(name: "m", kind: .matrix, format: .int4, rows: 96, cols: 512, rowOffset: 0, offset: 4096, byteCount: 96 * 512 + 96 * 4 * 2),
        ])
        XCTAssertNotEqual(a.digest, b.digest, "the digest covers the format")
        XCTAssertEqual(a.digest, QuantizedLayout(items: a.items).digest)
    }

    func testOpenAcceptsAMatchingFileAndRejectsEveryMismatch() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let dir = try tempDirectory()
        let url = QuantizedWeightFile.url(snapshot: "s", mode: .balanced, directory: dir)
        var payload = Data(count: Self.layout.payloadBytes)
        payload[0] = 0xab
        payload[599] = 0xcd
        try write(Self.header(), payload: payload, to: url)
        let file = try XCTUnwrap(QuantizedWeightFile.open(url: url, layout: Self.layout, modelId: Self.modelId, mode: .balanced, device: device))
        XCTAssertEqual(file.bytes, 4096 + 768)
        XCTAssertGreaterThanOrEqual(file.buffer.length, file.bytes)
        let bytes = file.buffer.contents().assumingMemoryBound(to: UInt8.self)
        XCTAssertEqual(bytes[4096], 0xab)
        XCTAssertEqual(bytes[4096 + 599], 0xcd)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        var otherDigest = Self.header()
        otherDigest.digest = Data(repeating: 7, count: 32)
        var otherVersion = Self.header()
        otherVersion.version = 2
        var otherCount = Self.header()
        otherCount.itemCount = 2
        let rejected: [(String, QuantizedWeightFile.Header, Data)] = [
            ("wrong mode", Self.header(mode: .compact), payload),
            ("wrong model id", Self.header(modelId: "other-fedcba9876543210"), payload),
            ("wrong version", otherVersion, payload),
            ("wrong digest", otherDigest, payload),
            ("wrong item count", otherCount, payload),
            ("short file", Self.header(), payload.dropLast()),
            ("long file", Self.header(), payload + Data(count: 1)),
        ]
        for (name, header, body) in rejected {
            try write(header, payload: body, to: url)
            XCTAssertNil(QuantizedWeightFile.open(url: url, layout: Self.layout, modelId: Self.modelId, mode: .balanced, device: device), name)
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path), "\(name): the file must be removed so the next load rebuilds it")
        }
        XCTAssertNil(QuantizedWeightFile.open(url: url, layout: Self.layout, modelId: Self.modelId, mode: .balanced, device: device), "absent")
    }

    func testModeTable() {
        for mode in ModelPrecisionMode.allCases {
            XCTAssertEqual(mode.format(for: .narrow), .bf16, "\(mode)")
        }
        XCTAssertEqual(ModelPrecisionMode.allCases.filter(\.isExperimental), [.compact], "the measured verdict: compact only")
        XCTAssertEqual(ModelPrecisionMode.full.format(for: .backbone), .bf16)
        XCTAssertEqual(ModelPrecisionMode.full.format(for: .tower), .bf16)
        XCTAssertEqual(ModelPrecisionMode.balanced.format(for: .backbone), .int8)
        XCTAssertEqual(ModelPrecisionMode.balanced.format(for: .tower), .int8)
        XCTAssertEqual(ModelPrecisionMode.compact.format(for: .backbone), .int4)
        XCTAssertEqual(ModelPrecisionMode.compact.format(for: .tower), .int8)
        XCTAssertEqual(ModelPrecisionMode.full.expertFormat, .fp8BlockScaled)
        XCTAssertEqual(ModelPrecisionMode.balanced.expertFormat, .fp8BlockScaled)
        XCTAssertEqual(ModelPrecisionMode.compact.expertFormat, .int4)
    }

    /// The recorder walks the real tensor list; mapping the shards is cheap
    /// and nothing is read.
    func testRecorderIsDeterministicAndPayloadsAreInRange() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 weights needed") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        let store = try SafeTensorsStore(directory: loc.directory, device: device)
        let inPlace = InPlaceSource(store: store, config: config, device: device)
        func plan(_ mode: ModelPrecisionMode) throws -> QuantizationPlan {
            let recorder = LayoutRecorder(inPlace: inPlace, mode: mode)
            _ = try ModelWeights(source: recorder, config: config)
            return recorder.plan
        }
        let balanced = try plan(.balanced), compact = try plan(.compact)
        XCTAssertEqual(try plan(.balanced).layout, balanced.layout)
        XCTAssertEqual(try plan(.compact).layout, compact.layout)
        XCTAssertNotEqual(balanced.layout.digest, compact.layout.digest)
        let gb = 1e9
        XCTAssertGreaterThan(Double(balanced.layout.payloadBytes), 2 * gb)
        XCTAssertLessThan(Double(balanced.layout.payloadBytes), 4 * gb)
        XCTAssertGreaterThan(Double(compact.layout.payloadBytes), 17 * gb)
        XCTAssertLessThan(Double(compact.layout.payloadBytes), 23 * gb)
        print("quantized payload: balanced \(balanced.layout.payloadBytes) bytes, compact \(compact.layout.payloadBytes) bytes")
        let embeddings = 1 + 1 + config.towerLayers + config.numCodebooks - 1
        XCTAssertEqual(compact.pinnedNames.count, embeddings)
        XCTAssertEqual(balanced.pinnedNames.count, embeddings + config.numLayers * config.numExperts * 6)
        XCTAssertEqual(compact.layout.items.filter { $0.kind == .experts }.count, config.numLayers * 3)
        XCTAssertTrue(balanced.layout.items.allSatisfy { $0.kind != .experts })
        for item in balanced.layout.items + compact.layout.items {
            XCTAssertEqual(item.offset % 256, 0, item.name)
            XCTAssertGreaterThanOrEqual(item.offset, QuantizedLayout.headerBytes)
        }
        XCTAssertEqual(balanced.layout[.init(name: "lm_head.weight", rowOffset: Int(config.eos))]?.rows, 1)
        XCTAssertEqual(balanced.layout[.init(name: "lm_head.weight", rowOffset: Int(config.semanticStart))]?.rows, config.codebookSize)
        XCTAssertNil(balanced.layout[.init(name: "lm_head.weight", rowOffset: 0)])
    }

    /// Mode balanced end to end: the first load builds the file, the second
    /// hits it, and on a short prompt both agree with mode full on the most
    /// likely main token.
    func testBalancedModeBuildsOnceAndAgreesWithFullOnArgmax() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 weights needed") }
        let dir = try tempDirectory()
        let tokenizer = try Tokenizer(url: loc.directory.appendingPathComponent("tokenizer.json"))
        let prompt = try Drama3Prompt.build(text: "Hello there.", tokenizer: tokenizer)
        func logits(_ model: DramaModel) throws -> [Float] {
            try GenerationSession(model: model, prompt: prompt).currentCandidateLogits
        }
        func argmax(_ l: [Float]) -> Int { l.indices.max { l[$0] < l[$1] }! }
        let full: [Float]
        do {
            let model = try DramaModel(location: loc, capacity: 2048, schedule: .default)
            XCTAssertEqual(model.mode, .full)
            XCTAssertNil(model.quantizedCache)
            full = try logits(model)
        }
        let url = QuantizedWeightFile.url(snapshot: loc.directory.lastPathComponent, mode: .balanced, directory: dir)
        let firstLogits: [Float]
        do {
            let first = try DramaModel(location: loc, mode: .balanced, capacity: 2048, schedule: .default, quantizedDirectory: dir)
            let info = try XCTUnwrap(first.quantizedCache)
            XCTAssertEqual(info.url, url)
            XCTAssertNotNil(info.buildSeconds)
            XCTAssertEqual(info.bytes, Int((try FileManager.default.attributesOfItem(atPath: url.path))[.size] as! UInt64))
            XCTAssertEqual(first.tuningKey.mode, "balanced")
            firstLogits = try logits(first)
            print("balanced cache: \(info.bytes) bytes built in \(info.buildSeconds!) s")
        }
        let second = try DramaModel(location: loc, mode: .balanced, capacity: 2048, schedule: .default, quantizedDirectory: dir)
        let hit = try XCTUnwrap(second.quantizedCache)
        XCTAssertNil(hit.buildSeconds)
        XCTAssertEqual(hit.url, url)
        let secondLogits = try logits(second)
        XCTAssertEqual(firstLogits, secondLogits, "the built file and the opened file must hold the same weights")
        XCTAssertEqual(argmax(firstLogits), argmax(full))
        print("balanced against full: candidate logits relative L2 \(Autotune.relL2(full, firstLogits)), argmax \(argmax(full))")
    }
}
