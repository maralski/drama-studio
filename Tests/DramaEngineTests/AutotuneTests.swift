import Metal
import XCTest
@testable import DramaEngine

final class AutotuneTests: XCTestCase {
    static let key = TuningKey(chip: "Apple M3 Max", os: "26.5.2", kernels: String(repeating: "ab", count: 32), model: "snap-0123456789abcdef", schema: 2)

    func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("drama-tuning-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func report(_ key: TuningKey, schedule: KernelSchedule = .default) -> TuningReport {
        TuningReport(key: key, created: Date(timeIntervalSince1970: 1_800_000_000), schedule: schedule, entries: [
            TuningEntryResult(name: "gemv.decode", defaultLabel: "t32x4sg", chosenLabel: "t16x8sg", defaultMs: 0.31, chosenMs: 0.27, gainPercent: 12.9, relL2: 0,
                              rounds: 5, rejected: [.init(label: "t64x1sg", reason: "relative L2 2.00e-01 above 1e-05", relL2: 0.2)]),
        ], wallSeconds: 12.5)
    }

    func testFileNameFollowsEveryKeyField() {
        let base = Self.key
        XCTAssertEqual(base.fileName, Self.key.fileName)
        XCTAssertEqual(base.fileName.count, 16 + ".json".count)
        var variants: [TuningKey] = []
        var k = base; k.chip = "Apple M4 Max"; variants.append(k)
        k = base; k.os = "26.5.3"; variants.append(k)
        k = base; k.kernels = String(repeating: "ba", count: 32); variants.append(k)
        k = base; k.model = "snap-fedcba9876543210"; variants.append(k)
        k = base; k.schema = 3; variants.append(k)
        for v in variants { XCTAssertNotEqual(v.fileName, base.fileName, "\(v)") }
        XCTAssertEqual(Set(variants.map(\.fileName)).count, variants.count)
    }

    func testSavedReportLoadsBackEqual() throws {
        let dir = try tempDirectory()
        var schedule = KernelSchedule.default
        schedule.gemvDecode = .init(tokenTile: 16, simdgroups: 8)
        schedule.moeTile = .init(bn: 64, rows: 32, bk: 32)
        let saved = report(Self.key, schedule: schedule)
        let url = try TuningCache.save(saved, directory: dir)
        XCTAssertEqual(url.lastPathComponent, Self.key.fileName)
        let hit = try XCTUnwrap(TuningCache.load(key: Self.key, directory: dir))
        XCTAssertEqual(hit.report, saved)
        XCTAssertEqual(hit.url, url)
        XCTAssertEqual(hit.report.schedule.gemvDecode.tokenTile, 16)
    }

    func testMissingFileIsAMiss() throws {
        XCTAssertNil(TuningCache.load(key: Self.key, directory: try tempDirectory()))
    }

    func testFileWithOtherKeyFieldsIsAMiss() throws {
        let dir = try tempDirectory()
        var other = Self.key
        other.os = "26.5.3"
        let data = try JSONEncoder().encode(report(other))
        try data.write(to: TuningCache.url(for: Self.key, directory: dir))
        XCTAssertNil(TuningCache.load(key: Self.key, directory: dir))
    }

    func testFileWithInvalidScheduleIsAMiss() throws {
        let dir = try tempDirectory()
        var schedule = KernelSchedule.default
        schedule.attentionSplitDecode = 5
        try TuningCache.save(report(Self.key, schedule: schedule), directory: dir)
        XCTAssertNil(TuningCache.load(key: Self.key, directory: dir))
    }

    /// A candidate that doubles its output, one that leaves NaN behind and one
    /// whose kernel is missing are all rejected and never chosen.
    func testBrokenCandidatesAreRejected() throws {
        let ctx = try MetalContext()
        let n = 512, kDim = 2048
        let w = KernelBench.bf16Weights(ctx.device, rows: n, cols: kDim)
        let x = Buf(ctx.device, floats: kDim, label: "x")
        var rng = SeededGenerator(seed: 9)
        for i in 0..<kDim { x.floats[i] = Float(rng.uniformOpenZero()) * 2 - 1 }
        let y = Buf(ctx.device, floats: n, label: "y")
        let candidates = [
            Autotune.Candidate(label: "default") { _ in },
            Autotune.Candidate(label: "doubled") { $0.gemvDecode.simdgroups = 8 },
            Autotune.Candidate(label: "nan") { $0.gemvDecode.tokenTile = 64 },
            Autotune.Candidate(label: "missing") { $0.gemvDecode.tokenTile = 16 },
            Autotune.Candidate(label: "same") { $0.gemvDecode.tokenTile = 8 },
        ]
        let entry = Autotune.Entry(
            name: "gemv.synthetic", candidates: candidates, layers: 3,
            encode: { enc, k, _ in
                let shape = k.schedule.gemvDecode
                if shape.tokenTile == 16 { _ = try k.ctx.pipeline("no_such_kernel") }
                if shape.tokenTile == 64 { return }
                try k.gemv(enc, w, x: x, y: y, m: 1)
                if shape.simdgroups == 8 { try k.residualAdd(enc, x: y, y: y, count: n) }
            },
            outputs: { y.read(n) },
            seedInputs: { for i in 0..<n { y.floats[i] = .nan } })
        var options = Autotune.Options()
        options.rounds = 2
        options.minGainPercent = 0
        let report = try Autotune.run(entries: [entry], ctx: ctx, precision: .exact, key: Self.key,
                                      makeCommandBuffer: { let cb = ctx.queue.makeCommandBuffer()!; return (cb, cb.makeComputeCommandEncoder()!) }, options: options)
        let result = try XCTUnwrap(report.entries.first)
        XCTAssertEqual(Set(result.rejected.map(\.label)), ["doubled", "nan", "missing"])
        let doubled = try XCTUnwrap(result.rejected.first { $0.label == "doubled" })
        XCTAssertEqual(try XCTUnwrap(doubled.relL2), 1, accuracy: 1e-4)
        XCTAssertEqual(try XCTUnwrap(result.rejected.first { $0.label == "nan" }).reason, "non-finite output")
        XCTAssertTrue(try XCTUnwrap(result.rejected.first { $0.label == "missing" }).reason.contains("no_such_kernel"))
        XCTAssertTrue(["default", "same"].contains(result.chosenLabel))
        XCTAssertEqual(result.rounds, 2)
        XCTAssertEqual(result.relL2, 0)
        XCTAssertNotEqual(report.schedule.gemvDecode.simdgroups, 8)
    }

    /// A schedule with every field tuned away from the default.
    static var allTuned: KernelSchedule {
        var s = KernelSchedule.default
        s.gemvDecode = .init(tokenTile: 1, simdgroups: 1)
        s.gemvNarrow = .init(tokenTile: 1, simdgroups: 8)
        s.gemmTile = .init(bm: 64, bn: 64, bk: 16)
        s.moeTile = .init(bn: 64, rows: 16, bk: 16)
        s.attentionSplitDecode = 16
        s.attentionSplitPrefill = 4
        s.gdnPartsDecode = 8
        s.gdnPartsPrefill = 2
        return s
    }

    func testLosingPrefillCheckRevertsOnlyThePrefillFields() {
        let tuned = Self.allTuned
        let lost = ForwardCheck(rows: 128, defaultMs: 222, tunedMs: 233, kept: false)
        let (s, notes) = Autotune.applyForwardChecks(schedule: tuned, defaults: .default, checks: [lost])
        XCTAssertEqual(s.gemvNarrow, KernelSchedule.default.gemvNarrow)
        XCTAssertEqual(s.gemmTile, KernelSchedule.default.gemmTile)
        XCTAssertEqual(s.moeTile, KernelSchedule.default.moeTile)
        XCTAssertEqual(s.attentionSplitPrefill, KernelSchedule.default.attentionSplitPrefill)
        XCTAssertEqual(s.gdnPartsPrefill, KernelSchedule.default.gdnPartsPrefill)
        XCTAssertEqual(s.gemvDecode, tuned.gemvDecode)
        XCTAssertEqual(s.attentionSplitDecode, tuned.attentionSplitDecode)
        XCTAssertEqual(s.gdnPartsDecode, tuned.gdnPartsDecode)
        XCTAssertEqual(Set(notes.keys), ["gemv.narrow", "gemm.prefill", "moe.tiled", "attention.prefill", "gdn.prefill"])
        XCTAssertEqual(Set(notes.values), [Autotune.revertedNote])
    }

    func testLosingDecodeCheckRevertsOnlyTheDecodeFields() {
        let tuned = Self.allTuned
        let lost = ForwardCheck(rows: 1, defaultMs: 17, tunedMs: 18, kept: false)
        let (s, notes) = Autotune.applyForwardChecks(schedule: tuned, defaults: .default, checks: [lost])
        XCTAssertEqual(s.gemvDecode, KernelSchedule.default.gemvDecode)
        XCTAssertEqual(s.attentionSplitDecode, KernelSchedule.default.attentionSplitDecode)
        XCTAssertEqual(s.gdnPartsDecode, KernelSchedule.default.gdnPartsDecode)
        XCTAssertEqual(s.gemvNarrow, tuned.gemvNarrow)
        XCTAssertEqual(s.gemmTile, tuned.gemmTile)
        XCTAssertEqual(s.moeTile, tuned.moeTile)
        XCTAssertEqual(s.attentionSplitPrefill, tuned.attentionSplitPrefill)
        XCTAssertEqual(s.gdnPartsPrefill, tuned.gdnPartsPrefill)
        XCTAssertEqual(Set(notes.keys), ["gemv.decode", "attention.decode", "gdn.decode"])
    }

    /// A kept check changes nothing, and a reverted field that already held
    /// the default gets no note.
    func testKeptCheckAndDefaultFieldsLeaveNoTrace() {
        let kept = ForwardCheck(rows: 128, defaultMs: 222, tunedMs: 214, kept: true)
        let (s, notes) = Autotune.applyForwardChecks(schedule: Self.allTuned, defaults: .default, checks: [kept])
        XCTAssertEqual(s, Self.allTuned)
        XCTAssertTrue(notes.isEmpty)
        var partly = KernelSchedule.default
        partly.moeTile = .init(bn: 64, rows: 16, bk: 16)
        let lost = ForwardCheck(rows: 128, defaultMs: 222, tunedMs: 233, kept: false)
        let (s2, notes2) = Autotune.applyForwardChecks(schedule: partly, defaults: .default, checks: [lost])
        XCTAssertEqual(s2, .default)
        XCTAssertEqual(Array(notes2.keys), ["moe.tiled"])
    }

    func testSavedForwardChecksLoadBack() throws {
        let dir = try tempDirectory()
        var saved = report(Self.key)
        saved.forwardChecks = [ForwardCheck(rows: 128, defaultMs: 221.9, tunedMs: 233.1, kept: false), ForwardCheck(rows: 1, defaultMs: 17.1, tunedMs: 15.9, kept: true)]
        saved.entries[0].note = Autotune.revertedNote
        try TuningCache.save(saved, directory: dir)
        let hit = try XCTUnwrap(TuningCache.load(key: Self.key, directory: dir))
        XCTAssertEqual(hit.report, saved)
    }

    func testOlderReportVersionIsAMiss() throws {
        let dir = try tempDirectory()
        var old = report(Self.key)
        old.version = 1
        try TuningCache.save(old, directory: dir)
        XCTAssertNil(TuningCache.load(key: Self.key, directory: dir))
    }

    /// The autotuner's own validation over the full domain of every census
    /// entry on the real weights; nothing is timed.
    func testCensusValidatesEveryCandidate() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 weights needed") }
        let model = try DramaModel(location: loc, schedule: .default)
        let entries = try Autotune.census(model: model)
        XCTAssertEqual(entries.map(\.name), ["gemv.decode", "gemv.narrow", "gemm.prefill", "moe.tiled", "attention.decode", "attention.prefill", "gdn.decode", "gdn.prefill"])
        XCTAssertEqual(Set(entries.map(\.name)), Set(Autotune.fields.map(\.entry)))
        // The MoE workload routes real hidden states, not the noise pool.
        let H = model.config.hiddenSize
        let moeInput = try model.probeMoEInput(rows: 300)
        let tiled = try XCTUnwrap(entries.first { $0.name == "moe.tiled" })
        tiled.seedInputs()
        XCTAssertEqual(model.backbone.h.read(H), Array(moeInput.prefix(H)))
        for entry in entries {
            let (accepted, rejected) = try Autotune.validate(entry, ctx: model.ctx, precision: model.precision, makeCommandBuffer: model.commandBuffer, tolerance: 1e-5)
            XCTAssertTrue(rejected.isEmpty, "\(entry.name): \(rejected)")
            XCTAssertEqual(accepted.count, entry.candidates.count, entry.name)
            let worst = accepted.map(\.relL2).max() ?? 0
            print("census \(entry.name): \(accepted.count) candidates, worst relative L2 \(worst)")
        }
    }
}
