import Metal
import XCTest
@testable import DramaEngine

/// The kernel schedule's domains: every value has a pipeline, out-of-domain
/// values are rejected, and the variants agree with the default.
final class ScheduleTests: XCTestCase {
    func testEveryDomainValueHasAPipeline() throws {
        let ctx = try MetalContext()
        var names: [String] = ["attention", "quantize_bf16_i8", "quantize_bf16_i4", "quantize_fp8_i4"]
        for format in WeightFormat.allCases {
            names += KernelSchedule.gemvTokenTiles.map { "gemv_\(format.kernelTag)_t\($0)" }
            names += KernelSchedule.gemmTiles.map { "gemm_\(format.kernelTag)_\($0.label)" }
        }
        for format in ExpertFormat.allCases {
            names += ["moe_gate_up_\(format.kernelTag)", "moe_down_\(format.kernelTag)"]
            names += KernelSchedule.moeTiles.flatMap { ["moe_gate_up_\(format.kernelTag)_tiled_\($0.label)", "moe_down_\(format.kernelTag)_tiled_\($0.label)"] }
        }
        names += KernelSchedule.gdnParts.map { "gdn_recurrent_p\($0)" }
        for name in names {
            XCTAssertNoThrow(try ctx.pipeline(name), name)
        }
        XCTAssertEqual(KernelSchedule.moeTiles.count, 10)
        let threads = try ctx.pipeline("attention").maxTotalThreadsPerThreadgroup
        XCTAssertGreaterThanOrEqual(threads, 32 * KernelSchedule.attentionSplits.max()!)
        for sg in KernelSchedule.moeRowSimdgroups {
            XCTAssertLessThanOrEqual(32 * sg, try ctx.pipeline("moe_gate_up_fp8").maxTotalThreadsPerThreadgroup, "simdgroups \(sg)")
        }
    }

    func testValidateRejectsOutOfDomainValues() throws {
        XCTAssertNoThrow(try KernelSchedule.default.validate())
        let broken: [(String, (inout KernelSchedule) -> Void)] = [
            ("gemvDecode.tokenTile 12", { $0.gemvDecode.tokenTile = 12 }),
            ("gemvNarrow.simdgroups 3", { $0.gemvNarrow.simdgroups = 3 }),
            ("gemmTile 48x32x16", { $0.gemmTile = .init(bm: 48, bn: 32, bk: 16) }),
            ("moeTile 64x64x16", { $0.moeTile = .init(bn: 64, rows: 64, bk: 16) }),
            ("moeRowSimdgroups 3", { $0.moeRowSimdgroups = 3 }),
            ("attentionSplitDecode 5", { $0.attentionSplitDecode = 5 }),
            ("attentionSplitPrefill 2", { $0.attentionSplitPrefill = 2 }),
            ("gdnPartsDecode 3", { $0.gdnPartsDecode = 3 }),
            ("gdnPartsPrefill 16", { $0.gdnPartsPrefill = 16 }),
        ]
        for (name, mutate) in broken {
            var s = KernelSchedule.default
            mutate(&s)
            XCTAssertThrowsError(try s.validate(), name)
        }
    }

    /// Every GEMV shape forms the same products in the same order per token, so
    /// the whole domain must match the default bit for bit, at M = 1 and above,
    /// in every weight format.
    func testGemvVariantsBitIdentical() throws {
        var f = try KernelParityTests.Fixture()
        for format in WeightFormat.allCases {
            for (n, kDim, m) in [(256, 2048, 128), (8192, 2048, 44), (2048, 4096, 5), (32, 2048, 100), (1, 2048, 128), (512, 2048, 1), (1, 2048, 1), (2048, 512, 7)] {
                let w = try f.matrix(format, rows: n, cols: kDim)
                let x = f.floats(m * kDim, label: "x")
                let y = Buf(f.ctx.device, floats: m * n, label: "y")
                var reference: [Float]?
                for shape in KernelSchedule.gemvShapes {
                    var s = KernelSchedule.default
                    s.gemvDecode = shape
                    s.gemvNarrow = shape
                    var k = Kernels(ctx: f.ctx, precision: .exact, schedule: s)
                    k.forceGemv = true
                    y.zero()
                    try f.run { enc in try k.gemv(enc, w, x: x, y: y, m: m) }
                    let got = y.read(m * n)
                    if let reference {
                        XCTAssertEqual(got, reference, "\(format) n=\(n) k=\(kDim) m=\(m) \(shape.label)")
                    } else {
                        XCTAssertTrue(got.allSatisfy { $0.isFinite })
                        reference = got
                    }
                }
            }
        }
    }

    /// Splits 4 and 16 differ from 8 by the order the partial softmaxes combine.
    func testAttentionSplitsAgree() throws {
        var f = try KernelParityTests.Fixture()
        let heads = 16, kvHeads = 2, keys = 300
        let kcache = f.floats(keys * kvHeads * 256, label: "k"), vcache = f.floats(keys * kvHeads * 256, label: "v")
        for rows in [1, 16] {
            let pos0 = keys - rows
            let q = f.floats(rows * heads * 256, label: "q")
            let out = Buf(f.ctx.device, floats: rows * heads * 256, label: "out")
            var reference: [Float]?
            for split in [8] + KernelSchedule.attentionSplits.filter({ $0 != 8 }) {
                var s = KernelSchedule.default
                s.attentionSplitDecode = split
                s.attentionSplitPrefill = split
                let k = Kernels(ctx: f.ctx, precision: .exact, schedule: s)
                out.zero()
                try f.run { enc in try k.attention(enc, out: out, q: q, kcache: kcache, vcache: vcache, heads: heads, kvHeads: kvHeads, rows: rows, pos0: pos0) }
                let got = out.read(out.count)
                XCTAssertTrue(got.allSatisfy { $0.isFinite }, "rows \(rows) split \(split)")
                if let reference {
                    let err = KernelParityTests.relL2(reference, got)
                    XCTAssertLessThan(err, 1e-5, "rows \(rows) split \(split)")
                    print("attention rows \(rows) split \(split): \(err)")
                } else { reference = got }
            }
        }
    }
}
