import Metal
import XCTest
@testable import DramaEngine

/// Every fused decode kernel must reproduce the separate dispatches it
/// replaces bit for bit, and a concurrent encoder with the engine's barriers
/// must reproduce the serial one.
final class FusionTests: XCTestCase {
    func run(_ ctx: MetalContext, concurrent: Bool, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = concurrent ? cb.makeComputeCommandEncoder(dispatchType: .concurrent)! : cb.makeComputeCommandEncoder()!
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    func testFusedBlocksMatchSeparateDispatchesBitForBit() throws {
        let ctx = try MetalContext()
        for block in FusionReplica(ctx: ctx).allBlocks() {
            block.reset()
            try run(ctx, concurrent: false) { try block.encode($0, fused: false) }
            let reference = block.outputBits()
            XCTAssertTrue(reference.allSatisfy { $0.contains { $0 != 0 } }, "\(block.name): an output stayed zero")
            for (concurrent, fused) in [(false, true), (true, false), (true, true)] {
                block.reset()
                try run(ctx, concurrent: concurrent) { try block.encode($0, fused: fused) }
                let got = block.outputBits()
                for (i, (a, b)) in zip(reference, got).enumerated() {
                    let diff = zip(a, b).filter { $0 != $1 }.count
                    XCTAssertEqual(diff, 0, "\(block.name) output \(i), fused \(fused), concurrent \(concurrent): \(diff) of \(a.count) differ")
                }
            }
        }
    }

    /// The tally sees every dispatch, and a concurrent encoder gets a barrier
    /// before each one except the later members of a parallel group.
    func testParallelGroupsSkipBarriers() throws {
        let ctx = try MetalContext()
        var k = Kernels(ctx: ctx, precision: .exact)
        let tally = DispatchTally()
        k.tally = tally
        let x = Buf(ctx.device, floats: 256, label: "x"), y = Buf(ctx.device, floats: 256, label: "y")
        try run(ctx, concurrent: true) { enc in
            try k.residualAdd(enc, x: x, y: y, count: 256)
            try k.parallel(enc) {
                try k.zero(enc, y)
                try k.zero(enc, Buf(ctx.device, floats: 256, label: "z"))
            }
            try k.residualAdd(enc, x: x, y: y, count: 256)
        }
        XCTAssertEqual(tally.counts, ["residual_add": 2, "zero_u32": 2])
        XCTAssertEqual(k.fence.barriers, 3)
    }

    /// Dependent dispatches on a serial encoder place no barriers.
    func testSerialEncoderPlacesNoBarriers() throws {
        let ctx = try MetalContext()
        let k = Kernels(ctx: ctx, precision: .exact)
        let x = Buf(ctx.device, floats: 256, label: "x"), y = Buf(ctx.device, floats: 256, label: "y")
        try run(ctx, concurrent: false) { enc in
            try k.residualAdd(enc, x: x, y: y, count: 256)
            try k.residualAdd(enc, x: x, y: y, count: 256)
        }
        XCTAssertEqual(k.fence.barriers, 0)
    }
}
