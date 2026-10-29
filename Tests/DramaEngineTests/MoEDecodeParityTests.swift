// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import DramaEngine

/// The three-dispatch decode MoE layer against the twelve dispatches of the
/// M > 1 path at one row, bit for bit, on a synthetic layer of the real
/// shapes, at both precisions. No model needed.
final class MoEDecodeParityTests: XCTestCase {
    func testDecodeLayerMatchesTwelveDispatches() throws {
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("config.json of drama-3 needed for the MoE shapes") }
        let config = try ModelConfig(url: loc.directory.appendingPathComponent("config.json"))
        for precision in [Precision.exact, Precision.reference] {
            let r = try MoEDecodeRig(ctx: try MetalContext(), config: config, precision: precision, copies: 1)
            let layer = r.layers[0]
            XCTAssertEqual(Set(layer.ids.readInts(8)).count, 8)
            let x0 = r.x.read(r.x.count)
            try MoEDecodeRig.run(r.ctx) { enc in try r.shippedLayer(enc, layer) }
            let want = r.x.read(r.x.count).map(\.bitPattern)
            XCTAssertNotEqual(want, x0.map(\.bitPattern))
            XCTAssertTrue(r.x.read(r.x.count).allSatisfy(\.isFinite))

            x0.withUnsafeBufferPointer { _ = memcpy(r.x.floats, $0.baseAddress!, x0.count * 4) }
            r.act.zero(); r.ids.zero(); r.rw.zero()
            try MoEDecodeRig.run(r.ctx) { enc in try r.decodeLayer(enc, layer) }
            XCTAssertEqual(r.x.read(r.x.count).map(\.bitPattern), want, "round \(precision.mimicBF16)")
            XCTAssertEqual(r.ids.readInts(8), layer.ids.readInts(8), "round \(precision.mimicBF16)")
            XCTAssertEqual(r.rw.read(8).map(\.bitPattern), layer.rw.read(8).map(\.bitPattern), "round \(precision.mimicBF16)")
        }
    }
}
