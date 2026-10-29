// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class SmokeTests: XCTestCase {
    func testLibraryLoads() throws {
        let ctx = try MetalContext()
        XCTAssertNoThrow(try ctx.pipeline("residual_add"))
    }
}
