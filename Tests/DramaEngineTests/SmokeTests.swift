import XCTest
@testable import DramaEngine

final class SmokeTests: XCTestCase {
    func testLibraryLoads() throws {
        let ctx = try MetalContext()
        XCTAssertNoThrow(try ctx.pipeline("residual_add"))
    }
}
