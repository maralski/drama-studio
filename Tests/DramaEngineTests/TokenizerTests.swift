import DramaEngine
import Foundation
import XCTest

final class TokenizerTests: XCTestCase {
    private struct Case: Decodable {
        let text: String
        let ids: [Int32]
    }

    private static let tokenizerURL = FileManager.default.homeDirectoryForCurrentUser.appending(
        path: ".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/tokenizer.json"
    )

    private static let fixtureURL = URL(filePath: #filePath)
        .deletingLastPathComponent()
        .appending(path: "Fixtures/tokenizer_cases.json")

    nonisolated(unsafe) private static var cached: Tokenizer?

    private func tokenizer() throws -> Tokenizer {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.tokenizerURL.path), "drama-3 tokenizer not in HF cache")
        if let cached = Self.cached { return cached }
        let loaded = try Tokenizer(url: Self.tokenizerURL)
        Self.cached = loaded
        return loaded
    }

    private func cases() throws -> [Case] {
        try JSONDecoder().decode([Case].self, from: Data(contentsOf: Self.fixtureURL))
    }

    func testMatchesHuggingFaceOnCorpus() throws {
        let tokenizer = try tokenizer()
        let cases = try cases()
        XCTAssertGreaterThanOrEqual(cases.count, 300)
        for testCase in cases {
            XCTAssertEqual(tokenizer.encode(testCase.text), testCase.ids, "text: \(testCase.text.debugDescription)")
        }
    }

    func testDecodeRoundTripsNormalizedText() throws {
        let tokenizer = try tokenizer()
        // Swift's String == is canonical equivalence, so compare bytes to catch NFC rewrites.
        let normalized = try cases().map(\.text).filter {
            Array($0.utf8) == Array($0.precomposedStringWithCanonicalMapping.utf8)
        }
        XCTAssertGreaterThan(normalized.count, 300)
        for text in normalized {
            XCTAssertEqual(Array(tokenizer.decode(tokenizer.encode(text)).utf8), Array(text.utf8), "text: \(text.debugDescription)")
        }
    }

    func testSpecialTokenIds() throws {
        let tokenizer = try tokenizer()
        let expected: [String: Int32] = [
            "<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "<|pad|>": 248077,
            "<|voice|>": 248081, "<|speaker:0|>": 248083, "<|semantic:0|>": 248099, "<|semantic:4095|>": 252194,
        ]
        for (token, id) in expected {
            XCTAssertEqual(tokenizer.id(forToken: token), id, token)
            XCTAssertEqual(tokenizer.encode(token), [id], token)
        }
        XCTAssertEqual(tokenizer.vocabularySize, 252216)
    }

    func testLoadTime() throws {
        _ = try tokenizer()
        let clock = ContinuousClock()
        var tokenizer: Tokenizer?
        let elapsed = try clock.measure { tokenizer = try Tokenizer(url: Self.tokenizerURL) }
        XCTAssertNotNil(tokenizer)
        print("tokenizer load: \(elapsed)")
    }
}
