import XCTest
@testable import DramaEngine

final class PrefixCacheTests: XCTestCase {
    /// A session that reuses the cached prefix states must produce the same
    /// logits as a fresh prefill of the whole prompt.
    func testCachedPrefixMatchesFreshPrefill() throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        let model = try DramaModel(location: loc, capacity: 2048)
        let tokenizer = try Tokenizer(url: loc.directory.appendingPathComponent("tokenizer.json"))
        var rng = SeededGenerator(seed: 11)
        let codes = (0..<10).map { c in (0..<150).map { _ in Int32(rng.next() % (c == 0 ? 4096 : 1024)) } }
        let reference = Drama3Prompt.Reference(transcript: "A made up reference clip.", codes: codes)
        let a = try Drama3Prompt.buildWithPrefix(text: "First sentence here.", tokenizer: tokenizer, references: [reference])
        let b = try Drama3Prompt.buildWithPrefix(text: "A different second sentence, longer than the first one.", tokenizer: tokenizer, references: [reference],
                                                 history: [.init(text: "First sentence here.", codes: codes.map { Array($0.prefix(20)) })])
        XCTAssertEqual(a.prefixRows, b.prefixRows)
        XCTAssertGreaterThan(a.prefixRows, 150)
        _ = try GenerationSession(model: model, prompt: a.rows, prefixRows: a.prefixRows)
        XCTAssertNotNil(model.prefixCache)
        let t0 = Date()
        let cached = try GenerationSession(model: model, prompt: b.rows, prefixRows: b.prefixRows)
        let cachedSeconds = Date().timeIntervalSince(t0)
        let cachedLogits = cached.currentCandidateLogits
        model.clearPrefixCache()
        let t1 = Date()
        let fresh = try GenerationSession(model: model, prompt: b.rows, prefixRows: 0)
        let freshSeconds = Date().timeIntervalSince(t1)
        let freshLogits = fresh.currentCandidateLogits
        let maxDiff = zip(cachedLogits, freshLogits).map { abs($0 - $1) }.max() ?? 0
        print("prefix cache: \(String(format: "%.2f", cachedSeconds))s cached vs \(String(format: "%.2f", freshSeconds))s fresh; max logit diff \(maxDiff)")
        XCTAssertEqual(maxDiff, 0, "cached prefix must reproduce the fresh prefill exactly")
        XCTAssertLessThan(cachedSeconds, freshSeconds)
    }
}
