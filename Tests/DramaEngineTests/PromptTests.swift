import XCTest
@testable import DramaEngine

final class PromptTests: XCTestCase {
    struct Codes: Decodable { let transcript: String?; let text: String?; let codes: [[Int32]] }
    struct PromptCase: Decodable { let name: String; let text: String; let references: [Codes]; let history: [Codes]; let rows: [[Int32]]?; let error: String? }
    struct ChunkCase: Decodable { let name: String; let text: String; let chunks: [String]; let by_length: [String]; let segments: [[String]] }
    struct Fixture: Decodable { let prompts: [PromptCase]; let chunking: [ChunkCase] }

    static let fixture: Fixture = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/prompt_cases.json")
        return try! JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }()

    func testPromptsMatchReference() throws {
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("drama-3 not in the Hugging Face cache") }
        let tokenizer = try Tokenizer(url: loc.directory.appendingPathComponent("tokenizer.json"))
        for c in Self.fixture.prompts {
            let refs = c.references.map { Drama3Prompt.Reference(transcript: $0.transcript!, codes: $0.codes) }
            let history = c.history.map { Drama3Prompt.HistoryTurn(text: $0.text!, codes: $0.codes) }
            do {
                let rows = try Drama3Prompt.build(text: c.text, tokenizer: tokenizer, references: refs, history: history)
                XCTAssertNil(c.error, "\(c.name): reference raised \(c.error ?? "")")
                XCTAssertEqual(rows, c.rows ?? [], c.name)
            } catch {
                XCTAssertNotNil(c.error, "\(c.name): unexpected \(error)")
            }
        }
    }

    func testChunkingMatchesReference() {
        for c in Self.fixture.chunking {
            XCTAssertEqual(Drama3Chunking.split(c.text), c.chunks, c.name)
            XCTAssertEqual(Drama3Chunking.splitByLength(c.text), c.by_length, c.name)
            XCTAssertEqual(Drama3Chunking.parseSpeakerSegments(c.text).map { [$0.0, $0.1] }, c.segments, c.name)
        }
    }

    func testHistoryCandidates() {
        var h = Drama3History()
        XCTAssertEqual(h.candidates.map { $0.count }, [0])
        h.push(text: "a", codes: [[1]])
        XCTAssertEqual(h.candidates.map { $0.map(\.text) }, [["a"], []])
        h.push(text: "b", codes: [[1]])
        h.push(text: "c", codes: [[1]])
        XCTAssertEqual(h.candidates.map { $0.map(\.text) }, [["a", "c"], ["a"], []])
    }

    func testSamplerMatchesReferenceRules() {
        var s = Sampler(seed: 7)
        // top_k 1 is greedy whatever the noise.
        let logits: [Float] = [0.1, 3.0, 0.2, 2.9]
        for _ in 0..<20 { XCTAssertEqual(s.sample(logits, temperature: 0.7, topK: 1, topP: 0.7), 1) }
        // top_p tiny keeps only the best token.
        for _ in 0..<20 { XCTAssertEqual(s.sample(logits, temperature: 1.0, topK: 30, topP: 0.01), 1) }
        // top_k 2 only ever returns the two best.
        var seen = Set<Int>()
        for _ in 0..<200 { seen.insert(s.sample(logits, temperature: 1.0, topK: 2, topP: 1.0)) }
        XCTAssertEqual(seen, [1, 3])
    }
}
