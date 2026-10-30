// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
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

    func testSentenceChunking() {
        XCTAssertEqual(Drama3Chunking.splitSentences("The quick brown fox jumps over the lazy dog. She sells sea shells by the sea shore."),
                       ["The quick brown fox jumps over the lazy dog.", "She sells sea shells by the sea shore."])
        XCTAssertEqual(Drama3Chunking.splitSentences("Pi is 3.14159! Really? Yes."), ["Pi is 3.14159!", "Really?", "Yes."])
        XCTAssertEqual(Drama3Chunking.splitSentences("She paused [sighs. no] and <whisper>left. quietly</whisper>. Then ran."),
                       ["She paused [sighs. no] and <whisper>left. quietly</whisper>.", "Then ran."])
        XCTAssertEqual(Drama3Chunking.splitSentences("She paused [sighs. no] and <whisper>left. quietly</whisper>. Then she ran all the way home."),
                       ["She paused [sighs. no] and <whisper>left. quietly</whisper>.", "Then she ran all the way home."])
        XCTAssertEqual(Drama3Chunking.splitSentencesTagged("<|speaker:0|> Hi there, friend. How are you? <|speaker:1|> Fine thanks, and you?"),
                       ["<|speaker:0|> Hi there, friend.", "<|speaker:0|> How are you?", "<|speaker:1|> Fine thanks, and you?"])
        XCTAssertEqual(Drama3Chunking.splitSentences("   "), [])
    }

    func testShortPoemIntroductionKeepsTheFollowingSentenceSeparate() {
        let introduction = "Here is a poem for you."
        let opening = "The morning light begins to break, Across the quiet, misty lake."
        for separator in ["\n", " "] {
            XCTAssertEqual(Drama3Chunking.splitSentencesTagged(introduction + separator + opening),
                           [introduction, opening])
        }
    }

    func testSentenceBoundariesPreserveProtectedSpansAndSpeakerTags() {
        XCTAssertEqual(Drama3Chunking.splitSentencesTagged("First. <|speaker:2|> [pause. wait] <whisper>Stay. quiet.</whisper> Now. Pi is 3.14. Go."),
                       ["<|speaker:0|> First.", "<|speaker:2|> [pause. wait] <whisper>Stay. quiet.</whisper> Now.", "<|speaker:2|> Pi is 3.14.", "<|speaker:2|> Go."])
        XCTAssertEqual(Drama3Chunking.splitSentences("Hello <|special.token|> there. Bye."),
                       ["Hello <|special.token|> there.", "Bye."])
    }

    func testSentenceEndPunctuationStaysWithItsSentence() {
        XCTAssertEqual(Drama3Chunking.splitSentences("Wait... Then go."), ["Wait...", "Then go."])
        XCTAssertEqual(Drama3Chunking.splitSentences("Really?! Yes!"), ["Really?!", "Yes!"])
        XCTAssertEqual(Drama3Chunking.splitSentences("She said \"Stay.\" Then left."), ["She said \"Stay.\"", "Then left."])
        XCTAssertEqual(Drama3Chunking.splitSentences("Tail ends “here.” Next."), ["Tail ends “here.”", "Next."])
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
