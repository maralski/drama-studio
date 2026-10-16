import XCTest
@testable import DramaEngine

final class PromptDirectingTests: XCTestCase {
    static let captain = VoiceDesign(name: "Captain", description: "gravelly old male voice, about seventy, weathered and slow, salt in every word",
                                     scene: "a storm-battered tavern at the edge of the harbor")
    static let captainSystem = """
        Scene: a storm-battered tavern at the edge of the harbor
        Characters:
        - Captain: gravelly old male voice, about seventy, weathered and slow, salt in every word
        Voice every line of the script in order, with <|audio_separator|> between consecutive lines.
        """

    static func codes(_ frames: Int, _ seed: Int32 = 1) -> [[Int32]] {
        (0..<10).map { book in (0..<frames).map { Int32($0) + seed + Int32(book) } }
    }

    func testSystemPromptListsSceneThenCharacterThenInstruction() {
        XCTAssertEqual(Self.captain.systemPrompt, Self.captainSystem)
        let owl = VoiceDesign(name: " Night Owl ", description: "smooth warm baritone, a late-night radio host, relaxed and intimate, close to the mic", scene: "  ")
        XCTAssertEqual(owl.systemPrompt, """
            Characters:
            - Night Owl: smooth warm baritone, a late-night radio host, relaxed and intimate, close to the mic
            Voice every line of the script in order, with <|audio_separator|> between consecutive lines.
            """)
    }

    func testDesignLayoutSpeaksEveryTurnByNameWithoutReferences() throws {
        let rendered = try Drama3Prompt.render(text: "Pray ye never find out.", character: Self.captain,
                                               history: [Take(text: "Ahoy there.", codes: Self.codes(2))])
        XCTAssertEqual(rendered, "<|im_start|>system\n" + Self.captainSystem + "<|im_end|>\n"
            + "<|im_start|>user\nCaptain: Ahoy there.<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:2><|im_end|>\n"
            + "<|im_start|>user\nCaptain: Pray ye never find out.<|im_end|>\n<|im_start|>assistant\n<|voice|>")
    }

    func testCharacterAndReferencesAreMutuallyExclusive() {
        let reference = Drama3Prompt.Reference(transcript: "A clip.", codes: Self.codes(2))
        XCTAssertThrowsError(try Drama3Prompt.render(text: "Hi.", references: [reference], character: Self.captain)) { XCTAssertEqual($0 as? PromptError, .voiceConflict) }
        XCTAssertThrowsError(try GenerationPlan(text: "Hi.", references: [reference], character: Self.captain, config: GenerationConfig())) { XCTAssertEqual($0 as? PromptError, .voiceConflict) }
        XCTAssertThrowsError(try Drama3Prompt.render(text: "Hi.", character: VoiceDesign(name: "X", description: "loud <|audio_separator|> voice")))
        XCTAssertThrowsError(try Drama3Prompt.render(text: "Hi.", character: VoiceDesign(name: " ", description: "a voice")))
    }

    func testDesignRowsCarryTheSeparatorTokenAndOpenTheSystemMessage() throws {
        guard let loc = ModelLocation.huggingFaceCache() else { throw XCTSkip("drama-3 not in the Hugging Face cache") }
        let tokenizer = try Tokenizer(url: loc.directory.appendingPathComponent("tokenizer.json"))
        let rows = try Drama3Prompt.build(text: "Pray ye never find out.", tokenizer: tokenizer, character: Self.captain)
        let ids = rows.map { $0[0] }
        let opening = tokenizer.encode("<|im_start|>system\n")
        XCTAssertEqual(Array(ids.prefix(opening.count)), opening)
        XCTAssertEqual(ids.filter { $0 == 252_196 }.count, 1)
        XCTAssertEqual(tokenizer.encode(Drama3Prompt.audioSeparator), [252_196])
        XCTAssertFalse(ids.contains(tokenizer.encode("<|speaker:0|>")[0]), "design turns carry no speaker tag")
        XCTAssertTrue(rows.allSatisfy { $0.dropFirst().allSatisfy { $0 == 0 } }, "no code rows without references")
    }

    func testRetakeTextCuesEverySentenceWithTheNote() {
        XCTAssertEqual(DirectorsNote.retakeText(line: "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out.", note: "softer and warmer"),
                       "Direction: softer and warmer\nLine: [softer and warmer] Ye want to know what's out past the reef? [softer and warmer] Pray ye never find out.")
        XCTAssertEqual(DirectorsNote.retakeText(line: "<|speaker:0|><whisper>Don't move.</whisper> They can hear us [gasps] right now!", note: " more [scared] <now> |  please\n"),
                       "Direction: more scared now please\nLine: [more scared now please] Don't move. [more scared now please] They can hear us right now!")
        XCTAssertEqual(DirectorsNote.retakeText(line: "One line, no end", note: "faster"), "Direction: faster\nLine: [faster] One line, no end")
    }

    func testNotesAreCleanedAndCapped() {
        XCTAssertEqual(DirectorsNote.clean(String(repeating: "a", count: 200))?.count, 120)
        XCTAssertNil(DirectorsNote.clean(" [] <> | "))
        XCTAssertEqual(DirectorsNote.clean("more amused than before; drop the hurt"), "more amused than before; drop the hurt")
    }

    func testDesignedRetakeUsesTheFirstTakeAsVoiceAndTheLastTwoAsHistory() throws {
        let line = "[low and menacing] Ye want to know? Pray."
        let first = Take(text: "Captain: [low and menacing] Ye want to know? Pray.", codes: Self.codes(3))
        let second = Take(text: "Captain: Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray.", codes: Self.codes(4))
        let third = Take(text: "Captain: Direction: faster\nLine: [faster] Ye want to know? [faster] Pray.", codes: Self.codes(5))
        let plan = try GenerationPlan(text: line, references: [], character: Self.captain, previousTakes: [first, second, third], note: "softer and warmer", config: GenerationConfig())
        let turn = "Captain: Direction: softer and warmer\nLine: [softer and warmer] Ye want to know? [softer and warmer] Pray."
        XCTAssertEqual(plan, try GenerationPlan(text: line, references: [], character: Self.captain, previousTakes: [first, second, third], note: "softer and warmer", config: GenerationConfig()))
        XCTAssertNil(plan.character)
        XCTAssertEqual(plan.references, [Drama3Prompt.Reference(transcript: "Ye want to know? Pray.", codes: first.codes)])
        XCTAssertEqual(plan.history, [second, third])
        XCTAssertEqual(plan.chunks, [GenerationPlan.Chunk(turn: turn, line: line)])
        XCTAssertEqual(plan.turnText, turn)
        XCTAssertEqual(try Drama3Prompt.render(text: plan.chunks[0].turn, references: plan.references, history: plan.history),
                       "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\n[Reference 1]\nText: <|speaker:0|> Ye want to know? Pray.\nSpeech: <codes:3><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + second.text + "<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:4><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + third.text + "<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:5><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + turn + "<|im_end|>\n<|im_start|>assistant\n<|voice|>")
    }

    func testFirstDesignTakeUsesTheDesignLayoutAndRecordsTheNamedTurn() throws {
        let plan = try GenerationPlan(text: "Ahoy. Who goes there in the storm tonight?", references: [], character: Self.captain, config: GenerationConfig())
        XCTAssertEqual(plan.character, Self.captain)
        XCTAssertTrue(plan.references.isEmpty)
        XCTAssertEqual(plan.chunks.map(\.turn), ["Ahoy. Who goes there in the storm tonight?"], "the builder writes the name")
        XCTAssertEqual(plan.turnText, "Captain: Ahoy. Who goes there in the storm tonight?")
    }

    func testDefaultVoiceRetakeClonesTheFirstTake() throws {
        let first = Take(text: "It was a dark and stormy night.", codes: Self.codes(3))
        let plan = try GenerationPlan(text: "It was a dark and stormy night.", references: [], previousTakes: [first], note: "whispered", config: GenerationConfig())
        XCTAssertEqual(plan.references, [Drama3Prompt.Reference(transcript: "It was a dark and stormy night.", codes: first.codes)])
        XCTAssertEqual(plan.history, [first])
        XCTAssertEqual(plan.turnText, "Direction: whispered\nLine: [whispered] It was a dark and stormy night.")
    }

    func testClonedRetakeKeepsTheClipAndSplitsNothing() throws {
        let clip = Drama3Prompt.Reference(transcript: "My own voice.", codes: Self.codes(6))
        let line = "The first sentence is long enough to stand. The second one is long enough too."
        let take = Take(text: line, codes: Self.codes(4))
        let retake = try GenerationPlan(text: line, references: [clip], previousTakes: [take], note: "slower, with more pauses", config: GenerationConfig())
        XCTAssertEqual(retake.references, [clip])
        XCTAssertEqual(retake.chunks.count, 1)
        XCTAssertEqual(retake.history, [take])
        let plain = try GenerationPlan(text: line, references: [clip], config: GenerationConfig())
        XCTAssertEqual(plain.chunks.count, 2, "without a note the line is cut into sentences")
        XCTAssertEqual(plain.turnText, line)
        XCTAssertEqual(try GenerationPlan(text: line, references: [clip], note: " | ", config: GenerationConfig()), plain, "an empty note is no note")
        XCTAssertThrowsError(try GenerationPlan(text: "[laughs]", references: [clip], note: "angrier", config: GenerationConfig()))
    }

    func testSpokenTextDropsNameAndDirection() {
        XCTAssertEqual(DirectorsNote.spokenText(ofTurn: "Captain: Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray.", character: Self.captain), "Ye want to know? Pray.")
        XCTAssertEqual(DirectorsNote.spokenText(ofTurn: "<|speaker:0|> Hello <whisper>there</whisper> [laughs] .", character: nil), "Hello there.")
    }
}
