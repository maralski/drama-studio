// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class PromptDirectingTests: XCTestCase {
    func testCustomDesignNamesStayOutOfFreshReusedAndDirectedPrompts() throws {
        for name in ["My narrator", "Dr. Élodie", "声の案内人", "A voice: with punctuation"] {
            let design = VoiceDesign(name: name, description: "calm, clear storyteller", scene: "a quiet room")
            let voice = Take(text: name + ": An earlier sample.", context: .drama3(codes: Self.codes(3)))
            let line = "First sentence. Second sentence."
            for (previous, note) in [([], nil), ([voice], nil), ([voice], "warmer")] as [([Take], String?)] {
                let plan = try GenerationPlan(text: line, references: [], character: design,
                                              previousTakes: previous, note: note, config: GenerationConfig())
                for chunk in plan.chunks {
                    let prompt = try Drama3Prompt.render(text: chunk.turn, references: plan.references,
                                                        character: plan.character, history: plan.history)
                    XCTAssertFalse(prompt.contains(name + ":"), name)
                    XCTAssertTrue(prompt.contains("<|im_start|>user\n<|speaker:0|>"), name)
                }
                if previous.isEmpty {
                    let prompt = try Drama3Prompt.render(text: plan.chunks[0].turn, character: plan.character)
                    XCTAssertTrue(prompt.contains("calm, clear storyteller"))
                    XCTAssertTrue(prompt.contains("Scene: a quiet room"))
                } else {
                    XCTAssertEqual(plan.references.map(\.transcript), ["An earlier sample."])
                    XCTAssertEqual(plan.history.map(\.text), ["An earlier sample."])
                }
                XCTAssertTrue(plan.turnText.hasPrefix(name + ": "), "Display names remain in project metadata")
            }
        }
    }

    func testDesignedVoiceUsesSpeakerTagInsteadOfSpeakingDisplayName() throws {
        let duke = VoiceDesign(name: "The Duke", description: "silky, theatrical villain with a rich low voice, amused and unhurried")
        let line = "Oh, you came all this way to stop me? [laughs] How delightfully brave of you."
        let prompt = try Drama3Prompt.render(text: line, character: duke)
        XCTAssertTrue(prompt.contains("Characters:\n- <|speaker:0|>: silky, theatrical villain"))
        XCTAssertTrue(prompt.contains("<|im_start|>user\n<|speaker:0|>: " + line))
        XCTAssertFalse(prompt.contains("The Duke:"), "A display name must not become spoken input")
    }

    func testDesignedVoiceReferenceDoesNotSpeakNameInLaterTakes() throws {
        let duke = VoiceDesign(name: "The Duke", description: "silky theatrical villain")
        let line = "Oh, you came all this way to stop me? [laughs] How delightfully brave of you."
        let voice = Take(text: "The Duke: " + line, context: .drama3(codes: Self.codes(3)))
        let older = Take(text: "The Duke: A different line.", context: .drama3(codes: Self.codes(4)))
        let plan = try GenerationPlan(text: line, references: [], character: duke, previousTakes: [voice, older], config: GenerationConfig())
        XCTAssertEqual(plan.chunks.map(\.turn), ["Oh, you came all this way to stop me?", "[laughs] How delightfully brave of you."])
        XCTAssertEqual(plan.history.map(\.text), ["A different line."])
        let prompt = try Drama3Prompt.render(text: plan.chunks[1].turn, references: plan.references, history: plan.history)
        XCTAssertFalse(prompt.contains("The Duke:"), "Saved named turns must also be normalized when used as reference history")
        XCTAssertEqual(plan.turnText, "The Duke: " + line, "Project metadata keeps the display name")
    }

    func testDesignedVoiceDoesNotDuplicateAnExistingSpeakerTag() throws {
        let duke = VoiceDesign(name: "The Duke", description: "silky theatrical villain")
        let expected = try Drama3Prompt.render(text: "How delightfully brave of you.", character: duke)
        for tag in ["<|speaker:0|>", "<|speaker:5|>"] {
            XCTAssertEqual(try Drama3Prompt.render(text: tag + " How delightfully brave of you.", character: duke), expected)
        }
    }

    static let captain = VoiceDesign(name: "Captain", description: "gravelly old male voice, about seventy, weathered and slow, salt in every word",
                                     scene: "a storm-battered tavern at the edge of the harbor")
    static let captainSystem = """
        Scene: a storm-battered tavern at the edge of the harbor
        Characters:
        - <|speaker:0|>: gravelly old male voice, about seventy, weathered and slow, salt in every word
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
            - <|speaker:0|>: smooth warm baritone, a late-night radio host, relaxed and intimate, close to the mic
            Voice every line of the script in order, with <|audio_separator|> between consecutive lines.
            """)
    }

    func testDesignLayoutTagsEveryTurnWithoutSpeakingTheName() throws {
        let rendered = try Drama3Prompt.render(text: "Pray ye never find out.", character: Self.captain,
                                               history: [Drama3Prompt.HistoryTurn(text: "Ahoy there.", codes: Self.codes(2))])
        XCTAssertEqual(rendered, "<|im_start|>system\n" + Self.captainSystem + "<|im_end|>\n"
            + "<|im_start|>user\n<|speaker:0|>: Ahoy there.<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:2><|im_end|>\n"
            + "<|im_start|>user\n<|speaker:0|>: Pray ye never find out.<|im_end|>\n<|im_start|>assistant\n<|voice|>")
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
        XCTAssertTrue(ids.contains(tokenizer.encode("<|speaker:0|>")[0]), "design turns identify the voice with a speaker token")
        XCTAssertTrue(rows.allSatisfy { $0.dropFirst().allSatisfy { $0 == 0 } }, "no code rows without references")
    }

    func testRetakeTextCuesEverySentenceWithTheNote() {
        XCTAssertEqual(DirectorsNote.retakeText(line: "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out.", note: "softer and warmer"),
                       "Direction: softer and warmer\nLine: [softer and warmer] [low and menacing] Ye want to know what's out past the reef? [softer and warmer] [pause] Pray ye never find out.")
        XCTAssertEqual(DirectorsNote.retakeText(line: "<|speaker:0|><whisper>Don't move.</whisper> They can hear us [gasps] right now!", note: " more [scared] <now> |  please\n"),
                       "<|speaker:0|> Direction: more scared now please\nLine: [more scared now please] <whisper>Don't move.</whisper> [more scared now please] They can hear us [gasps] right now!")
        XCTAssertEqual(DirectorsNote.retakeText(line: "One line, no end", note: "faster"), "Direction: faster\nLine: [faster] One line, no end")
    }

    func testLiteralRecoveryPreservesCuesWordsSpeakerAndTheWholeUserDirection() {
        let line = "<|speaker:2|> [gasps] <whisper>Stay here.</whisper>"
        let direction = String(repeating: "a", count: 120)
        XCTAssertEqual(DirectorsNote.literalRetry(line: line, direction: direction),
                       "<|speaker:2|> Direction: " + direction + ". Enunciate each word precisely\nLine: [" + direction + "] [Enunciate each word precisely] [gasps] <whisper>Stay here.</whisper>")
        XCTAssertEqual(DirectorsNote.spokenWords(line), "Stay here.")
        XCTAssertEqual(DirectorsNote.literalRetry(line: "Direction: north. Line: two.", direction: nil),
                       "Direction: Enunciate each word precisely\nLine: [Enunciate each word precisely] Direction: north. [Enunciate each word precisely] Line: two.")
    }

    func testNotesAreCleanedAndCapped() {
        XCTAssertEqual(DirectorsNote.clean(String(repeating: "a", count: 200))?.count, 120)
        XCTAssertNil(DirectorsNote.clean(" [] <> | "))
        XCTAssertEqual(DirectorsNote.clean("more amused than before; drop the hurt"), "more amused than before; drop the hurt")
    }

    func testDesignedRetakeUsesTheFirstTakeAsVoiceAndTheLastTwoAsHistory() throws {
        let line = "[low and menacing] Ye want to know? Pray."
        let first = Take(text: "Captain: [low and menacing] Ye want to know? Pray.", context: .drama3(codes: Self.codes(3)))
        let second = Take(text: "Captain: Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray.", context: .drama3(codes: Self.codes(4)))
        let third = Take(text: "Captain: Direction: faster\nLine: [faster] Ye want to know? [faster] Pray.", context: .drama3(codes: Self.codes(5)))
        let secondScript = "Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray."
        let thirdScript = "Direction: faster\nLine: [faster] Ye want to know? [faster] Pray."
        let plan = try GenerationPlan(text: line, references: [], character: Self.captain, previousTakes: [first, second, third], note: "softer and warmer", config: GenerationConfig())
        let turn = "Captain: Direction: softer and warmer\nLine: [softer and warmer] [low and menacing] Ye want to know? [softer and warmer] Pray."
        let script = "Direction: softer and warmer\nLine: [softer and warmer] [low and menacing] Ye want to know?"
        XCTAssertEqual(plan, try GenerationPlan(text: line, references: [], character: Self.captain, previousTakes: [first, second, third], note: "softer and warmer", config: GenerationConfig()))
        XCTAssertNil(plan.character)
        XCTAssertEqual(plan.references, [Drama3Prompt.Reference(transcript: "Ye want to know? Pray.", codes: first.dramaCodes)])
        XCTAssertEqual(plan.history, [Drama3Prompt.HistoryTurn(text: secondScript, codes: second.dramaCodes), Drama3Prompt.HistoryTurn(text: thirdScript, codes: third.dramaCodes)])
        XCTAssertEqual(plan.chunks, [GenerationPlan.Chunk(turn: script, line: "[low and menacing] Ye want to know?"),
                                     GenerationPlan.Chunk(turn: "Direction: softer and warmer\nLine: [softer and warmer] Pray.", line: "Pray.")])
        XCTAssertEqual(plan.turnText, turn)
        XCTAssertEqual(try Drama3Prompt.render(text: plan.chunks[0].turn, references: plan.references, history: plan.history),
                       "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\n[Reference 1]\nText: <|speaker:0|> Ye want to know? Pray.\nSpeech: <codes:3><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + secondScript + "<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:4><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + thirdScript + "<|im_end|>\n<|im_start|>assistant\n<|voice|><codes:5><|im_end|>\n"
                       + "<|im_start|>user\n<|speaker:0|> " + script + "<|im_end|>\n<|im_start|>assistant\n<|voice|>")
    }

    func testFirstDesignTakeUsesTheDesignLayoutAndRecordsTheNamedTurn() throws {
        let plan = try GenerationPlan(text: "Ahoy. Who goes there in the storm tonight?", references: [], character: Self.captain, config: GenerationConfig())
        XCTAssertEqual(plan.character, Self.captain)
        XCTAssertTrue(plan.references.isEmpty)
        XCTAssertEqual(plan.chunks.map(\.turn), ["Ahoy.", "Who goes there in the storm tonight?"], "the builder writes the name for each sentence")
        XCTAssertEqual(plan.turnText, "Captain: Ahoy. Who goes there in the storm tonight?")
    }

    func testSentenceChunksDoNotDependOnTheVoice() throws {
        let lines = ["Here is a poem for you.", "The morning light begins to break, Across the quiet, misty lake."]
        let text = lines.joined(separator: "\n")
        let reference = Drama3Prompt.Reference(transcript: "An arbitrary reference voice.", codes: Self.codes(8))
        let plans = [
            try GenerationPlan(text: text, references: [], config: GenerationConfig()),
            try GenerationPlan(text: text, references: [reference], config: GenerationConfig()),
            try GenerationPlan(text: text, references: [], character: Self.captain, config: GenerationConfig())
        ]
        for plan in plans {
            XCTAssertEqual(plan.chunks.map(\.line), lines)
            XCTAssertEqual(plan.chunks.map(\.turn), lines)
        }
    }

    func testDefaultVoiceRetakeClonesTheFirstTake() throws {
        let first = Take(text: "It was a dark and stormy night.", context: .drama3(codes: Self.codes(3)))
        let plan = try GenerationPlan(text: "It was a dark and stormy night.", references: [], previousTakes: [first], note: "whispered", config: GenerationConfig())
        XCTAssertEqual(plan.references, [Drama3Prompt.Reference(transcript: "It was a dark and stormy night.", codes: first.dramaCodes)])
        XCTAssertEqual(plan.history, [Drama3Prompt.HistoryTurn(first)])
        XCTAssertEqual(plan.turnText, "Direction: whispered\nLine: [whispered] It was a dark and stormy night.")
    }

    func testATakeOfTheSameTurnIsTheVoiceButNeverHistory() throws {
        let line = "It was a dark and stormy night."
        let first = Take(text: line, context: .drama3(codes: Self.codes(3)))
        let plain = try GenerationPlan(text: line, references: [], previousTakes: [first], config: GenerationConfig())
        XCTAssertEqual(plain.references.map(\.codes), [first.dramaCodes], "still the voice")
        XCTAssertEqual(plain.history, [], "replaying the same turn copies its audio")
        let angrier = Take(text: "Direction: angrier\nLine: [angrier] It was a dark and stormy night.", context: .drama3(codes: Self.codes(4)))
        let again = try GenerationPlan(text: line, references: [], previousTakes: [first, angrier], note: "angrier", config: GenerationConfig())
        XCTAssertEqual(again.history, [Drama3Prompt.HistoryTurn(first)], "the same note again leaves out the take it would copy")
    }

    func testClonedRetakeKeepsTheClipAndIsolatesSentences() throws {
        let clip = Drama3Prompt.Reference(transcript: "My own voice.", codes: Self.codes(6))
        let line = "The first sentence is long enough to stand. The second one is long enough too."
        let take = Take(text: line, context: .drama3(codes: Self.codes(4)))
        let retake = try GenerationPlan(text: line, references: [clip], previousTakes: [take], note: "slower, with more pauses", config: GenerationConfig())
        XCTAssertEqual(retake.references, [clip])
        XCTAssertEqual(retake.chunks.map(\.line), ["The first sentence is long enough to stand.", "The second one is long enough too."])
        XCTAssertEqual(retake.history, [Drama3Prompt.HistoryTurn(take)])
        let plain = try GenerationPlan(text: line, references: [clip], config: GenerationConfig())
        XCTAssertEqual(plain.chunks.count, 2, "without a note the line is cut into sentences")
        XCTAssertEqual(plain.turnText, line)
        XCTAssertEqual(try GenerationPlan(text: line, references: [clip], note: " | ", config: GenerationConfig()), plain, "an empty note is no note")
        XCTAssertThrowsError(try GenerationPlan(text: "[laughs]", references: [clip], note: "angrier", config: GenerationConfig()))
    }

    func testSpokenTextDropsNameAndDirection() {
        XCTAssertEqual(DirectorsNote.spokenText(ofTurn: "<|speaker:2|> Direction: angrier\nLine: [angrier] Stay here.", character: nil), "Stay here.")
        XCTAssertEqual(DirectorsNote.spokenText(ofTurn: "Captain: Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray.", character: Self.captain), "Ye want to know? Pray.")
        XCTAssertEqual(DirectorsNote.spokenText(ofTurn: "<|speaker:0|> Hello <whisper>there</whisper> [laughs] .", character: nil), "Hello there.")
    }
}
