// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class MingCuesTests: XCTestCase {
    func testReactionCuesBecomeMingTags() {
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[laughs]"), .tag("[Laughter]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[chuckles]"), .tag("[Laughter]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[sighs]"), .tag("[Sigh]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[coughs]"), .tag("[Cough]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[breathes heavily]"), .tag("[Breathing]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[gasps]"), .tag("[Surprise-ah]"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[ Laughs ]"), .tag("[Laughter]"), "spacing and case do not matter")
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[laughs nervously]"), .tag("[Laughter]"), "leading words decide")
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[Laughter]"), .tag("[Laughter]"), "a typed Ming tag passes through")
    }

    func testStylesLiftToEmotionsAndTheRestIsUnsupported() {
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[with bright eager excitement]"), .style("happy"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[heavy, deflated sadness]"), .style("sad"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[furious, barely holding it together]"), .style("angry"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[breathless and terrified]"), .style("fearful"))
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[icy calm]"), .unsupported)
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("[pause]"), .unsupported)
        XCTAssertEqual(SpeechEngineID.ming.cueSupport("<whisper>"), .unsupported)
        XCTAssertEqual(SpeechEngineID.drama3.cueSupport("[anything at all]"), .native)
    }

    func testEveryTagInTheTableIsOneMingSpells() {
        for case .tag(let tag) in MingCues.table.values {
            XCTAssertTrue(MingCues.tags.contains(tag), "\(tag) is not a Ming tag")
        }
        XCTAssertEqual(SpeechEngineID.ming.capabilities.cues, .fixed(MingCues.tags))
        XCTAssertEqual(MingCues.tags.count, 14)
    }

    func testTheWordCheckNeverExpectsAMingTagToBeSpoken() {
        for tag in MingCues.tags {
            XCTAssertEqual(DirectorsNote.spokenWords("Oh \(tag) no, really."), "Oh no, really.", tag)
            XCTAssertEqual(WordCoverage.expectedTokens("Oh \(tag) no.", locale: Locale(identifier: "en_US")),
                           WordCoverage.expectedTokens("Oh no.", locale: Locale(identifier: "en_US")), tag)
        }
    }

    func testTranslationSwapsTagsKeepsWordsAndReportsTheRest() {
        let translation = MingCues.translate(lines: ["[laughs] Oh no. [sighs] <whisper>not again</whisper> fine.", "[laughs nervously] Go [pause] now."], note: nil)
        XCTAssertEqual(translation.lines.map(\.text), ["[Laughter] Oh no. [Sigh] not again fine.", "[Laughter] Go now."])
        XCTAssertEqual(translation.lines.map { $0.style?.emotion }, [nil, nil])
        XCTAssertEqual(translation.notices, [.unsupportedCue("<whisper>"), .approximatedCue("[laughs nervously]", as: "[Laughter]"), .unsupportedCue("[pause]")])
    }

    func testAStyleHoldsForLaterSentencesUntilAnotherReplacesIt() {
        let translation = MingCues.translate(lines: ["[heavy, deflated sadness] It is over.", "Nothing left.", "[with bright eager excitement] Or is it? [icy calm] No."], note: nil)
        XCTAssertEqual(translation.lines.map { $0.style?.emotion }, [.sad, .sad, .happy])
        XCTAssertEqual(translation.lines.map(\.text), ["It is over.", "Nothing left.", "Or is it? No."])
        XCTAssertEqual(translation.notices, [.unsupportedCue("[icy calm]")])
    }

    func testANoteNamesTheEmotionAndWinsOverCues() {
        let directed = MingCues.translate(lines: ["[heavy, deflated sadness] It is over."], note: "much angrier, bite the words")
        XCTAssertEqual(directed.lines.map { $0.style?.emotion }, [.angry])
        XCTAssertEqual(directed.notices, [.unsupportedCue("[heavy, deflated sadness]")])
        let vague = MingCues.translate(lines: ["It is over."], note: "softer and warmer")
        XCTAssertEqual(vague.lines.map { $0.style?.emotion }, [nil])
        XCTAssertEqual(vague.notices, [.unsupportedNote("softer and warmer")])
    }

    func testSpeakerTagsPickTheSpeakerAndLeaveTheText() {
        let translation = MingCues.translate(lines: ["<|speaker:1|> [laughs] Hi there.", "<|speaker:0|> Hello."], note: nil)
        XCTAssertEqual(translation.lines.map(\.speaker), [1, 0])
        XCTAssertEqual(translation.lines.map(\.text), ["[Laughter] Hi there.", "Hello."])
        XCTAssertEqual(translation.notices, [])
    }

    func testCapabilitiesTableNamesBothEngines() {
        XCTAssertEqual(SpeechEngineID.drama3.capabilities.cues, .freeForm)
        XCTAssertTrue(SpeechEngineID.drama3.capabilities.history)
        XCTAssertFalse(SpeechEngineID.ming.capabilities.history)
        XCTAssertFalse(SpeechEngineID.ming.capabilities.passageTags)
        XCTAssertEqual(SpeechEngineID.ming.capabilities.samplesPerStep, 3528)
        XCTAssertEqual(SpeechEngineID.drama3.capabilities.samplesPerStep, 2048)
        XCTAssertEqual(SpeechEngineID.ming.capabilities.builtInVoices.count, 241)
        XCTAssertTrue(SpeechEngineID.ming.capabilities.builtInVoices.contains("US_Emma"))
        XCTAssertEqual(SpeechEngineID.ming.capabilities.languages, .only(["zh", "en", "ja", "ko", "de", "fr", "es"]))
    }

    /// Catches Chinese refused because the language recognizer names it with
    /// a script (`zh-Hans`, `zh-Hant`), and a language Ming does not speak
    /// passing the pre-flight, so the API answered 500 from the word check, or
    /// a stream failed after its 200, instead of a 400 that names the limit.
    func testOutlineTakesMingsLanguagesAndRefusesOthers() throws {
        for text in ["你好，今天天气很好。我们去公园散步吧。", "早安。今天天氣很好，我們去公園散步吧。", "Guten Morgen. Wie geht es dir heute?",
                     "Buenos días. ¿Cómo estás hoy? Hace un tiempo precioso.", "こんにちは。今日はいい天気ですね。"] {
            XCTAssertNoThrow(try SpeechOutline(SpeechRequest(text: text), engine: .ming), text)
        }
        let italian = SpeechRequest(text: "Buongiorno, come stai oggi? Spero che tu stia molto bene.")
        XCTAssertThrowsError(try SpeechOutline(italian, engine: .ming)) { error in
            guard case .engineUnsupported(.ming, let what)? = error as? PromptError else { return XCTFail("\(error)") }
            XCTAssertTrue(what.contains("Italian"), what)
        }
        XCTAssertNoThrow(try SpeechOutline(italian, engine: .drama3))
    }
}
