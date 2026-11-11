// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@testable import DramaEngine
import XCTest

final class BuiltInVoiceTests: XCTestCase {
    private let emma = BuiltInVoice(engine: .ming, name: "US_Emma")
    private let line = "The lamp was lit before the fog rolled in."

    /// Catches a hard-coded or truncated list, and a voice left out of every heading.
    func testMingsVoicesComeFromThePackageCatalogUnderFourHeadings() {
        let voices = SpeechEngineID.ming.builtInVoices
        XCTAssertEqual(voices.count, 241)
        XCTAssertEqual(Set(voices).count, 241)
        XCTAssertEqual(SpeechEngineID.drama3.builtInVoices, [])
        let headings = Dictionary(grouping: voices, by: \.group).mapValues(\.count)
        XCTAssertEqual(headings, ["English": 3, "Chinese": 33, "Chinese characters": 8, "Chinese, in one emotion": 197])
        XCTAssertEqual(voices.filter { $0.group == "English" }.map(\.name), ["US_Emma", "UK_George", "UK_Charlotte"])
        XCTAssertEqual(BuiltInVoice(engine: .ming, name: "灵晓声_高兴").note, "happy")
        XCTAssertNil(emma.note)
        XCTAssertEqual(voices.filter { $0.note != nil }.count, 197)
    }

    func testAKeyNamesAVoiceWithOrWithoutItsEngine() {
        XCTAssertEqual(BuiltInVoice.named("US_Emma"), emma)
        XCTAssertEqual(BuiltInVoice.named("us_emma"), emma)
        XCTAssertEqual(BuiltInVoice.named("ming:US_Emma"), emma)
        XCTAssertEqual(BuiltInVoice.named("Ming:uk_george"), BuiltInVoice(engine: .ming, name: "UK_George"))
        XCTAssertEqual(BuiltInVoice.named("水浒传_武松")?.group, "Chinese characters")
        XCTAssertNil(BuiltInVoice.named("drama3:US_Emma"))
        XCTAssertNil(BuiltInVoice.named("Nobody"))
        XCTAssertEqual(emma.qualifiedName, "ming:US_Emma")
        for voice in SpeechEngineID.ming.builtInVoices { XCTAssertEqual(BuiltInVoice.named(voice.qualifiedName), voice) }
    }

    /// Catches a Ming voice spoken by Drama 3 in whatever voice Drama 3 picks.
    func testAnotherEngineRefusesTheVoiceAndNamesItsOwner() throws {
        var request = SpeechRequest(text: line)
        request.builtInVoice = emma
        XCTAssertNoThrow(try SpeechOutline(request, engine: .ming))
        XCTAssertThrowsError(try SpeechOutline(request, engine: .drama3)) {
            XCTAssertEqual(($0 as? PromptError)?.description, "Drama 3 can't speak in US_Emma, a built-in Ming voice; switch to Ming or choose another voice")
        }
        request.builtInVoice = BuiltInVoice(engine: .ming, name: "US_Nobody")
        XCTAssertThrowsError(try SpeechOutline(request, engine: .ming)) {
            XCTAssertEqual(($0 as? PromptError)?.description, "Ming can't speak in US_Nobody; it has no built-in voice of that name")
        }
    }

    func testARequestTakesOneVoice() async throws {
        let reference = ReferenceVoice(name: "Keeper", transcript: "Hello there.",
                                       encodings: [try await MockSynthesizer(engine: .ming, pace: .instant).encodeVoice(samples: [Float](repeating: 0.1, count: 44_100), transcript: "Hello there.")])
        var cloned = SpeechRequest(text: line, references: [reference])
        cloned.builtInVoice = emma
        var designed = SpeechRequest(text: line, character: VoiceDesign(name: "Captain", description: "gravelly old male voice"))
        designed.builtInVoice = emma
        for request in [cloned, designed] {
            XCTAssertThrowsError(try SpeechOutline(request, engine: .ming)) { XCTAssertEqual($0 as? PromptError, .voiceConflict) }
        }
    }

    /// Catches every request speaking as US_Emma whatever voice it names.
    func testMingSpeaksTheNamedVoiceThenContinuesFromItsTake() throws {
        var request = SpeechRequest(text: line)
        XCTAssertEqual(try MingPlan(request).voice, .builtIn("US_Emma"))
        request.builtInVoice = BuiltInVoice(engine: .ming, name: "UK_George")
        XCTAssertEqual(try MingPlan(request).voice, .builtIn("UK_George"))
        XCTAssertEqual(try MingPlan(request).voice.references, [])
        XCTAssertTrue(String(describing: try MingPlan(request).voice.openingTask).contains("UK_George"), "the name reaches Ming's task")

        let earlier = try MockSynthesizer.mingReference(transcript: line, sampleCount: 44_100)
        request.previousTakes = [Take(text: line, context: .ming(reference: earlier, sampleCount: 44_100))]
        XCTAssertEqual(try MingPlan(request).voice, .clone([earlier]))
    }
}
