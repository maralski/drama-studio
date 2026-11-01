// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class ScriptLanguageTests: XCTestCase {
    func testInstalledLocaleSelectionHonorsPreferencesAndDoesNotDependOnAssetOrder() {
        let australian = Locale(identifier: "en_AU"), american = Locale(identifier: "en_US"), canadian = Locale(identifier: "en_CA")
        for installed in [[canadian, american, australian], [american, australian, canadian], [australian, canadian, american]] {
            XCTAssertEqual(ScriptLanguage.preferredLocale(language: "en", installed: installed, preferences: ["en-AU"]), australian)
            XCTAssertEqual(ScriptLanguage.preferredLocale(language: "en", installed: installed, preferences: ["en-US", "en-AU"]), american)
            XCTAssertEqual(ScriptLanguage.preferredLocale(language: "en", installed: installed, preferences: ["fr-FR"]), australian)
        }
        XCTAssertNil(ScriptLanguage.preferredLocale(language: "fr", installed: [australian], preferences: ["en-AU"]))
        XCTAssertEqual(ScriptLanguage.preferredLocale(language: "zh-Hans", installed: [Locale(identifier: "zh_Hant_TW"), Locale(identifier: "zh_Hans_CN")],
                                                     preferences: ["zh-Hant-TW"]), Locale(identifier: "zh_Hans_CN"))
    }

    func testConfidenceAndMixedLanguagePolicy() throws {
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["en": 0.99], segments: [["en": 0.99], ["en": 0.95]]), "en")
        XCTAssertThrowsError(try ScriptLanguage.choose(overall: ["en": 0.5, "fr": 0.5], segments: []))
        XCTAssertThrowsError(try ScriptLanguage.choose(overall: [:], segments: []))
        XCTAssertThrowsError(try ScriptLanguage.choose(overall: ["en": 0.9], segments: [["en": 0.99], ["ja": 0.99]]))
    }

    func testUnsupportedRecognizerLocaleAndMissingAssetsFailBeforeGeneration() throws {
        let english = Locale(identifier: "en_US")
        XCTAssertThrowsError(try ScriptLanguage.requireLocale(language: "en", recognizerAvailable: false, locale: english, installed: [english]))
        XCTAssertThrowsError(try ScriptLanguage.requireLocale(language: "xx", recognizerAvailable: true, locale: nil, installed: [english]))
        XCTAssertThrowsError(try ScriptLanguage.requireLocale(language: "fr", recognizerAvailable: true, locale: Locale(identifier: "fr_FR"), installed: [english]))
        let australian = Locale(identifier: "en_AU")
        XCTAssertEqual(try ScriptLanguage.requireLocale(language: "en", recognizerAvailable: true, locale: english,
                                                        installed: [australian]), australian)
        XCTAssertEqual(try ScriptLanguage.requireLocale(language: "en", recognizerAvailable: true, locale: english,
                                                        installed: [australian, Locale(identifier: "en-US")]).identifier(.bcp47), "en-US")
    }

    func testActualEnglishCueScriptIsIdentifiedWithoutReadingTheCueAsSpeech() throws {
        XCTAssertEqual(try ScriptLanguage.identify("[with bright eager excitement] They said yes! [gasps] They actually said yes, and we start on Monday!"), "en")
    }
    func testMixedScriptFailsBeforeAnyModelLoadOrAudioEvent() async throws {
        guard #available(macOS 26, *) else { throw XCTSkip("native word verification requires macOS 26") }
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let engine = DramaSynthesizer(location: ModelLocation(directory: missing), codecDirectory: missing)
        var events = 0
        do {
            for try await _ in engine.synthesize(SpeechRequest(text: "This is a complete English sentence. これは日本語で書かれた完全な文章です。")) { events += 1 }
            XCTFail("A mixed-language request became successful")
        } catch let error as SpeechVerificationError {
            XCTAssertTrue(error.localizedDescription.contains("mixes languages"))
        }
        XCTAssertEqual(events, 0)
        XCTAssertFalse(engine.isLoaded)
    }

}
