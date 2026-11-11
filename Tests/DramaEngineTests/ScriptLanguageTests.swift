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

    /// Catches a one-word line refused as uncertain, or read in whatever
    /// language the recognizer guessed: "Hi." came back Catalan at 0.80.
    func testALineTooShortToTellIsReadInTheFallbackLanguage() throws {
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["en": 0.23, "nl": 0.13], segments: [], short: true, fallback: "en"), "en")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["ca": 0.80, "tr": 0.07], segments: [], short: true, fallback: "en"), "en")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["it": 0.93], segments: [], short: true, fallback: "fr"), "fr")
        XCTAssertEqual(try ScriptLanguage.choose(overall: [:], segments: [], short: true, fallback: "en"), "en")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["es": 0.98], segments: [], short: true, fallback: "en"), "es", "a short line the recognizer is sure of keeps its language")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["ja": 1.0], segments: [], short: true, fallback: "en"), "ja", "another script is never the fallback's language")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["zh-Hant": 0.80, "zh-Hans": 0.20], segments: [], short: true, fallback: "en"), "zh-Hant")
        XCTAssertThrowsError(try ScriptLanguage.choose(overall: ["bg": 0.53, "ru": 0.42], segments: [], short: true, fallback: "en"))
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["ca": 0.80], segments: [], short: true), "ca", "without a fallback, detection decides as before")

        XCTAssertEqual(try ScriptLanguage.choose(overall: ["es": 0.80], segments: [], fallback: "en"), "es", "a longer line needs only 0.75")
        XCTAssertEqual(try ScriptLanguage.choose(overall: ["en": 0.5, "nl": 0.4], segments: [], fallback: "en"), "en", "an uncertain longer line is read in the fallback")
        XCTAssertThrowsError(try ScriptLanguage.choose(overall: ["en": 0.9], segments: [["en": 0.99], ["ja": 0.99]], fallback: "en"), "the fallback never hides mixed languages")

        for line in ["Hi.", "Goodbye.", "No.", "Hello.", "Okay.", "[laughs] Stop!", "42."] {
            XCTAssertEqual(try ScriptLanguage.identify(line, fallback: "en"), "en", line)
        }
        XCTAssertEqual(try ScriptLanguage.identify("こんにちは。", fallback: "en"), "ja")
        XCTAssertEqual(try ScriptLanguage.identify("Guten Morgen. Wie geht es dir heute?", fallback: "en"), "de")
        XCTAssertThrowsError(try ScriptLanguage.identify("Goodbye."), "detection alone cannot tell")
    }

    func testARequestsLanguageIsItsOwnBeforeItsTexts() throws {
        var request = SpeechRequest(text: "Guten Morgen. Wie geht es dir heute?")
        XCTAssertEqual(try ScriptLanguage.language(of: request), "de")
        request.language = "en"
        XCTAssertEqual(try ScriptLanguage.language(of: request), "en")
        XCTAssertEqual(try ScriptLanguage.language(of: SpeechRequest(text: "Goodbye.")), ScriptLanguage.userLanguage)
    }

    func testAnAppleModelChecksOnlyWhenTheRecognizerRunsAndTheLanguageIsInstalled() {
        let english = Locale(identifier: "en_US")
        XCTAssertNil(ScriptLanguage.installedLocale(recognizerAvailable: false, locale: english, installed: [english]))
        XCTAssertNil(ScriptLanguage.installedLocale(recognizerAvailable: true, locale: nil, installed: [english]))
        XCTAssertNil(ScriptLanguage.installedLocale(recognizerAvailable: true, locale: Locale(identifier: "fr_FR"), installed: [english]))
        let australian = Locale(identifier: "en_AU")
        XCTAssertEqual(ScriptLanguage.installedLocale(recognizerAvailable: true, locale: english, installed: [australian]), australian)
        XCTAssertEqual(ScriptLanguage.installedLocale(recognizerAvailable: true, locale: english,
                                                      installed: [australian, Locale(identifier: "en-US")])?.identifier(.bcp47), "en-US")
    }

    /// The request, not the Mac, is at fault, so this is a bad request and
    /// fails before any recognizer starts.
    func testALanguageNoRecognizerKnowsFailsAsTheRequestsLanguage() {
        guard #available(macOS 26, *) else { return }
        XCTAssertThrowsError(try LocalWordVerifier.prepare(lines: ["ᎣᏏᏲ. ᏙᎯᏧ?"], language: "chr", config: GenerationConfig(), shouldContinue: { true })) {
            guard case SpeechVerificationError.language = $0 else { return XCTFail("Unexpected \($0)") }
            XCTAssertEqual($0.localizedDescription,
                           "Cannot check the generated words: Drama Studio's speech recognizer doesn't know Cherokee, so it can't check words in it.")
        }
    }

    /// whisper names some languages by older codes.
    func testTheIndependentRecognizerHearsEachLanguageByItsWhisperCode() {
        XCTAssertEqual(LocalWordVerifier.recognitionLanguage("zh-Hans"), "zh")
        XCTAssertEqual(LocalWordVerifier.recognitionLanguage("es-MX"), "es")
        XCTAssertEqual(LocalWordVerifier.recognitionLanguage("nb"), "no")
        XCTAssertEqual(LocalWordVerifier.recognitionLanguage("jv"), "jw")
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
