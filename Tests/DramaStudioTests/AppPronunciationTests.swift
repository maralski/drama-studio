import Foundation
import XCTest
@testable import DramaStudio

final class AppPronunciationTests: XCTestCase {
    private func dictionary(_ rules: PronunciationRule...) -> PronunciationDictionary { PronunciationDictionary(rules: rules) }

    func testWholeWordVersusSubstring() {
        let whole = dictionary(PronunciationRule(match: "GIF", sayAs: "jif"))
        XCTAssertEqual(whole.apply(to: "GIF, gifts and gif."), "jif, gifts and jif.")
        let substring = dictionary(PronunciationRule(match: "ph", sayAs: "f", wholeWord: false))
        XCTAssertEqual(substring.apply(to: "phone graph"), "fone graf")
        let abbreviation = dictionary(PronunciationRule(match: "Dr.", sayAs: "Doctor"))
        XCTAssertEqual(abbreviation.apply(to: "Dr. Who met Dr.Strange and dr. No"), "Doctor Who met Dr.Strange and Doctor No")
    }

    func testCaseSensitivity() {
        let sensitive = dictionary(PronunciationRule(match: "US", sayAs: "United States", caseSensitive: true))
        XCTAssertEqual(sensitive.apply(to: "The US and us."), "The United States and us.")
        let insensitive = dictionary(PronunciationRule(match: "nasa", sayAs: "Nasa"))
        XCTAssertEqual(insensitive.apply(to: "NASA, Nasa, nasa"), "Nasa, Nasa, Nasa")
    }

    func testTagsCuesAndMarkupAreNeverRewritten() {
        let rules = dictionary(
            PronunciationRule(match: "Smith", sayAs: "Smyth"),
            PronunciationRule(match: "speaker", sayAs: "X", wholeWord: false),
            PronunciationRule(match: "whisper", sayAs: "Y"),
            PronunciationRule(match: "laughs", sayAs: "Z")
        )
        XCTAssertEqual(
            rules.apply(to: "<|speaker:0|> Smith [laughs at Smith] <whisper>Smith whispers</whisper> laughs"),
            "<|speaker:0|> Smyth [laughs at Smith] <whisper>Smyth whispers</whisper> Z"
        )
    }

    func testLongestMatchFirstAndNoChainedRewrites() {
        let city = dictionary(PronunciationRule(match: "New", sayAs: "Noo"), PronunciationRule(match: "New York", sayAs: "Nyork"))
        XCTAssertEqual(city.apply(to: "New York is new"), "Nyork is Noo")
        let chain = dictionary(PronunciationRule(match: "a", sayAs: "b", wholeWord: false), PronunciationRule(match: "b", sayAs: "c", wholeWord: false))
        XCTAssertEqual(chain.apply(to: "ab"), "bc")
        XCTAssertEqual(dictionary().apply(to: "untouched"), "untouched")
        XCTAssertEqual(dictionary(PronunciationRule(match: "", sayAs: "x")).apply(to: "untouched"), "untouched")
    }

    func testSaveLoadRoundTripAndMissingFile() throws {
        let url = try temporaryDirectory().appending(path: "nested/DramaStudio/pronunciation.json")
        XCTAssertEqual(try PronunciationDictionary.load(from: url), PronunciationDictionary())

        let rules = dictionary(PronunciationRule(match: "Nguyen", sayAs: "win"), PronunciationRule(match: "SQL", sayAs: "sequel", wholeWord: false, caseSensitive: true))
        try rules.save(to: url)
        XCTAssertEqual(try PronunciationDictionary.load(from: url), rules)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual((json["rules"] as? [[String: Any]])?.first?["sayAs"] as? String, "win")

        try Data("{not json".utf8).write(to: url)
        XCTAssertThrowsError(try PronunciationDictionary.load(from: url))
        XCTAssertTrue(PronunciationDictionary.defaultURL.path.hasSuffix("Library/Application Support/DramaStudio/pronunciation.json"))
    }
}
