// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class WordCoverageTests: XCTestCase {
    private let english = Locale(identifier: "en_US")
    private let englishLocales = [Locale(identifier: "en_US"), Locale(identifier: "en_AU")]

    private func assertMatch(_ pairs: [(String, String)], _ expected: Bool, file: StaticString = #filePath, line: UInt = #line) {
        for locale in englishLocales {
            for (first, second) in pairs {
                for (script, heard) in [(first, second), (second, first)] {
                    XCTAssertEqual(WordCoverage.matches(line: script, transcript: heard, locale: locale), expected,
                                   "\(locale.identifier): \(script) | \(heard)", file: file, line: line)
                }
            }
        }
    }

    func testWrittenNumbersMatchEveryCommonSpokenReading() {
        assertMatch([
            ("It costs 1,200 dollars.", "It costs twelve hundred dollars."),
            ("It costs 1,200 dollars.", "It costs one thousand two hundred dollars."),
            ("Agent 007 reports for duty.", "Agent double oh seven reports for duty."),
            ("Agent 007 reports for duty.", "Agent zero zero seven reports for duty."),
            ("Call 555 0142 now.", "Call five five five oh one four two now."),
            ("Call 555-0142 now.", "Call 555 0142 now."),
            ("Call 555-0142 now.", "Call five five five, oh one four two now."),
            ("It weighs 4.99 grams.", "It weighs four point nine nine grams."),
            ("The rate is 2.5 percent.", "The rate is two point five percent."),
            ("Meet at 3:30.", "Meet at three thirty."),
            ("Meet at 3:30.", "Meet at 330."),
            ("Meet at 3:30.", "Meet at 3.30."),
            ("In 2026 we left.", "In twenty twenty six we left."),
            ("In 2026 we left.", "In two thousand twenty-six we left."),
            ("In 2026 we left.", "In two thousand and twenty six we left."),
            ("In 1998 we left.", "In nineteen ninety-eight we left."),
            ("In 1905 we left.", "In nineteen oh five we left."),
            ("Board the 747 now.", "Board the seven forty seven now."),
            ("It costs $4.99.", "It costs four ninety-nine."),
            ("It costs $4.99.", "It costs 4 99."),
            ("They raised $2.5 million.", "They raised 2.5 million dollars."),
            ("They raised $2.5 million.", "They raised two point five million dollars."),
            ("They raised $2,500,000.", "They raised 2.5 million dollars."),
            ("About 100 came.", "About a hundred came."),
            ("About 700 came.", "About seven hundred came."),
            ("Back in the 1990s.", "Back in the nineties."),
            ("Back in the 1990s.", "Back in the nineteen nineties."),
            ("Back in the 90s.", "Back in the '90s."),
            ("Take the 5th door.", "Take the fifth door."),
            ("It fell 12%.", "It fell twelve percent."),
        ], true)
        assertMatch([("Call 555 0142 now.", "Call five five five one four two now."),
                     ("Room 0142 is open.", "Room one four two is open."),
                     ("Call 555. 0142 is mine.", "Call 5550142 is mine.")], false)
    }

    func testSpokenReadingsOfDifferentKindsAndSeparateNumbersNeverMerge() {
        assertMatch([
            ("Count down: three, two, one.", "Count down: three hundred twenty one."),
            ("Count down: three, two, one.", "Count down: 321."),
            ("Rooms 5, 6 and 7 are free.", "Rooms fifty six and seven are free."),
            ("Chapter 2, 3 and 4.", "Chapter 23 and 4."),
            ("In 2019, 300 people came.", "In 201, 9300 people came."),
            ("one two three four five six seven", "one million two hundred thirty four thousand five hundred sixty seven"),
            ("I have two, oh well.", "I have twenty well."),
        ], false)
        assertMatch([
            ("Call me on 555 0142 when you land.", "Call me on 555 0142 when you land"),
            ("Call me on 555 0142 when you land.", "Call me on 555-0142 when you land."),
            ("Call me on 555 0142 when you land.", "Call me on 5550142 when you land."),
            ("Call me on five five five, oh one four two, when you land.", "Call me on 555 0142 when you land"),
            ("Call me on five five five, oh one four two, when you land.", "Call me on 555-0142 when you land."),
            ("Call me on five five five, oh one four two, when you land.", "Call me on 5550142 when you land."),
            ("Three. Two. One. Go!", "Three, two, one, go!"),
            ("In 2019, 300 people came.", "In 2019, 300 people came."),
        ], true)
    }

    func testClockTimesAndLeadingZerosDoNotReadAsCardinals() {
        assertMatch([
            ("Meet at 3:30.", "Meet at three hundred and thirty."),
            ("Meet at 1:30.", "Meet at one hundred and thirty."),
            ("Agent 007.", "Agent seven."),
        ], false)
        assertMatch([("Meet at 3:30.", "Meet at 330."), ("Meet at 330.", "Meet at three hundred and thirty.")], true)
    }

    func testCurrenciesMustAgreeButASymbolMayGoUnspoken() {
        assertMatch([
            ("It costs 50 cents.", "It costs 50 dollars."),
            ("It costs £5.", "It costs five dollars."),
            ("It costs €5.", "It costs five dollars."),
            ("It costs £5.", "It costs $5."),
            ("I paid 5 dollars.", "I paid 5."),
        ], false)
        assertMatch([
            ("They raised $2.5 million.", "They raised 2.5 million dollars."),
            ("It costs $4.99.", "It costs four ninety nine dollars."),
            ("It costs $4.99 a month.", "It costs four ninety-nine a month."),
            ("I paid 5 dollars.", "I paid $5."),
            ("I paid one dollar.", "I paid $1."),
            ("It costs 50¢.", "It costs fifty cents."),
            ("It costs £5.", "It costs five pounds."),
        ], true)
    }

    func testWrongNumbersAndWordsStillFail() {
        assertMatch([
            ("Deliveries grew by 18% this quarter.", "Deliveries grew by 80% this quarter."),
            ("In 1998 we left.", "In 1989 we left."),
            ("It costs $4.99.", "It costs $9.99."),
            ("They raised 2.5 million dollars.", "They raised $25 million."),
            ("The value is 2.5.", "The value is 25."),
            ("The value is 2.5.", "The value is twenty five."),
            ("Meet at 3:30.", "Meet at 3:40."),
            ("Agent 007 reports for duty.", "Agent 008 reports for duty."),
            ("The new routes open on Monday.", "The new roads open on Monday."),
            ("The queen arrives at noon.", "The king arrives at noon."),
            ("The ship left the harbour, and nobody saw it again.", "The ship left the harbour."),
        ], false)
    }

    func testBritishAndAmericanSpellingsMatch() {
        assertMatch([
            ("The boat is in the harbour.", "The boat is in the harbor."),
            ("What colour is the theatre?", "What color is the theater?"),
            ("All right, let us go.", "Alright, let us go."),
            ("Ok, I will organise the grey programme.", "Okay, I will organize the gray program."),
            ("We travelled to analyse it.", "We traveled to analyze it."),
        ], true)
        assertMatch([("I promise.", "I promised."), ("The boat is in the harbour.", "The boat is in the harbors.")], false)
    }

    func testCompoundSpacingMatches() {
        assertMatch([("The website is down.", "The web site is down."), ("Read the ebook.", "Read the e book.")], true)
        assertMatch([("The website is down.", "The web is down.")], false)
    }

    func testMissingOpeningSentenceIsRejectedEvenWithALongReaction() {
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] The garden gate is closed. The meeting begins on Monday.",
                                            transcript: "The meeting begins on Monday.", locale: english))
    }

    func testEveryWordIncludingNegationAndRepetitionsIsRequiredInOrder() {
        let line = "No, no, he did not leave the gate open."
        for heard in ["No he did not leave the gate open", "No no he did leave the gate open",
                      "No no he did not leave the open gate", "No no he did not leave the gate open again"] {
            XCTAssertFalse(WordCoverage.matches(line: line, transcript: heard, locale: english), heard)
        }
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: line, locale: english))
    }

    func testReactionsOnlyPermitExtraNonverbalSoundsWithoutDeletingRequiredWords() {
        XCTAssertTrue(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! He left.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "[laughs] He left.", transcript: "Hah! He left.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! Left.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "He left.", transcript: "Ha ha! He left.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! She left.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "[clears throat] He left.", transcript: "Ahem. He left.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] He left.", transcript: "Ahem. Left.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] Ahem, he left.", transcript: "He left.", locale: english))
    }

    func testCommonEnglishContractionsDoNotLoseNegation() {
        XCTAssertTrue(WordCoverage.matches(line: "I can't leave. What's the plan?", transcript: "I cannot leave. What is the plan?", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "What's the plan?", transcript: "whats the plan?", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "What's the plan?", transcript: "what the plan?", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "I can't leave.", transcript: "I can leave.", locale: english))
    }

    func testMarkupIsNotSpeechButTaggedWordsAreStillRequired() {
        let line = "<|speaker:2|> [icy calm] <whisper>Wait here.</whisper> Then <emphasis>go.</emphasis>"
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: "Wait here. Then go.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "Wait here.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs]", transcript: "ha ha", locale: english))
    }

    func testTestedContractionsNumbersAndUnicodeNamesAreEquivalent() {
        XCTAssertTrue(WordCoverage.matches(line: "I’m here. You don't have 24 tickets, Élodie.",
                                           transcript: "I am here. You do not have twenty-four tickets, Elodie.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "You don't have 24 tickets, Élodie.",
                                            transcript: "You have twenty-four tickets, Élodie.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "Peter and Élodie are here.", transcript: "Peter is here.", locale: english))
    }

    func testAnUncertainMatchingWordCannotCertifySpeech() {
        let line = "[clears throat] The garden gate is closed."
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The garden gate is closed.", locale: english, uncertainSpans: ["garden"]))
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: "Ha! The garden gate is closed.", locale: english, uncertainSpans: ["Ha!"]))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The garden gate is closed.", locale: english, uncertainSpans: ["closed."]))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "He left.", locale: english, uncertainSpans: ["He"]))
    }

    func testWrittenTitlesAreEquivalentButAnAcronymIsNotANameTitle() throws {
        try WordCoverage.prepare()
        XCTAssertTrue(WordCoverage.matches(line: "His name is Mister Hops.", transcript: "His name is Mr. Hops.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "Doctor Elodie is here.", transcript: "Dr. Elodie is here.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "MR results are ready.", transcript: "Mister results are ready.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "His name is Mister Hops.", transcript: "His name is Mr. Jones.", locale: english))
    }

    func testExplicitRecognizerReactionAnnotationsMatchOnlyRequestedCueFamilies() {
        for (cue, annotation) in [("gasps", "*gasp*"), ("gasps", "(gasps)"), ("gasps", "[Gasping]"),
                                  ("laughs", "(laughter)"), ("chuckles", "*chuckling*"),
                                  ("sighs", "(sigh)"), ("coughs", "[coughing]"),
                                  ("clears throat", "(clears throat)"), ("hesitates", "*hesitation*"),
                                  ("pause", "(pause)"), ("whispers", "(whispering)")] {
            XCTAssertTrue(WordCoverage.matches(line: "[\(cue)] Please wait here.",
                                               transcript: "\(annotation) Please wait here.", locale: english), cue + annotation)
        }
        XCTAssertTrue(WordCoverage.matches(line: "<whisper>Please wait here.</whisper>",
                                           transcript: "(whispering) Please wait here.", locale: english))
        for (line, heard) in [("Please wait here.", "*gasp* Please wait here."),
                              ("[laughs] Please wait here.", "*gasp* Please wait here."),
                              ("[gasps] Please wait here.", "Gasp, please wait here."),
                              ("[gasps] Please say gasp.", "*gasp* Please say."),
                              ("[gasps] He waits here.", "(gasps) Waits here."),
                              ("[gasps] No, no, he did not wait.", "(gasps) No, he did not wait."),
                              ("[gasps] The garden gate is closed.", "(gasps) The golden gate is closed."),
                              ("[gasps] The garden gate is closed.", "(gasps) The Gordon gate is closed.")] {
            XCTAssertFalse(WordCoverage.matches(line: line, transcript: heard, locale: english), line + heard)
        }
        XCTAssertFalse(WordCoverage.matches(line: "[gasps] Please wait here.", transcript: "(gasps) Please wait here.",
                                            locale: english, uncertainSpans: ["Please"]))
        XCTAssertTrue(WordCoverage.matches(line: "[gasps] Please wait here.", transcript: "(gasps) Please wait here.",
                                           locale: english, uncertainSpans: ["(gasps)"]))
    }

    func testOldConfigurationDefaultsToRequiredChecks() throws {
        let old = Data(#"{"temperature":0.7,"maxRetries":0,"minSpokenRatio":0}"#.utf8)
        let config = try JSONDecoder().decode(GenerationConfig.self, from: old)
        XCTAssertEqual(config.wordVerification, .required)
        XCTAssertEqual(config.maxRetries, 0)
        XCTAssertEqual(try GenerationPlan(text: "Hello.", references: [], config: config).streaming, .sentence)
        XCTAssertEqual(GenerationConfig.make(expressiveness: 0.5).wordVerification, .required)
    }

    func testNumericSignsDecimalsAndPercentCannotDisappear() {
        for (written, spoken) in [("-12", "minus twelve"), ("−12.5", "minus twelve point five"),
                                  ("+12.5", "plus twelve point five"), ("12.5%", "twelve point five percent"),
                                  (".5%", "zero point five percent")] {
            XCTAssertTrue(WordCoverage.matches(line: "The result is \(written).", transcript: "The result is \(spoken).", locale: english), written)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The temperature is -12 degrees.", transcript: "The temperature is twelve degrees.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "The result is +12.", transcript: "The result is twelve.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "Revenue rose 12.5%.", transcript: "Revenue rose twelve five.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "Revenue rose 12.5%.", transcript: "Revenue rose twelve point five.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "The result is 12.5.", transcript: "The result is twelve five.", locale: english))
    }

    func testNumericOrdinalsMatchTheirSpokenForms() {
        for (written, spoken) in [("1st", "first"), ("2nd", "second"), ("3rd", "third"), ("4th", "fourth"),
                                  ("11th", "eleventh"), ("12th", "twelfth"), ("13th", "thirteenth"),
                                  ("20th", "twentieth"), ("21st", "twenty first"), ("22nd", "twenty second"),
                                  ("23rd", "twenty third"), ("101st", "one hundred first")] {
            XCTAssertTrue(WordCoverage.matches(line: "The \(spoken) door is open.", transcript: "The \(written) door is open.", locale: english), written)
            XCTAssertTrue(WordCoverage.matches(line: "The \(written) door is open.", transcript: "The \(spoken) door is open.", locale: english), written)
        }
        XCTAssertTrue(WordCoverage.matches(line: "The twenty first door is open.", transcript: "The ２１ＳＴ door is open.", locale: english))
    }

    func testOrdinalFormattingCannotHideWrongNumbersOrMissingWords() {
        for heard in ["The 2nd door is open.", "The 1st door is open again.", "The 1st is open.",
                      "The 1th door is open.", "The 01st door is open.", "The +1st door is open.",
                      "The -1st door is open.", "The - 1st door is open.", "The + 1st door is open.",
                      "The −\n1st door is open.", "The 1st5 door is open."] {
            XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: heard, locale: english), heard)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The first door is not open.", transcript: "The 1st door is open.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: "The 1st door is open.", locale: english, uncertainSpans: ["1st"]))
        XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: "The 1st door is open.", locale: Locale(identifier: "de_DE")))
    }

    func testClockFormattingMatchesOnlyTheExpectedTimeAtItsPosition() {
        let line = "The meeting begins on Monday at nine thirty."
        for heard in ["The meeting begins on Monday at 9.30.", "The meeting begins on Monday at 9:30."] {
            XCTAssertTrue(WordCoverage.matches(line: line, transcript: heard, locale: english), heard)
        }
        XCTAssertTrue(WordCoverage.matches(line: "The meeting begins at 9:30.", transcript: "The meeting begins at nine thirty.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "Meet at nine thirty and leave at ten forty five.", transcript: "Meet at 9.30 and leave at 10.45.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "The value is at nine point three. Meet at nine thirty.",
                                           transcript: "The value is at 9.30. Meet at 9.30.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "[sighs] Meet at nine thirty.", transcript: "Ha! Meet at 9.30.", locale: english))
    }

    func testClockFormattingPreservesDecimalsSignsAndConfidence() {
        let line = "The meeting begins at nine thirty."
        for time in ["9.45", "10.30", "9.300", "9.30%", "9.30kg", "-9.30", "+9.30", "9.30.5", "9:30:45", "29.30"] {
            XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at \(time).", locale: english), time)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The meeting does not begin at nine thirty.", transcript: "The meeting begins at 9.30.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at 9.30.", locale: english, uncertainSpans: ["9.30"]))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at nine point three.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "The value is at 9.30.", transcript: "The value is at nine point three.", locale: english))
        // A written pair "9.30" reads as "nine thirty", as in a price or an Australian clock time.
        XCTAssertTrue(WordCoverage.matches(line: "The value is 9.30.", transcript: "The value is nine thirty.", locale: english))
        XCTAssertTrue(WordCoverage.matches(line: "Set it at nine thirty percent.", transcript: "Set it at 9.30 %.", locale: english))
        for heard in ["Meet at 9.30-5.", "Meet at 9.30/5.", "Meet at 9.30 / 5.", "Meet at 9.30 +5.",
                      "Meet at 9.30 × 5.", "Meet at 9.30 * 5.", "Meet at 9.30 ÷ 5.", "Meet at 9.30 – 5."] {
            XCTAssertFalse(WordCoverage.matches(line: "Meet at nine thirty five.", transcript: heard, locale: english), heard)
        }
        XCTAssertFalse(WordCoverage.matches(line: "Meet at nine thirty zero point five.", transcript: "Meet at 9.30 .5.", locale: english))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at 9.30.", locale: Locale(identifier: "fr_FR")))
    }
}
