// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class WordCoverageTests: XCTestCase {
    private let english = Locale(identifier: "en_US")
    private let englishLocales = [Locale(identifier: "en_US"), Locale(identifier: "en_AU")]

    /// Checks each pair in every English locale, with either side as the
    /// script, heard by each recognizer profile.
    private func assertMatch(_ pairs: [(String, String)], _ expected: Bool,
                             heardBy profiles: [WordCoverage.RecognizerProfile] = [.exact, .apple],
                             file: StaticString = #filePath, line: UInt = #line) {
        assertHeard(pairs + pairs.map { ($1, $0) }, expected, heardBy: profiles, file: file, line: line)
    }

    /// Checks each (script, transcript) pair in one direction only.
    private func assertHeard(_ pairs: [(String, String)], _ expected: Bool,
                             heardBy profiles: [WordCoverage.RecognizerProfile] = [.exact, .apple],
                             file: StaticString = #filePath, line: UInt = #line) {
        for locale in englishLocales {
            for profile in profiles {
                for (script, heard) in pairs {
                    XCTAssertEqual(WordCoverage.matches(line: script, transcript: heard, locale: locale, profile: profile), expected,
                                   "\(locale.identifier) \(profile): \(script) | \(heard)", file: file, line: line)
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
            ("Meet at 3:30.", "Meet at 3.30."),
            ("In 2026 we left.", "In twenty twenty six we left."),
            ("In 2026 we left.", "In two thousand twenty-six we left."),
            ("In 2026 we left.", "In two thousand and twenty six we left."),
            ("In 1998 we left.", "In nineteen ninety-eight we left."),
            ("In 1905 we left.", "In nineteen oh five we left."),
            ("It costs $4.99.", "It costs four ninety-nine."),
            ("It costs $4.99.", "It costs 4 99."),
            ("They raised $2.5 million.", "They raised 2.5 million dollars."),
            ("They raised $2.5 million.", "They raised two point five million dollars."),
            ("They raised $2,500,000.", "They raised 2.5 million dollars."),
            ("About 100 came.", "About a hundred came."),
            ("About 700 came.", "About seven hundred came."),
            ("Back in the 1990s.", "Back in the nineteen nineties."),
            ("Back in the 90s.", "Back in the '90s."),
            ("Take the 5th door.", "Take the fifth door."),
            ("It fell 12%.", "It fell twelve percent."),
        ], true)
        assertHeard([("Board the 747 now.", "Board the seven forty seven now.")], true)
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
            ("Rooms 5 6 and 7 are free.", "Rooms fifty six and seven are free."),
            ("Board the seven forty seven.", "Board the seven hundred forty seven."),
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
            ("Oh oh oh, look at that.", "Oh, oh, oh, look at that."),
            ("Oh, one more thing.", "Oh one more thing."),
        ], true)
    }

    func testACenturyMatchesOnlyItsOwnHundreds() {
        assertMatch([
            ("The castle dates from the 1600s.", "The castle dates from the sixteen hundreds."),
            ("It was built in the 1900s.", "It was built in the nineteen hundreds."),
        ], true)
        assertHeard([("Back in the 1990s.", "Back in the 90s."), ("Back in the 1990s.", "Back in the nineties.")], true)
        assertHeard([("Back in the 90s.", "Back in the 1990s."), ("Back in the nineties.", "Back in the 1990s."),
                     ("Back in the nineteen nineties.", "Back in the 90s."),
                     ("The castle dates from the 1600s.", "The castle dates from the 00s.")], false)
        assertMatch([
            ("Back in the 1990s.", "Back in the eighteen nineties."),
            ("Back in the 1990s.", "Back in the 1890s."),
            ("Back in the 2020s.", "Back in the nineteen twenties."),
            ("Back in the 2020s.", "Back in the 1920s."),
            ("The castle dates from the 1600s.", "The castle dates from the 1700s."),
            ("The castle dates from the 1600s.", "The castle dates from the seventeen hundreds."),
            ("The castle dates from the 2000s.", "The castle dates from the 1900s."),
            ("The castle dates from the 1600s.", "The castle dates from the sixteen hundred."),
        ], false)
    }

    func testAHyphenJoinsOnlyIdentifiersAndOtherwiseReadsToOrThrough() {
        assertMatch([
            ("Pages 10-20 are missing.", "Pages ten to twenty are missing."),
            ("Read pages 10-20 before class.", "Read pages 10 through 20 before class."),
            ("Read pages 10-20 before class.", "Read pages ten thru twenty before class."),
            ("Pages 10-20 are missing.", "Pages 10 to 20 are missing."),
            ("The final score was 21 to 17.", "The final score was 21-17."),
            ("The final score was twenty-one to seventeen.", "The final score was 21-17."),
            ("Call 555-0142 now.", "Call 5550142 now."),
            ("Call 555-0142 now.", "Call five five five oh one four two now."),
            ("Call 1-800-555-0142 now.", "Call 18005550142 now."),
            ("The answer is 5 - 3.", "The answer is five minus three."),
        ], true)
        assertMatch([
            ("Pages 10-20 are missing.", "Pages 1020 are missing."),
            ("Read pages 10 to 20 before class.", "Read pages 10 through 20 before class."),
            ("Read pages 10-20 before class.", "Read pages 10 through 30 before class."),
            ("Read pages 10-20 before class.", "Read pages 10 and 20 before class."),
            ("The final score was 21-17.", "The final score was 2117."),
            ("The answer is 5 - 3.", "The answer is 53."),
        ], false)
    }

    func testARangeWordMayGoUnsaidOnlyBetweenNumbersReadDigitByDigit() {
        assertMatch([
            ("They had a 2-1 win.", "They had a two one win."),
            ("They had a 2-1 win.", "They had a two to one win."),
            ("They had a 2-1 win.", "They had a 2 1 win."),
            ("She works a 9-5 job.", "She works a nine five job."),
            ("They kept the 3-0 lead.", "They kept the three oh lead."),
        ], true)
        assertMatch([
            ("They had a 2-1 win.", "They had a twenty one win."),
            ("They had a 2-1 win.", "They had a 21 win."),
            ("They had a 2-1 win.", "They had a two two win."),
            ("They had a 2-1 win.", "They had a two and one win."),
            ("Pages 10-20 are missing.", "Pages ten twenty are missing."),
            ("Pages 10-20 are missing.", "Pages 10, 20 are missing."),
            ("The final score was 21-17.", "The final score was twenty one seventeen."),
            ("The answer is 5 - 3.", "The answer is five three."),
        ], false)
    }

    func testNumberWordsWithNoSingleReadingAreSeveralNumbers() {
        assertMatch([
            ("Nothing was the same since 9/11.", "Nothing was the same since nine eleven."),
            ("The shop is open 24/7.", "The shop is open twenty four seven."),
            ("Rooms 7 11 13 are free.", "Rooms seven eleven thirteen are free."),
            ("Rooms 7, 11, 13 are free.", "Rooms seven eleven thirteen are free."),
            ("Take twelve five now.", "Take 12 5 now."),
            ("Take twelve five now.", "Take twelve, five now."),
        ], true)
        assertMatch([
            ("Take twelve five now.", "Take 125 now."),
            ("Take twelve five now.", "Take 12.5 now."),
            ("Take twelve five now.", "Take 1 25 now."),
            ("Rooms 7 11 13 are free.", "Rooms seven one eleven three are free."),
            ("Rooms 7 11 13 are free.", "Rooms seventy one, one hundred thirteen are free."),
            ("The shop is open 24/7.", "The shop is open two forty seven."),
            ("Nothing was the same since 9/11.", "Nothing was the same since ninety one one."),
            ("Board the 747 now.", "Board the seventy four seven now."),
            ("Nothing was the same since nine eleven.", "Nothing was the same since 9, 1."),
        ], false)
    }

    func testABareHundredOrThousandIsANumberOnlyOnItsOwn() {
        assertMatch([
            ("He came third in the hundred metre sprint.", "He came third in the 100 metre sprint."),
            ("They sailed past the thousand islands.", "They sailed past the 1000 islands."),
            ("It was built in the nineteen hundreds.", "It was built in the 1900s."),
            ("There were hundreds of people.", "There were hundreds of people."),
        ], true)
        assertMatch([
            ("He came third in the hundred metre sprint.", "He came third in the 200 metre sprint."),
            ("There were hundreds of people.", "There were 100s of people."),
            ("There were hundreds of people.", "There were 100 of people."),
            ("The code is 25 100.", "The code is two five hundred."),
        ], false)
    }

    func testUnspacedCommaListsAreSeparateNumbers() {
        assertMatch([("Count 3,2,1 and go.", "Count 3, 2, 1 and go."),
                     ("Count 3,2,1 and go.", "Count three, two, one and go."),
                     ("The town has 12,500 residents.", "The town has twelve thousand five hundred residents.")], true)
        assertMatch([("Count 3,2,1 and go.", "Count 321 and go."),
                     ("Count 3,2,1 and go.", "Count three hundred twenty one and go.")], false)
    }

    func testPunctuationInsideASpokenDigitRunDoesNotMatter() {
        assertMatch([
            ("Three, two, one, go!", "Three two one go"),
            ("Three, two, one, go!", "3 2 1 go"),
            ("Three. Two. One. Go!", "3, 2, 1, go!"),
            ("I'll take two. Three if you have them.", "I'll take two three if you have them"),
            ("Table for two. Seven o'clock.", "Table for two seven o'clock"),
            ("Rooms 5, 6 and 7 are free.", "Rooms five six and seven are free."),
            ("Agent 007 reports for duty.", "Agent oh oh seven reports for duty."),
            ("Agent 007 reports for duty.", "Agent zero, zero, seven reports for duty."),
        ], true)
        assertMatch([
            ("Three, two, one, go!", "321 go"),
            ("Three, two, one, go!", "Three hundred twenty one go"),
            ("Rooms 5, 6 and 7 are free.", "Rooms fifty six and seven are free."),
            ("Table for two. Seven o'clock.", "Table for twenty seven o'clock"),
            ("Two, three.", "Twenty three."),
            ("Three two one go", "Three, two, go"),
            ("Agent 007 reports for duty.", "Agent oh seven reports for duty."),
            ("Meet at 10:5.", "Meet at ten five."),
        ], false)
    }

    func testClockTimesAndLeadingZerosDoNotReadAsCardinals() {
        assertMatch([
            ("Meet at 3:30.", "Meet at three hundred and thirty."),
            ("Meet at 1:30.", "Meet at one hundred and thirty."),
            ("Agent 007.", "Agent seven."),
        ], false)
        assertMatch([("Meet at 330.", "Meet at three hundred and thirty.")], true)
    }

    func testAppleFormattingLossesAreForgivenOnlyInAppleTranscripts() {
        let lossy = [
            ("The train leaves at 1:30 this afternoon.", "The train leaves at 130 this afternoon."),
            ("The train leaves at one thirty this afternoon.", "The train leaves at 130 this afternoon."),
            ("Meet at 3:30.", "Meet at 330."),
            ("Meet at 10:30.", "Meet at 1030."),
            ("Board the seven forty seven now.", "Board the 747 now."),
            ("Revenue reached 2.5 million dollars.", "Revenue reached $25 million."),
            ("Revenue reached two point five million dollars.", "Revenue reached $25 million."),
            ("It grew to 12.5 billion.", "It grew to 125 billion."),
            ("It grew to 1.25 billion.", "It grew to 125 billion."),
        ]
        assertHeard(lossy, true, heardBy: [.apple])
        assertHeard(lossy, false, heardBy: [.exact])
        assertHeard([
            ("Revenue reached 3.5 million dollars.", "Revenue reached $25 million."),
            ("Revenue reached 25 million dollars.", "Revenue reached $2.5 million."),
            ("Revenue reached 2 million dollars.", "Revenue reached $20 million."),
            ("It costs 2.5 dollars.", "It costs $25."),
            ("It weighs 2.5 hundred grams.", "It weighs 25 hundred grams."),
            ("The train leaves at 1:30 this afternoon.", "The train leaves at one hundred and thirty this afternoon."),
        ], false)
        assertMatch([
            ("In 1998 we left.", "In nineteen ninety-eight we left."),
            ("In 2026 we left.", "In twenty twenty six we left."),
            ("In 1905 we left.", "In nineteen oh five we left."),
        ], true)
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

    func testAmountsMatchTheirCommonSpokenForms() {
        assertMatch([
            ("It costs $4.99.", "It costs four dollars ninety nine."),
            ("It costs $4.99.", "It costs four dollars and ninety nine cents."),
            ("It costs $5.03.", "It costs five dollars and three cents."),
            ("It costs $5.00.", "It costs five dollars."),
            ("It costs $1.", "It costs a dollar."),
            ("It costs $1.50.", "It costs a dollar fifty."),
            ("It costs $0.99.", "It costs ninety nine cents."),
            ("It costs £5.50.", "It costs five pounds fifty."),
            ("A stamp now costs 50 cents.", "A stamp now costs ¢50."),
            ("A stamp now costs fifty cents.", "A stamp now costs ¢50."),
            ("A stamp now costs 50 cents.", "A stamp now costs 50¢."),
            ("A stamp now costs 50 cents.", "A stamp now costs $.50."),
            ("A stamp now costs 50 cents.", "A stamp now costs $0.50."),
            ("A stamp now costs fifty cents.", "A stamp now costs $.50."),
        ], true)
        assertMatch([
            ("It costs $4.99.", "It costs four dollars nine."),
            ("It costs $4.99.", "It costs nine dollars ninety nine."),
            ("It costs $4.99.", "It costs four dollars ninety."),
            ("It costs $5.50.", "It costs five pounds fifty."),
            ("It costs $1.", "It costs a pound."),
            ("A stamp now costs 50 cents.", "A stamp now costs fifty dollars."),
            ("A stamp now costs $.50.", "A stamp now costs fifty dollars."),
        ], false)
    }

    func testAppleMayWriteCentsAsABareDollarAmount() {
        let cents = [
            ("A stamp now costs 50 cents.", "A stamp now costs $50."),
            ("A stamp now costs ¢50.", "A stamp now costs $50."),
            ("It costs seventy-five cents.", "It costs $75."),
        ]
        assertHeard(cents, true, heardBy: [.apple])
        assertHeard(cents, false, heardBy: [.exact])
        assertHeard([("It costs fifty dollars.", "It costs $50.")], true)
        assertHeard([
            ("It costs $75.", "It costs 75 cents."),
            ("It costs 75 cents.", "It costs $75.00."),
            ("It costs 75 cents.", "It costs $750."),
            ("It costs 75 cents.", "It costs €75."),
            ("It costs 75 cents.", "It costs $76."),
            ("It costs 750 cents.", "It costs $750."),
        ], false)
    }

    func testWrongNumbersAndWordsStillFail() {
        assertMatch([
            ("Deliveries grew by 18% this quarter.", "Deliveries grew by 80% this quarter."),
            ("In 1998 we left.", "In 1989 we left."),
            ("It costs $4.99.", "It costs $9.99."),
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
        assertMatch([("I promise.", "I promised."), ("The boat is in the harbour.", "The boat is in the harbors."),
                     ("The pan is aluminium.", "The pan is aluminum."), ("It is our speciality.", "It is our specialty.")], false)
    }

    func testAUnitAfterANumberMatchesItsAbbreviationButNoOtherUnit() {
        assertMatch([
            ("The parcel weighs 2.75 kilograms.", "The parcel weighs 2.75 kg."),
            ("It weighs 1 kg.", "It weighs one kilogram."),
            ("It weighs 2.75kg.", "It weighs two point seven five kilograms."),
            ("It is 25°C outside.", "It is 25 degrees Celsius outside."),
            ("Drive at 50 km/h.", "Drive at 50 kilometres per hour."),
            ("Drive at 50 mph.", "Drive at fifty miles per hour."),
            ("The board is 6 in wide.", "The board is six inches wide."),
            ("It takes 5 min.", "It takes five minutes."),
        ], true)
        assertMatch([
            ("It is 5 km away.", "It is 5 miles away."),
            ("It weighs 5 kg.", "It weighs 5 g."),
            ("It is 25°C outside.", "It is 25 degrees Fahrenheit outside."),
            ("Add 5 ml.", "Add 5 l."),
            ("It takes 5 min.", "It takes 5 seconds."),
            ("It is 5 km away.", "It is 5 away."),
            ("It weighs 5 kg.", "It weighs 6 kilograms."),
            ("It took hours.", "It took hour."),
            ("Wait a second.", "Wait a sec."),
        ], false)
    }

    func testBritishUnitSpellingsNameTheSameUnit() throws {
        assertMatch([
            ("Add 250 ml of warm milk.", "Add 250 millilitres of warm milk."),
            ("Add 250 ml of warm milk.", "Add 250 milliliters of warm milk."),
            ("Add 250 ml of warm milk.", "Add two hundred fifty millilitres of warm milk."),
            ("It is 5 km away.", "It is 5 kilometres away."),
            ("Pour 2 l of water.", "Pour 2 litres of water."),
            ("Pour 2 cl of rum.", "Pour 2 centilitres of rum."),
        ], true)
        assertMatch([
            ("Add 250 ml of warm milk.", "Add 250 centilitres of warm milk."),
            ("Add 250 ml of warm milk.", "Add 250 millimetres of warm milk."),
            ("It is 5 km away.", "It is 5 metres away."),
        ], false)
        let url = try XCTUnwrap(EngineResources.bundle().url(forResource: "EnglishSpokenForms", withExtension: "json"))
        let units = try JSONDecoder().decode(WordCoverage.EnglishForms.self, from: Data(contentsOf: url)).units.keys
        let metric = units.filter { $0.hasSuffix("ter") || $0.hasSuffix("ters") }
        XCTAssertGreaterThanOrEqual(metric.count, 14)
        assertMatch(metric.map { ("It is 5 \($0).", "It is 5 \($0.replacingOccurrences(of: "ter", with: "tre")).") }, true)
    }

    func testAPossessiveApostropheIsSilentButAContractionIsNot() {
        assertMatch([
            ("My neighbour's favourite flavour.", "My neighbors favorite flavor."),
            ("My neighbours' favourite flavour.", "My neighbor's favorite flavor."),
            ("He's here.", "He is here."),
        ], true)
        assertMatch([
            ("It's tail is long.", "Its tail is long."),
            ("My neighbour's favourite flavour.", "My neighbour favourite flavour."),
            ("My neighbour's favourite flavour.", "My neighbourhood favourite flavour."),
        ], false)
    }

    func testCompoundSpacingMatches() {
        assertMatch([("The website is down.", "The web site is down."), ("Read the ebook.", "Read the e book.")], true)
        assertMatch([("The website is down.", "The web is down.")], false)
    }

    func testANumberGluedToAWordIsTheSameWordAndNumber() {
        assertMatch([("Running the Drama 3 model.", "Running the Drama3 model."),
                     ("Running the Drama three model.", "Running the Drama3 model.")], true)
        assertMatch([("Running the Drama 3 model.", "Running the Drama2 model."),
                     ("Running the Drama 3 model.", "Running the Drama model.")], false)
    }

    func testMissingOpeningSentenceIsRejectedEvenWithALongReaction() {
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] The garden gate is closed. The meeting begins on Monday.",
                                            transcript: "The meeting begins on Monday.", locale: english, profile: .exact))
    }

    func testEveryWordIncludingNegationAndRepetitionsIsRequiredInOrder() {
        let line = "No, no, he did not leave the gate open."
        for heard in ["No he did not leave the gate open", "No no he did leave the gate open",
                      "No no he did not leave the open gate", "No no he did not leave the gate open again"] {
            XCTAssertFalse(WordCoverage.matches(line: line, transcript: heard, locale: english, profile: .exact), heard)
        }
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: line, locale: english, profile: .exact))
    }

    func testReactionsOnlyPermitExtraNonverbalSoundsWithoutDeletingRequiredWords() {
        XCTAssertTrue(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! He left.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "[laughs] He left.", transcript: "Hah! He left.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! Left.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "He left.", transcript: "Ha ha! He left.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "Ha ha! She left.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "[clears throat] He left.", transcript: "Ahem. He left.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] He left.", transcript: "Ahem. Left.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "[clears throat] Ahem, he left.", transcript: "He left.", locale: english, profile: .exact))
    }

    func testCommonEnglishContractionsDoNotLoseNegation() {
        XCTAssertTrue(WordCoverage.matches(line: "I can't leave. What's the plan?", transcript: "I cannot leave. What is the plan?", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "What's the plan?", transcript: "whats the plan?", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "What's the plan?", transcript: "what the plan?", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "I can't leave.", transcript: "I can leave.", locale: english, profile: .exact))
    }

    func testMarkupIsNotSpeechButTaggedWordsAreStillRequired() {
        let line = "<|speaker:2|> [icy calm] <whisper>Wait here.</whisper> Then <emphasis>go.</emphasis>"
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: "Wait here. Then go.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "Wait here.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs]", transcript: "ha ha", locale: english, profile: .exact))
    }

    func testTestedContractionsNumbersAndUnicodeNamesAreEquivalent() {
        XCTAssertTrue(WordCoverage.matches(line: "I’m here. You don't have 24 tickets, Élodie.",
                                           transcript: "I am here. You do not have twenty-four tickets, Elodie.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "You don't have 24 tickets, Élodie.",
                                            transcript: "You have twenty-four tickets, Élodie.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "Peter and Élodie are here.", transcript: "Peter is here.", locale: english, profile: .exact))
    }

    func testAnUncertainMatchingWordCannotCertifySpeech() {
        let line = "[clears throat] The garden gate is closed."
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The garden gate is closed.", locale: english, profile: .exact, uncertainSpans: ["garden"]))
        XCTAssertTrue(WordCoverage.matches(line: line, transcript: "Ha! The garden gate is closed.", locale: english, profile: .exact, uncertainSpans: ["Ha!"]))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The garden gate is closed.", locale: english, profile: .exact, uncertainSpans: ["closed."]))
        XCTAssertFalse(WordCoverage.matches(line: "[laughs] He left.", transcript: "He left.", locale: english, profile: .exact, uncertainSpans: ["He"]))
    }

    func testWrittenTitlesAreEquivalentButAnAcronymIsNotANameTitle() throws {
        try WordCoverage.prepare()
        XCTAssertTrue(WordCoverage.matches(line: "His name is Mister Hops.", transcript: "His name is Mr. Hops.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "Doctor Elodie is here.", transcript: "Dr. Elodie is here.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "MR results are ready.", transcript: "Mister results are ready.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "His name is Mister Hops.", transcript: "His name is Mr. Jones.", locale: english, profile: .exact))
    }

    func testACuedLineDropsAnySoundLabelButNeverAWord() {
        assertHeard([
            ("[gasps] Please wait here.", "*gasp* Please wait here."),
            ("[gasps] Please wait here.", "[Gasping] Please wait here."),
            ("[clears throat] Please wait here.", "(clears throat) Please wait here."),
            ("[whispers] Please wait here.", "(whispering) Please wait here."),
            ("<whisper>Please wait here.</whisper>", "(whispering) Please wait here."),
            ("[laughs] Please wait here.", "*gasp* Please wait here."),
            ("[giggles] Stop it, that tickles!", "(laughs) Stop it, that tickles."),
            ("[scoffs] As if anyone believes that story.", "*sniff* As if anyone believes that story."),
            ("[scoffs] As if anyone believes that story.", "(sighs) As if anyone believes that story."),
            ("[yawns] Is it morning already?", "(exhales) Is it morning already? [ Silence ]"),
            ("[coughs] Sorry, it's the dust in here.", "[coughing] Sorry, it's the dust in here. [BLANK_AUDIO]"),
            ("[icy calm] Put the letter down.", "[SOUND] Put the letter down."),
            ("[sighs] I said never.", "I said *never*."),
        ], true)
        assertHeard([
            ("Please wait here.", "*gasp* Please wait here."),
            ("<|speaker:1|> Please wait here.", "(sighs) Please wait here."),
            ("[gasps] Please wait here.", "Gasp, please wait here."),
            ("[gasps] Please say gasp.", "*gasp* Please say."),
            ("[gasps] He waits here.", "(gasps) Waits here."),
            ("[gasps] No, no, he did not wait.", "(gasps) No, he did not wait."),
            ("[sighs] I suppose we should start again.", "(sigh)"),
            ("[gasps] The door was open the whole time!", "[BLANK_AUDIO]"),
            ("[sighs] I suppose we should start again.", "*Sigh* I suppose we should stop again."),
            ("[coughs] Sorry, it's the dust in here.", "*coughs* Sorry, it's the smoke in here."),
            ("[groans] Not another Monday morning.", "(sigh) Not enough."),
            ("[sighs] I said never.", "I said *ever*."),
        ], false)
        XCTAssertFalse(WordCoverage.matches(line: "[gasps] Please wait here.", transcript: "(gasps) Please wait here.",
                                            locale: english, profile: .exact, uncertainSpans: ["Please"]))
        XCTAssertTrue(WordCoverage.matches(line: "[gasps] Please wait here.", transcript: "(gasps) Please wait here.",
                                           locale: english, profile: .exact, uncertainSpans: ["(gasps)"]))
    }

    func testAnyCueLetsAnInterjectionBeInsertedButNeverReplaceAWord() {
        assertHeard([
            ("[groans] Not another Monday morning.", "Ugh, not another Monday morning."),
            ("[groans] Not another Monday morning.", "Oh. Not another Monday morning."),
            ("[yawns] Is it morning already?", "Ugh. Is it morning already?"),
            ("[icy calm] Put the letter down.", "Mm. Put the letter down."),
            ("[sarcastic and deadpan] Oh good, another fire drill.", "Pfft. Oh good, another fire drill."),
        ], true)
        assertHeard([
            ("Not another Monday morning.", "Ugh, not another Monday morning."),
            ("<|speaker:0|> Not another Monday morning.", "Oh. Not another Monday morning."),
            ("[groans] Not another Monday morning.", "Ugh, not another morning."),
            ("[groans] Oh, not again.", "Ugh, not again."),
            ("[groans] Oh, not again.", "Not again."),
            ("[sarcastic and deadpan] Oh good, another fire drill.", "Oh good, another fire alarm."),
            ("[Enunciate each word precisely] The quick brown fox.", "The quick brown ugh."),
        ], false)
    }

    func testAParenthesizedStageDirectionIsACueButSpokenParenthesesAreWords() {
        let line = "(laughing) That is the worst idea I have ever heard."
        assertHeard([
            (line, "That is the worst idea I have ever heard."),
            (line, "(laughs) That is the worst idea I have ever heard."),
            (line, "Ha ha. That is the worst idea I have ever heard."),
            ("Fine. (Sighs heavily) Let us go.", "Fine. Let us go."),
            ("I know (beat) you did it.", "I know, you did it."),
        ], true)
        assertHeard([
            (line, "That is the worst idea I ever heard."),
            (line, "That is the best idea I have ever heard."),
            ("I told him (twice) already.", "I told him already."),
            ("(And I mean it) go home.", "Go home."),
            ("(She laughs) Go home.", "Go home."),
            ("(laughing at the sheer nerve of it) Go home.", "Go home."),
            ("I told him (twice) already.", "Ugh, I told him twice already."),
        ], false)
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
            XCTAssertTrue(WordCoverage.matches(line: "The result is \(written).", transcript: "The result is \(spoken).", locale: english, profile: .exact), written)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The temperature is -12 degrees.", transcript: "The temperature is twelve degrees.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "The result is +12.", transcript: "The result is twelve.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "Revenue rose 12.5%.", transcript: "Revenue rose twelve five.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "Revenue rose 12.5%.", transcript: "Revenue rose twelve point five.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "The result is 12.5.", transcript: "The result is twelve five.", locale: english, profile: .exact))
    }

    func testNumericOrdinalsMatchTheirSpokenForms() {
        for (written, spoken) in [("1st", "first"), ("2nd", "second"), ("3rd", "third"), ("4th", "fourth"),
                                  ("11th", "eleventh"), ("12th", "twelfth"), ("13th", "thirteenth"),
                                  ("20th", "twentieth"), ("21st", "twenty first"), ("22nd", "twenty second"),
                                  ("23rd", "twenty third"), ("101st", "one hundred first")] {
            XCTAssertTrue(WordCoverage.matches(line: "The \(spoken) door is open.", transcript: "The \(written) door is open.", locale: english, profile: .exact), written)
            XCTAssertTrue(WordCoverage.matches(line: "The \(written) door is open.", transcript: "The \(spoken) door is open.", locale: english, profile: .exact), written)
        }
        XCTAssertTrue(WordCoverage.matches(line: "The twenty first door is open.", transcript: "The ２１ＳＴ door is open.", locale: english, profile: .exact))
    }

    func testOrdinalFormattingCannotHideWrongNumbersOrMissingWords() {
        for heard in ["The 2nd door is open.", "The 1st door is open again.", "The 1st is open.",
                      "The 1th door is open.", "The 01st door is open.", "The +1st door is open.",
                      "The -1st door is open.", "The - 1st door is open.", "The + 1st door is open.",
                      "The −\n1st door is open.", "The 1st5 door is open."] {
            XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: heard, locale: english, profile: .exact), heard)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The first door is not open.", transcript: "The 1st door is open.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: "The 1st door is open.", locale: english, profile: .exact, uncertainSpans: ["1st"]))
        XCTAssertFalse(WordCoverage.matches(line: "The first door is open.", transcript: "The 1st door is open.", locale: Locale(identifier: "de_DE"), profile: .exact))
    }

    func testClockFormattingMatchesOnlyTheExpectedTimeAtItsPosition() {
        let line = "The meeting begins on Monday at nine thirty."
        for heard in ["The meeting begins on Monday at 9.30.", "The meeting begins on Monday at 9:30."] {
            XCTAssertTrue(WordCoverage.matches(line: line, transcript: heard, locale: english, profile: .exact), heard)
        }
        XCTAssertTrue(WordCoverage.matches(line: "The meeting begins at 9:30.", transcript: "The meeting begins at nine thirty.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "Meet at nine thirty and leave at ten forty five.", transcript: "Meet at 9.30 and leave at 10.45.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "The value is at nine point three. Meet at nine thirty.",
                                           transcript: "The value is at 9.30. Meet at 9.30.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "[sighs] Meet at nine thirty.", transcript: "Ha! Meet at 9.30.", locale: english, profile: .exact))
    }

    func testClockFormattingPreservesDecimalsSignsAndConfidence() {
        let line = "The meeting begins at nine thirty."
        for time in ["9.45", "10.30", "9.300", "9.30%", "9.30kg", "-9.30", "+9.30", "9.30.5", "9:30:45", "29.30"] {
            XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at \(time).", locale: english, profile: .exact), time)
        }
        XCTAssertFalse(WordCoverage.matches(line: "The meeting does not begin at nine thirty.", transcript: "The meeting begins at 9.30.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at 9.30.", locale: english, profile: .exact, uncertainSpans: ["9.30"]))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at nine point three.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "The value is at 9.30.", transcript: "The value is at nine point three.", locale: english, profile: .exact))
        // A written pair "9.30" reads as "nine thirty", as in a price or an Australian clock time.
        XCTAssertTrue(WordCoverage.matches(line: "The value is 9.30.", transcript: "The value is nine thirty.", locale: english, profile: .exact))
        XCTAssertTrue(WordCoverage.matches(line: "Set it at nine thirty percent.", transcript: "Set it at 9.30 %.", locale: english, profile: .exact))
        for heard in ["Meet at 9.30-5.", "Meet at 9.30/5.", "Meet at 9.30 / 5.", "Meet at 9.30 +5.",
                      "Meet at 9.30 × 5.", "Meet at 9.30 * 5.", "Meet at 9.30 ÷ 5.", "Meet at 9.30 – 5."] {
            XCTAssertFalse(WordCoverage.matches(line: "Meet at nine thirty five.", transcript: heard, locale: english, profile: .exact), heard)
        }
        XCTAssertTrue(WordCoverage.matches(line: "Meet at nine thirty zero point five.", transcript: "Meet at 9.30 .5.", locale: english, profile: .exact))
        XCTAssertFalse(WordCoverage.matches(line: line, transcript: "The meeting begins at 9.30.", locale: Locale(identifier: "fr_FR"), profile: .exact))
    }
}
