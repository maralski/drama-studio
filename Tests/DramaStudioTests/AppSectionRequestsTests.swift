import DramaEngine
import XCTest
@testable import DramaStudio

final class AppSectionRequestsTests: XCTestCase {
    private typealias Segment = NarrationDocument.Segment

    private let voice = ReferenceVoice(name: "Ann", transcript: "a", codes: [[1]])
    private let config = GenerationConfig()

    private func take(_ n: Int32, frames: Int = 1) -> Take {
        Take(text: "take \(n)", codes: Array(repeating: Array(repeating: n, count: frames), count: 10))
    }

    /// Four narrator sections, a second speaker between the second and third.
    private func document() -> NarrationDocument {
        NarrationDocument(title: "Deck", source: .pptx, segments: [
            Segment(index: 1, label: "Slide 1", text: "  One.\n"),
            Segment(index: 2, label: "Slide 2", text: "Two."),
            Segment(index: 3, label: "Slide 3", text: "Aside.", speakerTag: 1),
            Segment(index: 4, label: "Slide 4", text: "Three."),
            Segment(index: 5, label: "Slide 5", text: "Four."),
        ])
    }

    func testQueueSkipsSkippedAndEmptySegments() {
        var document = document()
        document.segments[1].skip = true
        document.segments[3].text = " \n "
        XCTAssertEqual(SectionRequests.queue(document), [document.segments[0].id, document.segments[2].id, document.segments[4].id])
    }

    func testOneRequestPerSegmentWithTrimmedTextAndTheSegmentSpeakersVoice() {
        let document = document()
        let narrator = SectionRequests.casting(forSpeaker: 0, assignments: [0: .modelDefault, 1: .library(voice.id)], library: [voice])
        let request = SectionRequests.request(for: document.segments[0], in: document, casting: narrator, takes: [:], config: config)
        XCTAssertEqual(request.text, "One.")
        XCTAssertEqual(request.references, [])
        XCTAssertNil(request.character)
        XCTAssertEqual(request.previousTakes, [])

        let aside = SectionRequests.casting(forSpeaker: 1, assignments: [0: .modelDefault, 1: .library(voice.id)], library: [voice])
        XCTAssertEqual(aside.references, [voice], "speaker 1's library voice casts speaker 1's section")
        let design = VoiceDesign(name: "Nana", description: "warm")
        let designed = SectionRequests.casting(forSpeaker: 2, assignments: [2: .designed(design)], library: [], sessionVoices: [.designed(design): take(9)])
        XCTAssertEqual(designed.character, design)
        XCTAssertEqual(designed.voiceTake, take(9))
        XCTAssertEqual(SectionRequests.casting(forSpeaker: 3, assignments: [3: .surprise], library: []).recordsVoice, .surprise)
    }

    func testHistoryIsTheFirstAndLatestEarlierSectionOfTheSameSpeaker() {
        let document = document()
        let ids = document.segments.map(\.id)
        let takes = [ids[0]: take(1), ids[1]: take(2), ids[2]: take(7), ids[3]: take(3), ids[4]: take(4)]
        let casting = SectionRequests.casting(forSpeaker: 0, assignments: [:], library: [])
        func previous(_ index: Int) -> [Take] {
            SectionRequests.request(for: document.segments[index], in: document, casting: casting, takes: takes, config: config).previousTakes
        }
        XCTAssertEqual(previous(0), [])
        XCTAssertEqual(previous(1), [take(1)])
        XCTAssertEqual(previous(3), [take(1), take(2)], "section 3 of the narrator: the aside's take is another speaker's")
        XCTAssertEqual(previous(4), [take(1), take(3)], "regenerating a section never uses a later one")
        let aside = SectionRequests.request(for: document.segments[2], in: document, casting: SectionRequests.casting(forSpeaker: 1, assignments: [:], library: []), takes: takes, config: config)
        XCTAssertEqual(aside.previousTakes, [], "speaker 1 has no earlier section")
    }

    func testSessionVoiceAnchorsTheHistory() {
        let document = document()
        let ids = document.segments.map(\.id)
        let voiceTake = take(9)
        let casting = SectionRequests.casting(forSpeaker: 0, assignments: [0: .surprise], library: [], sessionVoices: [.surprise: voiceTake])
        XCTAssertEqual(casting.voiceTake, voiceTake)
        let second = SectionRequests.request(for: document.segments[1], in: document, casting: casting, takes: [ids[0]: voiceTake], config: config)
        XCTAssertEqual(second.previousTakes, [voiceTake], "the first section's take is the session voice, passed once")
        let fourth = SectionRequests.request(for: document.segments[3], in: document, casting: casting, takes: [ids[0]: voiceTake, ids[1]: take(2)], config: config)
        XCTAssertEqual(fourth.previousTakes, [voiceTake, take(2)])
    }

    func testLibraryVoiceKeepsOnlyTheLatestSectionAsHistory() {
        let document = document()
        let ids = document.segments.map(\.id)
        let casting = SectionRequests.casting(forSpeaker: 0, assignments: [0: .library(voice.id)], library: [voice])
        let request = SectionRequests.request(for: document.segments[3], in: document, casting: casting, takes: [ids[0]: take(1), ids[1]: take(2)], config: config)
        XCTAssertEqual(request.references, [voice])
        XCTAssertEqual(request.previousTakes, [take(2)], "the reference is the voice, so no anchor take")
    }

    func testEmptyAndSkippedTakesAreNeverHistory() {
        var document = document()
        let ids = document.segments.map(\.id)
        document.segments[1].skip = true
        let casting = SectionRequests.casting(forSpeaker: 0, assignments: [:], library: [])
        let request = SectionRequests.request(for: document.segments[3], in: document, casting: casting, takes: [ids[0]: take(1, frames: 0), ids[1]: take(2)], config: config)
        XCTAssertEqual(request.previousTakes, [], "an empty take and a skipped section's take are left out")

        let surprise = SectionRequests.casting(forSpeaker: 0, assignments: [0: .surprise], library: [], sessionVoices: [.surprise: take(5, frames: 0)])
        XCTAssertEqual(SectionRequests.request(for: document.segments[0], in: document, casting: surprise, takes: [:], config: config).previousTakes, [])
    }

    func testSpeakersAreTheDistinctSegmentSpeakersNarratorFirst() {
        XCTAssertEqual(SectionRequests.speakers(of: document()).map(\.name), ["Narrator", "Speaker 1"])
        XCTAssertEqual(SectionRequests.speakers(of: document()).map(\.index), [0, 1])
    }

    func testFlattenJoinsNarratedSegmentsAndTagsOnlySeveralSpeakers() {
        var document = document()
        XCTAssertEqual(document.flattened(), "<|speaker:0|>One.\n\n<|speaker:0|>Two.\n\n<|speaker:1|>Aside.\n\n<|speaker:0|>Three.\n\n<|speaker:0|>Four.")
        document.segments[2].skip = true
        document.segments[4].text = ""
        XCTAssertEqual(document.flattened(), "One.\n\nTwo.\n\nThree.", "one speaker left: no tags")
    }
}
