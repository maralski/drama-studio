import DramaEngine
import DramaServer
import XCTest
@testable import DramaStudio

@MainActor
final class AppNarrationSessionTests: XCTestCase {
    private var directory: URL!
    private var synthesizer: RecordingSynthesizer!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        synthesizer = RecordingSynthesizer()
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func document() -> NarrationDocument {
        NarrationDocument(title: "Deck", source: .pptx, segments: [
            .init(index: 1, label: "Slide 1: Welcome", text: "Good morning, everyone."),
            .init(index: 2, label: "Slide 2: Agenda", text: "First we go through the numbers. Then we take your questions."),
            .init(index: 3, label: "Slide 3", text: "", skip: true),
            .init(index: 4, label: "Slide 4: Aside", text: "A word from our guest.", speakerTag: 1),
            .init(index: 5, label: "Slide 5: Close", text: "Thank you."),
        ])
    }

    private func makeSession(pronunciation: URL? = nil) -> SessionModel {
        let session = SessionModel(synthesizer: synthesizer, library: VoiceLibrary(directory: directory.appending(path: "voices")),
                                   pronunciationURL: pronunciation ?? directory.appending(path: "pronunciation.json"))
        session.playsWhileGenerating = false
        return session
    }

    private func finish(_ session: SessionModel, timeout: Double = 10) async {
        let deadline = Date().addingTimeInterval(timeout)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(session.state.isGenerating, "the run finished")
    }

    func testImportingADeckOpensItAsADocumentWithItsFirstSectionSelected() throws {
        let session = makeSession()
        session.script = "An old script."
        let deck = directory.appending(path: "deck.pptx")
        try PPTXFixture.package(PPTXFixture.standard).write(to: deck)
        session.open(deck)
        XCTAssertNil(session.notice)
        XCTAssertEqual(session.document?.title, "Quarterly Review")
        XCTAssertEqual(session.document?.segments.count, 3)
        XCTAssertEqual(session.selectedSegmentID, session.document?.segments.first?.id)
        XCTAssertEqual(session.editorText, "Good morning, everyone.\nToday we cover the numbers.")
        XCTAssertEqual(session.script, "")
        XCTAssertEqual(session.title, "Quarterly Review")

        let notes = directory.appending(path: "notes.txt")
        try "Plain words.".write(to: notes, atomically: true, encoding: .utf8)
        session.open(notes)
        XCTAssertNil(session.document, "text imports as a plain script")
        XCTAssertEqual(session.script, "Plain words.")
    }

    func testGenerateNarratesEachQueuedSectionWithHistoryFromTheSpeakersEarlierSection() async throws {
        let session = makeSession()
        session.open(document())
        let document = try XCTUnwrap(session.document)
        XCTAssertEqual(session.selectedSegmentID, document.segments[0].id)
        XCTAssertTrue(session.canGenerate)
        XCTAssertEqual(try session.castSpeakers.get().map(\.name), ["Narrator", "Speaker 1"])

        session.generate()
        await finish(session)
        XCTAssertEqual(session.state, .ready)
        let requests = synthesizer.requests
        XCTAssertEqual(requests.map(\.text), ["Good morning, everyone.", "First we go through the numbers. Then we take your questions.", "A word from our guest.", "Thank you."])
        let takes = document.segments.map { session.chosenTake(for: $0.id) }
        XCTAssertEqual(takes.map { $0 != nil }, [true, true, false, true, true], "one take per queued section")
        XCTAssertEqual(requests[1].previousTakes, [takes[0]!.take], "section 2 follows section 1")
        XCTAssertEqual(requests[2].previousTakes, [], "the guest has no earlier section")
        XCTAssertEqual(requests[3].previousTakes, [takes[0]!.take, takes[1]!.take], "the narrator's first and latest sections")

        let narration = session.narrationTakes
        XCTAssertEqual(narration.map(\.segmentID), [0, 1, 3, 4].map { document.segments[$0].id })
        XCTAssertEqual(narration.map(\.sampleRange.lowerBound), narration.indices.map { i in narration[..<i].reduce(0) { $0 + $1.sampleRange.count } })
        XCTAssertEqual(session.audio, narration.compactMap { session.chosenTake(for: $0.segmentID)?.samples }.flatMap { $0 })
        XCTAssertEqual(narration[1].sentences.map(\.text), ["First we go through the numbers.", "Then we take your questions."])
        XCTAssertEqual(narration[1].take, takes[1]?.take, "the timeline carries each section's codes")
        XCTAssertTrue(session.canExportSections)
        XCTAssertTrue(session.canExportSlidePackage)
    }

    func testNarrateThisSectionReplacesOnlyThatSectionsChosenTake() async throws {
        let session = makeSession()
        session.open(document())
        let document = try XCTUnwrap(session.document)
        session.generate()
        await finish(session)
        let before = document.segments.map { session.chosenTake(for: $0.id) }

        session.selectSegment(document.segments[1].id)
        XCTAssertTrue(session.canNarrateSection)
        session.narrateSelectedSection()
        await finish(session)

        XCTAssertEqual(synthesizer.requests.count, 5)
        XCTAssertEqual(synthesizer.requests.last?.text, "First we go through the numbers. Then we take your questions.")
        XCTAssertEqual(synthesizer.requests.last?.previousTakes, [before[0]!.take], "history is the earlier section, never the later ones")
        let section = try XCTUnwrap(session.sectionTakes[document.segments[1].id])
        XCTAssertEqual(section.takes.map(\.number), [1, 2])
        XCTAssertEqual(section.chosenID, section.takes[1].id)
        XCTAssertNotEqual(section.takes[1].samples, before[1]!.samples)
        for index in [0, 3, 4] {
            XCTAssertEqual(session.chosenTake(for: document.segments[index].id)?.id, before[index]?.id, "section \(index + 1) kept its take")
        }
        XCTAssertEqual(session.visibleTakes.count, 2, "the strip shows the selected section's takes")
        XCTAssertEqual(session.audio, session.narrationTakes.compactMap { session.chosenTake(for: $0.segmentID)?.samples }.flatMap { $0 })

        session.selectTake(section.takes[0].id)
        XCTAssertEqual(session.sectionTakes[document.segments[1].id]?.chosenID, section.takes[0].id)
        XCTAssertEqual(session.audio, session.narrationTakes.compactMap { session.chosenTake(for: $0.segmentID)?.samples }.flatMap { $0 })
        XCTAssertEqual(session.playhead, before[0]!.samples.count, "playback starts at the section")
        XCTAssertTrue(session.state.isPlaying)
    }

    func testSkippingASectionRemovesItFromTheAudioAndTheQueue() async throws {
        let session = makeSession()
        session.open(document())
        let document = try XCTUnwrap(session.document)
        session.generate()
        await finish(session)
        let first = try XCTUnwrap(session.chosenTake(for: document.segments[0].id))

        session.setSkip(true, for: document.segments[0].id)
        XCTAssertEqual(session.narrationTakes.first?.segmentID, document.segments[1].id)
        XCTAssertEqual(session.audio.count, session.narrationTakes.reduce(0) { $0 + $1.sampleRange.count })
        XCTAssertFalse(session.audio.starts(with: first.samples.prefix(100)))
        XCTAssertEqual(SectionRequests.queue(try XCTUnwrap(session.document)).count, 3)

        session.generate()
        await finish(session)
        XCTAssertEqual(synthesizer.requests.suffix(3).map(\.text), ["First we go through the numbers. Then we take your questions.", "A word from our guest.", "Thank you."])
        XCTAssertEqual(synthesizer.requests[4].previousTakes, [], "the skipped first section is no longer history")
        XCTAssertEqual(session.chosenTake(for: document.segments[0].id)?.id, first.id, "a skipped section keeps its take")

        session.setSkip(false, for: document.segments[0].id)
        XCTAssertTrue(session.audio.starts(with: first.samples))
    }

    func testFlattenToScriptTurnsTheDocumentIntoAPlainScript() async throws {
        let session = makeSession()
        session.open(document())
        session.generate()
        await finish(session)
        session.flattenToScript()
        XCTAssertNil(session.document)
        XCTAssertEqual(session.script, "<|speaker:0|>Good morning, everyone.\n\n<|speaker:0|>First we go through the numbers. Then we take your questions.\n\n<|speaker:1|>A word from our guest.\n\n<|speaker:0|>Thank you.")
        XCTAssertEqual(session.editorText, session.script)
        XCTAssertTrue(session.sectionTakes.isEmpty)
        XCTAssertTrue(session.audio.isEmpty)
        XCTAssertEqual(session.state, .idle)
        XCTAssertEqual(try session.castSpeakers.get().map(\.index), [0, 1])
    }

    func testSaveThenOpenRoundTripsSourceCastingSettingsTakesAndAudio() async throws {
        let session = makeSession()
        session.open(document())
        let voice = ReferenceVoice(name: "Ann", transcript: "Hi.", codes: Array(repeating: [1, 2, 3], count: 10))
        try session.library.add(voice)
        session.setVoice(.library(voice.id), forSpeaker: 1)
        session.expressiveness = 0.8
        session.quickPreview = true
        session.generate()
        await finish(session)
        let bundle = directory.appending(path: "Deck.dramaproj")
        session.saveProject(to: bundle)
        XCTAssertNil(session.notice)
        XCTAssertEqual(session.recentProjects, [bundle])

        let reopened = makeSession()
        XCTAssertFalse(reopened.canExport)
        reopened.open(bundle)
        XCTAssertNil(reopened.notice)
        XCTAssertEqual(reopened.source, session.source)
        XCTAssertEqual(reopened.speakerVoices, [1: .library(voice.id)])
        XCTAssertEqual(reopened.expressiveness, 0.8)
        XCTAssertTrue(reopened.quickPreview)
        XCTAssertEqual(reopened.narrationTakes, session.narrationTakes, "ranges, sentences and codes survive")
        XCTAssertEqual(reopened.audio, session.audio)
        XCTAssertEqual(reopened.state, .ready)
        XCTAssertTrue(reopened.canExport)
        XCTAssertTrue(reopened.canExportSections)
        XCTAssertEqual(reopened.chosenTake(for: session.document!.segments[1].id)?.duration, session.chosenTake(for: session.document!.segments[1].id)?.duration)
        XCTAssertEqual(reopened.recentProjects, [bundle])

        reopened.selectSegment(session.document!.segments[4].id)
        reopened.narrateSelectedSection()
        await finish(reopened)
        let last = try XCTUnwrap(synthesizer.requests.last)
        XCTAssertEqual(last.previousTakes.count, 2, "a reopened project still has codes for history")

        let folder = directory.appending(path: "slides")
        reopened.exportSlidePackage(to: folder, as: .wav16)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), ["README.txt", "slide-01.wav", "slide-02.wav", "slide-04.wav", "slide-05.wav"])
        let captions = directory.appending(path: "deck.srt")
        reopened.exportCaptions(to: captions, as: .srt)
        XCTAssertTrue(try String(contentsOf: captions, encoding: .utf8).contains("First we go through the numbers."))
    }

    func testPlainScriptProjectRoundTripsItsExportedTake() async throws {
        let session = makeSession()
        session.script = "A plain line that is long enough. Another one that is also long."
        session.generate()
        await finish(session)
        let bundle = directory.appending(path: "Plain.dramaproj")
        session.saveProject(to: bundle)

        let reopened = makeSession()
        reopened.openProject(from: bundle)
        XCTAssertEqual(reopened.script, "A plain line that is long enough. Another one that is also long.")
        XCTAssertEqual(reopened.audio, session.audio)
        XCTAssertEqual(reopened.takes.count, 1)
        XCTAssertEqual(reopened.takes.first?.sentences.map(\.text), ["A plain line that is long enough.", "Another one that is also long."])
        XCTAssertEqual(reopened.captionTakes.first?.sentences, session.captionTakes.first?.sentences)
        XCTAssertTrue(reopened.canExportCaptions)
    }

    func testPronunciationRewritesTheRequestTextOnly() async throws {
        let url = directory.appending(path: "rules.json")
        try PronunciationDictionary(rules: [PronunciationRule(match: "GIF", sayAs: "jif")]).save(to: url)
        let session = makeSession(pronunciation: url)
        session.script = "Open the GIF in the viewer now. Then send it to the team."
        session.generate()
        await finish(session)
        XCTAssertEqual(synthesizer.requests.last?.text, "Open the jif in the viewer now. Then send it to the team.")
        XCTAssertEqual(session.takes.last?.sentences.map(\.text), ["Open the GIF in the viewer now.", "Then send it to the team."], "sentences keep the script's spelling for highlighting")

        session.retake(note: "as a GIF fan")
        await finish(session)
        XCTAssertEqual(synthesizer.requests.last?.note, "as a GIF fan")
        XCTAssertEqual(synthesizer.requests.last?.text, "Open the jif in the viewer now. Then send it to the team.")

        try PronunciationDictionary().save(to: url)
        session.generate()
        await finish(session)
        XCTAssertEqual(synthesizer.requests.last?.text, "Open the GIF in the viewer now. Then send it to the team.", "the dictionary is read again at every Generate")
    }

    func testHighlightNamesTheSentenceAtAPosition() async throws {
        let session = makeSession()
        session.script = "<|speaker:0|> The first sentence is right here. The second sentence ends it all."
        session.generate()
        await finish(session)
        let spans = session.sentenceSpans
        XCTAssertEqual(spans.map(\.text), ["<|speaker:0|> The first sentence is right here.", "<|speaker:0|> The second sentence ends it all."])
        XCTAssertEqual(spans.last?.range.upperBound, session.audio.count)
        XCTAssertEqual(session.highlight(at: 0), SpokenHighlight(segmentID: nil, index: 0, texts: spans.map(\.text)))
        XCTAssertEqual(session.highlight(at: spans[1].range.lowerBound)?.index, 1)
        XCTAssertNil(session.highlight(at: session.audio.count))

        session.open(document())
        session.generate()
        await finish(session)
        let document = try XCTUnwrap(session.document)
        let second = try XCTUnwrap(session.narrationTakes.first { $0.segmentID == document.segments[1].id })
        let highlight = try XCTUnwrap(session.highlight(at: second.sampleRange.lowerBound + 5))
        XCTAssertEqual(highlight, SpokenHighlight(segmentID: document.segments[1].id, index: 0, texts: ["First we go through the numbers.", "Then we take your questions."]))
        XCTAssertEqual(session.playingSegment(at: second.sampleRange.upperBound - 1), document.segments[1].id)
    }

    func testEditorEditsTheSelectedSectionAndCuesGoIntoIt() throws {
        let session = makeSession()
        session.open(document())
        let document = try XCTUnwrap(session.document)
        XCTAssertEqual(session.editorText, "Good morning, everyone.")
        session.selectSegment(document.segments[1].id)
        session.editorText = "Numbers first."
        XCTAssertEqual(session.document?.segments[1].text, "Numbers first.")
        XCTAssertEqual(session.document?.segments[0].text, "Good morning, everyone.")
        session.insert(PerformanceCue.all[0])
        XCTAssertEqual(session.document?.segments[1].text, "Numbers first. [laughs]")
        XCTAssertEqual(session.script, "", "a document has no plain script")
        session.setSpeaker(2, for: document.segments[1].id)
        XCTAssertEqual(try session.castSpeakers.get().map(\.index), [0, 1, 2])
    }
}
