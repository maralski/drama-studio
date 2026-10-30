// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import DramaEngine
import XCTest
@testable import DramaStudio

/// The mock engine, keeping every request it was sent. Request n's audio is
/// scaled by n, so takes of the same text have different samples.
final class RecordingSynthesizer: SpeechSynthesizer, @unchecked Sendable {
    private let mock = MockSynthesizer(pace: .instant)
    private let lock = NSLock()
    private var sent: [SpeechRequest] = []

    var requests: [SpeechRequest] { lock.withLock { sent } }

    func load(onPhase: @escaping @Sendable (String) -> Void) async throws { try await mock.load(onPhase: onPhase) }
    var isLoaded: Bool { get async { await mock.isLoaded } }
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        try await mock.encodeReference(audioURL: audioURL, transcript: transcript, name: name)
    }
    func decode(frames: [Frame]) async throws -> [Float] { try await mock.decode(frames: frames) }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        let gain = Float(lock.withLock {
            sent.append(request)
            return sent.count
        })
        let events = mock.synthesize(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in events {
                        switch event {
                        case .partialAudio(let samples, let index, let frames):
                            continuation.yield(.partialAudio(samples: samples.map { $0 * gain }, chunkIndex: index, frameRange: frames))
                        case .audio(let samples, let index): continuation.yield(.audio(samples: samples.map { $0 * gain }, chunkIndex: index))
                        case .finished(var result):
                            result.samples = result.samples.map { $0 * gain }
                            continuation.yield(.finished(result))
                        default: continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@MainActor
final class AppDirectingTests: XCTestCase {
    private var directory: URL!
    private var synthesizer: RecordingSynthesizer!
    private let design = VoiceDesign(name: "Nana", description: "warm elderly woman", scene: "a quiet bedroom")

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        synthesizer = RecordingSynthesizer()
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession(script: String = "[sighs] Once there was a lighthouse.") -> SessionModel {
        let session = SessionModel(synthesizer: synthesizer, library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        session.script = script
        return session
    }

    private func finish(_ action: () -> Void, in session: SessionModel) async {
        let before = synthesizer.requests.count
        action()
        XCTAssertEqual(synthesizer.requests.count, before + 1, "the action sent a request")
        let deadline = Date().addingTimeInterval(10)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testSurpriseVoiceIsReusedUntilItIsPickedAgain() async {
        let session = makeSession()
        session.setVoice(.surprise, forSpeaker: 0)
        await finish(session.generate, in: session)
        let first = synthesizer.requests[0]
        XCTAssertEqual(first.previousTakes, [])
        XCTAssertNil(first.character)
        let surprise = session.result?.take
        XCTAssertNotNil(surprise)
        XCTAssertEqual(session.sessionVoices[.surprise], surprise)

        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [surprise!], "the second take reuses the first take's voice")
        XCTAssertEqual(session.sessionVoices[.surprise], surprise, "a later take does not replace the session voice")

        session.setVoice(.modelDefault, forSpeaker: 0)
        session.setVoice(.surprise, forSpeaker: 0)
        XCTAssertNil(session.sessionVoices[.surprise])
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[2].previousTakes, [], "picking Surprise me again gives a fresh surprise")
    }

    func testDesignedVoiceSendsTheCharacterAndThenItsFirstTake() async {
        let session = makeSession()
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[0].character, design)
        XCTAssertEqual(synthesizer.requests[0].references.count, 0)
        let voice = session.result!.take
        XCTAssertTrue(voice.text.hasPrefix("Nana: "), voice.text)

        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[1].character, design)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [voice])
    }

    func testKeepThisVoiceAddsTheSpokenTranscriptToTheLibrary() async throws {
        let session = makeSession()
        let speaker = try session.prepared.get().speakers[0]
        session.setVoice(.designed(design), for: speaker)
        XCTAssertFalse(session.canKeepVoice(for: speaker), "nothing to keep before the first take")
        await finish(session.generate, in: session)
        let take = try XCTUnwrap(session.result?.take)
        XCTAssertTrue(session.canKeepVoice(for: speaker))

        session.keepVoice(for: speaker)
        let kept = try XCTUnwrap(session.library.voices.first)
        XCTAssertEqual(kept.name, "Nana")
        XCTAssertEqual(kept.transcript, "Once there was a lighthouse.")
        XCTAssertEqual(kept.codes, take.codes)
        XCTAssertEqual(session.voice(for: speaker), .library(kept.id))
        XCTAssertEqual(session.casting.references, [kept])
        XCTAssertNil(session.casting.character)
        XCTAssertTrue(session.sessionVoices.isEmpty)
    }

    func testTakesAreNumberedAndPreservedWhenTheScriptChanges() async {
        let session = makeSession()
        await finish(session.generate, in: session)
        await finish(session.generate, in: session)
        XCTAssertEqual(session.takes.map(\.number), [1, 2])
        XCTAssertEqual(session.takes.map(\.note), [nil, nil])
        XCTAssertEqual(session.selectedTakeID, session.takes[1].id)
        XCTAssertEqual(session.takes[1].samples, session.audio)
        session.selectTake(session.takes[0].id)
        session.useSelectedTakeForExport()
        XCTAssertEqual(session.keptTakeID, session.takes[0].id)

        session.script += " The end."
        XCTAssertFalse(session.canRetake, "takes of the old text can't be retaken")
        XCTAssertNotNil(session.retakeBlocker)
        await finish(session.generate, in: session)
        XCTAssertEqual(session.takes.map(\.number), [1, 2, 3])
        XCTAssertEqual(session.keptTakeID, session.takes[0].id)
        XCTAssertTrue(session.isOutdated(session.takes[0]))
        XCTAssertFalse(session.isOutdated(session.takes[2]))
    }

    func testChangingTheVoicePreservesEarlierTakes() async {
        let session = makeSession()
        await finish(session.generate, in: session)
        XCTAssertTrue(session.canRetake)
        session.setVoice(.designed(design), forSpeaker: 0)
        XCTAssertFalse(session.canRetake, "a retake would clone the first take, which is the old voice")
        XCTAssertTrue(session.retakeBlocker?.contains("voice changed") == true)
        await finish(session.generate, in: session)
        XCTAssertEqual(session.takes.map(\.number), [1, 2])
        XCTAssertTrue(session.canRetake)
    }

    func testRetakeSendsTheNoteAfterEveryTakeSoFar() async throws {
        let session = makeSession()
        XCTAssertFalse(session.canRetake, "a retake needs a take")
        await finish(session.generate, in: session)
        await finish(session.generate, in: session)
        let earlier = session.takes.map(\.take)
        await finish({ session.retake(note: "  softer [and] warmer ") }, in: session)

        let request = try XCTUnwrap(synthesizer.requests.last)
        XCTAssertEqual(request.note, "softer and warmer")
        XCTAssertEqual(request.text, try session.prepared.get().text)
        XCTAssertEqual(request.previousTakes, earlier)
        XCTAssertEqual(session.takes.map(\.number), [1, 2, 3])
        XCTAssertEqual(session.takes.last?.note, "softer and warmer")
        XCTAssertEqual(session.selectedTakeID, session.takes.last?.id)
        XCTAssertTrue(session.takes[2].take.text.hasPrefix("Direction: softer and warmer"), session.takes[2].take.text)
    }

    func testRetakeLeadsWithTheSessionVoiceUnlessItIsTheFirstTake() async throws {
        let session = makeSession()
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        let voice = session.takes[0].take
        await finish({ session.retake(note: "slower") }, in: session)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [voice], "the voice is already the first take")
        XCTAssertEqual(synthesizer.requests[1].character, design)

        session.script = "A new line to read."
        await finish(session.generate, in: session)
        let first = session.takes.last!.take
        XCTAssertNotEqual(first, voice)
        await finish({ session.retake(note: "angrier") }, in: session)
        XCTAssertEqual(synthesizer.requests[3].previousTakes, [voice, first])
    }

    func testRetakeIsOffForSeveralSpeakers() async {
        let session = makeSession(script: "ANN: Hi there.\nBEN: Hello.")
        session.screenplayMode = true
        await finish(session.generate, in: session)
        XCTAssertEqual(session.takes.count, 1)
        XCTAssertFalse(session.canRetake)
        XCTAssertTrue(session.retakeBlocker?.contains("one speaker") == true)
        session.retake(note: "softer")
        XCTAssertEqual(synthesizer.requests.count, 1)
    }

    func testExportWritesTheKeptTake() async throws {
        let session = makeSession()
        await finish(session.generate, in: session)
        await finish(session.generate, in: session)
        let first = session.takes[0]
        XCTAssertNotEqual(first.samples, session.takes[1].samples)
        XCTAssertTrue(session.exportSamples == first.samples, "auditioning a new take does not change the export selection")

        session.selectTake(first.id)
        session.useSelectedTakeForExport()
        session.selectTake(session.takes[1].id)
        XCTAssertEqual(session.audio, session.takes[1].samples)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "kept.wav")
        session.export(to: url, as: .wav16)
        let written = try AVAudioFile(forReading: url)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: written.processingFormat, frameCapacity: AVAudioFrameCount(written.length)))
        try written.read(into: buffer)
        let peak = UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)).map(abs).max() ?? 0
        XCTAssertEqual(Int(written.length), first.samples.count)
        XCTAssertEqual(peak, first.samples.map(abs).max() ?? 0, accuracy: 0.001, "the kept take's samples, not the selected one's")
    }

    func testTriedLineBecomesTheSessionVoiceOnlyForTheSameDesign() async throws {
        let session = makeSession()
        let speaker = try session.prepared.get().speakers[0]
        await finish({ session.tryLine("[sighs] Off to bed now.", as: design) }, in: session)
        let audition = try XCTUnwrap(session.audition)
        let tried = synthesizer.requests[0]
        XCTAssertEqual(tried.text, "[sighs] Off to bed now.")
        XCTAssertEqual(tried.character, design)
        XCTAssertEqual(tried.previousTakes, [])
        XCTAssertTrue(session.takes.isEmpty, "trying a line adds no take")
        XCTAssertTrue(session.audio.isEmpty, "trying a line leaves the transport alone")
        XCTAssertEqual(session.script, "[sighs] Once there was a lighthouse.")
        XCTAssertEqual(session.state, .idle)

        session.useDesign(design, for: speaker)
        XCTAssertEqual(session.voice(for: speaker), .designed(design))
        XCTAssertEqual(session.sessionVoices[.designed(design)], audition.take)
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [audition.take], "the voice heard is the voice used")

        var changed = design
        changed.description = "brisk young woman"
        await finish({ session.tryLine("Hello.", as: design) }, in: session)
        session.useDesign(changed, for: speaker)
        XCTAssertNil(session.sessionVoices[.designed(changed)], "a changed design was not heard")
    }

    func testReplacingAnAuditionedVoiceMarksTheSavedProjectDirty() async throws {
        let session = makeSession()
        let speaker = try session.prepared.get().speakers[0]
        session.setVoice(.designed(design), for: speaker)
        await finish(session.generate, in: session)
        XCTAssertTrue(session.saveProject(to: directory.appending(path: "Voice.dramaproj")))
        XCTAssertFalse(session.hasUnsavedChanges)
        await finish({ session.tryLine("A different sample to choose a new voice.", as: design) }, in: session)
        session.useDesign(design, for: speaker)
        XCTAssertTrue(session.hasUnsavedChanges)
        XCTAssertTrue(session.isOutdated(session.takes[0]))
    }

    func testTryingALineCountsAsBusy() {
        let session = makeSession()
        session.tryLine("Hello there.", as: design)
        XCTAssertTrue(session.state.isGenerating, "the API's busy check reads this")
        XCTAssertTrue(session.isAuditioning)
        session.cancelAudition()
        XCTAssertFalse(session.state.isGenerating)
        XCTAssertNil(session.audition)
    }

    func testCastPresetReplacesTheScriptAndDesignsTheNarrator() throws {
        let session = makeSession(script: "ANN: Hi.\nBEN: Hello.")
        session.screenplayMode = true
        let nana = try XCTUnwrap(Examples.cast.first { $0.id == "Nana" })
        session.cast(nana)
        XCTAssertEqual(session.script, nana.line)
        XCTAssertFalse(session.screenplayMode)
        let speakers = try session.prepared.get().speakers
        XCTAssertEqual(speakers.count, 1)
        XCTAssertEqual(session.voice(for: speakers[0]), .designed(nana.design))
        XCTAssertEqual(session.casting.character, nana.design)
    }

    func testDesignDraftNeedsANameAndADescription() {
        var draft = DesignDraft(editing: nil, line: "")
        XCTAssertNil(draft.design)
        draft.name = " Ivy "
        XCTAssertNil(draft.design)
        draft.description = "bright, quick"
        XCTAssertEqual(draft.design, VoiceDesign(name: "Ivy", description: "bright, quick"))
        draft.fill(Examples.cast[0])
        XCTAssertEqual(draft.design, Examples.cast[0].design)
        XCTAssertEqual(draft.line, Examples.cast[0].line)
        XCTAssertEqual(DesignDraft(editing: design, line: "x").design, design)
    }

    func testKeptSurpriseIsNamedSurpriseVoice() async throws {
        let session = makeSession(script: "Hello there.")
        let speaker = try session.prepared.get().speakers[0]
        session.setVoice(.surprise, for: speaker)
        await finish(session.generate, in: session)
        session.keepVoice(for: speaker)
        XCTAssertEqual(session.library.voices.map(\.name), ["Surprise voice"])
        XCTAssertEqual(session.library.voices.first?.transcript, "Hello there.")
    }
}
