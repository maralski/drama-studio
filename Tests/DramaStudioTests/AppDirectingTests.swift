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
    var engineDescription: EngineDescription { mock.engineDescription }
    var isTuned: Bool? { mock.isTuned }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding {
        try await mock.encodeVoice(samples: samples, transcript: transcript)
    }
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
                            result.wordCoverage = request.config.wordVerification == .required ? .recognized : .unchecked
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
        let deadline = Date().addingTimeInterval(10)
        while synthesizer.requests.count == before || session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(synthesizer.requests.count, before + 1, "the action sent a request")
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

    func testSaveSelectedModelDefaultTakeReusesItsVoiceAfterEditingAndReopening() async throws {
        let session = makeSession(script: "The first recording stays here.")
        await finish(session.generate, in: session)
        session.script = "These are the words of the voice I want to keep."
        await finish(session.generate, in: session)
        let selected = try XCTUnwrap(session.takes.last)
        session.script = "A different later take."
        await finish(session.generate, in: session)
        session.selectTake(selected.id)
        session.script = "The script has now been edited again."

        XCTAssertTrue(session.saveVoice(fromTake: selected.id, name: "  My narrator  "))
        let saved = try XCTUnwrap(session.library.voices.first)
        XCTAssertEqual(saved.name, "My narrator")
        XCTAssertEqual(saved.transcript, "These are the words of the voice I want to keep.")
        XCTAssertEqual(saved.dramaCodes, selected.take?.dramaCodes)
        XCTAssertEqual(session.voice(for: Speaker(index: 0, name: "Narrator")), .library(saved.id))
        XCTAssertEqual(session.takes.count, 3)
        XCTAssertNotEqual(session.visibleExportTakeID, selected.id, "Saving a voice must not change the export take")
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests.last?.text, "The script has now been edited again.")
        XCTAssertEqual(synthesizer.requests.last?.references, [saved])

        let project = directory.appending(path: "SavedVoice.dramaproj")
        XCTAssertTrue(session.saveProject(to: project))
        let reopened = makeSession()
        XCTAssertTrue(reopened.openProject(from: project))
        reopened.script = "Another script after reopening."
        await finish(reopened.generate, in: reopened)
        XCTAssertEqual(synthesizer.requests.last?.references, [saved])
    }

    func testSavingDesignedRetakeUsesItsOriginalWordsAndSelectedCodes() async throws {
        let session = makeSession(script: "Hello there.")
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        await finish({ session.retake(note: "softer") }, in: session)
        let take = try XCTUnwrap(session.takes.last)
        session.setVoice(.modelDefault, forSpeaker: 0)
        session.script = "Changed words."
        XCTAssertTrue(session.saveVoice(fromTake: take.id, name: "Selected performance"))
        let saved = try XCTUnwrap(session.library.voices.first)
        XCTAssertEqual(saved.transcript, "Hello there.")
        XCTAssertEqual(saved.dramaCodes, take.take?.dramaCodes)
    }

    func testSavedVoiceConditionsASingleNonzeroSpeakerAfterEditing() async throws {
        let session = makeSession(script: "<|speaker:2|>Hello there.")
        await finish(session.generate, in: session)
        let take = try XCTUnwrap(session.takes.last)
        XCTAssertTrue(session.saveVoice(fromTake: take.id, name: "Speaker two"))
        session.script = "<|speaker:2|>A different line."
        await finish(session.generate, in: session)
        XCTAssertTrue(session.casting.ignored.isEmpty)
        let request = try XCTUnwrap(synthesizer.requests.last)
        XCTAssertEqual(request.references.first?.dramaCodes, take.take?.dramaCodes)
        let plan = try GenerationPlan(request)
        let prompt = try Drama3Prompt.render(text: plan.chunks[0].turn, references: plan.references, history: plan.history)
        XCTAssertTrue(prompt.contains("\nText: <|speaker:2|> Hello there.\nSpeech: "), prompt)
        XCTAssertFalse(prompt.contains("\nText: <|speaker:0|>"), prompt)
    }

    func testSavingTakeRejectsInvalidSelectionBlankNameAndMixedSpeakers() async throws {
        let session = makeSession(script: "Hello there.")
        XCTAssertFalse(session.saveVoice(fromTake: UUID(), name: "Missing"))
        await finish(session.generate, in: session)
        let take = try XCTUnwrap(session.takes.last)
        XCTAssertFalse(session.saveVoice(fromTake: take.id, name: " \n "))
        XCTAssertTrue(session.library.voices.isEmpty)
        session.script = "<|speaker:0|>Hello. <|speaker:1|>Goodbye."
        await finish(session.generate, in: session)
        XCTAssertFalse(session.saveVoice(fromTake: try XCTUnwrap(session.takes.last).id, name: "Mixed"))
        XCTAssertTrue(session.library.voices.isEmpty)
        session.script = "Hello. <|speaker:1|>Goodbye."
        await finish(session.generate, in: session)
        XCTAssertFalse(session.saveVoice(fromTake: try XCTUnwrap(session.takes.last).id, name: "Implicit narrator"))
        XCTAssertTrue(session.library.voices.isEmpty)
    }

    func testIncompleteTakeCannotBeSavedAsAVoice() async throws {
        let session = makeSession(script: "Hello there.")
        await finish(session.generate, in: session)
        let project = directory.appending(path: "Incomplete.dramaproj")
        XCTAssertTrue(session.saveProject(to: project))
        let manifest = project.appending(path: "project.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
        var archive = try XCTUnwrap(json["archive"] as? [String: Any])
        archive["plainTakes"] = try XCTUnwrap(archive["plainTakes"] as? [[String: Any]]).map { entry in
            var entry = entry; entry["isIncomplete"] = true; return entry
        }
        json["archive"] = archive
        try JSONSerialization.data(withJSONObject: json).write(to: manifest)
        let reopened = makeSession()
        XCTAssertTrue(reopened.openProject(from: project))
        let take = try XCTUnwrap(reopened.takes.first)
        XCTAssertTrue(take.isIncomplete)
        XCTAssertFalse(reopened.saveVoice(fromTake: take.id, name: "Incomplete"))
        XCTAssertTrue(reopened.library.voices.isEmpty)
    }

    func testVoiceWriteFailureDoesNotChangeTheCast() async throws {
        let session = makeSession(script: "Hello there.")
        await finish(session.generate, in: session)
        let take = try XCTUnwrap(session.takes.first)
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
        try Data("not a directory".utf8).write(to: directory)
        XCTAssertFalse(session.saveVoice(fromTake: take.id, name: "Cannot save"))
        XCTAssertEqual(session.voice(for: Speaker(index: 0, name: "Narrator")), .modelDefault)
        XCTAssertTrue(session.library.voices.isEmpty)
    }

    func testLegacyProjectPreservesAudioAndUnverifiedVoiceWithoutReusingThem() async throws {
        let session = makeSession()
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        let original = try XCTUnwrap(session.takes.first)
        let url = directory.appending(path: "Legacy.dramaproj")
        XCTAssertTrue(session.saveProject(to: url))

        let checked = makeSession()
        XCTAssertTrue(checked.openProject(from: url))
        XCTAssertTrue(try XCTUnwrap(checked.takes.first).wordsChecked)
        XCTAssertEqual(checked.sessionVoices[.designed(design)], original.take)

        let manifest = url.appending(path: "project.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
        var archive = try XCTUnwrap(json["archive"] as? [String: Any])
        var voices = try XCTUnwrap(archive.removeValue(forKey: "recordedVoices") as? [Any])
        for index in stride(from: 1, to: voices.count, by: 2) {
            voices[index] = try XCTUnwrap((voices[index] as? [String: Any])?["take"])
        }
        archive["sessionVoices"] = voices
        archive["plainTakes"] = try XCTUnwrap(archive["plainTakes"] as? [[String: Any]]).map { entry in
            var entry = entry
            entry.removeValue(forKey: "wordsChecked")
            return entry
        }
        json["archive"] = archive
        json["version"] = 2
        try JSONSerialization.data(withJSONObject: json).write(to: manifest)

        let restored = makeSession()
        XCTAssertTrue(restored.openProject(from: url))
        XCTAssertEqual(restored.takes.first?.samples, original.samples)
        XCTAssertFalse(try XCTUnwrap(restored.takes.first).wordsChecked)
        XCTAssertFalse(restored.saveVoice(fromTake: try XCTUnwrap(restored.takes.first).id, name: "Unchecked"))
        XCTAssertTrue(restored.library.voices.isEmpty)
        XCTAssertTrue(restored.sessionVoices.isEmpty)
        let preserved = try XCTUnwrap(restored.projectSnapshot.project.archive?.recordedVoices?[.designed(design)])
        XCTAssertEqual(preserved.take, original.take)
        XCTAssertFalse(preserved.wordsChecked)
        await finish(restored.generate, in: restored)
        XCTAssertEqual(synthesizer.requests.last?.previousTakes, [])
    }

    func testNativeOnlyCheckedProjectKeepsAudioButDoesNotReuseItAsDualChecked() async throws {
        let session = makeSession()
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        let original = try XCTUnwrap(session.takes.first)
        let url = directory.appending(path: "NativeOnly.dramaproj")
        XCTAssertTrue(session.saveProject(to: url))
        let manifest = url.appending(path: "project.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
        var archive = try XCTUnwrap(json["archive"] as? [String: Any])
        archive["plainTakes"] = try XCTUnwrap(archive["plainTakes"] as? [[String: Any]]).map { entry in
            var entry = entry; entry.removeValue(forKey: "wordCheckRevision"); return entry
        }
        var voices = try XCTUnwrap(archive["recordedVoices"] as? [Any])
        for index in stride(from: 1, to: voices.count, by: 2) {
            var value = try XCTUnwrap(voices[index] as? [String: Any])
            value.removeValue(forKey: "wordCheckRevision"); voices[index] = value
        }
        archive["recordedVoices"] = voices; json["archive"] = archive; json["version"] = 3
        try JSONSerialization.data(withJSONObject: json).write(to: manifest)
        let restored = makeSession()
        XCTAssertTrue(restored.openProject(from: url))
        let preserved = try XCTUnwrap(restored.takes.first)
        XCTAssertEqual(preserved.samples, original.samples)
        XCTAssertTrue(preserved.wordsChecked, "Preserve the historical native-only check.")
        XCTAssertFalse(preserved.hasCurrentWordCheck)
        XCTAssertFalse(restored.saveVoice(fromTake: preserved.id, name: "Earlier check"))
        XCTAssertTrue(restored.library.voices.isEmpty)
        XCTAssertTrue(restored.sessionVoices.isEmpty)
        await finish(restored.generate, in: restored)
        XCTAssertEqual(synthesizer.requests.last?.previousTakes, [])
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
        XCTAssertEqual(kept.dramaCodes, take.dramaCodes)
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
        XCTAssertTrue(session.takes[2].take?.text.hasPrefix("Direction: softer and warmer") == true, session.takes[2].take?.text ?? "")
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
        let samples = session.auditionAudio
        XCTAssertEqual(samples.count, audition.take.sampleCount, "Each generated sample is queued once")
        session.stopPreview()
        session.replayAudition()
        XCTAssertTrue(session.isPlayingAudition)
        XCTAssertEqual(session.auditionAudio, samples)
        XCTAssertEqual(synthesizer.requests.count, 1, "Play Again replays the same voice without generating again")

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

    func testDesignPreviewUsesTheSameSettingsAsMainGeneration() async throws {
        let duke = try XCTUnwrap(Examples.cast.first { $0.design.name == "The Duke" })
        let session = makeSession(script: duke.line)
        session.expressiveness = 0.75
        await finish({ session.tryLine(duke.line, as: duke.design) }, in: session)
        session.setVoice(.designed(duke.design), forSpeaker: 0)
        await finish(session.generate, in: session)
        let preview = synthesizer.requests[0]
        let main = synthesizer.requests[1]
        XCTAssertEqual(preview.text, main.text)
        XCTAssertEqual(preview.character, main.character)
        XCTAssertEqual(preview.config, main.config)
        XCTAssertEqual(preview.previousTakes, main.previousTakes)
        XCTAssertEqual(preview.references, main.references)
    }

    func testDesignPreviewChecksWords() async throws {
        let session = makeSession()
        session.expressiveness = 0.25
        await finish({ session.tryLine("[gasps] Please keep the door closed.", as: design) }, in: session)
        let sent = try XCTUnwrap(synthesizer.requests.last)
        XCTAssertEqual(sent.config.wordVerification, .required)
        XCTAssertEqual(sent.config.maxRetries, 5)
        XCTAssertEqual(sent.config.minSpokenRatio, 0.6)
        XCTAssertNil(sent.config.stopAfterSilenceSeconds)
        XCTAssertEqual(sent.config.frameCap, .fromText(margin: 1.5, floorSeconds: 2))
        XCTAssertEqual(sent.config.temperature, 0.6, accuracy: 0.001)
        XCTAssertEqual(session.config.wordVerification, .required)
    }

    func testEveryVoicePresetUsesStructuredSpeakerIdentity() throws {
        XCTAssertFalse(Examples.cast.isEmpty)
        for preset in Examples.cast {
            let prompt = try Drama3Prompt.render(text: preset.line, character: preset.design)
            XCTAssertTrue(prompt.contains("<|im_start|>user\n<|speaker:0|>: " + preset.line), preset.design.name)
            XCTAssertTrue(prompt.contains(preset.design.description), preset.design.name)
            if let scene = preset.design.scene { XCTAssertTrue(prompt.contains(scene), preset.design.name) }
            var renamed = preset.design
            renamed.name = "A custom display name"
            XCTAssertEqual(try Drama3Prompt.render(text: preset.line, character: renamed), prompt,
                           "Renaming \(preset.design.name) must not change what the model speaks")
        }
    }

    func testDesignPreviewPlaysBeforeTheWholeSampleFinishes() async throws {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 200)),
                                   library: VoiceLibrary(directory: directory))
        session.script = "Keep the editor unchanged."
        let sample = "Hello there. " + String(repeating: "This second sentence takes longer to generate ", count: 8) + "."
        session.tryLine(sample, as: design)
        let deadline = Date().addingTimeInterval(5)
        while session.isAuditioning && !session.isPlayingAudition && Date() < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(session.isPlayingAudition, "The first completed sentence should already be playing")
        XCTAssertTrue(session.isAuditioning, "Playback must start before the rest of the sample finishes")
        XCTAssertFalse(session.auditionAudio.isEmpty)
        XCTAssertTrue(session.audio.isEmpty, "Preview audio must not replace the main transport")
        XCTAssertTrue(session.takes.isEmpty)
        XCTAssertEqual(session.script, "Keep the editor unchanged.")
        session.cancelAudition()
        XCTAssertFalse(session.isPlayingAudition)
        XCTAssertTrue(session.auditionAudio.isEmpty)
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
