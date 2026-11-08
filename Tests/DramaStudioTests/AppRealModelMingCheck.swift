// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

/// Drives the app's session on the real Ming model, and once on Drama 3, against
/// the same home as ``AppRealModelCheck`` plus `MING_WEIGHTS`. The home's
/// `Voices` holds at least one voice saved before clips were kept (JSON only,
/// as in a real library); it is copied, never changed. Every line runs
/// with the word check on. The wav of each take lands in `DRAMA_APP_CHECK_OUT`
/// (default: the work folder). Run under tools/with-gpu-lock:
///
///     DRAMA_APP_CHECK_HOME=<home> MING_WEIGHTS=<checkpoint> swift test --filter AppRealModelMingCheck
@MainActor
final class AppRealModelMingCheck: XCTestCase {
    private var session: SessionModel!
    private var out: URL!

    func testCloneCuesContinuityAndAMixedProjectOnTheRealModels() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["DRAMA_APP_CHECK_HOME"] else { throw XCTSkip("set DRAMA_APP_CHECK_HOME") }
        guard let mingPath = environment["MING_WEIGHTS"], case let ming = MingLocation(directory: URL(filePath: mingPath)), ming.isComplete else {
            throw XCTSkip("set MING_WEIGHTS to a complete Ming checkpoint")
        }
        let home = URL(filePath: path, directoryHint: .isDirectory)
        let work = FileManager.default.temporaryDirectory.appending(path: "ming-check-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: work) }
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: home.appending(path: "clip.aiff"), to: work.appending(path: "clip.aiff"))
        out = environment["DRAMA_APP_CHECK_OUT"].map { URL(filePath: $0, directoryHint: .isDirectory) } ?? work.appending(path: "out")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        let voices = work.appending(path: "Voices")
        let library = VoiceLibrary(directory: voices, codecDirectory: home.appending(path: "Codec"))
        let models = try FileManager.default.contentsOfDirectory(at: home.appending(path: "Models"), includingPropertiesForKeys: nil)
        let engines = try SpeechEngines(selected: .ming, factories: [
            .ming: .ming(location: ming),
            .drama3: .drama3(location: ModelLocation(directory: try XCTUnwrap(models.first)), codecDirectory: home.appending(path: "Codec"),
                             tuningDirectory: home.appending(path: "Tuning")),
        ], voices: library)
        session = SessionModel(synthesizer: engines, library: library, pronunciationURL: work.appending(path: "pronunciation.json"))
        let clip = work.appending(path: "clip.aiff")
        let transcript = "The lighthouse keeper climbed the stairs every evening to light the lamp."
        print("MING-CHECK memory at start \(Self.footprintGB()) GB")

        let voice = try await session.encodeVoice(clip: clip, transcript: transcript, name: "Ming clip")
        XCTAssertEqual(Set(voice.encodings.keys), [.ming])
        guard case .ming(let reference)? = voice.encodings[.ming] else { return XCTFail("a Ming encoding") }
        XCTAssertLessThanOrEqual(reference.sampleCount, try XCTUnwrap(voice.clipSampleCount), "the encoding covers the clip up to its trailing silence")
        XCTAssertGreaterThan(reference.sampleCount, try XCTUnwrap(voice.clipSampleCount) * 3 / 4)
        print("MING-CHECK (a) cloned \(voice.name): \(reference.latentRows) latent rows for \(reference.sampleCount) samples")
        session.setVoice(.library(voice.id), forSpeaker: 0)
        let a = try await generate("The lamp was lit before the fog rolled in.", as: "a-clone")
        XCTAssertEqual(a.take.take?.context.engine, .ming)

        let saved = try XCTUnwrap(try FileManager.default.contentsOfDirectory(at: home.appending(path: "Voices"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.sorted { $0.path < $1.path }.first, "a JSON-only voice in the home")
        try FileManager.default.copyItem(at: saved, to: voices.appending(path: saved.lastPathComponent))
        library.reload()
        let old = try XCTUnwrap(library.voices.first { $0.id.uuidString == saved.deletingPathExtension().lastPathComponent })
        let oldClip = voices.appending(path: "\(old.id.uuidString).wav")
        XCTAssertNil(old.clipSampleCount)
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldClip.path), "the old voice starts with no clip")
        session.setVoice(.library(old.id), forSpeaker: 0)
        let b = try await generate("Fog swallowed the harbor by midnight.", as: "b-old-voice")
        XCTAssertEqual(b.take.take?.context.engine, .ming)
        let stored = try XCTUnwrap(library.voice(old.id))
        XCTAssertEqual(Set(stored.encodings.keys), [.drama3, .ming])
        let onDisk = try JSONSerialization.jsonObject(with: Data(contentsOf: work.appending(path: "Voices/\(old.id.uuidString).json"))) as? [String: Any]
        XCTAssertEqual(Set((onDisk?["encodings"] as? [String: Any] ?? [:]).keys), ["drama3", "ming"], "the Ming encoding is on disk")
        let recreated = try ReferenceAudio.load(url: oldClip)
        XCTAssertEqual(recreated.count, (old.dramaCodes?[0].count ?? 0) * ModelConstants.samplesPerFrame, "the clip was recreated from the Drama 3 codes and kept")
        try ReferenceAudio.write(recreated, to: out.appending(path: "b-old-voice-recreated-clip.wav"))
        print("MING-CHECK (b) \(old.name) had no clip; recreated \(recreated.count) samples, now has \(stored.encodings.keys.map(\.rawValue).sorted())")

        let c = try await generate("[laughs] You should have seen his face. [sighs] Then it was over.", as: "c-cues")
        XCTAssertEqual(c.result.notices, [], "every cue maps to a tag")
        XCTAssertEqual(c.take.sentences.count, 2)

        session.setVoice(.surprise, forSpeaker: 0)
        let d1 = try await generate("Nobody answered the door.", as: "d1-first")
        XCTAssertEqual(session.casting.voiceTake, d1.take.take, "the first take is the session voice")
        let d2 = try await generate("So I knocked again, louder this time.", as: "d2-followup")
        XCTAssertEqual(d2.result.notices, [], "the previous Ming take carries the voice")
        session.retake(note: "much angrier")
        let d3 = try await settle(as: "d3-angrier-retake")
        XCTAssertEqual(d3.result.notices, [], "the note names an emotion Ming performs")

        session.setVoice(.designed(VoiceDesign(name: "Captain", description: "gravelly old male voice, weathered and slow")), forSpeaker: 0)
        let f = try await generate("You want to know what happened? Then sit down and listen.", as: "f-design")
        XCTAssertEqual(f.take.sentences.count, 2)
        print("MING-CHECK memory after Ming lines \(Self.footprintGB()) GB")

        session.engine = .drama3
        session.setVoice(.library(voice.id), forSpeaker: 0)
        let e = try await generate("The keeper never spoke of that night again.", as: "e-drama")
        XCTAssertEqual(e.take.take?.context.engine, .drama3)
        XCTAssertEqual(Set(try XCTUnwrap(library.voice(voice.id)).encodings.keys), [.drama3, .ming], "Drama encoded the Ming-cloned voice from its clip")
        print("MING-CHECK memory after the Drama line \(Self.footprintGB()) GB")
        let bundle = work.appending(path: "mixed.dramaproj")
        XCTAssertTrue(session.saveProject(to: bundle), session.notice ?? "save")
        let engineSequence = session.takes.map { $0.take?.context.engine }
        XCTAssertEqual(engineSequence, [.ming, .ming, .ming, .ming, .ming, .ming, .ming, .drama3])

        let reopened = SessionModel(synthesizer: engines, library: library, pronunciationURL: work.appending(path: "pronunciation.json"))
        XCTAssertTrue(reopened.openProject(from: bundle), reopened.notice ?? "open")
        XCTAssertEqual(reopened.takes.map { $0.take?.context.engine }, engineSequence)
        for take in reopened.takes {
            XCTAssertEqual(take.take?.sampleCount, take.samples.count)
            XCTAssertEqual(take.sentences.map(\.samples).reduce(0, +), take.samples.count)
        }
        XCTAssertEqual(reopened.audio.count, session.audio.count)
        reopened.togglePlayback()
        try await Task.sleep(for: .seconds(1.5))
        XCTAssertTrue(reopened.state.isPlaying)
        let position = reopened.position
        let highlight = try XCTUnwrap(reopened.highlight(at: position))
        let span = try XCTUnwrap(reopened.sentenceSpans.last { $0.range.contains(position) })
        XCTAssertEqual(highlight.texts[highlight.index], span.text)
        print("MING-CHECK (e) reopened \(reopened.takes.count) takes, \(reopened.sentenceSpans.count) sentences; playing at \(position): \"\(span.text)\"")
        reopened.togglePlayback()

        session.engine = .ming
        session.setVoice(.library(voice.id), forSpeaker: 0)
        _ = try await generate("And the light kept turning.", as: "g-ming-again")
        print("MING-CHECK memory after returning to Ming \(Self.footprintGB()) GB")
    }

    private func generate(_ script: String, as name: String) async throws -> (take: SessionTake, result: SpeechResult) {
        session.script = script
        session.generate()
        return try await settle(as: name)
    }

    private func settle(as name: String) async throws -> (take: SessionTake, result: SpeechResult) {
        let deadline = Date().addingTimeInterval(600)
        while session.state.isGenerating, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertNil(session.notice, name)
        if case .failed(let message) = session.state { XCTFail("\(name): \(message)") }
        let result = try XCTUnwrap(session.result, name)
        let take = try XCTUnwrap(session.takes.last, name)
        XCTAssertEqual(result.wordCoverage, .recognized, "\(name) passed the word check")
        XCTAssertEqual(take.take?.sampleCount, take.samples.count, "\(name): the context's length matches the audio")
        XCTAssertEqual(take.sentences.map(\.samples).reduce(0, +), take.samples.count, "\(name): sentence samples add up to the take")
        try ReferenceAudio.write(take.samples, to: out.appending(path: "\(name).wav"))
        print("MING-CHECK \(name): \(take.samples.count) samples (\(String(format: "%.2f", take.duration)) s), \(take.sentences.count) sentences, "
              + "\(result.timings.framesGenerated) steps in \(String(format: "%.2f", result.timings.decodeSeconds)) s decode, "
              + "\(String(format: "%.2f", result.timings.prefillSeconds)) s prefill, \(String(format: "%.2f", result.timings.codecSeconds)) s encode, "
              + "\(String(format: "%.2f", result.timings.verificationSeconds)) s verify, speed \(String(format: "%.1f", result.speed))x, notices \(result.notices)")
        return (take, result)
    }

    /// The process's physical footprint (what Activity Monitor calls Memory,
    /// which leaves out clean file-backed pages such as Drama 3's mapped
    /// weights) and its resident size, which counts them.
    private static func footprintGB() -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        guard result == KERN_SUCCESS else { return "unknown" }
        return String(format: "footprint %.2f GB, resident %.2f", Double(info.phys_footprint) / 1e9, Double(info.resident_size) / 1e9)
    }
}
