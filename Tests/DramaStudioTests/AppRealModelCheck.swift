// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

/// Drives the app's session on the real Drama 3 model, against a home laid
/// out like `DRAMA_STUDIO_HOME` (Models, Codec, Tuning, Voices) with saved
/// projects in `projects/` and a short clip at `clip.aiff`. Copies everything
/// it touches first. Run under tools/with-gpu-lock:
///
///     DRAMA_APP_CHECK_HOME=<home> swift test --filter AppRealModelCheck
@MainActor
final class AppRealModelCheck: XCTestCase {
    func testOpenPlayGenerateAndCloneOnTheRealModel() async throws {
        guard let path = ProcessInfo.processInfo.environment["DRAMA_APP_CHECK_HOME"] else { throw XCTSkip("set DRAMA_APP_CHECK_HOME") }
        let home = URL(filePath: path, directoryHint: .isDirectory)
        let work = FileManager.default.temporaryDirectory.appending(path: "app-check-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: work) }
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        for item in ["Voices", "projects", "clip.aiff"] {
            try FileManager.default.copyItem(at: home.appending(path: item), to: work.appending(path: item))
        }
        let models = try FileManager.default.contentsOfDirectory(at: home.appending(path: "Models"), includingPropertiesForKeys: nil)
        let engine = DramaSynthesizer(location: ModelLocation(directory: try XCTUnwrap(models.first)),
                                      codecDirectory: home.appending(path: "Codec"), tuningDirectory: home.appending(path: "Tuning"))
        let session = SessionModel(synthesizer: engine, library: VoiceLibrary(directory: work.appending(path: "Voices")),
                                   pronunciationURL: work.appending(path: "pronunciation.json"))

        let projects = try FileManager.default.contentsOfDirectory(at: work.appending(path: "projects"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "dramaproj" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        for project in projects {
            XCTAssertTrue(session.openProject(from: project), session.notice ?? project.lastPathComponent)
            let saved = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: project.appending(path: "project.json"))) as? [String: Any])
            XCTAssertEqual(saved["version"] as? Int, 4, "saved by the format 4 release")
            var expected: [[Int]] = []
            for take in saved["takes"] as? [[String: Any]] ?? [] {
                let range = take["sampleRange"] as? [Int] ?? []
                var cursor = range[0]
                for sentence in take["sentences"] as? [[String: Any]] ?? [] {
                    let end = min(cursor + (sentence["frames"] as? Int ?? 0) * 2048, range[1])
                    expected.append([cursor, end])
                    cursor = end
                }
            }
            let spans = session.captionTakes.flatMap { SentenceSpan.spans(of: $0.sentences, in: $0.sampleRange, segmentID: $0.segmentID) }
            XCTAssertEqual(spans.map { [$0.range.lowerBound, $0.range.upperBound] }, expected, "\(project.lastPathComponent): sentence spans as format 4 timed them")
            let savedCodes = (saved["takes"] as? [[String: Any]] ?? []).map { ($0["take"] as? [String: Any])?["codes"] as? [[Int]] }
            XCTAssertEqual(session.captionTakes.map { $0.take.map { $0.dramaCodes.map { $0.map(Int.init) } } }, savedCodes, "\(project.lastPathComponent): codes")
            print("APP-CHECK opened \(project.lastPathComponent): \(session.audio.count) samples, \(spans.count) sentences")

            session.togglePlayback()
            try await Task.sleep(for: .seconds(1.5))
            XCTAssertTrue(session.state.isPlaying, "\(project.lastPathComponent) plays")
            let position = session.position
            let highlight = try XCTUnwrap(session.highlight(at: position), "a sentence is highlighted at \(position)")
            let span = try XCTUnwrap(session.sentenceSpans.last { $0.range.contains(position) })
            XCTAssertEqual(highlight.texts[highlight.index], span.text)
            print("APP-CHECK playing \(project.lastPathComponent) at \(position): \"\(span.text)\" \(span.range)")
            session.togglePlayback()
        }

        let screenplay = try XCTUnwrap(projects.first { $0.lastPathComponent == "Screenplay.dramaproj" })
        XCTAssertTrue(session.openProject(from: screenplay))
        let takesBefore = session.takes.count
        session.script = "Herald: The lighthouse stood alone on the cliff."
        session.generate()
        let deadline = Date().addingTimeInterval(600)
        while session.state.isGenerating, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertNil(session.notice)
        print("APP-CHECK after generate: state \(session.state) notice \(session.notice ?? "none") prepared \(session.prepared)")
        let result = try XCTUnwrap(session.result)
        XCTAssertEqual(session.takes.count, takesBefore + 1)
        let take = try XCTUnwrap(session.takes.last)
        XCTAssertEqual(take.take?.sampleCount, take.samples.count, "the context's length matches the audio")
        XCTAssertEqual(take.sentences.map(\.samples).reduce(0, +), take.samples.count, "sentence samples add up to the take")
        print("APP-CHECK generated \(take.samples.count) samples, \(take.sentences.count) sentences, coverage \(result.wordCoverage)")

        let voice = try await session.encodeVoice(clip: work.appending(path: "clip.aiff"), transcript: "The lighthouse keeper climbed the stairs every evening to light the lamp.", name: "Check clip")
        let clip = try XCTUnwrap(try session.library.clip(for: voice.id), "the library keeps the clip")
        XCTAssertEqual(voice.clipSampleCount, clip.count)
        XCTAssertEqual(Double(clip.count) / 44_100, 3.74, accuracy: 0.05, "the 22.05 kHz clip is stored at 44.1 kHz")
        XCTAssertNotNil(voice.encodings[.drama3])
        print("APP-CHECK cloned \(voice.name): clip \(clip.count) samples, \(voice.dramaCodes?[0].count ?? 0) frames")

        for previewed in [voice, try XCTUnwrap(session.library.voices.first { $0.clipSampleCount == nil })] {
            session.togglePreview(previewed)
            try await Task.sleep(for: .seconds(3))
            XCTAssertNil(session.notice, previewed.name)
            XCTAssertEqual(session.previewingVoice, previewed.id, "\(previewed.name) previews")
            print("APP-CHECK previewed \(previewed.name), clip stored: \(previewed.clipSampleCount != nil)")
            session.stopPreview()
        }
    }
}
