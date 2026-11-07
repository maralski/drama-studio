// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import DramaEngine
import DramaServer
import XCTest
@testable import DramaStudio

/// Fixtures/FormatV4 holds projects, a recovery draft and a voice file saved by
/// the format 4 app, with what that app read back from each in expected.json.
/// FormatFixtureWriter wrote them at commit 15ac1c1; check that commit out to
/// write them again.
@MainActor
final class AppFormatMigrationTests: XCTestCase {
    private var directory: URL!
    /// A copy in the test's own folder: opening a project adds it to Recent
    /// Projects, which drops it again once the folder is gone.
    private var fixtures: URL!
    private var expected: [String: Any] {
        get throws { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appending(path: "expected.json"))) as? [String: Any]) }
    }

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        fixtures = directory.appending(path: "FormatV4")
        try FileManager.default.copyItem(at: Bundle.module.url(forResource: "FormatV4", withExtension: nil)!, to: fixtures)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession() -> SessionModel {
        let session = SessionModel(synthesizer: RecordingSynthesizer(), library: VoiceLibrary(directory: directory.appending(path: "Voices")),
                                   pronunciationURL: directory.appending(path: "pronunciation.json"))
        session.playsWhileGenerating = false
        return session
    }

    private static func sha(_ samples: [Float]) -> String {
        samples.withUnsafeBytes { SHA256.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
    }

    /// What the app reads back from a project, in the shape expected.json records it.
    private func readBack(_ session: SessionModel) -> [String: Any] {
        let all = session.takes + session.sectionTakes.values.flatMap(\.takes)
        return [
            "audioSHA": Self.sha(session.audio),
            "audioCount": session.audio.count,
            "cues": Captions.cues(for: session.captionTakes).map { [$0.startSample, $0.endSample, $0.text] as [Any] },
            "spans": session.captionTakes.flatMap { SentenceSpan.spans(of: $0.sentences, in: $0.sampleRange, segmentID: $0.segmentID) }
                .map { [$0.range.lowerBound, $0.range.upperBound, $0.text] as [Any] },
            "narration": session.captionTakes.map { take -> [String: Any] in
                ["segment": take.segmentID.uuidString, "start": take.sampleRange.lowerBound, "end": take.sampleRange.upperBound,
                 "sentenceSamples": take.sentences.map(\.samples), "codes": take.take?.dramaCodes ?? []]
            },
            "takes": Dictionary(uniqueKeysWithValues: all.map { take in
                (take.id.uuidString, ["text": take.take?.text ?? "", "codes": take.take?.dramaCodes ?? [], "audioSHA": Self.sha(take.samples),
                                      "sentenceSamples": take.sentences.map(\.samples)] as [String: Any])
            }),
        ]
    }

    /// A project without an archive gets fresh take ids each time it opens, so
    /// its takes compare by content.
    private func assertSame(_ actual: [String: Any], _ expected: Any?, _ message: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let canonical = { (value: Any) throws -> String in
            var value = try XCTUnwrap(value as? [String: Any])
            if let takes = value["takes"] as? [String: Any], takes.count == 1 { value["takes"] = Array(takes.values) }
            return String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self)
        }
        XCTAssertEqual(try canonical(actual), try canonical(try XCTUnwrap(expected)), message, file: file, line: line)
    }

    /// Every stored take object in a saved project.json.
    private static func storedTakes(in project: [String: Any]) -> [[String: Any]] {
        let narration = (project["takes"] as? [[String: Any]] ?? []).compactMap { $0["take"] as? [String: Any] }
        let archive = project["archive"] as? [String: Any] ?? [:]
        let plain = archive["plainTakes"] as? [[String: Any]] ?? []
        let sections = (archive["sectionTakes"] as? [Any] ?? []).compactMap { $0 as? [[String: Any]] }.flatMap { $0 }
        return narration + (plain + sections).compactMap { $0["take"] as? [String: Any] }
    }

    func testFormatFourProjectsAndRecoveryReadBackTheSameTimingCodesAndAudio() throws {
        let expected = try expected
        for name in ["sections", "plain", "legacy-v3", "Recovery"] {
            let session = makeSession()
            XCTAssertTrue(session.openProject(from: fixtures.appending(path: "\(name).dramaproj"), recovering: name == "Recovery"), session.notice ?? name)
            try assertSame(readBack(session), expected[name], "\(name) reads back as format 4 did")
        }
    }

    func testResavedProjectWritesFormatFiveAndReadsBackTheSame() throws {
        let expected = try expected
        for name in ["sections", "plain", "legacy-v3"] {
            let session = makeSession()
            XCTAssertTrue(session.openProject(from: fixtures.appending(path: "\(name).dramaproj")))
            let bundle = directory.appending(path: "\(name).dramaproj")
            XCTAssertTrue(session.saveProject(to: bundle), session.notice ?? name)

            let data = try Data(contentsOf: bundle.appending(path: "project.json"))
            let project = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(project["version"] as? Int, 5, name)
            XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("\"frames\""), "\(name): sentences are timed in samples only")
            let stored = Self.storedTakes(in: project)
            XCTAssertFalse(stored.isEmpty, name)
            XCTAssertTrue(stored.allSatisfy { Set($0.keys) == ["text", "context"] }, "\(name): codes live only inside an engine context")

            let reopened = makeSession()
            XCTAssertTrue(reopened.openProject(from: bundle))
            try assertSame(readBack(reopened), expected[name], "\(name) survives a format 5 save")
        }
    }

    func testFormatFourVoiceLoadsWithItsDramaEncodingAndNoClip() throws {
        let voices = fixtures.appending(path: "Voices")
        let (loaded, unreadable) = VoiceDirectory(url: voices).load()
        XCTAssertEqual(unreadable, [])
        let voice = try XCTUnwrap(loaded.first)
        let recorded = try XCTUnwrap(try expected["voice"] as? [String: Any])
        XCTAssertEqual(voice.id.uuidString, recorded["id"] as? String)
        XCTAssertEqual(voice.dramaCodes, recorded["codes"] as? [[Int32]])
        XCTAssertEqual(voice.encodings.keys.sorted { $0.rawValue < $1.rawValue }, [.drama3])
        XCTAssertNil(voice.clipSampleCount)
        XCTAssertFalse(FileManager.default.fileExists(atPath: voices.appending(path: "\(voice.id.uuidString).wav").path))
        XCTAssertEqual(voice.durationSeconds, Double((voice.dramaCodes?[0].count ?? 0) * 2048) / 44_100)
    }

    func testVoiceClipIsKeptBesideTheVoiceAndDeletedWithIt() async throws {
        let store = VoiceDirectory(url: directory.appending(path: "Voices"))
        let clip = (0..<30_000).map { Float(sin(Double($0) * 0.01)) * 0.5 }
        let voice = ReferenceVoice(name: "Ann", transcript: "Hi.", encodings: [.drama3(codes: Array(repeating: [1, 2], count: 10))])
        try store.add(voice, clip: clip)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: store.url.path).sorted(),
                       ["\(voice.id.uuidString).json", "\(voice.id.uuidString).wav"])
        let readBack = try await store.clip(for: voice.id)
        XCTAssertEqual(readBack, clip, "the clip reads back sample for sample")
        let reloaded = try XCTUnwrap(store.load().voices.first)
        XCTAssertEqual(reloaded.clipSampleCount, 30_000)
        XCTAssertEqual(reloaded.durationSeconds, 30_000 / 44_100)

        try store.remove(voice.id)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: store.url.path), [])
    }

    func testOldVoiceClipIsRecreatedFromItsDramaCodes() async throws {
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let (loaded, _) = VoiceDirectory(url: fixtures.appending(path: "Voices")).load()
        let voice = try XCTUnwrap(loaded.first)
        let clip = try await DramaCodec.recreateClip(of: voice)
        XCTAssertEqual(clip?.count, (voice.dramaCodes?[0].count ?? 0) * ModelConstants.samplesPerFrame)
        let ming = ReferenceVoice(name: "No Drama", transcript: "Hi.", encodings: [])
        let none = try await DramaCodec.recreateClip(of: ming)
        XCTAssertNil(none)
    }
}
