// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import DramaEngine
import DramaServer
import XCTest
@testable import DramaStudio

/// Writes projects, a recovery draft and a voice file in the format the app
/// saves today, with what the app reads back from each, so a later format can
/// be checked against them. Runs only with DRAMA_WRITE_FORMAT_FIXTURES set to
/// an output folder:
///
///     DRAMA_WRITE_FORMAT_FIXTURES=$PWD/Tests/DramaStudioTests/Fixtures/FormatV4 \
///       swift test --filter FormatFixtureWriter
@MainActor
final class FormatFixtureWriter: XCTestCase {
    func testWriteFixtures() async throws {
        guard let path = ProcessInfo.processInfo.environment["DRAMA_WRITE_FORMAT_FIXTURES"] else {
            throw XCTSkip("set DRAMA_WRITE_FORMAT_FIXTURES to write format fixtures")
        }
        let out = URL(filePath: path, directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: out)
        try FileManager.default.createDirectory(at: out.appending(path: "Voices"), withIntermediateDirectories: true)
        let work = FileManager.default.temporaryDirectory.appending(path: "format-fixtures-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: work) }
        let synthesizer = RecordingSynthesizer()
        func makeSession() -> SessionModel {
            let session = SessionModel(synthesizer: synthesizer, library: VoiceLibrary(directory: out.appending(path: "Voices")),
                                       pronunciationURL: work.appending(path: "pronunciation.json"))
            session.playsWhileGenerating = false
            return session
        }
        func finish(_ session: SessionModel) async {
            while session.state.isGenerating { try? await Task.sleep(for: .milliseconds(10)) }
        }

        let clip = work.appending(path: "ann.wav")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try AudioExporter.write((0..<(44_100 * 3 / 2)).map { Float(sin(Double($0) * 0.05) * 0.3) }, to: clip, as: .wavFloat)
        let sections = makeSession()
        let voice = try await sections.encodeVoice(clip: clip, transcript: "Hello from Ann.", name: "Ann")

        sections.open(NarrationDocument(title: "Deck", source: .pptx, segments: [
            .init(index: 1, label: "Slide 1", text: "Good morning, everyone. Welcome back."),
            .init(index: 2, label: "Slide 2", text: "", skip: true),
            .init(index: 3, label: "Slide 3", text: "A word from our guest.", speakerTag: 1),
            .init(index: 4, label: "Slide 4", text: "Thank you. See you next week."),
        ]))
        sections.setVoice(.library(voice.id), forSpeaker: 1)
        sections.generate()
        await finish(sections)
        sections.selectSegment(sections.document!.segments[0].id)
        sections.narrateSelectedSection()
        await finish(sections)
        XCTAssertTrue(sections.saveProject(to: out.appending(path: "sections.dramaproj")))
        try await RecoveryStore(url: out.appending(path: "Recovery.dramaproj")).update(sections.projectSnapshot)

        let plain = makeSession()
        plain.script = "Ann: First line here. Second line there.\nBo: A reply."
        plain.generate()
        await finish(plain)
        plain.script = "The second take, edited."
        plain.generate()
        await finish(plain)
        var snapshot = plain.projectSnapshot
        let partial = SessionTake(number: 3, note: "Stopped early", take: Take(text: "Cut short", codes: []),
                                  samples: (0..<(5 * 2048)).map { Float(sin(Double($0) * 0.02) * 0.2) }, isIncomplete: true)
        snapshot.project.archive?.plainTakes.append(ArchivedTake(partial))
        snapshot.archivedAudio[partial.id] = partial.samples
        try snapshot.save(to: out.appending(path: "plain.dramaproj"))

        var legacy = plain.projectSnapshot
        legacy.project.version = 3
        legacy.project.archive = nil
        try legacy.save(to: out.appending(path: "legacy-v3.dramaproj"))

        var expected: [String: Any] = ["version": StudioProject.currentVersion, "voice": ["id": voice.id.uuidString, "codes": voice.codes]]
        for name in ["sections", "plain", "legacy-v3", "Recovery"] {
            let reader = makeSession()
            XCTAssertTrue(reader.openProject(from: out.appending(path: "\(name).dramaproj"), recovering: name == "Recovery"))
            let all = reader.takes + reader.sectionTakes.values.flatMap(\.takes)
            expected[name] = [
                "audioSHA": Self.sha(reader.audio),
                "audioCount": reader.audio.count,
                "cues": Captions.cues(for: reader.captionTakes).map { [$0.startSample, $0.endSample, $0.text] as [Any] },
                "spans": reader.captionTakes.flatMap { SentenceSpan.spans(of: $0.sentences, in: $0.sampleRange, segmentID: $0.segmentID) }
                    .map { [$0.range.lowerBound, $0.range.upperBound, $0.text] as [Any] },
                "narration": reader.captionTakes.map { take -> [String: Any] in
                    ["segment": take.segmentID.uuidString, "start": take.sampleRange.lowerBound, "end": take.sampleRange.upperBound,
                     "sentenceSamples": take.sentences.map { $0.frames * ModelConstants.samplesPerFrame }, "codes": take.take?.codes ?? []]
                },
                "takes": Dictionary(uniqueKeysWithValues: all.map { take in
                    (take.id.uuidString, ["text": take.take.text, "codes": take.take.codes, "audioSHA": Self.sha(take.samples),
                                          "sentenceSamples": take.sentences.map { $0.frames * ModelConstants.samplesPerFrame }] as [String: Any])
                }),
            ]
        }
        let data = try JSONSerialization.data(withJSONObject: expected, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: out.appending(path: "expected.json"))
    }

    static func sha(_ samples: [Float]) -> String {
        samples.withUnsafeBytes { SHA256.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
    }
}
