// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation
import XCTest
@testable import DramaStudio

final class AppStudioProjectTests: XCTestCase {
    private func fixture() throws -> (project: StudioProject, audio: [Float]) {
        let document = try PPTXImporter.importPackage(ZipArchive(data: PPTXFixture.package(PPTXFixture.standard)), fallbackTitle: "deck")
        let takes = [
            NarrationTake(segmentID: document.segments[0].id, sampleRange: 0..<(20 * 2048), sentences: [.init(text: "Good morning, everyone.", frames: 12), .init(text: "Today we cover the numbers.", frames: 8)]),
            NarrationTake(segmentID: document.segments[1].id, sampleRange: 61_440..<91_001, sentences: [.init(text: "Revenue grew twelve percent.", frames: 15)]),
        ]
        var audio = [Float](repeating: 0, count: 91_001)
        for take in takes {
            for i in take.sampleRange { audio[i] = Float(sin(Double(i) * 0.01) * 0.5) }
        }
        let project = StudioProject(
            title: "Quarterly Review", script: "<|speaker:0|>Hello.", document: document,
            casting: [0: UUID(), 3: UUID()], expressiveness: 0.7, quickPreview: true, takes: takes
        )
        return (project, audio)
    }

    func testBundleRoundTripsProjectAndAudio() throws {
        let (project, audio) = try fixture()
        let bundle = try temporaryDirectory().appending(path: "Review.dramaproj")
        try project.save(to: bundle, audio: audio)

        let files = try FileManager.default.subpathsOfDirectory(atPath: bundle.path).sorted()
        XCTAssertEqual(files, ["audio", "audio/segment-01.wav", "audio/segment-02.wav", "project.json"])
        let loaded = try StudioProject.load(from: bundle)
        XCTAssertEqual(loaded, project)
        XCTAssertEqual(loaded.version, StudioProject.currentVersion)
        XCTAssertEqual(try loaded.audio(in: bundle), audio, "32-bit float segments rebuild the combined audio exactly, gap included")

        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: bundle.appending(path: "project.json"))) as? [String: Any])
        XCTAssertEqual((json["casting"] as? [String: String]).map { Set($0.keys) }, ["0", "3"], "casting is keyed by speaker index")
    }

    func testResavingReplacesStaleAudio() throws {
        var (project, audio) = try fixture()
        let bundle = try temporaryDirectory().appending(path: "Review.dramaproj")
        try project.save(to: bundle, audio: audio)
        project.takes.removeLast()
        try project.save(to: bundle, audio: audio)

        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: bundle.appending(path: "audio").path), ["segment-01.wav"])
        XCTAssertEqual(try StudioProject.load(from: bundle).takes.count, 1)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: bundle.deletingLastPathComponent().path).allSatisfy { !$0.hasPrefix(".") }, "no staging folder left behind")
    }

    func testProjectWithoutAudioOrDocument() throws {
        let project = StudioProject(title: "Script only", script: "Hi.", expressiveness: 0.5, quickPreview: false)
        let bundle = try temporaryDirectory().appending(path: "Script.dramaproj")
        try project.save(to: bundle)
        XCTAssertEqual(try StudioProject.load(from: bundle), project)
        XCTAssertEqual(try project.audio(in: bundle), [])
    }

    func testNewerFormatIsRejectedWithClearError() throws {
        var (project, _) = try fixture()
        project.version = StudioProject.currentVersion + 1
        let bundle = try temporaryDirectory().appending(path: "Future.dramaproj")
        try project.save(to: bundle)
        XCTAssertThrowsError(try StudioProject.load(from: bundle)) { XCTAssertEqual($0 as? ProjectError, .newerVersion(StudioProject.currentVersion + 1)) }
    }
}
