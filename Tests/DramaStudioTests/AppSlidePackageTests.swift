// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation
import XCTest
@testable import DramaStudio

final class AppSlidePackageTests: XCTestCase {
    private let document = NarrationDocument(title: "Quarterly Review", source: .pptx, segments: [
        .init(index: 1, label: "Slide 1: Welcome", text: "a"),
        .init(index: 2, label: "Slide 2: Numbers", text: "b", skip: true),
        .init(index: 3, label: "Slide 3: Questions", text: "c"),
    ])

    private func takes() -> [NarrationTake] {
        [
            NarrationTake(segmentID: document.segments[0].id, sampleRange: 0..<44_100, sentences: [.init(text: "a", frames: 21)]),
            NarrationTake(segmentID: document.segments[2].id, sampleRange: 44_100..<66_150, sentences: [.init(text: "c", frames: 10)]),
        ]
    }

    func testFilesAreNamedBySlideNumberSoSkippedSlidesLeaveGaps() {
        let plan = SlidePackage.plan(takes: takes(), document: document)
        XCTAssertEqual(plan.files.map(\.name), ["slide-01.wav", "slide-03.wav"])
        XCTAssertEqual(plan.files.map(\.slide), [1, 3])
        XCTAssertEqual(SlidePackage.plan(takes: takes(), document: document, format: .aac).files.map(\.name), ["slide-01.m4a", "slide-03.m4a"])
        let unknown = [NarrationTake(segmentID: UUID(), sampleRange: 0..<10, sentences: [])]
        XCTAssertEqual(SlidePackage.plan(takes: unknown, document: document).files, [])
    }

    func testReadmeExplainsInsertAudioAndListsTheFiles() {
        let readme = SlidePackage.plan(takes: takes(), document: document).readme
        XCTAssertTrue(readme.hasPrefix("Slide narration for Quarterly Review\n"))
        XCTAssertTrue(readme.contains("Insert > Audio > Audio on My PC"))
        XCTAssertTrue(readme.contains("does not change the deck"))
        XCTAssertTrue(readme.contains("slide-01.wav  Slide 1: Welcome  1.0 s\nslide-03.wav  Slide 3: Questions  0.5 s\n"))
    }

    func testExportWritesEachSlideAndTheReadme() throws {
        let samples = (0..<66_150).map { Float(sin(Double($0) * 0.02) * 0.4) }
        let folder = try temporaryDirectory().appending(path: "package")
        let plan = try SlidePackage.export(takes: takes(), document: document, samples: samples, to: folder, format: .wavFloat)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), ["README.txt", "slide-01.wav", "slide-03.wav"])
        XCTAssertEqual(try StudioProject.readSamples(from: folder.appending(path: "slide-03.wav")), Array(samples[44_100..<66_150]))
        XCTAssertEqual(try String(contentsOf: folder.appending(path: "README.txt"), encoding: .utf8), plan.readme)
    }
}

final class AppSentenceLocatorTests: XCTestCase {
    func testFindsSentencesPastTagsAndMarkupInOrder() {
        let text = "<|speaker:0|> Hello there.\n[laughs] How are <whisper>you</whisper>?  Hello there."
        let ranges = SentenceLocator.ranges(of: ["<|speaker:0|> Hello there.", "[laughs] How are <whisper>you</whisper>?", "Hello there."], in: text)
        let ns = text as NSString
        XCTAssertEqual(ranges.map { $0.map(ns.substring(with:)) }, ["Hello there.", "[laughs] How are <whisper>you</whisper>?", "Hello there."])
        XCTAssertEqual(ranges[2]?.location, ns.range(of: "Hello there.", options: .backwards).location, "the repeat is the second occurrence")
    }

    func testCollapsedWhitespaceStillMatchesTheEditorsSpacing() {
        let text = "One  two\n\nthree. Four five."
        let ranges = SentenceLocator.ranges(of: ["One two three.", "Four  five."], in: text)
        XCTAssertEqual(ranges.map { $0.map((text as NSString).substring(with:)) }, ["One  two\n\nthree.", "Four five."])
    }

    func testAMissingSentenceIsNilAndDoesNotMoveTheSearch() {
        let text = "Alpha. Beta."
        let ranges = SentenceLocator.ranges(of: ["Gamma.", "", "Beta.", "Alpha."], in: text)
        XCTAssertEqual(ranges.map { $0.map((text as NSString).substring(with:)) }, [nil, nil, "Beta.", nil])
        XCTAssertEqual(SentenceLocator.ranges(of: ["x"], in: ""), [nil])
    }

    func testRangesAreUTF16OffsetsPastEmojiAndAccents() {
        let text = "Café 🎉 time. Done."
        let ns = text as NSString
        let ranges = SentenceLocator.ranges(of: ["Café 🎉 time.", "Done."], in: text)
        XCTAssertEqual(ranges[0], NSRange(location: 0, length: ns.range(of: ".").location + 1))
        XCTAssertEqual(ranges[1].map(ns.substring(with:)), "Done.")
    }

    func testScriptTextsPreferTheOriginalLinesWhenTheyRewriteToWhatWasSpoken() {
        let rewrite = { (line: String) in line.replacingOccurrences(of: "GIF", with: "jif") }
        XCTAssertEqual(SentenceLocator.scriptTexts(spoken: ["A jif.", "Done."], original: ["A GIF.", "Done."], rewrite: rewrite), ["A GIF.", "Done."])
        XCTAssertEqual(SentenceLocator.scriptTexts(spoken: ["A jif. Done."], original: ["A GIF.", "Done."], rewrite: rewrite), ["A jif. Done."], "different chunking: keep what was spoken")
    }
}

final class AppRecentProjectsTests: XCTestCase {
    func testMostRecentFirstWithoutDuplicatesAndCapped() {
        let urls = (1...12).map { URL(filePath: "/p/\($0).dramaproj") }
        var recent: [URL] = []
        for url in urls { recent = RecentProjects.updated(recent, adding: url) }
        XCTAssertEqual(recent.count, 10)
        XCTAssertEqual(recent.first, urls[11])
        XCTAssertEqual(recent.last, urls[2])
        recent = RecentProjects.updated(recent, adding: URL(filePath: "/p/5.dramaproj/"))
        XCTAssertEqual(recent.count, 10)
        XCTAssertEqual(recent.first?.path, "/p/5.dramaproj")
        XCTAssertEqual(recent.filter { $0.lastPathComponent == "5.dramaproj" }.count, 1)
    }

    func testLoadDropsMissingFiles() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "AppRecentProjectsTests-\(UUID().uuidString)"))
        let existing = try temporaryDirectory().appending(path: "Here.dramaproj")
        try FileManager.default.createDirectory(at: existing, withIntermediateDirectories: true)
        RecentProjects.save([existing, URL(filePath: "/nowhere/Gone.dramaproj")], to: defaults)
        XCTAssertEqual(RecentProjects.load(from: defaults).map(\.lastPathComponent), ["Here.dramaproj"])
    }
}
