import AVFoundation
import Foundation
import XCTest
@testable import DramaStudio

final class AppSegmentExportTests: XCTestCase {
    private let document = NarrationDocument(title: "Deck", source: .pptx, segments: [
        .init(index: 1, label: "Slide 1: Welcome", text: "a"),
        .init(index: 2, label: "Slide 2: Quarterly results", text: "b"),
        .init(index: 3, label: "Slide 3: Questions, \"live\" (no notes)", text: "c"),
    ])

    private func takes() -> [NarrationTake] {
        [
            NarrationTake(segmentID: document.segments[1].id, sampleRange: 0..<44_100, sentences: [.init(text: "b", frames: 21)]),
            NarrationTake(segmentID: document.segments[2].id, sampleRange: 50_000..<72_050, sentences: [.init(text: "c", frames: 10)]),
            NarrationTake(segmentID: UUID(), sampleRange: 72_050..<72_050, sentences: []),
        ]
    }

    func testFileNamesComeFromTakeOrderAndLabelHead() {
        XCTAssertEqual(SegmentExport.plan(takes: takes(), document: document).map(\.file), ["01-slide-2.wav", "02-slide-3.wav", "03-segment-3.wav"])
        XCTAssertEqual(SegmentExport.plan(takes: takes(), document: document, format: .aac).first?.file, "01-slide-2.m4a")
        XCTAssertEqual(SegmentExport.slug("Chapter 2: Café society"), "chapter-2")
        XCTAssertEqual(SegmentExport.slug("Page 12"), "page-12")
        XCTAssertEqual(SegmentExport.slug("Ünïcode  Title!"), "unicode-title")
        XCTAssertEqual(SegmentExport.slug("::"), "segment")

        let many = (0..<120).map { _ in NarrationTake(segmentID: UUID(), sampleRange: 0..<0, sentences: []) }
        XCTAssertEqual(SegmentExport.plan(takes: many, document: document).last?.file, "120-segment-120.wav")
        XCTAssertEqual(SegmentExport.plan(takes: many, document: document).first?.file, "001-segment-1.wav", "names sort in order")
    }

    func testManifestQuotesLabelsWithCommasAndQuotes() {
        XCTAssertEqual(SegmentExport.manifestCSV(SegmentExport.plan(takes: takes(), document: document)), """
        index,label,seconds,file
        1,Slide 2: Quarterly results,1.000,01-slide-2.wav
        2,"Slide 3: Questions, ""live"" (no notes)",0.500,02-slide-3.wav
        3,Segment 3,0.000,03-segment-3.wav

        """)
    }

    func testExportWritesEachSegmentAndTheManifest() throws {
        let samples = (0..<72_050).map { Float(sin(Double($0) * 0.02) * 0.4) }
        let folder = try temporaryDirectory().appending(path: "export")
        let items = try SegmentExport.export(takes: takes(), document: document, samples: samples, to: folder, format: .wavFloat)

        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), ["01-slide-2.wav", "02-slide-3.wav", "03-segment-3.wav", "manifest.csv"])
        let second = try StudioProject.readSamples(from: folder.appending(path: items[1].file))
        XCTAssertEqual(second, Array(samples[50_000..<72_050]))
        XCTAssertEqual(try AVAudioFile(forReading: folder.appending(path: items[0].file)).length, 44_100)
        XCTAssertEqual(try String(contentsOf: folder.appending(path: "manifest.csv"), encoding: .utf8), SegmentExport.manifestCSV(items))
    }
}
