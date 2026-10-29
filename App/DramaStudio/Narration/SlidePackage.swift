// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A folder of per-slide audio files for a PowerPoint deck, named by slide
/// number so skipped slides leave gaps, with a README on adding them.
enum SlidePackage {
    struct File: Equatable, Sendable {
        var slide: Int
        var label: String
        var seconds: Double
        var name: String
    }

    struct Plan: Equatable, Sendable {
        var files: [File]
        var readme: String
    }

    static let readmeName = "README.txt"

    static func plan(takes: [NarrationTake], document: NarrationDocument, format: ExportFormat = .wav16) -> Plan {
        let files = slides(takes, in: document).map { take, segment in
            File(slide: segment.index, label: segment.label, seconds: take.seconds,
                 name: String(format: "slide-%02ld.%@", segment.index, format.fileExtension))
        }
        return Plan(files: files, readme: readme(title: document.title, files: files))
    }

    /// Takes paired with their slide, in take order; a take of an unknown segment is left out.
    private static func slides(_ takes: [NarrationTake], in document: NarrationDocument) -> [(NarrationTake, NarrationDocument.Segment)] {
        let segments = Dictionary(document.segments.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        return takes.compactMap { take in segments[take.segmentID].map { (take, $0) } }
    }

    static func readme(title: String, files: [File]) -> String {
        let listing = files.map { String(format: "%@  %@  %.1f s", $0.name, $0.label, $0.seconds) }.joined(separator: "\n")
        return """
        Slide narration for \(title)

        One audio file per slide, named by slide number: slide-01 is slide 1. \
        A slide without a file was skipped. Drama Studio does not change the deck; it only writes these files.

        To add a file to its slide in PowerPoint:
        1. Open the deck and go to the slide.
        2. Choose Insert > Audio > Audio on My PC (on a Mac, Insert > Audio > Audio from File).
        3. Pick the slide's file from this folder.
        4. Under Playback, choose Start: Automatically and Hide During Show if the narration should play by itself.

        Files:
        \(listing)

        """
    }

    @discardableResult
    static func export(takes: [NarrationTake], document: NarrationDocument, samples: [Float], to folder: URL, format: ExportFormat = .wav16) throws -> Plan {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let plan = plan(takes: takes, document: document, format: format)
        for (file, (take, _)) in zip(plan.files, slides(takes, in: document)) {
            try AudioExporter.write(Array(samples[take.sampleRange.clamped(to: 0..<samples.count)]), to: folder.appending(path: file.name), as: format)
        }
        try Data(plan.readme.utf8).write(to: folder.appending(path: readmeName), options: .atomic)
        return plan
    }
}
