// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation

enum ProjectError: LocalizedError, Equatable {
    case newerVersion(Int)

    var errorDescription: String? {
        switch self {
        case .newerVersion(let version): "This project was saved by a newer Drama Studio (format \(version)). Update the app to open it."
        }
    }
}

struct StudioProject: Codable, Sendable, Equatable {
    static let currentVersion = 2
    static let fileExtension = "dramaproj"

    var version = currentVersion
    var title: String
    var script: String
    var document: NarrationDocument?
    var casting: [Int: UUID] = [:]
    var expressiveness: Double
    var quickPreview: Bool
    var takes: [NarrationTake] = []
    var archive: ProjectArchive?

    /// Writes the bundle beside the destination first and swaps it in, so a
    /// failed save leaves the previous bundle intact and no stale audio survives.
    func save(to bundle: URL, audio: [Float] = [], archivedAudio: [UUID: [Float]] = [:]) throws {
        let fileManager = FileManager.default
        let staging = bundle.deletingLastPathComponent().appending(path: ".\(bundle.lastPathComponent)-\(UUID().uuidString)")
        try fileManager.createDirectory(at: staging.appending(path: "audio"), withIntermediateDirectories: true)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(self).write(to: staging.appending(path: "project.json"))
            if archive == nil && !audio.isEmpty {
                for (offset, take) in takes.enumerated() {
                    let range = take.sampleRange.clamped(to: 0..<audio.count)
                    try Self.write(samples: Array(audio[range]), to: staging.appending(path: Self.audioPath(take: offset)))
                }
            }
            for take in archive?.allTakes ?? [] {
                guard let samples = archivedAudio[take.id] else { throw CocoaError(.fileWriteUnknown) }
                try Self.write(samples: samples, to: staging.appending(path: take.audioPath))
            }
            if fileManager.fileExists(atPath: bundle.path) {
                _ = try fileManager.replaceItemAt(bundle, withItemAt: staging)
            } else {
                try fileManager.moveItem(at: staging, to: bundle)
            }
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    static func load(from bundle: URL) throws -> StudioProject {
        let data = try Data(contentsOf: bundle.appending(path: "project.json"))
        struct Header: Decodable { var version: Int }
        let version = try JSONDecoder().decode(Header.self, from: data).version
        guard version <= currentVersion else { throw ProjectError.newerVersion(version) }
        return try JSONDecoder().decode(StudioProject.self, from: data)
    }

    func audio(in bundle: URL) throws -> [Float] {
        var combined: [Float] = []
        for (offset, take) in takes.enumerated() {
            let path: String
            if let archive {
                let id = document == nil ? (archive.exportTake ?? archive.selectedTake) : archive.sectionExports[take.segmentID]
                guard let id, let stored = archive.allTakes.first(where: { $0.id == id }) else { throw CocoaError(.fileReadCorruptFile) }
                path = stored.audioPath
            } else {
                path = Self.audioPath(take: offset)
            }
            let url = bundle.appending(path: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                if archive != nil { throw CocoaError(.fileReadNoSuchFile) }
                continue
            }
            let samples = try Self.readSamples(from: url)
            let end = take.sampleRange.lowerBound + samples.count
            if combined.count < end { combined += [Float](repeating: 0, count: end - combined.count) }
            combined.replaceSubrange(take.sampleRange.lowerBound..<end, with: samples)
        }
        return combined
    }

    static func audioPath(take offset: Int) -> String {
        String(format: "audio/segment-%02ld.wav", offset + 1)
    }

    static func write(samples: [Float], to url: URL) throws {
        try AudioExporter.write(samples, to: url, as: .wavFloat)
    }

    /// Reads in blocks: one `read(into:)` can return fewer frames than the file holds.
    static func readSamples(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1 << 16)!
        var samples: [Float] = []
        samples.reserveCapacity(Int(file.length))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            guard buffer.frameLength > 0 else { break }
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }
        return samples
    }
}
