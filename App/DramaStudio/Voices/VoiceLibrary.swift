import DramaEngine
import Foundation

@MainActor @Observable
final class VoiceLibrary {
    private(set) var voices: [ReferenceVoice] = []
    private(set) var unreadableFiles: [String] = []
    let directory: URL

    nonisolated static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Voices", directoryHint: .isDirectory)
    }

    init(directory: URL = VoiceLibrary.defaultDirectory) {
        self.directory = directory
        reload()
    }

    func reload() {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        var loaded: [ReferenceVoice] = []
        var failed: [String] = []
        for file in files where file.pathExtension == "json" {
            do {
                loaded.append(try JSONDecoder().decode(ReferenceVoice.self, from: Data(contentsOf: file)))
            } catch {
                failed.append(file.lastPathComponent)
            }
        }
        voices = loaded.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        unreadableFiles = failed.sorted()
    }

    func add(_ voice: ReferenceVoice) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(voice).write(to: file(for: voice.id), options: .atomic)
        reload()
    }

    func remove(_ id: ReferenceVoice.ID) throws {
        try FileManager.default.removeItem(at: file(for: id))
        reload()
    }

    func voice(_ id: ReferenceVoice.ID?) -> ReferenceVoice? {
        voices.first { $0.id == id }
    }

    private func file(for id: ReferenceVoice.ID) -> URL {
        directory.appending(path: "\(id.uuidString).json")
    }
}

extension ReferenceVoice {
    var frameCount: Int { codes.first?.count ?? 0 }

    var frames: [Frame] {
        (0..<frameCount).map { t in
            let column = codes.map { $0[t] }
            return [column[0] + ModelConstants.semanticStart] + column
        }
    }
}
