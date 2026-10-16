import DramaEngine
import Foundation

/// Where the API keeps reference voices. The app passes its `VoiceLibrary`
/// so voices added over HTTP show up in the window; the CLI and tests use a
/// `VoiceDirectory`.
public protocol VoiceStore: Sendable {
    func allVoices() async -> [ReferenceVoice]
    func add(_ voice: ReferenceVoice) async throws
    func remove(_ id: ReferenceVoice.ID) async throws
}

public struct VoiceDirectory: VoiceStore {
    public let url: URL

    public static var defaultURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Voices", directoryHint: .isDirectory)
    }

    public init(url: URL = VoiceDirectory.defaultURL) { self.url = url }

    public func load() -> (voices: [ReferenceVoice], unreadable: [String]) {
        let files = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        var voices: [ReferenceVoice] = []
        var unreadable: [String] = []
        for file in files where file.pathExtension == "json" {
            do {
                voices.append(try JSONDecoder().decode(ReferenceVoice.self, from: Data(contentsOf: file)))
            } catch {
                unreadable.append(file.lastPathComponent)
            }
        }
        return (voices.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }, unreadable.sorted())
    }

    public func allVoices() async -> [ReferenceVoice] { load().voices }

    public func add(_ voice: ReferenceVoice) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try JSONEncoder().encode(voice).write(to: file(for: voice.id), options: .atomic)
    }

    public func remove(_ id: ReferenceVoice.ID) throws {
        try FileManager.default.removeItem(at: file(for: id))
    }

    private func file(for id: ReferenceVoice.ID) -> URL {
        url.appending(path: "\(id.uuidString).json")
    }
}

public enum VoiceLookup {
    public static func find(_ key: VoiceKey, in voices: [ReferenceVoice]) -> Result<ReferenceVoice, APIError> {
        if let id = UUID(uuidString: key), let voice = voices.first(where: { $0.id == id }) { return .success(voice) }
        for matches in [voices.filter { $0.name == key }, voices.filter { $0.name.caseInsensitiveCompare(key) == .orderedSame }] {
            if matches.count == 1 { return .success(matches[0]) }
            if matches.count > 1 { return .failure(.bad("More than one voice is named \"\(key)\"; pass its id.")) }
        }
        return .failure(.bad("No voice \"\(key)\" in the library. GET /v1/voices lists them."))
    }
}
