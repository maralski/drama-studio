import Foundation

/// The File > Open Recent Project list: most recent first, at most ten, in UserDefaults.
enum RecentProjects {
    static let key = "recentProjects"
    static let limit = 10

    static func updated(_ recent: [URL], adding url: URL, limit: Int = limit) -> [URL] {
        Array(([url] + recent.filter { $0.standardizedFileURL.path != url.standardizedFileURL.path }).prefix(limit))
    }

    /// Projects still on disk, most recent first.
    static func load(from defaults: UserDefaults = .standard) -> [URL] {
        (defaults.stringArray(forKey: key) ?? []).map { URL(filePath: $0) }.filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func save(_ recent: [URL], to defaults: UserDefaults = .standard) {
        defaults.set(recent.map(\.path), forKey: key)
    }
}
