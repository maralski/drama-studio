// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

struct AppVersion: Comparable, Equatable {
    let components: [Int]
    init?(_ text: String) {
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ $0.isASCII && $0.isNumber }) }),
              parts.allSatisfy({ Int($0) != nil }) else { return nil }
        components = parts.map { Int($0)! }
    }
    static func < (lhs: Self, rhs: Self) -> Bool { lhs.components.lexicographicallyPrecedes(rhs.components) }
}

struct AppRelease: Decodable {
    struct Asset: Decodable {
        let name: String
        let browser_download_url: URL
    }
    let tag_name: String
    let draft: Bool
    let prerelease: Bool
    let assets: [Asset]

    var version: String { tag_name.hasPrefix("v") ? String(tag_name.dropFirst()) : tag_name }

    func update(after current: String) throws -> (version: String, image: URL, checksums: URL)? {
        guard let installed = AppVersion(current) else { throw UpdateError.message("This build has no supported release version.") }
        guard !draft, !prerelease, let available = AppVersion(version), available > installed else { return nil }
        func asset(_ name: String) throws -> URL {
            let matches = assets.filter { $0.name == name }
            guard matches.count == 1 else { throw UpdateError.message("The release is missing its update assets.") }
            let url = matches[0].browser_download_url
            guard url.scheme == "https", url.host == "github.com", url.user == nil, url.password == nil,
                  url.path == "/maralski/drama-studio/releases/download/\(tag_name)/\(name)",
                  url.query == nil, url.fragment == nil else { throw UpdateError.message("The release asset URL is invalid.") }
            return url
        }
        return try (version, asset("DramaStudio-\(version).dmg"), asset("SHA256SUMS"))
    }

    static func checksum(in text: String, version: String) throws -> String {
        let matches = text.split(whereSeparator: \.isNewline).compactMap { line -> String? in
            let parts = line.split(whereSeparator: \.isWhitespace)
            guard parts.count == 2, parts[1] == "DramaStudio-\(version).dmg" else { return nil }
            return String(parts[0]).lowercased()
        }
        guard matches.count == 1, matches[0].count == 64,
              matches[0].allSatisfy({ $0.isASCII && $0.isHexDigit }) else {
            throw UpdateError.message("The release has no valid app checksum.")
        }
        return matches[0]
    }
}

enum UpdateError: LocalizedError {
    case message(String)
    var errorDescription: String? { switch self { case .message(let text): return text } }
}
