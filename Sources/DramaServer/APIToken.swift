// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The bearer token that guards the API from web pages that can reach
/// localhost. Health stays open so launchers can poll readiness.
public enum APIToken {
    public static var defaultURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/api-token")
    }

    public static func loadOrCreate(at url: URL = defaultURL) throws -> String {
        if let existing = try? String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), !existing.isEmpty {
            return existing
        }
        var generator = SystemRandomNumberGenerator()
        let token = (0..<32).map { _ in String(format: "%02x", UInt8.random(in: 0...255, using: &generator)) }.joined()
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data((token + "\n").utf8).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return token
    }

    public static func authorizes(_ request: HTTPRequest, token: String?) -> Bool {
        guard let token else { return true }
        if request.method == "GET", request.path == "/v1/health" { return true }
        guard let header = request.header("authorization"), header.lowercased().hasPrefix("bearer ") else { return false }
        let presented = Array(header.dropFirst("bearer ".count).trimmingCharacters(in: .whitespaces).utf8)
        let expected = Array(token.utf8)
        guard presented.count == expected.count else { return false }
        return zip(presented, expected).reduce(0) { $0 | ($1.0 ^ $1.1) } == 0
    }
}
