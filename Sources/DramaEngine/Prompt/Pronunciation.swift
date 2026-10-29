// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One rewrite: wherever `match` appears, the model hears `sayAs` instead.
public struct PronunciationRule: Codable, Sendable, Equatable, Hashable {
    /// The text to find. An empty match never applies.
    public var match: String
    /// What the model should say instead, spelled as it sounds.
    public var sayAs: String
    /// Match only where no letter, digit, or underscore touches either end.
    /// Default true.
    public var wholeWord: Bool
    /// Match case exactly. Default false.
    public var caseSensitive: Bool

    /// Creates a rule.
    public init(match: String, sayAs: String, wholeWord: Bool = true, caseSensitive: Bool = false) {
        self.match = match
        self.sayAs = sayAs
        self.wholeWord = wholeWord
        self.caseSensitive = caseSensitive
    }
}

/// Rules that rewrite words before the model hears them, as the app's
/// Settings and the HTTP API apply them. Apply it to ``SpeechRequest/text``
/// yourself; the engine does not.
///
/// ```swift
/// let rules = PronunciationDictionary(rules: [PronunciationRule(match: "Nguyen", sayAs: "Win")])
/// let request = SpeechRequest(text: rules.apply(to: "Ms Nguyen is here."))
/// ```
public struct PronunciationDictionary: Codable, Sendable, Equatable {
    /// The rules. When two could match at one place, the longer match wins,
    /// then the earlier rule.
    public var rules: [PronunciationRule]

    /// Creates a dictionary.
    public init(rules: [PronunciationRule] = []) {
        self.rules = rules
    }

    /// `~/Library/Application Support/DramaStudio/pronunciation.json`, the
    /// file the app and `drama serve` share.
    public static var defaultURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "DramaStudio/pronunciation.json")
    }

    /// Reads a dictionary from JSON; an empty one when the file does not exist.
    public static func load(from url: URL = defaultURL) throws -> PronunciationDictionary {
        guard FileManager.default.fileExists(atPath: url.path) else { return PronunciationDictionary() }
        return try JSONDecoder().decode(PronunciationDictionary.self, from: Data(contentsOf: url))
    }

    /// Writes the dictionary as pretty-printed JSON, creating the folder.
    public func save(to url: URL = defaultURL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    /// `text` with every rule applied in one left-to-right pass, so a
    /// replacement is never rewritten by another rule. Speaker tags,
    /// `[cues]`, and `<tags>` are left as they are.
    public func apply(to text: String) -> String {
        guard let matcher else { return text }
        let ns = text as NSString
        var out = ""
        var cursor = 0
        for protected in Self.protectedSpans.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            out += matcher.rewrite(ns.substring(with: NSRange(location: cursor, length: protected.range.location - cursor)))
            out += ns.substring(with: protected.range)
            cursor = protected.range.location + protected.range.length
        }
        return out + matcher.rewrite(ns.substring(from: cursor))
    }

    private static let protectedSpans = try! NSRegularExpression(pattern: #"<\|[^|]*\|>|\[[^\]]*\]|</?[A-Za-z][^<>]*>"#)

    private var matcher: Matcher? { Matcher(rules: rules) }

    private struct Matcher {
        let expression: NSRegularExpression
        let replacements: [String]

        init?(rules: [PronunciationRule]) {
            let usable = rules.filter { !$0.match.isEmpty }.enumerated()
                .sorted { ($0.element.match.count, -$0.offset) > ($1.element.match.count, -$1.offset) }
                .map(\.element)
            guard !usable.isEmpty else { return nil }
            let alternatives = usable.map { rule in
                var pattern = NSRegularExpression.escapedPattern(for: rule.match)
                if rule.wholeWord { pattern = #"(?<![\p{L}\p{N}_])"# + pattern + #"(?![\p{L}\p{N}_])"# }
                return "(" + (rule.caseSensitive ? pattern : "(?i:" + pattern + ")") + ")"
            }
            guard let expression = try? NSRegularExpression(pattern: alternatives.joined(separator: "|")) else { return nil }
            self.expression = expression
            self.replacements = usable.map(\.sayAs)
        }

        func rewrite(_ text: String) -> String {
            let ns = text as NSString
            var out = ""
            var cursor = 0
            for match in expression.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
                guard let rule = (1..<match.numberOfRanges).first(where: { match.range(at: $0).location != NSNotFound }) else { continue }
                out += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)) + replacements[rule - 1]
                cursor = match.range.location + match.range.length
            }
            return out + ns.substring(from: cursor)
        }
    }
}
