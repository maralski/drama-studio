import Foundation

public struct PronunciationRule: Codable, Sendable, Equatable, Hashable {
    public var match: String
    public var sayAs: String
    public var wholeWord: Bool
    public var caseSensitive: Bool

    public init(match: String, sayAs: String, wholeWord: Bool = true, caseSensitive: Bool = false) {
        self.match = match
        self.sayAs = sayAs
        self.wholeWord = wholeWord
        self.caseSensitive = caseSensitive
    }
}

public struct PronunciationDictionary: Codable, Sendable, Equatable {
    public var rules: [PronunciationRule]

    public init(rules: [PronunciationRule] = []) {
        self.rules = rules
    }

    public static var defaultURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "DramaStudio/pronunciation.json")
    }

    public static func load(from url: URL = defaultURL) throws -> PronunciationDictionary {
        guard FileManager.default.fileExists(atPath: url.path) else { return PronunciationDictionary() }
        return try JSONDecoder().decode(PronunciationDictionary.self, from: Data(contentsOf: url))
    }

    public func save(to url: URL = defaultURL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    /// One left-to-right pass, so a replacement is never rewritten by another rule.
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
