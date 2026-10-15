import Foundation

public enum TokenizerError: Error, Sendable {
    case invalidFile(String)
}

public final class Tokenizer: Sendable {
    private let idForToken: [String: Int32]
    private let tokenForId: [String?]
    private let addedTokens: AddedTokens
    private let splitter: NSRegularExpression
    private let bpe: BPE

    public var vocabularySize: Int { tokenForId.count }

    public init(url: URL) throws {
        let file = try TokenizerFile(url: url)
        idForToken = file.vocab.merging(file.added) { _, added in added }
        tokenForId = Self.invert(idForToken)
        addedTokens = AddedTokens(file.added)
        splitter = try NSRegularExpression(pattern: file.splitPattern)
        bpe = BPE(idForByte: try file.idsForBytes(), merges: try file.mergeTable())
    }

    public func id(forToken token: String) -> Int32? {
        idForToken[token]
    }

    public func encode(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        for segment in addedTokens.split(Array(text.utf8)) {
            switch segment {
            case .added(let id): ids.append(id)
            case .text(let bytes): encodeOrdinary(String(decoding: bytes, as: UTF8.self), into: &ids)
            }
        }
        return ids
    }

    public func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard tokenForId.indices.contains(Int(id)), let token = tokenForId[Int(id)] else { continue }
            ByteLevel.appendBytes(of: token, to: &bytes)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeOrdinary(_ text: String, into ids: inout [Int32]) {
        for piece in pieces(of: text.precomposedStringWithCanonicalMapping) {
            bpe.encode(piece.utf8, into: &ids)
        }
    }

    private func pieces(of text: String) -> [String] {
        let ns = text as NSString
        var pieces: [String] = []
        var cursor = 0
        splitter.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let range = match?.range else { return }
            if cursor < range.location {
                pieces.append(ns.substring(with: NSRange(location: cursor, length: range.location - cursor)))
            }
            pieces.append(ns.substring(with: range))
            cursor = NSMaxRange(range)
        }
        if cursor < ns.length { pieces.append(ns.substring(from: cursor)) }
        return pieces
    }

    private static func invert(_ idForToken: [String: Int32]) -> [String?] {
        var tokens = [String?](repeating: nil, count: Int(idForToken.values.max() ?? -1) + 1)
        for (token, id) in idForToken { tokens[Int(id)] = token }
        return tokens
    }
}
