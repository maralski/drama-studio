// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

struct TokenizerFile {
    let vocab: [String: Int32]
    let added: [String: Int32]
    let splitPattern: String
    private let merges: NSArray

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let root = try Self.object(try JSONSerialization.jsonObject(with: data), "root")
        let model = try Self.object(root["model"], "model")
        guard model["type"] as? String == "BPE" else { throw TokenizerError.invalidFile("model is not BPE") }
        guard (root["normalizer"] as? NSDictionary)?["type"] as? String == "NFC" else {
            throw TokenizerError.invalidFile("normalizer is not NFC")
        }
        vocab = try Self.vocab(model["vocab"])
        added = try Self.addedTokens(root["added_tokens"])
        splitPattern = try Self.splitPattern(root["pre_tokenizer"])
        guard let merges = model["merges"] as? NSArray else { throw TokenizerError.invalidFile("merges") }
        self.merges = merges
    }

    func idsForBytes() throws -> [Int32] {
        try (0...255).map { byte in
            guard let id = vocab[ByteLevel.token(forByte: UInt8(byte))] else {
                throw TokenizerError.invalidFile("byte \(byte) missing from vocab")
            }
            return id
        }
    }

    func mergeTable() throws -> [UInt64: BPE.Merge] {
        var table = [UInt64: BPE.Merge](minimumCapacity: merges.count)
        for (rank, element) in merges.enumerated() {
            guard let pair = element as? NSArray, pair.count == 2,
                  let left = (pair[0] as? String).map(Self.native), let right = (pair[1] as? String).map(Self.native),
                  let leftId = vocab[left], let rightId = vocab[right], let mergedId = vocab[left + right]
            else { throw TokenizerError.invalidFile("merge \(rank)") }
            table[BPE.pairKey(leftId, rightId)] = BPE.Merge(rank: Int32(rank), id: mergedId)
        }
        return table
    }

    private static func object(_ value: Any?, _ name: String) throws -> NSDictionary {
        guard let dictionary = value as? NSDictionary else { throw TokenizerError.invalidFile(name) }
        return dictionary
    }

    private static func vocab(_ value: Any?) throws -> [String: Int32] {
        let entries = try object(value, "vocab")
        var vocab = [String: Int32](minimumCapacity: entries.count)
        for (key, id) in entries {
            guard let token = key as? String, let id = id as? NSNumber else { throw TokenizerError.invalidFile("vocab") }
            vocab[native(token)] = id.int32Value
        }
        return vocab
    }

    // Bridged NSString keys hash through CoreFoundation; native copies make the 750k merge-table
    // lookups about 3x faster (0.85 s to 0.25 s measured).
    private static func native(_ string: String) -> String {
        var copy = string
        copy.makeContiguousUTF8()
        return copy
    }

    private static func addedTokens(_ value: Any?) throws -> [String: Int32] {
        guard let entries = value as? [NSDictionary] else { throw TokenizerError.invalidFile("added_tokens") }
        return try Dictionary(uniqueKeysWithValues: entries.map { entry in
            guard let content = entry["content"] as? String, let id = entry["id"] as? NSNumber,
                  entry["normalized"] as? Bool == false
            else { throw TokenizerError.invalidFile("added token \(entry)") }
            return (content, id.int32Value)
        })
    }

    private static func splitPattern(_ value: Any?) throws -> String {
        let steps = try object(value, "pre_tokenizer")["pretokenizers"] as? [NSDictionary] ?? []
        guard steps.count == 2,
              steps[0]["type"] as? String == "Split", steps[0]["behavior"] as? String == "Isolated",
              steps[0]["invert"] as? Bool == false,
              let pattern = (steps[0]["pattern"] as? NSDictionary)?["Regex"] as? String,
              steps[1]["type"] as? String == "ByteLevel", steps[1]["use_regex"] as? Bool == false,
              steps[1]["add_prefix_space"] as? Bool == false
        else { throw TokenizerError.invalidFile("pre_tokenizer is not Split(Isolated) then ByteLevel") }
        return pattern
    }
}
