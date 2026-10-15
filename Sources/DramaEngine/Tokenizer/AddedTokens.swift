struct AddedTokens: Sendable {
    enum Segment {
        case text(ArraySlice<UInt8>)
        case added(Int32)
    }

    private struct Entry: Sendable {
        let bytes: [UInt8]
        let id: Int32
    }

    private let entriesByFirstByte: [[Entry]]

    init(_ tokens: [String: Int32]) {
        var buckets = [[Entry]](repeating: [], count: 256)
        for (content, id) in tokens {
            let bytes = Array(content.utf8)
            guard let first = bytes.first else { continue }
            buckets[Int(first)].append(Entry(bytes: bytes, id: id))
        }
        entriesByFirstByte = buckets.map { $0.sorted { $0.bytes.count > $1.bytes.count } }
    }

    func split(_ utf8: [UInt8]) -> [Segment] {
        var segments: [Segment] = []
        var textStart = 0
        var index = 0
        while index < utf8.count {
            guard let entry = longestMatch(in: utf8, at: index) else {
                index += 1
                continue
            }
            if textStart < index { segments.append(.text(utf8[textStart..<index])) }
            segments.append(.added(entry.id))
            index += entry.bytes.count
            textStart = index
        }
        if textStart < utf8.count { segments.append(.text(utf8[textStart...])) }
        return segments
    }

    private func longestMatch(in utf8: [UInt8], at index: Int) -> Entry? {
        entriesByFirstByte[Int(utf8[index])].first { utf8[index...].starts(with: $0.bytes) }
    }
}
