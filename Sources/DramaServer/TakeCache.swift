import DramaEngine
import Foundation

/// `codes_base64`: 16-bit little-endian codes, codebook by codebook.
public enum TakeCodes {
    public static func encode(_ codes: [[Int32]]) -> String {
        var data = Data(capacity: codes.count * (codes.first?.count ?? 0) * 2)
        for code in codes.joined() { withUnsafeBytes(of: UInt16(code).littleEndian) { data.append(contentsOf: $0) } }
        return data.base64EncodedString()
    }

    public static func decode(_ base64: String) -> [[Int32]]? {
        guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else { return nil }
        let books = ModelConstants.numCodebooks
        guard !data.isEmpty, data.count % (2 * books) == 0 else { return nil }
        let values = stride(from: 0, to: data.count, by: 2).map { Int32(UInt16(data[data.startIndex + $0]) | UInt16(data[data.startIndex + $0 + 1]) << 8) }
        guard values.allSatisfy({ $0 < Int32(ModelConstants.codebookSize) }) else { return nil }
        let frames = values.count / books
        return (0..<books).map { Array(values[($0 * frames)..<(($0 + 1) * frames)]) }
    }
}

/// The last takes the server generated, so a client can retake by id.
public actor TakeCache {
    public struct Entry: Sendable, Equatable {
        public var id: UUID
        public var take: Take
        /// The script text the request spoke.
        public var line: String
        public var casting: VoiceCasting
        public var character: VoiceDesign?
        /// The cached takes this one was generated after, then itself.
        public var lineage: [UUID]
    }

    public static let capacity = 20
    private var entries: [Entry] = []

    public init() {}

    func insert(_ entry: Entry) {
        entries.removeAll { $0.id == entry.id }
        entries.append(entry)
        if entries.count > Self.capacity { entries.removeFirst(entries.count - Self.capacity) }
    }

    public func entry(_ id: UUID) -> Entry? { entries.first { $0.id == id } }
}
