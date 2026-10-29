// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// Bytes of a newline-delimited body to whole lines, `\n` or `\r\n` ended.
public struct LineSplitter: Sendable {
    private var pending = Data()

    public init() {}

    public mutating func feed(_ data: Data) -> [String] {
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 10) {
            var line = pending[pending.startIndex..<newline]
            if line.last == 13 { line = line.dropLast() }
            lines.append(String(decoding: line, as: UTF8.self))
            pending = Data(pending[(newline + 1)...])
        }
        return lines
    }

    /// The unterminated last line, if any.
    public mutating func finish() -> [String] {
        defer { pending = Data() }
        return pending.isEmpty ? [] : [String(decoding: pending, as: UTF8.self)]
    }
}

/// Text that arrives a line at a time, cut into the sentences the engine
/// speaks as soon as each one is complete.
public struct LiveText: Sendable {
    private var pending = ""
    private static let marks = Set(".!?;\u{3002}\u{FF01}\u{FF1F}\u{FF1B}\u{2026}")
    private static let closers = Set("\"'\u{201D}\u{2019})")

    public init() {}

    /// The sentences `line` completes. An empty line flushes the rest.
    public mutating func append(line: String) -> [String] {
        guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return flush() }
        pending = pending.isEmpty ? line : pending + " " + line
        guard let cut = Self.endOfLastSentence(in: pending) else { return [] }
        // Cut before splitting: the splitter folds a short unfinished tail into the sentence before it.
        let sentences = Drama3Chunking.splitSentencesTagged(String(pending[..<cut]))
        var tail = String(pending[cut...]).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty, !tail.hasPrefix("<|speaker:"), let last = sentences.last, last.hasPrefix("<|speaker:"), let close = last.range(of: "|>") {
            tail = last[..<close.upperBound] + " " + tail
        }
        pending = tail
        return sentences
    }

    public mutating func flush() -> [String] {
        defer { pending = "" }
        return Drama3Chunking.splitSentencesTagged(pending)
    }

    private static func endOfLastSentence(in text: String) -> String.Index? {
        guard var end = text.lastIndex(where: { marks.contains($0) }).map(text.index(after:)) else { return nil }
        while end < text.endIndex, closers.contains(text[end]) { end = text.index(after: end) }
        return end
    }
}

/// How a streaming response frames audio and progress.
enum StreamFraming: Sendable {
    /// Raw 16-bit PCM in HTTP chunks, and nothing else.
    case pcm
    /// Server-Sent Events.
    case events

    init?(_ delivery: Delivery) {
        switch delivery {
        case .pcmStream: self = .pcm
        case .events: self = .events
        case .file: return nil
        }
    }

    var headers: [String: String] {
        var headers = ["X-Drama-Sample-Rate": String(AudioEncoding.sampleRate), "X-Drama-Format": "pcm_s16le_mono"]
        switch self {
        case .pcm: headers["Content-Type"] = AudioFormat.pcm.contentType
        case .events:
            headers["Content-Type"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
        }
        return headers
    }

    func audio(_ samples: [Float]) -> Data {
        guard !samples.isEmpty else { return Data() }
        switch self {
        case .pcm: return AudioEncoding.pcm16(samples)
        case .events: return Self.event("audio", AudioEncoding.pcm16(samples).base64EncodedString())
        }
    }

    func sentence(index: Int, text: String, seconds: Double) -> Data {
        message("sentence", ["index": index, "text": text, "seconds": seconds])
    }

    func done(_ fields: [String: Any]) -> Data { message("done", fields) }

    func error(_ message: String) -> Data { self.message("error", ["message": message]) }

    private func message(_ name: String, _ fields: [String: Any]) -> Data {
        guard case .events = self else { return Data() }
        let rounded = fields.mapValues { value -> Any in
            guard let number = value as? Double else { return value }
            return NSDecimalNumber(string: String(format: "%.3f", number))
        }
        let json = (try? JSONSerialization.data(withJSONObject: rounded, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data("{}".utf8)
        return Self.event(name, String(decoding: json, as: UTF8.self))
    }

    private static func event(_ name: String, _ data: String) -> Data {
        Data("event: \(name)\ndata: \(data)\n\n".utf8)
    }
}
