// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

enum Captions {
    enum Format: String, CaseIterable, Sendable {
        case srt, webVTT

        var fileExtension: String { self == .srt ? "srt" : "vtt" }
        fileprivate var millisecondSeparator: String { self == .srt ? "," : "." }
    }

    struct Cue: Equatable, Sendable {
        var startSample: Int
        var endSample: Int
        var text: String
    }

    static func cues(for takes: [NarrationTake]) -> [Cue] {
        var cues: [Cue] = []
        for take in takes {
            var cursor = take.sampleRange.lowerBound
            for sentence in take.sentences {
                let end = min(cursor + sentence.frames * ModelConstants.samplesPerFrame, take.sampleRange.upperBound)
                let text = spokenText(sentence.text)
                if !text.isEmpty, end > cursor { cues.append(Cue(startSample: cursor, endSample: end, text: text)) }
                cursor = end
            }
        }
        return cues
    }

    static func render(_ takes: [NarrationTake], as format: Format) -> String {
        render(cues: cues(for: takes), as: format)
    }

    static func render(cues: [Cue], as format: Format) -> String {
        let separator = " --> "
        switch format {
        case .srt:
            return cues.enumerated().map { offset, cue in
                "\(offset + 1)\n\(timestamp(cue.startSample, format))\(separator)\(timestamp(cue.endSample, format))\n\(cue.text)\n"
            }.joined(separator: "\n")
        case .webVTT:
            let body = cues.map { cue in
                "\(timestamp(cue.startSample, format))\(separator)\(timestamp(cue.endSample, format))\n\(escapeVTT(cue.text))\n"
            }
            return (["WEBVTT\n"] + body).joined(separator: "\n")
        }
    }

    static func timestamp(_ samples: Int, _ format: Format) -> String {
        let rate = ModelConstants.sampleRate
        let milliseconds = (samples * 1000 + rate / 2) / rate
        let (hours, rest) = milliseconds.quotientAndRemainder(dividingBy: 3_600_000)
        let (minutes, rest2) = rest.quotientAndRemainder(dividingBy: 60_000)
        let (seconds, millis) = rest2.quotientAndRemainder(dividingBy: 1000)
        return String(format: "%02ld:%02ld:%02ld%@%03ld", hours, minutes, seconds, format.millisecondSeparator, millis)
    }

    static func spokenText(_ sentence: String) -> String {
        sentence
            .replacingOccurrences(of: #"<\|[^|]*\|>"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"</?[A-Za-z][^<>]*>"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    private static func escapeVTT(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }
}
