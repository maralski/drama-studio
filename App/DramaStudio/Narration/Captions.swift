import DramaEngine
import Foundation

/// One caption cue per spoken sentence, timed from the frames each sentence took.
enum Captions {
    enum Format: String, CaseIterable, Sendable {
        case srt, webVTT

        var fileExtension: String { self == .srt ? "srt" : "vtt" }
        fileprivate var millisecondSeparator: String { self == .srt ? "," : "." }
    }

    /// Times are sample positions in the combined audio.
    struct Cue: Equatable, Sendable {
        var start: Int
        var end: Int
        var text: String
    }

    static func cues(for takes: [NarrationTake]) -> [Cue] {
        var cues: [Cue] = []
        for take in takes {
            var cursor = take.sampleRange.lowerBound
            for sentence in take.sentences {
                let end = min(cursor + sentence.frames * ModelConstants.samplesPerFrame, take.sampleRange.upperBound)
                let text = spokenText(sentence.text)
                if !text.isEmpty, end > cursor { cues.append(Cue(start: cursor, end: end, text: text)) }
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
                "\(offset + 1)\n\(timestamp(cue.start, format))\(separator)\(timestamp(cue.end, format))\n\(cue.text)\n"
            }.joined(separator: "\n")
        case .webVTT:
            let body = cues.map { cue in
                "\(timestamp(cue.start, format))\(separator)\(timestamp(cue.end, format))\n\(escapeVTT(cue.text))\n"
            }
            return (["WEBVTT\n"] + body).joined(separator: "\n")
        }
    }

    /// `HH:MM:SS,mmm` for SRT, `HH:MM:SS.mmm` for WebVTT, rounded to the nearest millisecond.
    static func timestamp(_ samples: Int, _ format: Format) -> String {
        let rate = ModelConstants.sampleRate
        let milliseconds = (samples * 1000 + rate / 2) / rate
        let (hours, rest) = milliseconds.quotientAndRemainder(dividingBy: 3_600_000)
        let (minutes, rest2) = rest.quotientAndRemainder(dividingBy: 60_000)
        let (seconds, millis) = rest2.quotientAndRemainder(dividingBy: 1000)
        return String(format: "%02ld:%02ld:%02ld%@%03ld", hours, minutes, seconds, format.millisecondSeparator, millis)
    }

    /// The sentence as a viewer reads it: speaker tags and paired tag markup
    /// removed, `[cue]` markers kept, on one line.
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
