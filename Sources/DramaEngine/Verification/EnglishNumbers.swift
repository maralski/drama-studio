// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// English numbers as the set of values they can mean. One number becomes one
/// token, `#` followed by its readings joined with `|`, so token ordinals stay
/// one per spoken unit and two numbers match when their readings intersect.
/// Integer readings are digit strings ("1200", "007"); a written decimal or a
/// clock reads as "a.b"; a spoken "point" decimal reads as "pa.b"; a decade
/// reads as "1990s". An integer reading never contains a point, so "2.5" and
/// "25" cannot match.
enum EnglishNumbers {
    enum Piece {
        case word(String)
        case written(Set<String>)
    }

    private static let units: [String: Int] = Dictionary(uniqueKeysWithValues: zip(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"], 0...))
    private static let tens: [String: Int] = Dictionary(uniqueKeysWithValues: zip(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], stride(from: 20, through: 90, by: 10)))
    private static let decades: [String: String] = Dictionary(uniqueKeysWithValues: tens.keys.map {
        (String($0.dropLast()) + "ies", $0)
    })
    private static let largeScales = ["thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
                                      "trillion": 1_000_000_000_000]
    private static let scales = largeScales.merging(["hundred": 100]) { $1 }
    private static let currencies: Set<String> = ["dollar", "dollars", "cent", "cents", "pound", "pounds", "euro", "euros"]
    private static let mergedReadingLimit = 64

    static func token(_ readings: Set<String>) -> String { "#" + readings.sorted().joined(separator: "|") }

    static func isNumber(_ token: String) -> Bool { token.hasPrefix("#") }

    static func same(_ left: String, _ right: String) -> Bool {
        guard left != right else { return true }
        guard isNumber(left), isNumber(right) else { return false }
        return !Set(left.dropFirst().split(separator: "|")).isDisjoint(with: right.dropFirst().split(separator: "|"))
    }

    /// Readings of a written group. `integer` may hold thousands commas;
    /// `separator` is "." or ":"; `decade` marks a trailing "s" or "'s".
    static func written(integer: String?, separator: String?, fraction: String?, decade: Bool) -> Set<String>? {
        let digits = (integer ?? "0").replacingOccurrences(of: ",", with: "")
        let significant = digits.drop { $0 == "0" }
        let trimmed = significant.isEmpty ? "0" : String(significant)
        if decade {
            guard separator == nil, [2, 4].contains(digits.count), digits.hasSuffix("0") else { return nil }
            return digits.count == 4 ? [digits + "s", String(digits.suffix(2)) + "s"] : [digits + "s"]
        }
        guard let separator, let fraction else { return [digits, trimmed] }
        if separator == ":" {
            return fraction.count == 2 ? [trimmed + "." + fraction, trimmed + fraction] : [digits + ":" + fraction]
        }
        var readings: Set<String> = [trimmed + "." + fraction, "p" + trimmed + "." + fraction]
        let shortened = fraction.reversed().drop { $0 == "0" }
        if !shortened.isEmpty { readings.insert("p" + trimmed + "." + String(shortened.reversed())) }
        return readings
    }

    /// Groups number words and written numbers into number tokens, drops
    /// currency words beside a number, and applies `spellings` to the rest.
    /// `joined` is true when only whitespace or commas separate a piece from the
    /// one before, as in a phone number written "five five five, oh one four two".
    static func tokens(_ pieces: [(piece: Piece, joined: Bool)], spellings: [String: String]) -> [String] {
        enum Part { case word(String), number(Set<String>) }
        func word(_ index: Int) -> String? {
            guard index < pieces.count, case .word(let word) = pieces[index].piece else { return nil }
            return word
        }
        func joinedWord(_ index: Int) -> String? { index < pieces.count && pieces[index].joined ? word(index) : nil }
        func isDigit(_ word: String?) -> Bool { word == "oh" || word.flatMap { units[$0] }.map { $0 < 10 } == true }
        func startsRun(_ index: Int) -> Bool {
            guard let word = word(index) else { return false }
            if units[word] != nil || tens[word] != nil || decades[word] != nil { return true }
            if word == "a" { return joinedWord(index + 1).flatMap { scales[$0] } != nil }
            return ["double", "triple", "point"].contains(word) && isDigit(joinedWord(index + 1))
        }
        func runEnd(from start: Int) -> Int {
            guard startsRun(start) else { return start }
            var end = start + 1, hasPoint = word(start) == "point"
            while let next = joinedWord(end), let previous = word(end - 1), decades[previous] == nil {
                let continues: Bool
                switch next {
                case "double", "triple": continues = isDigit(joinedWord(end + 1))
                case "point": continues = !hasPoint && isDigit(joinedWord(end + 1)); hasPoint = true
                case "and": continues = scales[previous] != nil && joinedWord(end + 1).map { units[$0] != nil || tens[$0] != nil } == true
                default: continues = units[next] != nil || tens[next] != nil || decades[next] != nil || scales[next] != nil || next == "oh"
                }
                guard continues else { break }
                end += 1
            }
            return end
        }

        var parts: [(part: Part, joined: Bool)] = []
        var index = 0
        while index < pieces.count {
            let joined = pieces[index].joined
            if case .written(let readings) = pieces[index].piece {
                if let scale = joinedWord(index + 1).flatMap({ scales[$0] }) {
                    parts.append((.number(multiplied(readings, by: scale)), joined)); index += 2
                } else {
                    parts.append((.number(readings), joined)); index += 1
                }
                continue
            }
            let end = runEnd(from: index)
            if end > index {
                parts.append((.number(spokenReadings((index..<end).compactMap(word))), joined)); index = end
            } else {
                parts.append((.word(word(index)!), joined)); index += 1
            }
        }

        var merged: [Part] = []
        for (part, joined) in parts {
            if joined, case .number(let right) = part, case .number(let left)? = merged.last,
               let both = concatenated(left, right) {
                merged[merged.count - 1] = .number(both)
            } else {
                merged.append(part)
            }
        }

        var result: [String] = []
        for (offset, part) in merged.enumerated() {
            switch part {
            case .number(let readings):
                result.append(token(readings))
            case .word(let word):
                func isNumber(_ at: Int) -> Bool {
                    guard merged.indices.contains(at), case .number = merged[at] else { return false }
                    return true
                }
                if currencies.contains(word), isNumber(offset - 1) || isNumber(offset + 1) { continue }
                if let spelling = spellings[word] {
                    result += spelling.split(separator: " ").map(String.init)
                } else {
                    result.append(word)
                }
            }
        }
        return result
    }

    private static func isInteger(_ reading: String) -> Bool { !reading.isEmpty && reading.allSatisfy { ("0"..."9").contains($0) } }

    private static func concatenated(_ left: Set<String>, _ right: Set<String>) -> Set<String>? {
        // Inside a longer number every leading zero is spoken: "555 0142" never reads as "555142".
        func padded(_ readings: Set<String>) -> [String] {
            readings.filter { reading in
                isInteger(reading) && !readings.contains { $0.count > reading.count && $0.hasSuffix(reading) && $0.hasPrefix("0") }
            }.sorted()
        }
        let lefts = padded(left), rights = padded(right)
        guard !lefts.isEmpty, !rights.isEmpty else { return nil }
        var readings = Set<String>()
        for first in lefts {
            for second in rights {
                readings.insert(first + second)
                if second.count == 2 { readings.insert(first + "." + second) }
            }
        }
        return Set(readings.sorted().prefix(mergedReadingLimit))
    }

    private static func multiplied(_ readings: Set<String>, by scale: Int) -> Set<String> {
        Set(readings.compactMap { reading in
            let value = reading.hasPrefix("p") ? String(reading.dropFirst()) : reading
            guard !value.contains("s"), !value.contains(":"),
                  let decimal = Decimal(string: value, locale: Locale(identifier: "en_US_POSIX")) else { return nil }
            return NSDecimalNumber(decimal: decimal * Decimal(scale)).stringValue
        })
    }

    private static func spokenReadings(_ run: [String]) -> Set<String> {
        let readings = readings(run)
        return readings.isEmpty ? [run.joined(separator: " ")] : readings
    }

    private static func readings(_ run: [String]) -> Set<String> {
        guard let last = run.last else { return [] }
        if let tensWord = decades[last] {
            let years = readings(Array(run.dropLast()) + [tensWord]).filter { isInteger($0) && [2, 4].contains($0.count) && $0.hasSuffix("0") }
            return Set(years.flatMap { $0.count == 4 ? [$0 + "s", String($0.suffix(2)) + "s"] : [$0 + "s"] })
        }
        if let point = run.firstIndex(of: "point") {
            var fraction = run[(point + 1)...], scale: Int?
            if let last = fraction.last, let large = largeScales[last] { scale = large; fraction = fraction.dropLast() }
            guard let whole = point == 0 ? 0 : cardinal(run[..<point]), !fraction.isEmpty,
                  fraction.allSatisfy({ $0 == "oh" || units[$0].map { $0 < 10 } == true }) else { return [] }
            let decimal = "\(whole)." + fraction.map { $0 == "oh" ? "0" : String(units[$0]!) }.joined()
            return scale.map { multiplied([decimal], by: $0) } ?? ["p" + decimal]
        }
        var readings = Set<String>()
        if let value = cardinal(run[...]) { readings.insert(String(value)) }
        if let digits = digitSequence(run) { readings.insert(digits) }
        for split in 1..<min(3, run.count) {
            guard let first = small(run[..<split]) else { continue }
            let rest = run[split...]
            if let second = small(rest), second >= 10 {
                readings.formUnion(["\(first)\(second)", "\(first).\(second)"])
            } else if rest.count == 2, rest.first == "oh", let digit = units[rest.last!], digit < 10 {
                readings.formUnion(["\(first)0\(digit)", "\(first).0\(digit)"])
            }
        }
        if run.count > 1, let scale = largeScales[last] {
            readings.formUnion(multiplied(self.readings(Array(run.dropLast())), by: scale))
        }
        return readings
    }

    /// 0-99 from one or two words: "seven", "nineteen", "forty", "forty seven".
    private static func small(_ words: ArraySlice<String>) -> Int? {
        if words.count == 1 { return units[words.first!] ?? tens[words.first!] }
        guard words.count == 2, let ten = tens[words.first!], let unit = units[words.last!], (1...9).contains(unit) else { return nil }
        return ten + unit
    }

    /// Digit by digit: "five five five oh one", "double oh seven".
    private static func digitSequence(_ run: [String]) -> String? {
        var digits = "", repeats = 1
        for word in run {
            switch word {
            case "double": repeats = 2
            case "triple": repeats = 3
            case "oh": digits += String(repeating: "0", count: repeats); repeats = 1
            default:
                guard let digit = units[word], digit < 10 else { return nil }
                digits += String(repeating: String(digit), count: repeats); repeats = 1
            }
        }
        return repeats == 1 && !digits.isEmpty ? digits : nil
    }

    /// The standard cardinal, or nil when the words are not one well-formed
    /// number: "two thousand twenty six", "twelve hundred", "a hundred and five".
    private static func cardinal(_ words: ArraySlice<String>) -> Int? {
        var total = 0, index = words.startIndex, previousScale = Int.max
        func belowHundred(_ at: Int) -> (value: Int, next: Int)? {
            guard at < words.endIndex else { return nil }
            if let unit = units[words[at]] { return (unit, at + 1) }
            guard let ten = tens[words[at]] else { return nil }
            if at + 1 < words.endIndex, let unit = units[words[at + 1]], (1...9).contains(unit) { return (ten + unit, at + 2) }
            return (ten, at + 1)
        }
        func skipAnd() -> Bool {
            guard index < words.endIndex, words[index] == "and" else { return true }
            index += 1
            return belowHundred(index) != nil
        }
        while index < words.endIndex {
            var group: Int
            if words[index] == "a", index + 1 < words.endIndex, scales[words[index + 1]] != nil {
                group = 1; index += 1
            } else if let small = belowHundred(index) {
                group = small.value; index = small.next
            } else {
                return nil
            }
            if index < words.endIndex, words[index] == "hundred" {
                guard group > 0 else { return nil }
                group *= 100; index += 1
                guard skipAnd() else { return nil }
                if let small = belowHundred(index) { group += small.value; index = small.next }
            }
            if index < words.endIndex, let scale = largeScales[words[index]] {
                guard group > 0, group < 1000, scale < previousScale else { return nil }
                total += group * scale; previousScale = scale; index += 1
                guard skipAnd() else { return nil }
                continue
            }
            guard index == words.endIndex, previousScale == Int.max || group < 1000 else { return nil }
            return total + group
        }
        return previousScale == Int.max ? nil : total
    }
}
