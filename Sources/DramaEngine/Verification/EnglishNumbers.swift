// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// English numbers as the set of readings they can have. One number becomes one
/// token, `#` followed by its readings joined with `|`, so token ordinals stay
/// one per spoken unit and two numbers match when their readings intersect.
/// Each reading starts with its `Kind`, so a digit-by-digit "three two one"
/// and a cardinal "three hundred twenty one" never match each other, while a
/// written "321" carries both. A currency becomes its own token after the
/// number: `¤dollar` when spoken, `¤?dollar` when only a symbol implies it.
enum EnglishNumbers {
    enum Kind: Character {
        /// Digit by digit, leading zeros kept: "double oh seven" is 007.
        case digits = "d"
        /// "three hundred twenty one", "twelve hundred", "two point five million".
        case cardinal = "c"
        /// Two groups, the second always two digits: "seven forty seven" is 7.47,
        /// "nine thirty" is 9.30, "nineteen oh five" is 19.05.
        case paired = "t"
        /// "two point five" is 2.5.
        case decimal = "f"
        /// "nineties" is 90s.
        case decade = "s"
        /// Number words that form no reading, such as "twelve five"; they match
        /// only the same words.
        case words = "w"

        func reading(_ value: String) -> String { String(rawValue) + value }
    }

    enum Piece {
        case word(String)
        case written(Set<String>, currency: String?)
    }

    /// What separates a piece from the one before it.
    enum Gap { case space, comma, other }

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
    private static let currencyWords = ["dollar": "dollar", "dollars": "dollar", "cent": "cent", "cents": "cent",
                                        "pound": "pound", "pounds": "pound", "euro": "euro", "euros": "euro"]
    static let currencySymbols: [Character: String] = ["$": "dollar", "£": "pound", "€": "euro", "¢": "cent"]

    static func token(_ readings: Set<String>) -> String { "#" + readings.sorted().joined(separator: "|") }

    static func isNumber(_ token: String) -> Bool { token.hasPrefix("#") }

    private static func currency(_ token: String) -> Substring? {
        token.hasPrefix("¤") ? token.dropFirst().drop { $0 == "?" } : nil
    }

    static func isCurrency(_ token: String) -> Bool { token.hasPrefix("¤") }

    /// A currency symbol that the speaker may or may not have read aloud.
    static func isImpliedCurrency(_ token: String) -> Bool { token.hasPrefix("¤?") }

    static func same(_ left: String, _ right: String) -> Bool {
        guard left != right else { return true }
        if isNumber(left), isNumber(right) {
            return !Set(left.dropFirst().split(separator: "|")).isDisjoint(with: right.dropFirst().split(separator: "|"))
        }
        guard let first = currency(left), let second = currency(right) else { return false }
        return first == second
    }

    /// Readings of a written group. `integer` may hold thousands commas;
    /// `separator` is "." or ":"; `decade` marks a trailing "s" or "'s".
    static func written(integer: String?, separator: String?, fraction: String?, decade: Bool) -> Set<String>? {
        let digits = (integer ?? "0").replacingOccurrences(of: ",", with: "")
        let significant = digits.drop { $0 == "0" }
        let trimmed = significant.isEmpty ? "0" : String(significant)
        if decade {
            guard separator == nil else { return nil }
            return decadeReadings(digits)
        }
        guard let separator, let fraction else {
            guard digits == trimmed else { return [Kind.digits.reading(digits)] }
            var readings: Set = [Kind.digits.reading(digits), Kind.cardinal.reading(digits)]
            if (3...4).contains(digits.count) {
                readings.insert(Kind.paired.reading(digits.dropLast(2) + "." + digits.suffix(2)))
            }
            return readings
        }
        if separator == ":" {
            return [Kind.paired.reading(fraction.count == 2 ? trimmed + "." + fraction : digits + ":" + fraction)]
        }
        var readings: Set = [Kind.decimal.reading(trimmed + "." + fraction)]
        if fraction.count == 2 { readings.insert(Kind.paired.reading(trimmed + "." + fraction)) }
        let shortened = fraction.reversed().drop { $0 == "0" }
        if !shortened.isEmpty { readings.insert(Kind.decimal.reading(trimmed + "." + String(shortened.reversed()))) }
        return readings
    }

    private static func decadeReadings(_ digits: String) -> Set<String>? {
        guard [2, 4].contains(digits.count), digits.hasSuffix("0") else { return nil }
        let short = Kind.decade.reading(digits.suffix(2) + "s")
        return digits.count == 4 ? [Kind.decade.reading(digits + "s"), short] : [short]
    }

    /// Groups number words and written numbers into number tokens, turns a
    /// currency symbol or a currency word after a number into a currency token,
    /// and applies `spellings` to the other words.
    static func tokens(_ pieces: [(piece: Piece, gap: Gap)], spellings: [String: String]) -> [String] {
        enum Part { case word(String), number(Set<String>, currency: String?) }
        func word(_ index: Int) -> String? {
            guard index < pieces.count, case .word(let word) = pieces[index].piece else { return nil }
            return word
        }
        func joinedWord(_ index: Int) -> String? { index < pieces.count && pieces[index].gap == .space ? word(index) : nil }
        func isDigit(_ word: String?) -> Bool { word == "oh" || word.flatMap { units[$0] }.map { $0 < 10 } == true }
        func startsRun(_ index: Int) -> Bool {
            guard let word = word(index) else { return false }
            if units[word] != nil || tens[word] != nil || decades[word] != nil { return true }
            switch word {
            case "a": return joinedWord(index + 1).flatMap { scales[$0] } != nil
            case "oh": return joinedWord(index + 1) != "oh" && isDigit(joinedWord(index + 1)) && isDigit(joinedWord(index + 2))
            case "double", "triple", "point": return isDigit(joinedWord(index + 1))
            default: return false
            }
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

        var parts: [(part: Part, gap: Gap)] = []
        var index = 0
        while index < pieces.count {
            let gap = pieces[index].gap
            if case .written(let readings, let currency) = pieces[index].piece {
                let scaled = joinedWord(index + 1).flatMap { scales[$0] }.map { multiplied(readings, by: $0) } ?? []
                parts.append((.number(scaled.isEmpty ? readings : scaled, currency: currency), gap))
                index += scaled.isEmpty ? 1 : 2
                continue
            }
            let end = runEnd(from: index)
            if end > index {
                parts.append((.number(spokenReadings((index..<end).compactMap(word)), currency: nil), gap)); index = end
            } else {
                parts.append((.word(word(index)!), gap)); index += 1
            }
        }

        var merged: [Part] = []
        for (part, gap) in parts {
            if case .number(let right, nil) = part, case .number(let left, nil)? = merged.last,
               let both = joined(left, right, across: gap) {
                merged[merged.count - 1] = .number(both, currency: nil)
            } else {
                merged.append(part)
            }
        }

        var result: [String] = []
        for (offset, part) in merged.enumerated() {
            switch part {
            case .number(let readings, let currency):
                result.append(token(readings))
                if let currency { result.append("¤?" + currency) }
            case .word(let word):
                if let currency = currencyWords[word], offset > 0, case .number = merged[offset - 1] {
                    if result.last.map(isImpliedCurrency) == true { result.removeLast() }
                    result.append("¤" + currency)
                } else if let spelling = spellings[word] {
                    result += spelling.split(separator: " ").map(String.init)
                } else {
                    result.append(word)
                }
            }
        }
        return result
    }

    private static func values(_ readings: Set<String>, _ kind: Kind) -> [String] {
        readings.filter { $0.first == kind.rawValue }.map { String($0.dropFirst()) }.sorted()
    }

    /// One identifier from two digit groups, as in a phone number. Groups
    /// separated by a space join when both can be read digit by digit; across a
    /// comma only spoken digit runs join, so "three, two, one" stays three numbers.
    private static func joined(_ left: Set<String>, _ right: Set<String>, across gap: Gap) -> Set<String>? {
        func onlyDigits(_ readings: Set<String>) -> Bool { readings.allSatisfy { $0.first == Kind.digits.rawValue } }
        switch gap {
        case .space: break
        case .comma: guard onlyDigits(left), onlyDigits(right) else { return nil }
        case .other: return nil
        }
        let lefts = values(left, .digits), rights = values(right, .digits)
        guard !lefts.isEmpty, !rights.isEmpty else { return nil }
        var readings = Set<String>()
        for first in lefts {
            for second in rights {
                readings.insert(Kind.digits.reading(first + second))
                if gap == .space, second.count == 2, first.count <= 2, !first.hasPrefix("0") {
                    readings.insert(Kind.paired.reading(first + "." + second))
                }
            }
        }
        return readings
    }

    private static func multiplied(_ readings: Set<String>, by scale: Int) -> Set<String> {
        Set((values(readings, .cardinal) + values(readings, .decimal)).compactMap { value in
            Decimal(string: value, locale: Locale(identifier: "en_US_POSIX")).map {
                Kind.cardinal.reading(NSDecimalNumber(decimal: $0 * Decimal(scale)).stringValue)
            }
        })
    }

    private static func spokenReadings(_ run: [String]) -> Set<String> {
        let readings = readings(run)
        return readings.isEmpty ? [Kind.words.reading(run.joined(separator: " "))] : readings
    }

    private static func readings(_ run: [String]) -> Set<String> {
        guard let last = run.last else { return [] }
        if let tensWord = decades[last] {
            let spoken = readings(Array(run.dropLast()) + [tensWord])
            let years = values(spoken, .cardinal) + values(spoken, .paired).map { $0.replacingOccurrences(of: ".", with: "") }
            return Set(years.compactMap(decadeReadings).joined())
        }
        if let point = run.firstIndex(of: "point") {
            var fraction = run[(point + 1)...], scale: Int?
            if let last = fraction.last, let large = largeScales[last] { scale = large; fraction = fraction.dropLast() }
            guard let whole = point == 0 ? 0 : cardinal(run[..<point]), !fraction.isEmpty,
                  fraction.allSatisfy({ $0 == "oh" || units[$0].map { $0 < 10 } == true }) else { return [] }
            let decimal = Kind.decimal.reading("\(whole)." + fraction.map { $0 == "oh" ? "0" : String(units[$0]!) }.joined())
            return scale.map { multiplied([decimal], by: $0) } ?? [decimal]
        }
        var readings = Set<String>()
        if let value = cardinal(run[...]) { readings.insert(Kind.cardinal.reading(String(value))) }
        if let digits = digitSequence(run) { readings.insert(Kind.digits.reading(digits)) }
        for split in 1..<min(3, run.count) {
            guard let first = small(run[..<split]) else { continue }
            let rest = run[split...]
            if let second = small(rest), second >= 10 {
                readings.insert(Kind.paired.reading("\(first).\(second)"))
            } else if rest.count == 2, rest.first == "oh", let digit = units[rest.last!], digit < 10 {
                readings.insert(Kind.paired.reading("\(first).0\(digit)"))
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
