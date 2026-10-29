// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A JSON value. Parsing goes through JSONSerialization so booleans and
/// integers keep their kinds, which a JSON-RPC id depends on.
public enum JSON: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSON])
    case object([String: JSON])

    public init(parsing data: Data) throws {
        self = JSON(any: try JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed))
    }

    public init(any value: Any) {
        switch value {
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else if CFNumberIsFloatType(number) {
                self = .double(number.doubleValue)
            } else {
                self = .int(number.intValue)
            }
        case let string as String: self = .string(string)
        case let array as [Any]: self = .array(array.map(JSON.init(any:)))
        case let object as [String: Any]: self = .object(object.mapValues(JSON.init(any:)))
        default: self = .null
        }
    }

    /// One line, no trailing newline. Hand-written because JSONSerialization
    /// prints 2.354 as 2.3540000000000001.
    public var serialized: String {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value): return value.isFinite ? "\(value)" : "null"
        case .string(let value): return Self.quote(value)
        case .array(let values): return "[" + values.map(\.serialized).joined(separator: ",") + "]"
        case .object(let values): return "{" + values.sorted { $0.key < $1.key }.map { Self.quote($0.key) + ":" + $0.value.serialized }.joined(separator: ",") + "}"
        }
    }

    private static func quote(_ string: String) -> String {
        var out = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    public subscript(key: String) -> JSON? {
        if case .object(let values) = self { return values[key] }
        return nil
    }

    public var string: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var bool: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    public var double: Double? {
        switch self {
        case .int(let value): Double(value)
        case .double(let value): value
        default: nil
        }
    }

    public var object: [String: JSON]? {
        if case .object(let values) = self { return values }
        return nil
    }
}

extension JSON: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral, ExpressibleByNilLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(arrayLiteral elements: JSON...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSON)...) { self = .object(Dictionary(elements) { _, last in last }) }
    public init(nilLiteral: ()) { self = .null }
}
