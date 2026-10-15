enum ByteLevel {
    static let scalarForByte: [Unicode.Scalar] = {
        var next: UInt32 = 256
        return (0..<256).map { byte in
            if isPrintable(byte) { return Unicode.Scalar(UInt8(byte)) }
            defer { next += 1 }
            return Unicode.Scalar(next)!
        }
    }()

    static let byteForScalar: [Unicode.Scalar: UInt8] = Dictionary(
        uniqueKeysWithValues: scalarForByte.enumerated().map { ($1, UInt8($0)) }
    )

    static func token(forByte byte: UInt8) -> String {
        String(Character(scalarForByte[Int(byte)]))
    }

    static func appendBytes(of token: String, to bytes: inout [UInt8]) {
        let mapped = token.unicodeScalars.compactMap { byteForScalar[$0] }
        // Hugging Face falls back to the raw UTF-8 of any token holding an unmapped scalar.
        if mapped.count == token.unicodeScalars.count {
            bytes.append(contentsOf: mapped)
        } else {
            bytes.append(contentsOf: token.utf8)
        }
    }

    private static func isPrintable(_ byte: Int) -> Bool {
        (33...126).contains(byte) || (161...172).contains(byte) || (174...255).contains(byte)
    }
}
