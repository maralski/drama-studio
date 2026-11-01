// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One quantization path for recognition, WAV export and streamed PCM.
package enum PCM16 {
    package static func encode(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let value = Int16((max(-1, min(1, sample)) * 32767).rounded())
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    package static func wav(_ samples: [Float]) -> Data {
        let pcm = encode(samples)
        var header = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { header.append(contentsOf: $0) } }
        header.append(Data("RIFF".utf8))
        append(UInt32(36 + pcm.count))
        header.append(Data("WAVEfmt ".utf8))
        append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(UInt32(ModelConstants.sampleRate)); append(UInt32(ModelConstants.sampleRate * 2))
        append(UInt16(2)); append(UInt16(16))
        header.append(Data("data".utf8)); append(UInt32(pcm.count))
        return header + pcm
    }
}
