import Foundation
import Metal

/// The three expert matrices of a layer, in the address table's order.
enum ExpertMatrix: Int, CaseIterable {
    case gate, up, down

    /// The tensor name stem under `mlp.experts.<e>.`.
    var name: String {
        switch self {
        case .gate: return "gate_proj"
        case .up: return "up_proj"
        case .down: return "down_proj"
        }
    }

    func shape(_ config: ModelConfig) -> (rows: Int, cols: Int) {
        self == .down ? (config.hiddenSize, config.moeIntermediate) : (config.moeIntermediate, config.hiddenSize)
    }
}

/// A destination for a quantizer: weights at `offset`, scales or (scale, bias)
/// pairs at `paramOffset`, both inside `buffer`.
struct QuantizedSlot {
    let buffer: MTLBuffer
    let offset: Int
    let paramOffset: Int
}

extension Kernels {
    private struct QuantizeParams { var N: UInt32; var K: UInt32; var rowOffset: UInt32; var kind: UInt32 }

    /// Rows `rowOffset..<rowOffset + rows` of the bf16 [*, cols] tensor at
    /// `src` to int8 with half scales per row per 128 columns.
    func quantizeInt8(_ enc: MTLComputeCommandEncoder, src: TensorRef, rowOffset: Int, rows: Int, cols: Int, into slot: QuantizedSlot) throws {
        precondition(cols % 128 == 0)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(slot.buffer, offset: slot.offset, index: 1)
        enc.setBuffer(slot.buffer, offset: slot.paramOffset, index: 2)
        set(enc, QuantizeParams(N: UInt32(rows), K: UInt32(cols), rowOffset: UInt32(rowOffset), kind: 0), index: 3)
        try dispatch(enc, "quantize_bf16_i8", grid: MTLSize(width: cols / 128, height: rows, depth: 1), group: MTLSize(width: min(cols / 128, 16), height: 16, depth: 1))
    }

    /// The same rows to nibbles with a half (scale, bias) per row per 64 columns.
    func quantizeInt4(_ enc: MTLComputeCommandEncoder, src: TensorRef, rowOffset: Int, rows: Int, cols: Int, into slot: QuantizedSlot) throws {
        precondition(cols % 128 == 0)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(slot.buffer, offset: slot.offset, index: 1)
        enc.setBuffer(slot.buffer, offset: slot.paramOffset, index: 2)
        set(enc, QuantizeParams(N: UInt32(rows), K: UInt32(cols), rowOffset: UInt32(rowOffset), kind: 0), index: 3)
        try dispatch(enc, "quantize_bf16_i4", grid: MTLSize(width: cols / 64, height: rows, depth: 1), group: MTLSize(width: min(cols / 64, 16), height: 16, depth: 1))
    }

    /// One matrix kind of `count` fp8 experts to nibbles, expert-major: expert
    /// e's weights at `slot.offset + e * rows * cols / 2`, its params at
    /// `slot.paramOffset + e * rows * cols / 64 * 4`.
    func quantizeExpertsInt4(_ enc: MTLComputeCommandEncoder, experts: ExpertTable, kind: ExpertMatrix, count: Int, rows: Int, cols: Int, into slot: QuantizedSlot) throws {
        precondition(experts.format == .fp8BlockScaled && cols % 128 == 0 && rows % 128 == 0)
        enc.setBuffer(experts.table, offset: 0, index: 0)
        enc.setBuffer(slot.buffer, offset: slot.offset, index: 1)
        enc.setBuffer(slot.buffer, offset: slot.paramOffset, index: 2)
        set(enc, QuantizeParams(N: UInt32(rows), K: UInt32(cols), rowOffset: 0, kind: UInt32(kind.rawValue)), index: 3)
        try dispatch(enc, "quantize_fp8_i4", grid: MTLSize(width: cols / 64, height: rows, depth: count), group: MTLSize(width: min(cols / 64, 16), height: 16, depth: 1))
    }
}
