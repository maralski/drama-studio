import Foundation
import Metal

/// How a projection matrix is stored; picks the kernel family that reads it.
enum WeightFormat: String, CaseIterable {
    /// [N][K] ushort.
    case bf16
    /// [N][K] int8, plus [N][K/128] fp16 scales: symmetric, per row per 128 columns.
    case int8
    /// [N][K/2] bytes (element k in byte k/2, the even element in the low
    /// nibble), plus [N][K/64] (fp16 scale, fp16 bias): w = scale * q + bias, q in 0...15.
    case int4

    /// The infix of the GEMV and GEMM kernel names.
    var kernelTag: String {
        switch self {
        case .bf16: return "bf16"
        case .int8: return "i8"
        case .int4: return "i4"
        }
    }

    /// Bytes of an N x K matrix's weights, then of its scales or (scale, bias) pairs.
    func weightBytes(rows: Int, cols: Int) -> Int {
        switch self {
        case .bf16: return rows * cols * 2
        case .int8: return rows * cols
        case .int4: return rows * cols / 2
        }
    }

    func paramBytes(rows: Int, cols: Int) -> Int {
        switch self {
        case .bf16: return 0
        case .int8: return rows * (cols / 128) * 2
        case .int4: return rows * (cols / 64) * 4
        }
    }
}

/// How one layer's routed experts are stored.
enum ExpertFormat: CaseIterable {
    /// FP8 E4M3 bytes with fp32 128x128 block scales, as released.
    case fp8BlockScaled
    /// The dense int4 scheme per expert matrix: nibbles plus a half (scale, bias) per row per 64 columns.
    case int4

    /// The infix of the MoE kernel names.
    var kernelTag: String {
        switch self {
        case .fp8BlockScaled: return "fp8"
        case .int4: return "i4"
        }
    }
}

/// A projection matrix the GEMV and GEMM kernels read: `rows` x `cols`
/// weights at `offset`, and for a quantized format its scales at `paramOffset`.
struct WeightMatrix {
    let name: String
    let format: WeightFormat
    let rows: Int
    let cols: Int
    let buffer: MTLBuffer
    let offset: Int
    let paramOffset: Int
}

/// One layer's 256 experts as the MoE kernels address them: per expert the
/// GPU addresses of gate W, gate params, up W, up params, down W, down params.
struct ExpertTable {
    let table: MTLBuffer
    let format: ExpertFormat
}

/// Which precision table row a projection matrix falls under.
enum MatrixRole {
    case backbone
    case tower
    /// N <= 256: the router, in_proj_a, in_proj_b and the shared expert gate.
    case narrow
}

/// Where `ModelWeights` gets each item of its one ordered tensor list.
protocol WeightSource {
    func matrix(_ name: String, rows: Int, cols: Int, rowOffset: Int, role: MatrixRole) throws -> WeightMatrix
    /// Norms, conv taps, A_log, dt_bias and the q/k norms.
    func tensor(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef
    /// A bf16 lookup table the embedding kernels index, always read in place.
    func embedding(_ name: String, rows: Int, cols: Int) throws -> TensorRef
    /// `prefix` is the layer's name prefix; the experts hang off `prefix + "mlp.experts."`.
    func experts(layer: Int, prefix: String) throws -> ExpertTable
    /// Every Metal buffer a command buffer must declare to read the weights.
    var declaredBuffers: [MTLBuffer] { get }
}

/// Mode full: every weight read in place from the shards.
struct InPlaceSource: WeightSource {
    let store: SafeTensorsStore
    let config: ModelConfig
    let device: MTLDevice

    var declaredBuffers: [MTLBuffer] { store.shards }

    func tensor(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
        let ref = try store.tensor(name).expect(dtype, shape)
        guard ref.offset % 8 == 0 else { throw EngineError.badModel("\(name) is not 8-byte aligned") }
        return ref
    }

    func embedding(_ name: String, rows: Int, cols: Int) throws -> TensorRef {
        try tensor(name, .bf16, [rows, cols])
    }

    func matrix(_ name: String, rows: Int, cols: Int, rowOffset: Int, role: MatrixRole) throws -> WeightMatrix {
        let ref = try store.tensor(name)
        guard ref.dtype == .bf16, ref.shape.count == 2, ref.shape[1] == cols, rowOffset + rows <= ref.shape[0] else {
            throw EngineError.badModel("\(name): expected bf16 rows \(rowOffset)..<\(rowOffset + rows) of [*, \(cols)], found \(ref.dtype) \(ref.shape)")
        }
        guard ref.offset % 8 == 0 else { throw EngineError.badModel("\(name) is not 8-byte aligned") }
        return WeightMatrix(name: name, format: .bf16, rows: rows, cols: cols, buffer: ref.buffer, offset: ref.offset + rowOffset * cols * 2, paramOffset: 0)
    }

    /// Every expert's fp8 weights and block scales for one matrix kind of a layer.
    func expertRefs(prefix: String, kind: ExpertMatrix) throws -> [(weights: TensorRef, scales: TensorRef)] {
        let (rows, cols) = kind.shape(config)
        return try (0..<config.numExperts).map { e in
            let ep = prefix + "mlp.experts.\(e)." + kind.name
            return (try tensor(ep + ".weight", .f8e4m3, [rows, cols]), try tensor(ep + ".weight_scale_inv", .f32, [rows / 128, cols / 128]))
        }
    }

    func experts(layer: Int, prefix: String) throws -> ExpertTable {
        var table = [UInt64](repeating: 0, count: config.numExperts * 6)
        for kind in ExpertMatrix.allCases {
            for (e, refs) in try expertRefs(prefix: prefix, kind: kind).enumerated() {
                table[e * 6 + kind.rawValue * 2] = refs.weights.gpuAddress
                table[e * 6 + kind.rawValue * 2 + 1] = refs.scales.gpuAddress
            }
        }
        return ExpertTable(table: try ExpertTable.buffer(table, layer: layer, device: device), format: .fp8BlockScaled)
    }
}

extension ExpertTable {
    static func buffer(_ addresses: [UInt64], layer: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(bytes: addresses, length: addresses.count * 8, options: .storageModeShared) else {
            throw EngineError.badModel("cannot allocate expert table")
        }
        buffer.label = "experts.\(layer)"
        return buffer
    }
}
