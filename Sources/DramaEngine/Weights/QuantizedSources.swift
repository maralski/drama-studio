import Foundation
import Metal

/// Walks the model's tensor list once for a quantized mode and assigns each
/// item its place in the cache file. The matrices and tables it hands out
/// are placeholders on a dummy buffer; the `ModelWeights` built from it is
/// thrown away and its product is `plan`. Two walks give the same plan.
final class LayoutRecorder: WeightSource {
    let inPlace: InPlaceSource
    let mode: ModelPrecisionMode
    private var items: [QuantizedLayout.Item] = []
    private var sources: [BuildSource] = []
    private var pinnedNames: [String] = []
    private var cursor = QuantizedLayout.headerBytes
    private let placeholder: MTLBuffer

    init(inPlace: InPlaceSource, mode: ModelPrecisionMode) {
        self.inPlace = inPlace
        self.mode = mode
        placeholder = inPlace.device.makeBuffer(length: 256, options: .storageModeShared)!
    }

    var declaredBuffers: [MTLBuffer] { [] }
    var plan: QuantizationPlan { QuantizationPlan(layout: QuantizedLayout(items: items), sources: sources, pinnedNames: pinnedNames) }

    private func record(_ name: String, _ kind: QuantizedLayout.Kind, _ format: WeightFormat, rows: Int, cols: Int, rowOffset: Int, source: BuildSource) {
        let item = QuantizedLayout.Item(name: name, kind: kind, format: format, rows: rows, cols: cols, rowOffset: rowOffset, offset: cursor,
                                        byteCount: QuantizedLayout.aligned(format.weightBytes(rows: rows, cols: cols)) + format.paramBytes(rows: rows, cols: cols))
        cursor = QuantizedLayout.aligned(item.offset + item.byteCount)
        items.append(item)
        sources.append(source)
    }

    func matrix(_ name: String, rows: Int, cols: Int, rowOffset: Int, role: MatrixRole) throws -> WeightMatrix {
        let w = try inPlace.matrix(name, rows: rows, cols: cols, rowOffset: rowOffset, role: role)
        let slice = TensorRef(name: name, dtype: .bf16, shape: [rows, cols], buffer: w.buffer, offset: w.offset)
        let format = mode.format(for: role)
        record(name, .matrix, format, rows: rows, cols: cols, rowOffset: rowOffset, source: format == .bf16 ? .copy(slice) : .quantize(slice))
        return WeightMatrix(name: name, format: format, rows: rows, cols: cols, buffer: placeholder, offset: 0, paramOffset: 0)
    }

    func tensor(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
        let ref = try inPlace.tensor(name, dtype, shape)
        guard dtype == .bf16 else { throw EngineError.badModel("\(name): the cache file copies bf16 vectors only") }
        record(name, .vector, .bf16, rows: ref.count, cols: 1, rowOffset: 0, source: .copy(ref))
        return ref
    }

    func embedding(_ name: String, rows: Int, cols: Int) throws -> TensorRef {
        let ref = try inPlace.embedding(name, rows: rows, cols: cols)
        pinnedNames.append(name)
        return ref
    }

    func experts(layer: Int, prefix: String) throws -> ExpertTable {
        for kind in ExpertMatrix.allCases {
            let refs = try inPlace.expertRefs(prefix: prefix, kind: kind)
            switch mode.expertFormat {
            case .fp8BlockScaled:
                pinnedNames += refs.flatMap { [$0.weights.name, $0.scales.name] }
            case .int4:
                let (rows, cols) = kind.shape(inPlace.config)
                record(prefix + "mlp.experts." + kind.name, .experts, .int4, rows: refs.count * rows, cols: cols, rowOffset: 0, source: .experts(refs, kind))
            }
        }
        return ExpertTable(table: placeholder, format: mode.expertFormat)
    }
}

/// Modes balanced and compact: matrices and vectors from the cache file,
/// embeddings and (in balanced) the experts from pinned shard runs. The
/// expert table addresses come from the run buffers, never from the whole
/// shards, which a quantized mode never binds.
final class CacheSource: WeightSource {
    let file: QuantizedWeightFile
    let layout: QuantizedLayout
    let mode: ModelPrecisionMode
    let inPlace: InPlaceSource
    private let pinned: [String: TensorRef]
    private let runs: [MTLBuffer]

    init(file: QuantizedWeightFile, plan: QuantizationPlan, mode: ModelPrecisionMode, inPlace: InPlaceSource) throws {
        self.file = file
        self.layout = plan.layout
        self.mode = mode
        self.inPlace = inPlace
        let refs = try plan.pinnedNames.map { try inPlace.store.tensor($0) }
        let (repointed, runs) = try inPlace.store.pinned(refs, device: inPlace.device)
        pinned = Dictionary(uniqueKeysWithValues: zip(plan.pinnedNames, repointed))
        self.runs = runs
    }

    var declaredBuffers: [MTLBuffer] { [file.buffer] + runs }

    private func item(_ name: String, rowOffset: Int = 0, _ kind: QuantizedLayout.Kind) throws -> QuantizedLayout.Item {
        guard let item = layout[.init(name: name, rowOffset: rowOffset)], item.kind == kind else {
            throw EngineError.badModel("\(name) at row \(rowOffset) is not a \(kind.rawValue) of the quantized layout")
        }
        return item
    }

    func matrix(_ name: String, rows: Int, cols: Int, rowOffset: Int, role: MatrixRole) throws -> WeightMatrix {
        let item = try item(name, rowOffset: rowOffset, .matrix)
        guard item.rows == rows, item.cols == cols, item.format == mode.format(for: role) else {
            throw EngineError.badModel("\(name): the layout holds \(item.format) \(item.rows)x\(item.cols), the model wants \(mode.format(for: role)) \(rows)x\(cols)")
        }
        return WeightMatrix(name: name, format: item.format, rows: rows, cols: cols, buffer: file.buffer, offset: item.offset, paramOffset: item.paramOffset)
    }

    func tensor(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
        let item = try item(name, .vector)
        guard dtype == .bf16, item.rows == shape.reduce(1, *) else { throw EngineError.badModel("\(name): expected \(dtype) \(shape), the layout holds \(item.rows) bf16") }
        return TensorRef(name: name, dtype: dtype, shape: shape, buffer: file.buffer, offset: item.offset)
    }

    func embedding(_ name: String, rows: Int, cols: Int) throws -> TensorRef {
        guard let ref = pinned[name] else { throw EngineError.badModel("\(name) is not pinned") }
        return try ref.expect(.bf16, [rows, cols])
    }

    func experts(layer: Int, prefix: String) throws -> ExpertTable {
        let config = inPlace.config
        var addresses = [UInt64](repeating: 0, count: config.numExperts * 6)
        for kind in ExpertMatrix.allCases {
            switch mode.expertFormat {
            case .fp8BlockScaled:
                for e in 0..<config.numExperts {
                    let ep = prefix + "mlp.experts.\(e)." + kind.name
                    guard let w = pinned[ep + ".weight"], let s = pinned[ep + ".weight_scale_inv"] else { throw EngineError.badModel("\(ep) is not pinned") }
                    addresses[e * 6 + kind.rawValue * 2] = w.gpuAddress
                    addresses[e * 6 + kind.rawValue * 2 + 1] = s.gpuAddress
                }
            case .int4:
                let item = try item(prefix + "mlp.experts." + kind.name, .experts)
                let (rows, cols) = kind.shape(config)
                guard item.rows == config.numExperts * rows, item.cols == cols else { throw EngineError.badModel("\(item.name): unexpected shape \(item.rows)x\(item.cols)") }
                for e in 0..<config.numExperts {
                    addresses[e * 6 + kind.rawValue * 2] = file.buffer.gpuAddress + UInt64(item.offset + e * WeightFormat.int4.weightBytes(rows: rows, cols: cols))
                    addresses[e * 6 + kind.rawValue * 2 + 1] = file.buffer.gpuAddress + UInt64(item.paramOffset + e * WeightFormat.int4.paramBytes(rows: rows, cols: cols))
                }
            }
        }
        return ExpertTable(table: try ExpertTable.buffer(addresses, layer: layer, device: inPlace.device), format: mode.expertFormat)
    }
}
