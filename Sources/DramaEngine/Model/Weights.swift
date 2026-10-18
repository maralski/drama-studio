import Foundation
import Metal

struct LinearAttentionWeights {
    let inProjQKV: TensorRef   // [8192, 2048]
    let inProjZ: TensorRef     // [4096, 2048]
    let inProjA: TensorRef     // [32, 2048]
    let inProjB: TensorRef     // [32, 2048]
    let conv: TensorRef           // [8192, 1, 4]
    let aLog: TensorRef           // [32]
    let dtBias: TensorRef         // [32]
    let norm: TensorRef           // [128]
    let outProj: TensorRef     // [2048, 4096]
}

struct FullAttentionWeights {
    let qProj: TensorRef   // [8192, 2048] rows are [q | gate] per head
    let kProj: TensorRef   // [512, 2048]
    let vProj: TensorRef   // [512, 2048]
    let oProj: TensorRef   // [2048, 4096]
    let qNorm: TensorRef      // [256]
    let kNorm: TensorRef      // [256]
}

enum Mixer {
    case linear(LinearAttentionWeights)
    case full(FullAttentionWeights)
}

struct MoEWeights {
    let router: TensorRef           // [256, 2048]
    let sharedGate: TensorRef       // [512, 2048]
    let sharedUp: TensorRef         // [512, 2048]
    let sharedDown: TensorRef       // [2048, 512]
    let sharedExpertGate: TensorRef // [1, 2048]
    /// Per expert the GPU addresses of gate W, gate scales, up W, up scales, down W, down scales.
    let experts: MTLBuffer
}

struct BackboneLayerWeights {
    let inputNorm: TensorRef
    let postNorm: TensorRef
    let mixer: Mixer
    let moe: MoEWeights
}

struct BackboneWeights {
    let embed: TensorRef            // [252288, 2048]
    let finalNorm: TensorRef
    /// The only lm_head rows the engine reads: the eos row, and the semantic band.
    let lmHeadEos: TensorRef      // [1, 2048]
    let lmHeadSemantic: TensorRef // [4096, 2048]
    let layers: [BackboneLayerWeights]
}

struct TowerLayerWeights {
    let embedding: TensorRef   // [4096, 2048]
    let enorm: TensorRef
    let hnorm: TensorRef
    let ehProj: TensorRef   // [2048, 4096]
    let attentionNorm: TensorRef
    let ffnNorm: TensorRef
    let wqkv: TensorRef     // [9216, 2048] rows [Q | Gate | K | V]
    let wo: TensorRef       // [2048, 4096]
    let qNorm: TensorRef
    let kNorm: TensorRef
    let w1: TensorRef       // [6144, 2048]
    let w3: TensorRef       // [6144, 2048]
    let w2: TensorRef       // [2048, 6144]
    let finalNorm: TensorRef
}

struct TowerWeights {
    let codebookEmbeddings: TensorRef  // [40960, 2048]
    let shared: [TowerLayerWeights]
    let residual: [TowerLayerWeights]
    let heads: [TensorRef]          // 9 x [4096, 2048]
}

/// Every weight the engine reads, in place in the mapped shards.
struct ModelWeights {
    let backbone: BackboneWeights
    let tower: TowerWeights

    init(store: SafeTensorsStore, config: ModelConfig, device: MTLDevice) throws {
        let H = config.hiddenSize
        func t(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef {
            let ref = try store.tensor(name).expect(dtype, shape)
            guard ref.offset % 8 == 0 else { throw EngineError.badModel("\(name) is not 8-byte aligned") }
            return ref
        }
        func m(_ name: String, _ rows: Int, _ cols: Int, rowOffset: Int = 0) throws -> TensorRef {
            let ref = try store.tensor(name)
            guard ref.dtype == .bf16, ref.shape.count == 2, ref.shape[1] == cols, rowOffset + rows <= ref.shape[0] else {
                throw EngineError.badModel("\(name): expected bf16 rows \(rowOffset)..<\(rowOffset + rows) of [*, \(cols)], found \(ref.dtype) \(ref.shape)")
            }
            guard ref.offset % 8 == 0 else { throw EngineError.badModel("\(name) is not 8-byte aligned") }
            return TensorRef(name: name, dtype: .bf16, shape: [rows, cols], buffer: ref.buffer, offset: ref.offset + rowOffset * cols * 2)
        }
        func experts(layer: Int, prefix: String) throws -> MTLBuffer {
            let I = config.moeIntermediate
            var table = [UInt64](repeating: 0, count: config.numExperts * 6)
            for e in 0..<config.numExperts {
                let ep = prefix + "mlp.experts.\(e)."
                table[e * 6 + 0] = try t(ep + "gate_proj.weight", .f8e4m3, [I, H]).gpuAddress
                table[e * 6 + 1] = try t(ep + "gate_proj.weight_scale_inv", .f32, [I / 128, H / 128]).gpuAddress
                table[e * 6 + 2] = try t(ep + "up_proj.weight", .f8e4m3, [I, H]).gpuAddress
                table[e * 6 + 3] = try t(ep + "up_proj.weight_scale_inv", .f32, [I / 128, H / 128]).gpuAddress
                table[e * 6 + 4] = try t(ep + "down_proj.weight", .f8e4m3, [H, I]).gpuAddress
                table[e * 6 + 5] = try t(ep + "down_proj.weight_scale_inv", .f32, [H / 128, I / 128]).gpuAddress
            }
            guard let buffer = device.makeBuffer(bytes: table, length: table.count * 8, options: .storageModeShared) else {
                throw EngineError.badModel("cannot allocate expert table")
            }
            buffer.label = "experts.\(layer)"
            return buffer
        }
        var layers: [BackboneLayerWeights] = []
        for i in 0..<config.numLayers {
            let p = "model.language_model.layers.\(i)."
            let mixer: Mixer
            switch config.layerTypes[i] {
            case .linear:
                mixer = .linear(LinearAttentionWeights(
                    inProjQKV: try m(p + "linear_attn.in_proj_qkv.weight", 8192, H),
                    inProjZ: try m(p + "linear_attn.in_proj_z.weight", 4096, H),
                    inProjA: try m(p + "linear_attn.in_proj_a.weight", 32, H),
                    inProjB: try m(p + "linear_attn.in_proj_b.weight", 32, H),
                    conv: try t(p + "linear_attn.conv1d.weight", .bf16, [8192, 1, 4]),
                    aLog: try t(p + "linear_attn.A_log", .bf16, [32]),
                    dtBias: try t(p + "linear_attn.dt_bias", .bf16, [32]),
                    norm: try t(p + "linear_attn.norm.weight", .bf16, [128]),
                    outProj: try m(p + "linear_attn.out_proj.weight", H, 4096)))
            case .full:
                mixer = .full(FullAttentionWeights(
                    qProj: try m(p + "self_attn.q_proj.weight", 8192, H),
                    kProj: try m(p + "self_attn.k_proj.weight", 512, H),
                    vProj: try m(p + "self_attn.v_proj.weight", 512, H),
                    oProj: try m(p + "self_attn.o_proj.weight", H, 4096),
                    qNorm: try t(p + "self_attn.q_norm.weight", .bf16, [256]),
                    kNorm: try t(p + "self_attn.k_norm.weight", .bf16, [256])))
            }
            layers.append(BackboneLayerWeights(
                inputNorm: try t(p + "input_layernorm.weight", .bf16, [H]),
                postNorm: try t(p + "post_attention_layernorm.weight", .bf16, [H]),
                mixer: mixer,
                moe: MoEWeights(
                    router: try m(p + "mlp.gate.weight", config.numExperts, H),
                    sharedGate: try m(p + "mlp.shared_expert.gate_proj.weight", 512, H),
                    sharedUp: try m(p + "mlp.shared_expert.up_proj.weight", 512, H),
                    sharedDown: try m(p + "mlp.shared_expert.down_proj.weight", H, 512),
                    sharedExpertGate: try m(p + "mlp.shared_expert_gate.weight", 1, H),
                    experts: try experts(layer: i, prefix: p))))
        }
        backbone = BackboneWeights(
            embed: try t("model.language_model.embed_tokens.weight", .bf16, [config.vocabSize, H]),
            finalNorm: try t("model.language_model.norm.weight", .bf16, [H]),
            lmHeadEos: try m("lm_head.weight", 1, H, rowOffset: Int(config.eos)),
            lmHeadSemantic: try m("lm_head.weight", config.codebookSize, H, rowOffset: Int(config.semanticStart)),
            layers: layers)

        func towerLayer(_ p: String, embeddingName: String) throws -> TowerLayerWeights {
            let a = p + "transformer_layer.attention."
            let f = p + "transformer_layer.feed_forward."
            return TowerLayerWeights(
                embedding: try t(p + embeddingName + ".weight", .bf16, [config.codebookSize, H]),
                enorm: try t(p + "enorm.weight", .bf16, [H]),
                hnorm: try t(p + "hnorm.weight", .bf16, [H]),
                ehProj: try m(p + "eh_proj.weight", H, 2 * H),
                attentionNorm: try t(p + "transformer_layer.attention_norm.weight", .bf16, [H]),
                ffnNorm: try t(p + "transformer_layer.ffn_norm.weight", .bf16, [H]),
                wqkv: try m(a + "wqkv.weight", 9216, H),
                wo: try m(a + "wo.weight", H, 4096),
                qNorm: try t(a + "q_norm.weight", .bf16, [256]),
                kNorm: try t(a + "k_norm.weight", .bf16, [256]),
                w1: try m(f + "w1.weight", config.towerIntermediate, H),
                w3: try m(f + "w3.weight", config.towerIntermediate, H),
                w2: try m(f + "w2.weight", H, config.towerIntermediate),
                finalNorm: try t(p + "final_layernorm.weight", .bf16, [H]))
        }
        tower = TowerWeights(
            codebookEmbeddings: try t("audio_decoder.codebook_embeddings.weight", .bf16, [config.codebookSize * config.numCodebooks, H]),
            shared: try (0..<config.towerLayers).map { try towerLayer("audio_decoder_mtp_shared.layers.\($0).", embeddingName: "semantic_embedding") },
            residual: try (0..<(config.numCodebooks - 1)).map { try towerLayer("audio_decoder_mtp.layers.\($0).", embeddingName: "codebook_embedding") },
            heads: try (0..<(config.numCodebooks - 1)).map { try m("audio_decoder_mtp.heads.\($0).weight", config.codebookSize, H) })
    }
}
