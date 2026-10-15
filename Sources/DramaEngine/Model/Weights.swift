import Foundation
import Metal

struct LinearAttentionWeights {
    let inProjQKV: TensorRef   // [8192, 2048]
    let inProjZ: TensorRef     // [4096, 2048]
    let inProjA: TensorRef     // [32, 2048]
    let inProjB: TensorRef     // [32, 2048]
    let conv: TensorRef        // [8192, 1, 4]
    let aLog: TensorRef        // [32]
    let dtBias: TensorRef      // [32]
    let norm: TensorRef        // [128]
    let outProj: TensorRef     // [2048, 4096]
}

struct FullAttentionWeights {
    let qProj: TensorRef   // [8192, 2048] rows are [q | gate] per head
    let kProj: TensorRef   // [512, 2048]
    let vProj: TensorRef   // [512, 2048]
    let oProj: TensorRef   // [2048, 4096]
    let qNorm: TensorRef   // [256]
    let kNorm: TensorRef   // [256]
}

enum Mixer {
    case linear(LinearAttentionWeights)
    case full(FullAttentionWeights)
}

struct MoEWeights {
    let router: TensorRef          // [256, 2048]
    let sharedGate: TensorRef      // [512, 2048]
    let sharedUp: TensorRef        // [512, 2048]
    let sharedDown: TensorRef      // [2048, 512]
    let sharedExpertGate: TensorRef // [1, 2048]
    /// GPU addresses, per expert: gate W, gate scale, up W, up scale, down W, down scale.
    let table: MTLBuffer
}

struct BackboneLayerWeights {
    let inputNorm: TensorRef
    let postNorm: TensorRef
    let mixer: Mixer
    let moe: MoEWeights
}

struct BackboneWeights {
    let embed: TensorRef   // [252288, 2048]
    let finalNorm: TensorRef
    let lmHead: TensorRef  // [252288, 2048]
    let layers: [BackboneLayerWeights]
}

struct TowerLayerWeights {
    let embedding: TensorRef   // [4096, 2048]
    let enorm: TensorRef
    let hnorm: TensorRef
    let ehProj: TensorRef      // [2048, 4096]
    let attentionNorm: TensorRef
    let ffnNorm: TensorRef
    let wqkv: TensorRef        // [9216, 2048] rows [Q | Gate | K | V]
    let wo: TensorRef          // [2048, 4096]
    let qNorm: TensorRef
    let kNorm: TensorRef
    let w1: TensorRef          // [6144, 2048]
    let w3: TensorRef          // [6144, 2048]
    let w2: TensorRef          // [2048, 6144]
    let finalNorm: TensorRef
}

struct TowerWeights {
    let codebookEmbeddings: TensorRef  // [40960, 2048]
    let shared: [TowerLayerWeights]
    let residual: [TowerLayerWeights]
    let heads: [TensorRef]             // 9 x [4096, 2048]
}

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
        var layers: [BackboneLayerWeights] = []
        for i in 0..<config.numLayers {
            let p = "model.language_model.layers.\(i)."
            let mixer: Mixer
            switch config.layerTypes[i] {
            case .linear:
                mixer = .linear(LinearAttentionWeights(
                    inProjQKV: try t(p + "linear_attn.in_proj_qkv.weight", .bf16, [8192, H]),
                    inProjZ: try t(p + "linear_attn.in_proj_z.weight", .bf16, [4096, H]),
                    inProjA: try t(p + "linear_attn.in_proj_a.weight", .bf16, [32, H]),
                    inProjB: try t(p + "linear_attn.in_proj_b.weight", .bf16, [32, H]),
                    conv: try t(p + "linear_attn.conv1d.weight", .bf16, [8192, 1, 4]),
                    aLog: try t(p + "linear_attn.A_log", .bf16, [32]),
                    dtBias: try t(p + "linear_attn.dt_bias", .bf16, [32]),
                    norm: try t(p + "linear_attn.norm.weight", .bf16, [128]),
                    outProj: try t(p + "linear_attn.out_proj.weight", .bf16, [H, 4096])))
            case .full:
                mixer = .full(FullAttentionWeights(
                    qProj: try t(p + "self_attn.q_proj.weight", .bf16, [8192, H]),
                    kProj: try t(p + "self_attn.k_proj.weight", .bf16, [512, H]),
                    vProj: try t(p + "self_attn.v_proj.weight", .bf16, [512, H]),
                    oProj: try t(p + "self_attn.o_proj.weight", .bf16, [H, 4096]),
                    qNorm: try t(p + "self_attn.q_norm.weight", .bf16, [256]),
                    kNorm: try t(p + "self_attn.k_norm.weight", .bf16, [256])))
            }
            var table = [UInt64](repeating: 0, count: config.numExperts * 6)
            for e in 0..<config.numExperts {
                let ep = p + "mlp.experts.\(e)."
                table[e * 6 + 0] = try t(ep + "gate_proj.weight", .f8e4m3, [512, H]).gpuAddress
                table[e * 6 + 1] = try t(ep + "gate_proj.weight_scale_inv", .f32, [4, 16]).gpuAddress
                table[e * 6 + 2] = try t(ep + "up_proj.weight", .f8e4m3, [512, H]).gpuAddress
                table[e * 6 + 3] = try t(ep + "up_proj.weight_scale_inv", .f32, [4, 16]).gpuAddress
                table[e * 6 + 4] = try t(ep + "down_proj.weight", .f8e4m3, [H, 512]).gpuAddress
                table[e * 6 + 5] = try t(ep + "down_proj.weight_scale_inv", .f32, [16, 4]).gpuAddress
            }
            guard let tableBuffer = device.makeBuffer(bytes: table, length: table.count * 8, options: .storageModeShared) else {
                throw EngineError.badModel("cannot allocate expert table")
            }
            tableBuffer.label = "experts.\(i)"
            layers.append(BackboneLayerWeights(
                inputNorm: try t(p + "input_layernorm.weight", .bf16, [H]),
                postNorm: try t(p + "post_attention_layernorm.weight", .bf16, [H]),
                mixer: mixer,
                moe: MoEWeights(
                    router: try t(p + "mlp.gate.weight", .bf16, [config.numExperts, H]),
                    sharedGate: try t(p + "mlp.shared_expert.gate_proj.weight", .bf16, [512, H]),
                    sharedUp: try t(p + "mlp.shared_expert.up_proj.weight", .bf16, [512, H]),
                    sharedDown: try t(p + "mlp.shared_expert.down_proj.weight", .bf16, [H, 512]),
                    sharedExpertGate: try t(p + "mlp.shared_expert_gate.weight", .bf16, [1, H]),
                    table: tableBuffer)))
        }
        backbone = BackboneWeights(
            embed: try t("model.language_model.embed_tokens.weight", .bf16, [config.vocabSize, H]),
            finalNorm: try t("model.language_model.norm.weight", .bf16, [H]),
            lmHead: try t("lm_head.weight", .bf16, [config.vocabSize, H]),
            layers: layers)

        func towerLayer(_ p: String, embeddingName: String) throws -> TowerLayerWeights {
            let a = p + "transformer_layer.attention."
            let f = p + "transformer_layer.feed_forward."
            return TowerLayerWeights(
                embedding: try t(p + embeddingName + ".weight", .bf16, [config.codebookSize, H]),
                enorm: try t(p + "enorm.weight", .bf16, [H]),
                hnorm: try t(p + "hnorm.weight", .bf16, [H]),
                ehProj: try t(p + "eh_proj.weight", .bf16, [H, 2 * H]),
                attentionNorm: try t(p + "transformer_layer.attention_norm.weight", .bf16, [H]),
                ffnNorm: try t(p + "transformer_layer.ffn_norm.weight", .bf16, [H]),
                wqkv: try t(a + "wqkv.weight", .bf16, [9216, H]),
                wo: try t(a + "wo.weight", .bf16, [H, 4096]),
                qNorm: try t(a + "q_norm.weight", .bf16, [256]),
                kNorm: try t(a + "k_norm.weight", .bf16, [256]),
                w1: try t(f + "w1.weight", .bf16, [config.towerIntermediate, H]),
                w3: try t(f + "w3.weight", .bf16, [config.towerIntermediate, H]),
                w2: try t(f + "w2.weight", .bf16, [H, config.towerIntermediate]),
                finalNorm: try t(p + "final_layernorm.weight", .bf16, [H]))
        }
        tower = TowerWeights(
            codebookEmbeddings: try t("audio_decoder.codebook_embeddings.weight", .bf16, [config.codebookSize * config.numCodebooks, H]),
            shared: try (0..<config.towerLayers).map { try towerLayer("audio_decoder_mtp_shared.layers.\($0).", embeddingName: "semantic_embedding") },
            residual: try (0..<(config.numCodebooks - 1)).map { try towerLayer("audio_decoder_mtp.layers.\($0).", embeddingName: "codebook_embedding") },
            heads: try (0..<(config.numCodebooks - 1)).map { try t("audio_decoder_mtp.heads.\($0).weight", .bf16, [config.codebookSize, H]) })
    }
}
