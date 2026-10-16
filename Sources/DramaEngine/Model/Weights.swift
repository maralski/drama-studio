import Foundation
import Metal

struct LinearAttentionWeights {
    let inProjQKV: WeightMatrix   // [8192, 2048]
    let inProjZ: WeightMatrix     // [4096, 2048]
    let inProjA: WeightMatrix     // [32, 2048]
    let inProjB: WeightMatrix     // [32, 2048]
    let conv: TensorRef           // [8192, 1, 4]
    let aLog: TensorRef           // [32]
    let dtBias: TensorRef         // [32]
    let norm: TensorRef           // [128]
    let outProj: WeightMatrix     // [2048, 4096]
}

struct FullAttentionWeights {
    let qProj: WeightMatrix   // [8192, 2048] rows are [q | gate] per head
    let kProj: WeightMatrix   // [512, 2048]
    let vProj: WeightMatrix   // [512, 2048]
    let oProj: WeightMatrix   // [2048, 4096]
    let qNorm: TensorRef      // [256]
    let kNorm: TensorRef      // [256]
}

enum Mixer {
    case linear(LinearAttentionWeights)
    case full(FullAttentionWeights)
}

struct MoEWeights {
    let router: WeightMatrix           // [256, 2048]
    let sharedGate: WeightMatrix       // [512, 2048]
    let sharedUp: WeightMatrix         // [512, 2048]
    let sharedDown: WeightMatrix       // [2048, 512]
    let sharedExpertGate: WeightMatrix // [1, 2048]
    let experts: ExpertTable
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
    let lmHeadEos: WeightMatrix      // [1, 2048]
    let lmHeadSemantic: WeightMatrix // [4096, 2048]
    let layers: [BackboneLayerWeights]
}

struct TowerLayerWeights {
    let embedding: TensorRef   // [4096, 2048]
    let enorm: TensorRef
    let hnorm: TensorRef
    let ehProj: WeightMatrix   // [2048, 4096]
    let attentionNorm: TensorRef
    let ffnNorm: TensorRef
    let wqkv: WeightMatrix     // [9216, 2048] rows [Q | Gate | K | V]
    let wo: WeightMatrix       // [2048, 4096]
    let qNorm: TensorRef
    let kNorm: TensorRef
    let w1: WeightMatrix       // [6144, 2048]
    let w3: WeightMatrix       // [6144, 2048]
    let w2: WeightMatrix       // [2048, 6144]
    let finalNorm: TensorRef
}

struct TowerWeights {
    let codebookEmbeddings: TensorRef  // [40960, 2048]
    let shared: [TowerLayerWeights]
    let residual: [TowerLayerWeights]
    let heads: [WeightMatrix]          // 9 x [4096, 2048]
}

/// The model's one ordered tensor list, each item asked of a `WeightSource`.
struct ModelWeights {
    let backbone: BackboneWeights
    let tower: TowerWeights
    /// Every Metal buffer a command buffer must declare to read the weights.
    let declaredBuffers: [MTLBuffer]

    init(source: WeightSource, config: ModelConfig) throws {
        let H = config.hiddenSize
        func t(_ name: String, _ dtype: TensorDType, _ shape: [Int]) throws -> TensorRef { try source.tensor(name, dtype, shape) }
        func m(_ name: String, _ rows: Int, _ cols: Int, _ role: MatrixRole, rowOffset: Int = 0) throws -> WeightMatrix {
            try source.matrix(name, rows: rows, cols: cols, rowOffset: rowOffset, role: role)
        }
        var layers: [BackboneLayerWeights] = []
        for i in 0..<config.numLayers {
            let p = "model.language_model.layers.\(i)."
            let mixer: Mixer
            switch config.layerTypes[i] {
            case .linear:
                mixer = .linear(LinearAttentionWeights(
                    inProjQKV: try m(p + "linear_attn.in_proj_qkv.weight", 8192, H, .backbone),
                    inProjZ: try m(p + "linear_attn.in_proj_z.weight", 4096, H, .backbone),
                    inProjA: try m(p + "linear_attn.in_proj_a.weight", 32, H, .narrow),
                    inProjB: try m(p + "linear_attn.in_proj_b.weight", 32, H, .narrow),
                    conv: try t(p + "linear_attn.conv1d.weight", .bf16, [8192, 1, 4]),
                    aLog: try t(p + "linear_attn.A_log", .bf16, [32]),
                    dtBias: try t(p + "linear_attn.dt_bias", .bf16, [32]),
                    norm: try t(p + "linear_attn.norm.weight", .bf16, [128]),
                    outProj: try m(p + "linear_attn.out_proj.weight", H, 4096, .backbone)))
            case .full:
                mixer = .full(FullAttentionWeights(
                    qProj: try m(p + "self_attn.q_proj.weight", 8192, H, .backbone),
                    kProj: try m(p + "self_attn.k_proj.weight", 512, H, .backbone),
                    vProj: try m(p + "self_attn.v_proj.weight", 512, H, .backbone),
                    oProj: try m(p + "self_attn.o_proj.weight", H, 4096, .backbone),
                    qNorm: try t(p + "self_attn.q_norm.weight", .bf16, [256]),
                    kNorm: try t(p + "self_attn.k_norm.weight", .bf16, [256])))
            }
            layers.append(BackboneLayerWeights(
                inputNorm: try t(p + "input_layernorm.weight", .bf16, [H]),
                postNorm: try t(p + "post_attention_layernorm.weight", .bf16, [H]),
                mixer: mixer,
                moe: MoEWeights(
                    router: try m(p + "mlp.gate.weight", config.numExperts, H, .narrow),
                    sharedGate: try m(p + "mlp.shared_expert.gate_proj.weight", 512, H, .backbone),
                    sharedUp: try m(p + "mlp.shared_expert.up_proj.weight", 512, H, .backbone),
                    sharedDown: try m(p + "mlp.shared_expert.down_proj.weight", H, 512, .backbone),
                    sharedExpertGate: try m(p + "mlp.shared_expert_gate.weight", 1, H, .narrow),
                    experts: try source.experts(layer: i, prefix: p))))
        }
        backbone = BackboneWeights(
            embed: try source.embedding("model.language_model.embed_tokens.weight", rows: config.vocabSize, cols: H),
            finalNorm: try t("model.language_model.norm.weight", .bf16, [H]),
            lmHeadEos: try m("lm_head.weight", 1, H, .backbone, rowOffset: Int(config.eos)),
            lmHeadSemantic: try m("lm_head.weight", config.codebookSize, H, .backbone, rowOffset: Int(config.semanticStart)),
            layers: layers)

        func towerLayer(_ p: String, embeddingName: String) throws -> TowerLayerWeights {
            let a = p + "transformer_layer.attention."
            let f = p + "transformer_layer.feed_forward."
            return TowerLayerWeights(
                embedding: try source.embedding(p + embeddingName + ".weight", rows: config.codebookSize, cols: H),
                enorm: try t(p + "enorm.weight", .bf16, [H]),
                hnorm: try t(p + "hnorm.weight", .bf16, [H]),
                ehProj: try m(p + "eh_proj.weight", H, 2 * H, .tower),
                attentionNorm: try t(p + "transformer_layer.attention_norm.weight", .bf16, [H]),
                ffnNorm: try t(p + "transformer_layer.ffn_norm.weight", .bf16, [H]),
                wqkv: try m(a + "wqkv.weight", 9216, H, .tower),
                wo: try m(a + "wo.weight", H, 4096, .tower),
                qNorm: try t(a + "q_norm.weight", .bf16, [256]),
                kNorm: try t(a + "k_norm.weight", .bf16, [256]),
                w1: try m(f + "w1.weight", config.towerIntermediate, H, .tower),
                w3: try m(f + "w3.weight", config.towerIntermediate, H, .tower),
                w2: try m(f + "w2.weight", H, config.towerIntermediate, .tower),
                finalNorm: try t(p + "final_layernorm.weight", .bf16, [H]))
        }
        tower = TowerWeights(
            codebookEmbeddings: try source.embedding("audio_decoder.codebook_embeddings.weight", rows: config.codebookSize * config.numCodebooks, cols: H),
            shared: try (0..<config.towerLayers).map { try towerLayer("audio_decoder_mtp_shared.layers.\($0).", embeddingName: "semantic_embedding") },
            residual: try (0..<(config.numCodebooks - 1)).map { try towerLayer("audio_decoder_mtp.layers.\($0).", embeddingName: "codebook_embedding") },
            heads: try (0..<(config.numCodebooks - 1)).map { try m("audio_decoder_mtp.heads.\($0).weight", config.codebookSize, H, .tower) })
        declaredBuffers = source.declaredBuffers
    }
}
