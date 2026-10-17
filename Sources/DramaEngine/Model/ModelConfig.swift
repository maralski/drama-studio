import Foundation

/// The subset of `config.json` the engine depends on. Values the kernels hard
/// code (head dim 256, GDN head dim 128, 10 codebooks) are checked, not read.
package struct ModelConfig: Sendable {
    package enum LayerType: String, Sendable { case linear = "linear_attention", full = "full_attention" }

    package let hiddenSize: Int
    package let vocabSize: Int
    package let numLayers: Int
    package let layerTypes: [LayerType]
    package let numHeads: Int
    package let numKVHeads: Int
    package let headDim: Int
    package let rotaryDim: Int
    package let ropeTheta: Double
    package let rmsNormEps: Float
    package let linearKeyHeads: Int
    package let linearValueHeads: Int
    package let convKernel: Int
    package let numExperts: Int
    package let topK: Int
    package let moeIntermediate: Int
    package let sharedIntermediate: Int
    package let maxPositions: Int

    package let towerLayers: Int
    package let towerIntermediate: Int
    package let towerHeads: Int
    package let towerKVHeads: Int
    package let towerRotaryDim: Int
    package let towerRopeBase: Double
    package let towerNormEps: Float
    package let numCodebooks: Int
    package let codebookSize: Int
    package let vqBackboneCodebooks: Int

    package let semanticStart: Int32
    package let semanticEnd: Int32
    package let eos: Int32

    package init(url: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let text = root["text_config"] as? [String: Any],
              let audio = root["audio_decoder_config"] as? [String: Any] else {
            throw EngineError.badModel("config.json is not a fish_omni config")
        }
        func int(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = d[k] as? Int else { throw EngineError.badModel("config missing \(k)") }
            return v
        }
        func double(_ d: [String: Any], _ k: String) throws -> Double {
            guard let v = d[k] as? Double else { throw EngineError.badModel("config missing \(k)") }
            return v
        }
        guard root["model_type"] as? String == "fish_omni" else { throw EngineError.badModel("model_type is not fish_omni") }
        guard root["speech_template_version"] as? String == "reference-bracket-v1" else {
            throw EngineError.badModel("unsupported speech_template_version")
        }
        hiddenSize = try int(text, "hidden_size")
        vocabSize = try int(text, "vocab_size")
        numLayers = try int(text, "num_hidden_layers")
        guard let types = text["layer_types"] as? [String] else { throw EngineError.badModel("config missing layer_types") }
        layerTypes = try types.map {
            guard let t = LayerType(rawValue: $0) else { throw EngineError.badModel("layer type \($0)") }
            return t
        }
        numHeads = try int(text, "num_attention_heads")
        numKVHeads = try int(text, "num_key_value_heads")
        headDim = try int(text, "head_dim")
        let rope = text["rope_parameters"] as? [String: Any] ?? text
        ropeTheta = try double(rope, "rope_theta")
        let partial = rope["partial_rotary_factor"] as? Double ?? 0.25
        rotaryDim = Int(Double(headDim) * partial)
        rmsNormEps = Float(try double(text, "rms_norm_eps"))
        linearKeyHeads = try int(text, "linear_num_key_heads")
        linearValueHeads = try int(text, "linear_num_value_heads")
        convKernel = try int(text, "linear_conv_kernel_dim")
        numExperts = try int(text, "num_experts")
        topK = try int(text, "num_experts_per_tok")
        moeIntermediate = try int(text, "moe_intermediate_size")
        sharedIntermediate = try int(text, "shared_expert_intermediate_size")
        maxPositions = try int(text, "max_position_embeddings")

        towerLayers = try int(audio, "shared_enrichment_layers")
        towerIntermediate = try int(audio, "intermediate_size")
        towerHeads = try int(audio, "n_head")
        towerKVHeads = try int(audio, "n_local_heads")
        towerRotaryDim = Int(Double(try int(audio, "head_dim")) * (try double(audio, "rotary_percent")))
        towerRopeBase = try double(audio, "rope_base")
        towerNormEps = Float(try double(audio, "norm_eps"))
        numCodebooks = try int(audio, "num_codebooks")
        codebookSize = try int(audio, "vocab_size")
        vqBackboneCodebooks = try int(audio, "vq_backbone_codebooks")

        semanticStart = Int32(try int(root, "semantic_start_token_id"))
        semanticEnd = Int32(try int(root, "semantic_end_token_id"))
        eos = Int32(try int(root, "eos_token_id"))

        let audioHeadDim = try int(audio, "head_dim")
        let keyHeadDim = try int(text, "linear_key_head_dim")
        let valueHeadDim = try int(text, "linear_value_head_dim")
        let towerDim = try int(audio, "dim")
        let fixed: [(String, Bool)] = [
            ("hidden size 2048", hiddenSize == 2048),
            ("head dim 256", headDim == 256 && audioHeadDim == 256),
            ("rotary dim 64", rotaryDim == 64 && towerRotaryDim == 64),
            ("GDN head dims 128", keyHeadDim == 128 && valueHeadDim == 128),
            ("conv kernel 4", convKernel == 4),
            ("256 experts top 8", numExperts == 256 && topK == 8),
            ("expert intermediate 512", moeIntermediate == 512 && sharedIntermediate == 512),
            ("10 codebooks of 4096", numCodebooks == 10 && codebookSize == 4096),
            ("one backbone codebook", vqBackboneCodebooks == 1),
            ("tower dim 2048", towerDim == 2048),
            ("tower 4 shared layers", towerLayers == 4),
            ("audio decoder type", audio["decoder_type"] as? String == "codebook_mtp_shared"),
            ("semantic band", Int(semanticEnd - semanticStart) + 1 == codebookSize),
            ("fp8 experts", (root["quantization_config"] as? [String: Any])?["quant_method"] as? String == "bsgemm-moe"),
        ]
        for (name, ok) in fixed where !ok {
            throw EngineError.badModel("the kernels assume \(name)")
        }
    }
}
