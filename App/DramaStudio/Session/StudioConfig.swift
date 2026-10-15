import DramaEngine

extension GenerationConfig {
    static let defaultExpressiveness = 0.5

    /// The app's two controls over the engine defaults; every other field keeps
    /// the value the evaluation measured. Expressiveness 0.5 is the vendor's
    /// sampling, temperature 0.7 and top-p 0.7.
    static func make(expressiveness: Double, quickPreview: Bool) -> GenerationConfig {
        var config = GenerationConfig()
        config.temperature = 0.5 + 0.4 * expressiveness
        config.topP = 0.5 + 0.4 * expressiveness
        if quickPreview {
            config.minSpokenRatio = 0
            config.frameCap = .fromText(margin: 1.2, floorSeconds: 1.0)
        }
        return config
    }
}
