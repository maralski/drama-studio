import Foundation

/// xoshiro256** so runs are reproducible from a seed.
struct SeededGenerator: RandomNumberGenerator {
    private var s: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        var x = seed &+ 0x9E37_79B9_7F4A_7C15
        func splitmix() -> UInt64 {
            x &+= 0x9E37_79B9_7F4A_7C15
            var z = x
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        s = (splitmix(), splitmix(), splitmix(), splitmix())
    }

    init(state: SIMD4<UInt64>) { s = (state[0], state[1], state[2], state[3]) }

    /// The xoshiro state, so the GPU sampler can continue the same stream.
    var state: SIMD4<UInt64> {
        get { SIMD4(s.0, s.1, s.2, s.3) }
        set { s = (newValue[0], newValue[1], newValue[2], newValue[3]) }
    }

    mutating func next() -> UInt64 {
        let result = rotl(s.1 &* 5, 7) &* 9
        let t = s.1 << 17
        s.2 ^= s.0; s.3 ^= s.1; s.1 ^= s.2; s.0 ^= s.3
        s.2 ^= t
        s.3 = rotl(s.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 { (x << k) | (x >> (64 - k)) }

    /// Uniform in (0, 1].
    mutating func uniformOpenZero() -> Double {
        Double((next() >> 11) + 1) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func exponential() -> Double { -log(uniformOpenZero()) }
}

/// The reference sampler: sort, drop everything past top-k or past the top-p
/// mass (keeping the best), softmax, then argmax of probability / exponential noise.
public struct Sampler {
    var rng: SeededGenerator

    public init(seed: UInt64) { rng = SeededGenerator(seed: seed) }
    init(state: SIMD4<UInt64>) { rng = SeededGenerator(state: state) }

    var state: SIMD4<UInt64> {
        get { rng.state }
        set { rng.state = newValue }
    }

    /// Returns the index into `logits`.
    public mutating func sample(_ logits: [Float], temperature: Double, topK: Int, topP: Double) -> Int {
        let t = max(temperature, 1e-5)
        let n = logits.count
        var order = Array(0..<n)
        order.sort { logits[$0] > logits[$1] || (logits[$0] == logits[$1] && $0 < $1) }
        var values = order.map { Double(logits[$0]) / t }
        let maxValue = values[0]
        var expSum = 0.0
        for v in values { expSum += exp(v - maxValue) }
        var cumulative = 0.0
        var kept = 0
        for i in 0..<n {
            cumulative += exp(values[i] - maxValue) / expSum
            let dropped = (cumulative > topP || i >= topK) && i > 0
            if dropped { break }
            kept = i + 1
        }
        values.removeLast(n - kept)
        var keptSum = 0.0
        for v in values { keptSum += exp(v - maxValue) }
        var best = 0
        var bestScore = -Double.infinity
        for i in 0..<kept {
            let p = exp(values[i] - maxValue) / keptSum
            let score = p / rng.exponential()
            if score > bestScore { bestScore = score; best = i }
        }
        return order[best]
    }
}

/// A laughter loop in the code stream. The repetition-aware redraw forbids a
/// main token that repeats within 16 frames, so a stuck model settles into a
/// cycle of 17 or more: on the reliability study's two laughter loops the last
/// 48 main tokens matched the tokens 17 or 34 frames earlier exactly, while no
/// speech run matched above 0.75 at any lag from 17 to 40.
public enum MainTokenLoop {
    public static let windowFrames = 48
    public static let lags = 17...40
    public static let threshold = 0.9

    /// Whether the end of `mainTokens` repeats itself at some lag.
    public static func cycles(_ mainTokens: [Int32]) -> Bool {
        let n = mainTokens.count
        guard n >= windowFrames + lags.lowerBound else { return false }
        for lag in lags where n >= windowFrames + lag {
            var matches = 0
            for i in (n - windowFrames)..<n where mainTokens[i] == mainTokens[i - lag] { matches += 1 }
            if Double(matches) >= threshold * Double(windowFrames) { return true }
        }
        return false
    }
}

/// Main-token sampling over the candidates with repetition-aware resampling.
struct MainTokenSampler {
    static let windowFrames = 16
    static let retryTemperature = 1.0
    static let retryTopK = 30
    static let retryTopP = 0.95

    let config: ModelConfig
    /// Candidate ids in ascending order: eos, then the semantic band.
    let candidates: [Int32]

    init(config: ModelConfig) {
        self.config = config
        candidates = [config.eos] + (config.semanticStart...config.semanticEnd)
    }

    func isSemantic(_ id: Int32) -> Bool { id >= config.semanticStart && id <= config.semanticEnd }

    /// `history` is every main token so far, including the current row.
    func sample(candidateLogits: [Float], history: ArraySlice<Int32>, temperature: Double, topK: Int, topP: Double, sampler: inout Sampler) -> Int32 {
        let first = candidates[sampler.sample(candidateLogits, temperature: temperature, topK: topK, topP: topP)]
        let retry = candidates[sampler.sample(candidateLogits, temperature: Self.retryTemperature, topK: Self.retryTopK, topP: Self.retryTopP)]
        let window = history.suffix(Self.windowFrames)
        let repeated = window.contains(first) && isSemantic(first)
        return repeated ? retry : first
    }
}
