import Foundation

/// Which precision the weights are held at. Full is the released engine
/// bit for bit; the other two read a quantized cache file built on the first
/// load and keep the rest of the weights in place.
public enum ModelPrecisionMode: String, CaseIterable, Codable, Sendable {
    /// Every weight as released: bf16, and fp8 experts, all read in place.
    case full
    /// Dense and tower matrices int8, experts fp8 in place.
    case balanced
    /// Dense matrices int4, experts int4, tower matrices int8.
    case compact

    /// Whether the app may offer the mode as an Advanced option (false) or
    /// only as experimental (true). Decided by the measurements in the
    /// README's "Model precision" section: compact's speaker similarity
    /// (0.856) sits below every per-seed mean of full (0.866 to 0.874), its
    /// teacher-forced error is four to five times the bf16 oracle's own, and
    /// it ran to the frame cap without eos on the reference-chunking prompt.
    public var isExperimental: Bool { self == .compact }

    /// The storage format of a projection matrix by its role. Narrow matrices
    /// stay bf16 in every mode: they are about one percent of the bytes and
    /// gate the routing and the GDN recurrence.
    func format(for role: MatrixRole) -> WeightFormat {
        switch (self, role) {
        case (_, .narrow), (.full, _): return .bf16
        case (.balanced, .backbone), (.balanced, .tower), (.compact, .tower): return .int8
        case (.compact, .backbone): return .int4
        }
    }

    var expertFormat: ExpertFormat { self == .compact ? .int4 : .fp8BlockScaled }
}
