import Foundation

/// Per-device kernel schedule: which instantiation or dispatch shape each
/// kernel family runs with. Every field has a declared domain; the autotuner
/// searches the domains and the engine applies one schedule per process.
///
/// Path choice may depend on M == 1 against M > 1 and on N, never on the
/// chunk size, so a row's result is the same in any chunk (the prefix cache
/// relies on it). The defaults equal the constants the engine was tuned with
/// by hand, so `.default` reproduces that engine bit for bit.
package struct KernelSchedule: Codable, Equatable, Sendable {
    package struct GemvShape: Codable, Hashable, Sendable {
        package var tokenTile: Int
        package var simdgroups: Int
        package init(tokenTile: Int, simdgroups: Int) { self.tokenTile = tokenTile; self.simdgroups = simdgroups }
        package var label: String { "t\(tokenTile)x\(simdgroups)sg" }
    }
    package struct GemmTile: Codable, Hashable, Sendable {
        package var bm: Int
        package var bn: Int
        package var bk: Int
        package init(bm: Int, bn: Int, bk: Int) { self.bm = bm; self.bn = bn; self.bk = bk }
        package var label: String { "\(bm)x\(bn)x\(bk)" }
    }
    package struct MoETile: Codable, Hashable, Sendable {
        package var bn: Int
        package var rows: Int
        package var bk: Int
        package init(bn: Int, rows: Int, bk: Int) { self.bn = bn; self.rows = rows; self.bk = bk }
        package var label: String { "\(bn)x\(rows)x\(bk)" }
    }

    /// Dense projections with one row (decode), any N.
    package var gemvDecode = GemvShape(tokenTile: 32, simdgroups: 4)
    /// Dense projections with M > 1 and N <= 64, where a GEMM would have one or
    /// two column tiles walking K alone.
    package var gemvNarrow = GemvShape(tokenTile: 32, simdgroups: 4)
    /// Dense projections with M > 1 and N > 64; one tile for every such M.
    package var gemmTile = GemmTile(bm: 32, bn: 32, bk: 16)
    /// Routed experts with M > 1; one tile for every such M.
    package var moeTile = MoETile(bn: 32, rows: 32, bk: 16)
    /// Rows per threadgroup of the one-row expert kernels (M == 1).
    package var moeRowSimdgroups = 8
    /// Key split of the attention kernel for one query row.
    package var attentionSplitDecode = 8
    /// Key split of the attention kernel for a chunk of rows.
    package var attentionSplitPrefill = 8
    /// Threads per state column of the GDN recurrence for one row.
    package var gdnPartsDecode = 4
    /// Threads per state column of the GDN recurrence for a chunk of rows.
    package var gdnPartsPrefill = 4

    package init() {}
    package static let `default` = KernelSchedule()

    // Domains. Each list matches the kernels instantiated in the .metal
    // sources; ScheduleTests checks every value has a pipeline and agrees with
    // the default.
    package static let gemvTokenTiles = [1, 8, 16, 32, 64]
    package static let gemvSimdgroups = [1, 2, 4, 8]
    package static let gemmTiles: [GemmTile] = [
        GemmTile(bm: 32, bn: 32, bk: 16), GemmTile(bm: 32, bn: 32, bk: 32), GemmTile(bm: 64, bn: 32, bk: 16), GemmTile(bm: 32, bn: 64, bk: 16),
        GemmTile(bm: 64, bn: 64, bk: 16), GemmTile(bm: 64, bn: 64, bk: 32), GemmTile(bm: 16, bn: 64, bk: 16), GemmTile(bm: 16, bn: 32, bk: 16),
    ]
    /// {32, 64} x {16, 32, 64} x {16, 32} without bn 64 with rows 64, whose
    /// gate-up output staging would be the whole threadgroup memory budget.
    package static let moeTiles: [MoETile] = [32, 64].flatMap { bn in
        [16, 32, 64].flatMap { rows in
            [16, 32].compactMap { bk in bn == 64 && rows == 64 ? nil : MoETile(bn: bn, rows: rows, bk: bk) }
        }
    }
    package static let moeRowSimdgroups = [2, 4, 8, 16]
    package static let attentionSplits = [4, 8, 16]
    package static let gdnParts = [1, 2, 4, 8]

    package static let gemvShapes: [GemvShape] = gemvTokenTiles.flatMap { t in gemvSimdgroups.map { GemvShape(tokenTile: t, simdgroups: $0) } }

    /// Throws when any field is outside its domain.
    package func validate() throws {
        func check(_ ok: Bool, _ field: String, _ value: String) throws {
            if !ok { throw EngineError.badSchedule("\(field) \(value) is not in its domain") }
        }
        for (name, shape) in [("gemvDecode", gemvDecode), ("gemvNarrow", gemvNarrow)] {
            try check(Self.gemvTokenTiles.contains(shape.tokenTile) && Self.gemvSimdgroups.contains(shape.simdgroups), name, shape.label)
        }
        try check(Self.gemmTiles.contains(gemmTile), "gemmTile", gemmTile.label)
        try check(Self.moeTiles.contains(moeTile), "moeTile", moeTile.label)
        try check(Self.moeRowSimdgroups.contains(moeRowSimdgroups), "moeRowSimdgroups", "\(moeRowSimdgroups)")
        try check(Self.attentionSplits.contains(attentionSplitDecode), "attentionSplitDecode", "\(attentionSplitDecode)")
        try check(Self.attentionSplits.contains(attentionSplitPrefill), "attentionSplitPrefill", "\(attentionSplitPrefill)")
        try check(Self.gdnParts.contains(gdnPartsDecode), "gdnPartsDecode", "\(gdnPartsDecode)")
        try check(Self.gdnParts.contains(gdnPartsPrefill), "gdnPartsPrefill", "\(gdnPartsPrefill)")
    }
}
