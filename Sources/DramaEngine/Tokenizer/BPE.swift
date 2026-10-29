// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
struct BPE: Sendable {
    struct Merge: Sendable {
        let rank: Int32
        let id: Int32
    }

    private struct Symbol {
        var id: Int32
        var mergeWithNext: Merge?
    }

    let idForByte: [Int32]
    let merges: [UInt64: Merge]

    static func pairKey(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }

    func encode(_ bytes: some Collection<UInt8>, into ids: inout [Int32]) {
        var symbols = bytes.map { Symbol(id: idForByte[Int($0)], mergeWithNext: nil) }
        for index in symbols.indices { refresh(&symbols, at: index) }
        while let index = lowestRankIndex(symbols), let merge = symbols[index].mergeWithNext {
            symbols[index].id = merge.id
            symbols.remove(at: index + 1)
            refresh(&symbols, at: index - 1)
            refresh(&symbols, at: index)
        }
        ids.append(contentsOf: symbols.map(\.id))
    }

    private func refresh(_ symbols: inout [Symbol], at index: Int) {
        guard symbols.indices.contains(index) else { return }
        let next = index + 1
        symbols[index].mergeWithNext = next < symbols.count
            ? merges[Self.pairKey(symbols[index].id, symbols[next].id)]
            : nil
    }

    private func lowestRankIndex(_ symbols: [Symbol]) -> Int? {
        var best: (index: Int, rank: Int32)?
        for (index, symbol) in symbols.enumerated() {
            guard let rank = symbol.mergeWithNext?.rank else { continue }
            if rank < best?.rank ?? .max { best = (index, rank) }
        }
        return best?.index
    }
}
