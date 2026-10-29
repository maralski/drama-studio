// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation
import Metal

/// The exponent-coded form of the decode families: one file per matrix under
/// `<root>/<model>/`, built once per checkpoint on every core, verified weight
/// for weight against the bf16 source, and mapped like the shards on every
/// later load. A cache that is missing, partial, lists other matrices, was
/// built from other shard files, or whose files do not hash as the manifest
/// says is deleted and built again.
final class PackedWeightCache: @unchecked Sendable {
    /// Bumped when the file layout or `PackedBF16`'s section layout changes.
    static let layout = 2
    static let format = "exponent-coded bf16: per 256-weight tile a header word (base exponent, escape byte offset), 256 sign+mantissa bytes, 96 bytes of 3-bit exponent codes; then one byte per escaped exponent and 16 pad bytes; sections 16-byte aligned after a 64-byte file header of magic, layout, rows, cols, escaped"
    static let magic: UInt32 = 0x3150_4244
    static let fileHeaderBytes = 64
    static let manifestName = "manifest.json"

    static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Packed", isDirectory: true)
    }

    /// `DRAMA_PACKED=0` keeps every family bf16, for A/B runs of one binary.
    static var enabled: Bool { ProcessInfo.processInfo.environment["DRAMA_PACKED"] != "0" }

    struct Entry: Codable, Equatable {
        var matrix: DenseMatrix
        var file: String
        var escaped: Int
        /// Payload bytes after the file header: the sections, the escapes
        /// and the escape pad (`PackedBF16.storedByteCount`).
        var bytes: Int
        var sha256: String
    }

    struct Manifest: Codable, Equatable {
        var layout: Int
        var format: String
        var model: String
        /// The shard files the matrices were read from.
        var sources: [ShardFingerprint]
        var entries: [Entry]
    }

    enum Outcome: Equatable {
        case loaded
        case built(seconds: Double)
    }

    let directory: URL
    let outcome: Outcome
    private(set) var matrices: [DenseMatrix: PackedBF16] = [:]
    private var files: [MappedBuffer] = []

    /// Every mapped file, for the encoders' residency lists.
    var buffers: [MTLBuffer] { files.map(\.buffer) }
    /// Bytes on disk, headers included.
    var byteCount: Int { files.reduce(0) { $0 + $1.fileSize } }

    /// Opens the cache for `model` (the snapshot name and digest `TuningKey`
    /// uses) under `root`, building it first when it is not complete,
    /// verified and built from the shard files `sources` names. `log` gets
    /// one line per packed matrix while building.
    init(matrices: [(DenseMatrix, TensorRef)], model: String, sources: [ShardFingerprint], root: URL, device: MTLDevice, log: (String) -> Void) throws {
        directory = root.appendingPathComponent(model, isDirectory: true)
        Self.removeStaleBuilds(of: directory)
        let wanted = matrices.map(\.0)
        if let hit = try? Self.load(directory: directory, wanted: wanted, sources: sources, device: device) {
            (files, self.matrices) = hit
            outcome = .loaded
            return
        }
        let start = Date()
        log("packing \(matrices.count) matrices into \(directory.path)")
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
        try Self.build(matrices, model: model, sources: sources, into: directory, device: device, log: log)
        (files, self.matrices) = try Self.load(directory: directory, wanted: wanted, sources: sources, device: device)
        outcome = .built(seconds: Date().timeIntervalSince(start))
    }

    /// Every weight of every matrix, unpacked from the mapped files and
    /// compared with its bf16 source; the first mismatch, if any.
    func verify(against sources: [(DenseMatrix, TensorRef)]) -> (DenseMatrix, index: Int)? {
        let lock = NSLock()
        nonisolated(unsafe) var found: (DenseMatrix, index: Int)?
        DispatchQueue.concurrentPerform(iterations: sources.count) { i in
            let (matrix, source) = sources[i]
            guard let packed = matrices[matrix] else { lock.lock(); found = found ?? (matrix, -1); lock.unlock(); return }
            let bits = UnsafeBufferPointer(start: source.pointer.assumingMemoryBound(to: UInt16.self), count: source.count)
            if let index = packed.firstMismatch(against: bits) { lock.lock(); found = found ?? (matrix, index); lock.unlock() }
        }
        return found
    }

    static func fileName(_ m: DenseMatrix) -> String { "\(m.name).rows\(m.rowOffset).packed" }

    private static func buildPrefix(of directory: URL) -> String { directory.lastPathComponent + ".building-" }

    /// `<name>.building-<pid>-<uuid>` next to `directory`, so a later open can
    /// tell a build in progress from one a dead process left behind.
    private static func buildDirectory(for directory: URL) -> URL {
        directory.deletingLastPathComponent().appendingPathComponent(buildPrefix(of: directory) + "\(getpid())-" + UUID().uuidString, isDirectory: true)
    }

    /// Removes the build directories whose process is gone. This process's
    /// own builds stay: two loads in one process may build at once.
    private static func removeStaleBuilds(of directory: URL) {
        let parent = directory.deletingLastPathComponent()
        let prefix = buildPrefix(of: directory)
        for url in (try? FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil)) ?? [] where url.lastPathComponent.hasPrefix(prefix) {
            let pid = pid_t(url.lastPathComponent.dropFirst(prefix.count).prefix { $0 != "-" })
            if let pid, pid == getpid() || kill(pid, 0) == 0 || errno == EPERM { continue }
            try? FileManager.default.removeItem(at: url)
        }
    }

    private static func header(_ m: DenseMatrix, escaped: Int) -> Data {
        var words = [UInt32](repeating: 0, count: fileHeaderBytes / 4)
        words[0] = magic
        words[1] = UInt32(layout)
        words[2] = UInt32(m.rows)
        words[3] = UInt32(m.cols)
        words[4] = UInt32(escaped)
        return words.withUnsafeBytes { Data($0) }
    }

    private static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }

    /// Packs every matrix on all cores into a temporary directory next to
    /// `directory`, each file verified against its source before it is
    /// written, then renames the directory into place.
    private static func build(_ matrices: [(DenseMatrix, TensorRef)], model: String, sources: [ShardFingerprint], into directory: URL, device: MTLDevice, log: (String) -> Void) throws {
        let fm = FileManager.default
        let temp = buildDirectory(for: directory)
        try fm.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: temp) }
        let lock = NSLock()
        nonisolated(unsafe) var entries = [Entry?](repeating: nil, count: matrices.count)
        nonisolated(unsafe) var failure: Error?
        nonisolated(unsafe) var done = 0, bytesSoFar = 0
        DispatchQueue.concurrentPerform(iterations: matrices.count) { i in
            let (matrix, source) = matrices[i]
            do {
                let packed = PackedBF16(source, device: device)
                let bits = UnsafeBufferPointer(start: source.pointer.assumingMemoryBound(to: UInt16.self), count: source.count)
                if let index = packed.firstMismatch(against: bits) {
                    throw EngineError.badModel("\(matrix.name) rows \(matrix.rowOffset)+: packed weight \(index) does not unpack to its source")
                }
                let payload = Data(bytesNoCopy: packed.buffer.contents() + packed.headerOffset, count: packed.storedByteCount, deallocator: .none)
                let file = fileName(matrix)
                let url = temp.appendingPathComponent(file)
                guard fm.createFile(atPath: url.path, contents: nil) else { throw EngineError.badModel("cannot create \(url.path)") }
                let handle = try FileHandle(forWritingTo: url)
                try handle.write(contentsOf: header(matrix, escaped: packed.escapedWeights))
                try handle.write(contentsOf: payload)
                try handle.close()
                let entry = Entry(matrix: matrix, file: file, escaped: packed.escapedWeights, bytes: packed.storedByteCount, sha256: hex(SHA256.hash(data: payload)))
                lock.lock()
                entries[i] = entry
                done += 1
                bytesSoFar += packed.byteCount
                log("packed \(done)/\(matrices.count) matrices, \(bytesSoFar) bytes so far (\(matrix.name))")
                lock.unlock()
            } catch {
                lock.lock()
                failure = failure ?? error
                lock.unlock()
            }
        }
        if let failure { throw failure }
        let manifest = Manifest(layout: layout, format: format, model: model, sources: sources, entries: entries.map { $0! })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: temp.appendingPathComponent(manifestName), options: .atomic)
        try fm.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        if rename(temp.path, directory.path) != 0 {
            let code = errno
            // Another process finished its build first; `load` reads that one.
            guard code == EEXIST || code == ENOTEMPTY else {
                throw EngineError.badModel("cannot move the packed cache into \(directory.path): \(String(cString: strerror(code)))")
            }
        }
    }

    /// Maps every file the manifest lists, after checking the manifest names
    /// exactly `wanted` read from exactly `sources`, each file's header and
    /// size, and each payload's hash.
    private static func load(directory: URL, wanted: [DenseMatrix], sources: [ShardFingerprint], device: MTLDevice) throws -> ([MappedBuffer], [DenseMatrix: PackedBF16]) {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent(manifestName)))
        guard manifest.layout == layout, manifest.format == format, manifest.sources == sources,
              manifest.entries.count == wanted.count, Set(manifest.entries.map(\.matrix)) == Set(wanted) else {
            throw EngineError.badModel("packed cache manifest does not match")
        }
        var files: [MappedBuffer] = []
        var matrices: [DenseMatrix: PackedBF16] = [:]
        for e in manifest.entries {
            // `m` is one of `wanted` after the set check above; the counts
            // come from the manifest and are checked before any arithmetic.
            let m = e.matrix
            let offsets = PackedBF16.sectionOffsets(tiles: m.rows * m.cols / PackedBF16.tile)
            guard (0...m.rows * m.cols).contains(e.escaped), e.bytes == offsets.escape + e.escaped + PackedBF16.escapePad else {
                throw EngineError.badModel("packed file \(e.file) has an impossible size in the manifest")
            }
            let mapped = try MappedBuffer(url: directory.appendingPathComponent(e.file), device: device)
            let words = mapped.buffer.contents().assumingMemoryBound(to: UInt32.self)
            guard mapped.fileSize == fileHeaderBytes + e.bytes,
                  words[0] == magic, words[1] == UInt32(layout), words[2] == UInt32(m.rows), words[3] == UInt32(m.cols), words[4] == UInt32(e.escaped) else {
                throw EngineError.badModel("packed file \(e.file) does not match its manifest entry")
            }
            files.append(mapped)
            matrices[m] = PackedBF16(n: m.rows, k: m.cols, buffer: mapped.buffer,
                                     headerOffset: fileHeaderBytes + offsets.header, mantOffset: fileHeaderBytes + offsets.mant,
                                     codeOffset: fileHeaderBytes + offsets.code, escapeOffset: fileHeaderBytes + offsets.escape,
                                     escapedWeights: e.escaped)
        }
        let lock = NSLock()
        nonisolated(unsafe) var corrupt: String?
        DispatchQueue.concurrentPerform(iterations: files.count) { i in
            let e = manifest.entries[i]
            let payload = Data(bytesNoCopy: files[i].buffer.contents() + fileHeaderBytes, count: e.bytes, deallocator: .none)
            if hex(SHA256.hash(data: payload)) != e.sha256 { lock.lock(); corrupt = corrupt ?? e.file; lock.unlock() }
        }
        if let corrupt { throw EngineError.badModel("packed file \(corrupt) does not hash as its manifest entry says") }
        return (files, matrices)
    }
}
