import CryptoKit
import Foundation
import Metal

/// What a tuned schedule is valid for: this GPU, this OS, these kernel
/// sources, this checkpoint, this schedule layout. Any change is a miss.
package struct TuningKey: Codable, Equatable, Sendable {
    /// `MTLDevice.name`.
    package var chip: String
    /// `ProcessInfo` operating system version as "26.5.2".
    package var os: String
    /// SHA-256 hex of the kernel source string MetalContext compiled.
    package var kernels: String
    /// Snapshot directory name, a dash, and the first 16 hex digits of the
    /// SHA-256 of model.safetensors.index.json.
    package var model: String
    /// Bumped when KernelSchedule's fields change without a kernel change.
    package var schema: Int

    package static let currentSchema = 2

    package init(chip: String, os: String, kernels: String, model: String, schema: Int = TuningKey.currentSchema) {
        self.chip = chip
        self.os = os
        self.kernels = kernels
        self.model = model
        self.schema = schema
    }

    package init(ctx: MetalContext, location: ModelLocation) throws {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let index = try Data(contentsOf: location.directory.appendingPathComponent("model.safetensors.index.json"))
        self.init(chip: ctx.device.name,
                  os: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
                  kernels: ctx.kernelDigest,
                  model: location.directory.lastPathComponent + "-" + Self.hex(SHA256.hash(data: index)).prefix(16))
    }

    /// First 16 hex digits of the SHA-256 over the fields, plus ".json".
    package var fileName: String {
        let text = [chip, os, kernels, model, String(schema)].joined(separator: "\n")
        return String(Self.hex(SHA256.hash(data: Data(text.utf8))).prefix(16)) + ".json"
    }

    static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }
}

/// One tuned entry: what was compared, what was chosen, and why.
package struct TuningEntryResult: Codable, Equatable, Sendable {
    package struct Rejected: Codable, Equatable, Sendable {
        package var label: String
        package var reason: String
        /// Set when the candidate ran and was out of tolerance.
        package var relL2: Double?
    }
    package var name: String
    package var defaultLabel: String
    package var chosenLabel: String
    package var defaultMs: Double
    package var chosenMs: Double
    package var gainPercent: Double
    /// Relative L2 of the chosen candidate's outputs against the default's.
    package var relL2: Double
    /// Timing rounds measured; 0 when the budget ran out before this entry.
    package var rounds: Int
    package var rejected: [Rejected]
    /// Why the row does not show the entry's own winner, when it does not.
    package var note: String? = nil
}

/// One whole-forward timing of the default schedule against the tuned one,
/// medians over interleaved rounds. `kept` false means the side of the
/// schedule this row count exercises was reverted to the default.
package struct ForwardCheck: Codable, Equatable, Sendable {
    package var rows: Int
    package var defaultMs: Double
    package var tunedMs: Double
    package var kept: Bool
}

package struct TuningReport: Codable, Equatable, Sendable {
    package static let currentVersion = 2
    package var version: Int = TuningReport.currentVersion
    package var key: TuningKey
    package var created: Date
    package var schedule: KernelSchedule
    package var entries: [TuningEntryResult]
    package var wallSeconds: Double
    package var forwardChecks: [ForwardCheck] = []
}

/// JSON files under Application Support, one per `TuningKey`.
package enum TuningCache {
    package static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Tuning", isDirectory: true)
    }

    package static func url(for key: TuningKey, directory: URL = defaultDirectory) -> URL {
        directory.appendingPathComponent(key.fileName)
    }

    /// The report for `key`, or nil when there is none, its key fields
    /// disagree, or its schedule fails validation.
    package static func load(key: TuningKey, directory: URL = defaultDirectory) -> (report: TuningReport, url: URL)? {
        let url = url(for: key, directory: directory)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let report = try? decoder.decode(TuningReport.self, from: data),
              report.version == TuningReport.currentVersion, report.key == key,
              (try? report.schedule.validate()) != nil else { return nil }
        return (report, url)
    }

    @discardableResult
    package static func save(_ report: TuningReport, directory: URL = defaultDirectory) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = url(for: report.key, directory: directory)
        try encoder.encode(report).write(to: url, options: .atomic)
        return url
    }
}
