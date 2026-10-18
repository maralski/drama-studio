import CryptoKit
import Foundation
import Metal

/// What a tuned schedule is valid for: this GPU, this OS, these kernel
/// sources, this checkpoint, this schedule layout. Any change is a miss.
public struct TuningKey: Codable, Equatable, Sendable {
    /// `MTLDevice.name`.
    public var chip: String
    /// `ProcessInfo` operating system version as "26.5.2".
    public var os: String
    /// SHA-256 hex of the kernel source string MetalContext compiled.
    public var kernels: String
    /// Snapshot directory name, a dash, and the first 16 hex digits of the
    /// SHA-256 of model.safetensors.index.json.
    public var model: String
    /// Bumped when KernelSchedule's fields change without a kernel change.
    public var schema: Int

    public static let currentSchema = 2

    public init(chip: String, os: String, kernels: String, model: String, schema: Int = TuningKey.currentSchema) {
        self.chip = chip
        self.os = os
        self.kernels = kernels
        self.model = model
        self.schema = schema
    }

    public init(ctx: MetalContext, location: ModelLocation) throws {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let index = try Data(contentsOf: location.directory.appendingPathComponent("model.safetensors.index.json"))
        self.init(chip: ctx.device.name,
                  os: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
                  kernels: ctx.kernelDigest,
                  model: location.directory.lastPathComponent + "-" + Self.hex(SHA256.hash(data: index)).prefix(16))
    }

    /// First 16 hex digits of the SHA-256 over the fields, plus ".json".
    public var fileName: String {
        let text = [chip, os, kernels, model, String(schema)].joined(separator: "\n")
        return String(Self.hex(SHA256.hash(data: Data(text.utf8))).prefix(16)) + ".json"
    }

    static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }
}

/// One tuned entry: what was compared, what was chosen, and why.
public struct TuningEntryResult: Codable, Equatable, Sendable {
    public struct Rejected: Codable, Equatable, Sendable {
        public var label: String
        public var reason: String
        /// Set when the candidate ran and was out of tolerance.
        public var relL2: Double?
    }
    public var name: String
    public var defaultLabel: String
    public var chosenLabel: String
    public var defaultMs: Double
    public var chosenMs: Double
    public var gainPercent: Double
    /// Relative L2 of the chosen candidate's outputs against the default's.
    public var relL2: Double
    /// Timing rounds measured; 0 when the budget ran out before this entry.
    public var rounds: Int
    public var rejected: [Rejected]
    /// Why the row does not show the entry's own winner, when it does not.
    public var note: String? = nil
}

/// One whole-forward timing of the default schedule against the tuned one,
/// medians over interleaved rounds. `kept` false means the side of the
/// schedule this row count exercises was reverted to the default.
public struct ForwardCheck: Codable, Equatable, Sendable {
    public var rows: Int
    public var defaultMs: Double
    public var tunedMs: Double
    public var kept: Bool
}

public struct TuningReport: Codable, Equatable, Sendable {
    public static let currentVersion = 2
    public var version: Int = TuningReport.currentVersion
    public var key: TuningKey
    public var created: Date
    public var schedule: KernelSchedule
    public var entries: [TuningEntryResult]
    public var wallSeconds: Double
    public var forwardChecks: [ForwardCheck] = []
}

/// JSON files under Application Support, one per `TuningKey`.
public enum TuningCache {
    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Tuning", isDirectory: true)
    }

    public static func url(for key: TuningKey, directory: URL = defaultDirectory) -> URL {
        directory.appendingPathComponent(key.fileName)
    }

    /// The report for `key`, or nil when there is none, its key fields
    /// disagree, or its schedule fails validation.
    public static func load(key: TuningKey, directory: URL = defaultDirectory) -> (report: TuningReport, url: URL)? {
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
    public static func save(_ report: TuningReport, directory: URL = defaultDirectory) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = url(for: report.key, directory: directory)
        try encoder.encode(report).write(to: url, options: .atomic)
        return url
    }
}
