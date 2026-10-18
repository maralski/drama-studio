import Foundation
import Metal

/// One part of an install the engine needs, and whether it is in place.
public struct SetupItem: Sendable, Equatable {
    /// The parts of a working install, in the order the engine uses them.
    public enum Component: String, Sendable, CaseIterable {
        /// `config.json`, `model.safetensors.index.json` and every shard the
        /// index names. The released snapshot has eight shards, 39.6 GB, in a
        /// 41.5 GB download.
        case weights
        /// `tokenizer.json` in the snapshot.
        case tokenizer
        /// `codec.pth` in the snapshot, the source of the Core ML codec models.
        /// Needed only until the codec models are built.
        case codecCheckpoint
        /// `DramaCodecEncoder`, `DramaCodecDecoder` and `DramaCodecProbe` in the codec directory,
        /// as `.mlpackage` or compiled `.mlmodelc`: 809 MB, 353 MB and 353 MB.
        case codecModels
        /// The kernel schedule tuned for this Mac, kept as one JSON file in
        /// the tuning directory.
        case tuning
    }

    /// Whether a component is in place, and what to do when it is not.
    public enum State: Sendable, Equatable {
        /// In place.
        case ready
        /// The engine cannot load until someone carries out `fix`.
        case missing(fix: String)
        /// The engine builds it during its first load. `fix` says how to
        /// build it ahead of time.
        case builtOnFirstLoad(fix: String)
    }

    /// Which part of the install this item describes.
    public var component: Component
    /// Whether the part is in place.
    public var state: State
    /// The file or directory the item is about, when there is one.
    public var url: URL?
    /// One line for a person saying what was found.
    public var summary: String

    /// The action that puts the component in place; nil when it is ready.
    public var fix: String? {
        switch state {
        case .ready: nil
        case .missing(let fix), .builtOnFirstLoad(let fix): fix
        }
    }

    /// Whether the engine cannot load until the component is fixed.
    public var isMissing: Bool {
        if case .missing = state { true } else { false }
    }

    /// Creates an item.
    public init(component: Component, state: State, url: URL?, summary: String) {
        self.component = component
        self.state = state
        self.url = url
        self.summary = summary
    }
}

/// What ``DramaSetup/check(location:codecDirectory:tuningDirectory:)`` found:
/// one ``SetupItem`` per ``SetupItem/Component``, in the order of
/// `SetupItem.Component.allCases`.
public struct SetupReport: Sendable, Equatable {
    /// The snapshot that was checked; nil when none was found.
    public var location: ModelLocation?
    /// One item per component.
    public var items: [SetupItem]

    /// Creates a report. `items` must hold one item per component.
    public init(location: ModelLocation?, items: [SetupItem]) {
        self.location = location
        self.items = items
    }

    /// The item for `component`.
    public subscript(component: SetupItem.Component) -> SetupItem {
        items.first { $0.component == component }!
    }

    /// True when nothing is missing, so ``DramaSynthesizer/load(onPhase:)``
    /// can succeed. The first load may still tune the kernels.
    public var isReady: Bool { !items.contains(where: \.isMissing) }

    /// True when every component is ready, so the first load does no work
    /// beyond mapping the weights and compiling the kernels.
    public var isPrepared: Bool { items.allSatisfy { $0.state == .ready } }

    /// The fix of every item that is not ready, in component order.
    public var fixes: [String] { items.compactMap(\.fix) }
}

/// Why ``DramaSetup/prepare(location:codecDirectory:python:tuningDirectory:progress:)``
/// stopped.
public enum DramaSetupError: Error, CustomStringConvertible {
    /// The weights, the tokenizer, or the codec checkpoint are missing.
    /// Downloading them is the user's call, so `prepare` does not.
    case notReady(SetupReport)
    /// The codec models must be built and no Python was named. `command` is
    /// the shell text that builds them.
    case pythonRequired(command: String)
    /// The conversion script exited with `status`. `output` is the end of
    /// what it printed.
    case conversionFailed(command: String, status: Int32, output: String)

    /// A message for a person, with the fixes or the command to run.
    public var description: String {
        switch self {
        case .notReady(let report):
            return "The Drama 3 install is incomplete:\n" + report.items.filter(\.isMissing).map { "- \($0.summary) \($0.fix ?? "")" }.joined(separator: "\n")
        case .pythonRequired(let command):
            return "The Core ML codec models are missing. Build them once with a Python that has torch 2.7.1 and coremltools:\n\(command)"
        case .conversionFailed(let command, let status, let output):
            return "The codec conversion exited with status \(status):\n\(command)\n\(output)"
        }
    }
}

/// Checks and completes an install without the app: the weights, the
/// tokenizer, the Core ML codec models, and the kernel tuning.
///
/// ``check(location:codecDirectory:tuningDirectory:)`` reads files only and
/// never loads the model. ``prepare(location:codecDirectory:python:tuningDirectory:progress:)``
/// builds what is missing and can be built: the codec models, through a
/// Python you name, and the kernel tuning, through one model load.
public enum DramaSetup: Sendable {
    /// The shell command that downloads the model into the Hugging Face cache.
    public static let downloadCommand = "uvx --from huggingface_hub hf download fishaudio/drama-3"

    /// Where the suggested Python environment for the codec conversion lives:
    /// `~/.cache/drama-codec-venv`.
    public static var suggestedPythonEnvironment: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/drama-codec-venv", directoryHint: .isDirectory)
    }

    /// The shell command that creates ``suggestedPythonEnvironment`` with the
    /// packages the codec conversion needs. It needs `uv`.
    public static var pythonSetupCommand: String {
        let venv = suggestedPythonEnvironment
        let python = venv.appending(path: "bin/python")
        return "uv venv --python 3.12 \(shellQuoted(venv.path)) && uv pip install --python \(shellQuoted(python.path)) \"torch==2.7.1\" coremltools \"numpy<2.4\" einops soundfile"
    }

    /// `convert_codec.py` inside DramaEngine's resource bundle; nil when the
    /// bundle is missing.
    public static var conversionScript: URL? {
        MetalContext.resourceBundle().url(forResource: "CodecConversion", withExtension: nil)?.appending(path: "convert_codec.py")
    }

    /// The shell command that builds the codec models from `location`'s
    /// `codec.pth` into `codecDirectory` with `python`. It takes a few minutes
    /// on the CPU.
    public static func codecConversionCommand(python: URL, location: ModelLocation, codecDirectory: URL = DramaCodec.defaultDirectory) -> String {
        let script = conversionScript?.path ?? "Sources/DramaEngine/CodecConversion/convert_codec.py"
        return [python.path, "-I", "-B", script, "--codec", codecCheckpoint(location).path, "--out", codecDirectory.path]
            .map(shellQuoted).joined(separator: " ")
    }

    /// Reports each component of the install. Reads files only: it never
    /// loads the model, starts Python, or writes anything.
    ///
    /// - Parameters:
    ///   - location: The snapshot to check, usually ``ModelLocation/huggingFaceCache(repo:)``.
    ///     Nil reports the weights as not downloaded.
    ///   - codecDirectory: Where the Core ML codec models live. Defaults to
    ///     ``DramaCodec/defaultDirectory``.
    ///   - tuningDirectory: Where tuned kernel schedules live. Defaults to
    ///     ``DramaSynthesizer/defaultTuningDirectory``.
    public static func check(location: ModelLocation?, codecDirectory: URL = DramaCodec.defaultDirectory,
                             tuningDirectory: URL = DramaSynthesizer.defaultTuningDirectory) -> SetupReport {
        let codecModels = codecModelsItem(location: location, codecDirectory: codecDirectory)
        return SetupReport(location: location, items: [
            weightsItem(location),
            tokenizerItem(location),
            codecCheckpointItem(location, codecModelsReady: codecModels.state == .ready),
            codecModels,
            tuningItem(location: location, tuningDirectory: tuningDirectory),
        ])
    }

    /// Builds whatever is missing and can be built, then returns once the
    /// first load of the engine has nothing left to do.
    ///
    /// 1. When the codec models are missing, runs the bundled
    ///    `convert_codec.py` with `python`, a few minutes on the CPU. With no
    ///    `python` it throws ``DramaSetupError/pythonRequired(command:)``
    ///    with the commands to run instead. It never downloads Python.
    /// 2. When this Mac has no tuned kernel schedule, loads the model once,
    ///    which tunes and caches it (about 30 s after the load), and then
    ///    releases the model. A tuning failure keeps the default schedule
    ///    and does not throw, as in ``DramaSynthesizer/load(onPhase:)``.
    ///
    /// Calling it again after it succeeds returns without loading anything.
    ///
    /// - Parameters:
    ///   - location: The snapshot. Its weights, tokenizer and, while the
    ///     codec models are missing, `codec.pth` must be present, or it
    ///     throws ``DramaSetupError/notReady(_:)``.
    ///   - codecDirectory: Where to find or build the codec models.
    ///   - python: A Python 3 with torch 2.7.1, coremltools, numpy, einops
    ///     and soundfile. ``pythonSetupCommand`` makes one.
    ///   - tuningDirectory: Where tuned kernel schedules live.
    ///   - progress: Called with one line per step, and with each line the
    ///     conversion script prints.
    public static func prepare(location: ModelLocation, codecDirectory: URL = DramaCodec.defaultDirectory, python: URL? = nil,
                               tuningDirectory: URL = DramaSynthesizer.defaultTuningDirectory,
                               progress: @escaping @Sendable (String) -> Void = { _ in }) async throws {
        var report = check(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
        guard !report.items.contains(where: { $0.isMissing && $0.component != .codecModels }) else { throw DramaSetupError.notReady(report) }
        if report[.codecModels].isMissing {
            guard let python else {
                let suggested = suggestedPythonEnvironment.appending(path: "bin/python")
                throw DramaSetupError.pythonRequired(command: pythonSetupCommand + "\n" + codecConversionCommand(python: suggested, location: location, codecDirectory: codecDirectory))
            }
            progress("Building the Core ML codec models with \(python.path), a few minutes")
            try await runConversion(python: python, location: location, codecDirectory: codecDirectory, progress: progress)
            report = check(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
            guard report.isReady else { throw DramaSetupError.notReady(report) }
        }
        if report.isPrepared { return }
        progress("Loading Drama 3 to tune the kernels for this Mac")
        try await DramaSynthesizer(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory).load(onPhase: progress)
    }

    static func codecCheckpoint(_ location: ModelLocation) -> URL { location.directory.appending(path: "codec.pth") }

    static func weightsItem(_ location: ModelLocation?) -> SetupItem {
        guard let location else {
            return SetupItem(component: .weights, state: .missing(fix: "Run \(downloadCommand), or pass the folder that holds config.json as ModelLocation(directory:)."),
                             url: nil, summary: "No Drama 3 snapshot in the Hugging Face cache.")
        }
        let dir = location.directory
        var expected = ["config.json", "model.safetensors.index.json"]
        if let data = try? Data(contentsOf: dir.appending(path: "model.safetensors.index.json")),
           let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = index["weight_map"] as? [String: String] {
            expected += Set(map.values).sorted()
        }
        let absent = expected.filter { !FileManager.default.fileExists(atPath: dir.appending(path: $0).path) }
        guard absent.isEmpty else {
            let listed = absent.prefix(4).joined(separator: ", ") + (absent.count > 4 ? " and \(absent.count - 4) more" : "")
            return SetupItem(component: .weights, state: .missing(fix: "Run \(downloadCommand) to finish the download."),
                             url: dir, summary: "Missing from \(dir.path): \(listed).")
        }
        let shards = expected.dropFirst(2)
        let bytes = shards.reduce(UInt64(0)) { $0 + fileSize(dir.appending(path: $1)) }
        return SetupItem(component: .weights, state: .ready, url: dir,
                         summary: "Weights: \(shards.count) shards, \(gigabytes(bytes)) at \(dir.path).")
    }

    static func tokenizerItem(_ location: ModelLocation?) -> SetupItem {
        guard let url = location?.directory.appending(path: "tokenizer.json") else {
            return SetupItem(component: .tokenizer, state: .missing(fix: "Run \(downloadCommand)."), url: nil, summary: "No tokenizer: the snapshot is missing.")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            return SetupItem(component: .tokenizer, state: .missing(fix: "Run \(downloadCommand) to finish the download."), url: url, summary: "\(url.path) is missing.")
        }
        return SetupItem(component: .tokenizer, state: .ready, url: url, summary: "Tokenizer at \(url.path).")
    }

    static func codecCheckpointItem(_ location: ModelLocation?, codecModelsReady: Bool) -> SetupItem {
        let url = location.map(codecCheckpoint)
        if let url, FileManager.default.fileExists(atPath: url.path) {
            return SetupItem(component: .codecCheckpoint, state: .ready, url: url, summary: "Codec checkpoint at \(url.path).")
        }
        if codecModelsReady {
            return SetupItem(component: .codecCheckpoint, state: .ready, url: url, summary: "No codec.pth, which is fine: the codec models are built.")
        }
        return SetupItem(component: .codecCheckpoint, state: .missing(fix: "Run \(downloadCommand) to finish the download."), url: url,
                         summary: "codec.pth is missing, and the codec models are built from it.")
    }

    static func codecModelsItem(location: ModelLocation?, codecDirectory: URL) -> SetupItem {
        if DramaCodec.modelsExist(in: codecDirectory) {
            return SetupItem(component: .codecModels, state: .ready, url: codecDirectory, summary: "Core ML codec models at \(codecDirectory.path).")
        }
        let suggested = suggestedPythonEnvironment.appending(path: "bin/python")
        let convert = location.map { codecConversionCommand(python: suggested, location: $0, codecDirectory: codecDirectory) }
            ?? "DramaSetup.prepare(location:python:) once the snapshot is downloaded"
        return SetupItem(component: .codecModels,
                         state: .missing(fix: "Build them once: \(pythonSetupCommand), then call DramaSetup.prepare(location:python:) with \(suggested.path), or run \(convert)"),
                         url: codecDirectory, summary: "No Core ML codec models in \(codecDirectory.path).")
    }

    static func tuningItem(location: ModelLocation?, tuningDirectory: URL) -> SetupItem {
        let later = SetupItem.State.builtOnFirstLoad(fix: "The first load tunes the kernels for this Mac, about 30 s, and caches the result. DramaSetup.prepare does it ahead of time.")
        guard let device = MTLCreateSystemDefaultDevice() else {
            return SetupItem(component: .tuning, state: .missing(fix: "Run on a Mac with Apple silicon."), url: nil, summary: "No Metal device.")
        }
        let source: String
        do { source = try MetalContext.kernelSources() } catch {
            return SetupItem(component: .tuning, state: .missing(fix: "Ship DramaEngine's resource bundle, \(MetalContext.resourceBundleName), with the app."),
                             url: nil, summary: "The Metal kernel sources are missing: \(error).")
        }
        guard let location, let key = try? TuningKey(chip: device.name, kernelDigest: MetalContext.digest(source), location: location) else {
            return SetupItem(component: .tuning, state: later, url: nil, summary: "Not checked: the weights are missing.")
        }
        guard let (report, url) = TuningCache.load(key: key, directory: tuningDirectory) else {
            return SetupItem(component: .tuning, state: later, url: tuningDirectory, summary: "No kernel schedule tuned for \(device.name) on macOS \(key.os) yet.")
        }
        return SetupItem(component: .tuning, state: .ready, url: url,
                         summary: "Kernels tuned for \(device.name) on macOS \(key.os), \(report.created.formatted(date: .abbreviated, time: .omitted)).")
    }

    static func runConversion(python: URL, location: ModelLocation, codecDirectory: URL, progress: @escaping @Sendable (String) -> Void) async throws {
        let command = codecConversionCommand(python: python, location: location, codecDirectory: codecDirectory)
        guard let script = conversionScript else {
            throw DramaSetupError.conversionFailed(command: command, status: -1, output: "convert_codec.py is missing from \(MetalContext.resourceBundleName).")
        }
        let run = ConversionRun(progress: progress)
        run.process.executableURL = python
        run.process.arguments = ["-I", "-B", script.path, "--codec", codecCheckpoint(location).path, "--out", codecDirectory.path]
        let status = try await run.start()
        guard status == 0 else { throw DramaSetupError.conversionFailed(command: command, status: status, output: run.tail) }
    }

    static func fileSize(_ url: URL) -> UInt64 {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
        return (attributes?[.size] as? NSNumber)?.uint64Value ?? 0
    }

    static func gigabytes(_ bytes: UInt64) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }

    static func shellQuoted(_ text: String) -> String {
        text.allSatisfy { $0.isLetter || $0.isNumber || "/._-=".contains($0) } ? text : "'" + text.replacingOccurrences(of: "'", with: #"'\''"#) + "'"
    }
}

/// A child Python process and the tail of its output. Unchecked because
/// `Process` is not Sendable; the lock guards `lines`, and `terminate` is
/// safe from any thread once the process runs.
private final class ConversionRun: @unchecked Sendable {
    let process = Process()
    private let progress: @Sendable (String) -> Void
    private let lock = NSLock()
    private var lines: [String] = []
    private var partial = Data()

    init(progress: @escaping @Sendable (String) -> Void) { self.progress = progress }

    var tail: String { lock.withLock { lines.suffix(40).joined(separator: "\n") } }

    func start() async throws -> Int32 {
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [self] handle in receive(handle.availableData) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                process.terminationHandler = { _ in continuation.resume() }
                do { try process.run() } catch {
                    process.terminationHandler = nil
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: { [self] in
            if process.isRunning { process.terminate() }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        receive(pipe.fileHandleForReading.readDataToEndOfFile())
        receive(Data("\n".utf8))
        try Task.checkCancellation()
        return process.terminationStatus
    }

    private func receive(_ data: Data) {
        let complete: [String] = lock.withLock {
            partial.append(data)
            guard let end = partial.lastIndex(of: UInt8(ascii: "\n")) else { return [] }
            let text = String(decoding: partial[..<end], as: UTF8.self)
            partial.removeSubrange(...end)
            let found = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
            lines += found
            return found
        }
        complete.forEach(progress)
    }
}
