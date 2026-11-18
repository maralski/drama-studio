// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import DramaEngine
import DramaServer
import Foundation

struct Args {
    var values: [String: String] = [:]
    var flags: Set<String> = []
    var positional: [String] = []
    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") { values[key] = argv[i + 1]; i += 1 } else { flags.insert(key) }
            } else { positional.append(a) }
            i += 1
        }
    }
    subscript(_ k: String) -> String? { values[k] }
    func int(_ k: String, _ d: Int) -> Int { values[k].flatMap(Int.init) ?? d }
    func double(_ k: String, _ d: Double) -> Double { values[k].flatMap(Double.init) ?? d }
}

func awaitBlocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let group = DispatchGroup()
    group.enter()
    nonisolated(unsafe) var result: Result<T, Error>?
    Task.detached { defer { group.leave() }; do { result = .success(try await body()) } catch { result = .failure(error) } }
    group.wait()
    return try result!.get()
}

func fail(_ m: String) -> Never { FileHandle.standardError.write((m + "\n").data(using: .utf8)!); exit(1) }

/// drama setup [--model <dir>] [--ming-weights <dir>]: what each engine's
/// install has and what it lacks, with the fix for each gap. Reads files only.
/// Exits 1 when no engine is ready.
func setup(_ args: Args) -> Never {
    let reports = [
        DramaSetup.check(location: args["model"].map { ModelLocation(directory: URL(fileURLWithPath: $0)) } ?? .huggingFaceCache()),
        DramaSetup.check(ming: args["ming-weights"].map { MingLocation(directory: URL(fileURLWithPath: $0)) } ?? MingLocation.find()),
    ]
    for report in reports {
        print("\(report.engine.capabilities.name), \(report.engine.install.repo): \(report.isReady ? "ready" : "not ready")")
        for item in report.items {
            let mark = switch item.state {
            case .ready: "ok     "
            case .missing: "missing"
            case .builtOnFirstLoad: "later  "
            }
            print("  \(mark) \(item.component.rawValue): \(item.summary)")
            if let fix = item.fix { print("          \(fix)") }
        }
    }
    exit(reports.contains(where: \.isReady) ? 0 : 1)
}

/// drama serve [--port 8780] [--no-token] [--engine drama3|ming] [--model <dir>] [--ming-weights <dir>] [--voices <dir>]:
/// the app's HTTP API, headless, on every installed engine. `--engine`, else
/// DRAMA_ENGINE, picks the one requests use when they name none; it defaults
/// to Drama 3 when installed. DRAMA_PREVIEW_ENGINE=1 serves mocks of both
/// engines, as in the app; MING_WEIGHTS or the Hugging Face cache locates Ming.
/// `--voices` names the voice library, by default the app's.
func serve(_ args: Args) throws -> Never {
    guard let port = UInt16(exactly: args.int("port", Int(APIService.defaultPort))) else { fail("--port must be between 0 and 65535") }
    let named = args["engine"] ?? ProcessInfo.processInfo.environment["DRAMA_ENGINE"]
    let chosen = named.map { name -> SpeechEngineID in
        guard let engine = SpeechEngineID(rawValue: name) else { fail("--engine must be drama3 or ming") }
        return engine
    }
    var factories: [SpeechEngineID: SpeechEngines.Factory] = [:]
    var gaps: [SpeechEngineID: String] = [:]
    if ProcessInfo.processInfo.environment["DRAMA_PREVIEW_ENGINE"] == "1" {
        for engine in SpeechEngineID.allCases { factories[engine] = .mock(engine) }
    } else {
        let drama = args["model"].map { ModelLocation(directory: URL(fileURLWithPath: $0)) } ?? ModelLocation.huggingFaceCache()
        if let drama, drama.isComplete, DramaCodec.modelsExist() {
            factories[.drama3] = .drama3(location: drama, codecDirectory: DramaCodec.defaultDirectory)
        } else {
            gaps[.drama3] = "no complete Drama 3 install; pass --model <dir> or run `drama setup` for what is missing"
        }
        if let ming = args["ming-weights"].map({ MingLocation(directory: URL(fileURLWithPath: $0)) }) ?? MingLocation.find(), ming.isComplete {
            factories[.ming] = .ming(location: ming)
        } else {
            gaps[.ming] = "no complete Ming checkpoint; pass --ming-weights <dir>, set MING_WEIGHTS, or run `drama setup` for what is missing"
        }
    }
    guard let engine = chosen ?? SpeechEngineID.allCases.first(where: { factories[$0] != nil }) else {
        fail(SpeechEngineID.allCases.compactMap { gaps[$0] }.joined(separator: "\n"))
    }
    guard factories[engine] != nil else { fail(gaps[engine] ?? "\(engine.capabilities.name) isn't installed") }
    for (other, gap) in gaps where other != engine { FileHandle.standardError.write("\(other.capabilities.name) not served: \(gap)\n".data(using: .utf8)!) }
    let voices = args["voices"].map { VoiceDirectory(url: URL(fileURLWithPath: $0, isDirectory: true)) } ?? VoiceDirectory()
    let synthesizer = try SpeechEngines(selected: engine, factories: factories, voices: voices)
    let served = SpeechEngineID.allCases.filter { factories[$0] != nil }.map(\.capabilities.name).joined(separator: " and ")
    let token = args.flags.contains("no-token") ? nil : try APIToken.loadOrCreate()
    let service = APIService(synthesizer: synthesizer, voices: voices, token: token, version: "cli")
    let server = APIServer(service: service, port: port) { state in
        switch state {
        case .listening(let port):
            print("Drama API on http://127.0.0.1:\(port) (\(service.info.kind.rawValue): \(served); \(engine.capabilities.name) by default)")
            print(token == nil ? "no bearer token (--no-token)" : "bearer token in \(APIToken.defaultURL.path)")
            fflush(stdout)
        case .failed(let message):
            fail("API server failed: \(message)")
        case .stopped, .starting:
            break
        }
    }
    server.start()
    Task.detached {
        do {
            try await synthesizer.load { FileHandle.standardError.write("loading: \($0)\n".data(using: .utf8)!) }
            FileHandle.standardError.write("model loaded\n".data(using: .utf8)!)
        } catch {
            fail("model failed to load: \(error)")
        }
    }
    dispatchMain()
}

let usage = "usage: drama <serve|setup|check-words> [options]"
let args = Args(Array(CommandLine.arguments.dropFirst()))
if args.flags.contains("help") || args.positional.contains("-h") || args.positional.first == "help" {
    if args.positional.first == "serve" {
        print("usage: drama serve [--port N] [--engine drama3|ming] [--model DIR] [--ming-weights DIR] [--voices DIR] [--no-token]")
    } else {
        print(usage)
    }
    exit(0)
}
switch args.positional.first {
case "serve": try serve(args)
case "setup": setup(args)
case "check-words": exit(WordCheckDiagnostic.run(Array(args.positional.dropFirst())))
default: fail(usage)
}
