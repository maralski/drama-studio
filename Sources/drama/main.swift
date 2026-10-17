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

func loadModel(_ args: Args) throws -> DramaModel {
    guard let loc = args["model"].map({ ModelLocation(directory: URL(fileURLWithPath: $0)) }) ?? ModelLocation.huggingFaceCache() else {
        fail("no model found; pass --model <dir> or run: hf download fishaudio/drama-3")
    }
    var precision = Precision.exact
    if args["precision"] == "reference" { precision = .reference }
    let t0 = Date()
    let model = try DramaModel(location: loc, precision: precision, capacity: args.int("capacity", DramaModel.defaultCapacity),
                               schedule: args.flags.contains("untuned") ? .default : nil)
    FileHandle.standardError.write("loaded \(loc.directory.lastPathComponent) in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s\n".data(using: .utf8)!)
    return model
}

func readFrames(_ path: String) throws -> [Frame] {
    let a = try NPY.read(url: URL(fileURLWithPath: path))
    guard a.shape.count == 2, a.shape[1] == 11 else { fail("\(path): expected [N, 11]") }
    let v = a.int64Values
    return (0..<a.shape[0]).map { r in (0..<11).map { Int32(v[r * 11 + $0]) } }
}

/// drama dump --rows prompt_rows.npy [--frames frames.npy | --steps N] --out out.npz
func dump(_ args: Args) throws {
    guard let outPath = args["out"], args["rows"] != nil || args["text"] != nil else { fail("usage: drama dump (--rows prompt_rows.npy | --text <text>) [--frames frames.npy | --steps N] --out out.npz") }
    let forced = try args["frames"].map(readFrames)
    let steps = forced?.count ?? args.int("steps", 8)
    let temperature = args.double("temperature", 0.7), topK = args.int("top-k", 30), topP = args.double("top-p", 0.7)
    let seed = UInt64(args.int("seed", 0))
    let model = try loadModel(args)
    let prompt: [Frame]
    if let rowsPath = args["rows"] { prompt = try readFrames(rowsPath) } else {
        let tokenizer = try Tokenizer(url: model.location.directory.appendingPathComponent("tokenizer.json"))
        prompt = try Drama3Prompt.build(text: args["text"]!, tokenizer: tokenizer)
    }
    let t0 = Date()
    let session = try GenerationSession(model: model, prompt: prompt, trace: true)
    let prefillSeconds = Date().timeIntervalSince(t0)
    var sampler = Sampler(seed: seed)
    var hidden: [[Float]] = [], cand: [[Float]] = [], towerLogits: [[[Float]]] = [], frames: [Frame] = []
    let t1 = Date()
    for i in 0..<steps {
        hidden.append(session.lastHidden)
        let frame: Frame
        let trace: StepTrace
        if let forced {
            frame = forced[i]
            trace = try session.force(frame)
        } else {
            let r = try session.step(temperature: temperature, topK: topK, topP: topP, sampler: &sampler)
            guard let f = r.frame else { hidden.removeLast(); break }
            frame = f
            trace = r.trace
        }
        cand.append(trace.candidateLogits)
        towerLogits.append(trace.towerLogits)
        frames.append(frame)
    }
    hidden.append(session.lastHidden)
    cand.append(session.currentCandidateLogits)
    let decodeSeconds = Date().timeIntervalSince(t1)
    let H = model.config.hiddenSize
    let F = frames.count
    let meta = "{\"engine\": \"metal\", \"precision\": \"\(args["precision"] ?? "exact")\", \"seed\": \(seed), \"temperature\": \(temperature), \"top_k\": \(topK), \"top_p\": \(topP), \"prefill_seconds\": \(prefillSeconds), \"decode_seconds\": \(decodeSeconds), \"steps\": \(F)}"
    var entries: [(String, NPY.Array)] = [
        ("prompt_rows", .int64(prompt.flatMap { $0.map(Int64.init) }, shape: [prompt.count, 11])),
        ("frames", .int64(frames.flatMap { $0.map(Int64.init) }, shape: [F, 11])),
        ("hidden", .float32(hidden.flatMap { $0 }, shape: [F + 1, H])),
        ("cand_logits", .float32(cand.flatMap { $0 }, shape: [F + 1, model.config.codebookSize + 1])),
        ("tower_logits", .float32(towerLogits.flatMap { $0.flatMap { $0 } }, shape: [F, model.config.numCodebooks - 1, model.config.codebookSize])),
        ("layer_out_step0", .float32(session.layerOutStep0.flatMap { $0 }, shape: [model.config.numLayers, H])),
        ("tower_hidden_step0", .float32(session.towerHiddenStep0.flatMap { $0 }, shape: [session.towerHiddenStep0.count, H])),
    ]
    entries.append(("meta", NPY.Array(dtype: .uint8, shape: [meta.utf8.count], data: Data(meta.utf8))))
    try NPY.writeNPZ(entries, to: URL(fileURLWithPath: outPath))
    print("prefill \(prompt.count) rows: \(String(format: "%.2f", prefillSeconds))s; \(F) steps: \(String(format: "%.2f", decodeSeconds))s (\(String(format: "%.1f", Double(F) / max(decodeSeconds, 1e-9))) frames/s); wrote \(outPath)")
}

func scheduleLine(_ model: DramaModel, args: Args) -> String {
    switch model.scheduleSource {
    case .cache(let url): return "schedule: tuned (\(url.path))"
    case .given: return args.flags.contains("untuned") ? "schedule: default (--untuned)" : "schedule: given"
    case .default: return "schedule: default (no tuning cache; run drama tune)"
    }
}

func bench(_ args: Args) throws {
    let model = try loadModel(args)
    print(scheduleLine(model, args: args))
    if args.flags.contains("warm") { print(String(format: "warm-up walk: %.2fs", model.benchWarmUp())) }
    let t0 = Date(); _ = try model.benchBackbone(rows: 1, skip: []); print(String(format: "first forward (page faults): %.2fs", Date().timeIntervalSince(t0)))
    for rows in [1, 8, 32, 128, 300] {
        var line = "rows \(rows):"
        for skip in [Set<String>(), ["moe"], ["experts"], ["shared"], ["linear"], ["gdn"], ["full"], ["attn"], ["moe", "linear", "full"]] {
            let t = try model.benchBackbone(rows: rows, skip: skip)
            line += String(format: "  %@=%.1fms", skip.isEmpty ? "all" : "no-" + skip.sorted().joined(separator: "-"), t * 1000)
        }
        let tt = try model.benchTower(rows: rows)
        line += String(format: "  tower=%.1fms", tt * 1000)
        print(line)
    }
}

/// Decodes frames with the Core ML codec and writes a 44.1 kHz float WAV.
func writeWav(frames: [Frame], to wavPath: String) throws {
    guard DramaCodec.modelsExist() else { fail("the Core ML codec models are missing; run tools/convert_codec.py") }
    let codec = try awaitBlocking { try await DramaCodec(directory: DramaCodec.defaultDirectory) }
    let t0 = Date()
    let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
    let samples = try DramaSynthesizer.decodeBlocking(codec, codes: codes)
    try writeWav(samples: samples, to: wavPath)
    print(String(format: "decoded %.1fs of audio with the Core ML codec in %.2fs -> %@", Double(samples.count) / 44_100, Date().timeIntervalSince(t0), wavPath))
}

func writeWav(samples: [Float], to wavPath: String) throws {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 1, interleaved: false)!
    let file = try AVAudioFile(forWriting: URL(fileURLWithPath: wavPath), settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
    try file.write(from: buffer)
}

/// drama decode --frames frames.npy --wav out.wav
func decodeCommand(_ args: Args) throws {
    guard let framesPath = args["frames"], let wavPath = args["wav"] else { fail("usage: drama decode --frames frames.npy --wav out.wav") }
    try writeWav(frames: try readFrames(framesPath), to: wavPath)
}

/// drama generate --text "..." [--ref clip_codes.npy --ref-text "..."] --out frames.npy
func generate(_ args: Args) throws {
    guard let text = args["text"], let outPath = args["out"] else { fail("usage: drama generate --text <text> [--ref codes.npy --ref-text <transcript>] --out frames.npy") }
    let model = try loadModel(args)
    let tokenizer = try Tokenizer(url: model.location.directory.appendingPathComponent("tokenizer.json"))
    var refs: [Drama3Prompt.Reference] = []
    if let refPath = args["ref"] {
        let a = try NPY.read(url: URL(fileURLWithPath: refPath))
        guard a.shape.count == 2, a.shape[0] == 10 else { fail("\(refPath): expected [10, T]") }
        let v = a.int64Values.map(Int32.init)
        let codes = (0..<10).map { c in Array(v[(c * a.shape[1])..<((c + 1) * a.shape[1])]) }
        refs.append(Drama3Prompt.Reference(transcript: args["ref-text"] ?? "", codes: codes))
    }
    var config = GenerationConfig()
    config.temperature = args.double("temperature", 0.7); config.topK = args.int("top-k", 30); config.topP = args.double("top-p", 0.7)
    config.maxNewFrames = args.int("max-frames", 600)
    config.chunking = args["chunking"].flatMap(GenerationConfig.Chunking.init(rawValue:)) ?? .reference
    if let s = args["seed"].flatMap(UInt64.init) { config.seed = s }
    if args.flags.contains("warm") { model.warmUp() }
    let gen = FrameGenerator(model: model, tokenizer: tokenizer)
    if args.flags.contains("no-retry") { config.minSpokenRatio = 0 }
    if DramaCodec.modelsExist(), !args.flags.contains("no-codec") {
        let codec = try awaitBlocking { try await DramaCodec(directory: DramaCodec.defaultDirectory) }
        gen.codec = { codes, window in try DramaSynthesizer.decodeBlocking(codec, codes: codes, window: window) }
        FileHandle.standardError.write("codec probe: Core ML\n".data(using: .utf8)!)
    } else {
        config.stopAfterSilenceSeconds = nil
    }
    var all: [Frame] = []
    var audio: [Float] = []
    _ = try gen.generate(text: text, references: refs, config: config) { event in
        switch event {
        case .chunkStarted(let i, let n, let t): FileHandle.standardError.write("chunk \(i + 1)/\(n): \(t.prefix(60))\n".data(using: .utf8)!)
        case .retry(_, let attempt, let spoken, let expected):
            FileHandle.standardError.write(String(format: "  retry %d: spoke %.1fs of an expected %.1fs\n", attempt, spoken, expected).data(using: .utf8)!)
        case .chunkFinished(let r):
            all += r.frames
            audio += r.samples
            FileHandle.standardError.write(String(format: "  %d rows prefilled in %.2fs; %d frames in %.2fs (%.1f frames/s, GPU %.1f ms/frame) eos=%@ attempts=%d\n", r.promptRows, r.prefillSeconds, r.frames.count, r.decodeSeconds, Double(r.frames.count) / max(r.decodeSeconds, 1e-9), r.gpuSeconds * 1000 / Double(max(r.frames.count, 1)), r.finishedByEOS ? "yes" : "no", r.attempts).data(using: .utf8)!)
        default: break
        }
    }
    try NPY.write(.int64(all.flatMap { $0.map(Int64.init) }, shape: [all.count, 11]), to: URL(fileURLWithPath: outPath))
    print("wrote \(all.count) frames (\(String(format: "%.1f", Double(all.count) * 2048 / 44100))s of audio) to \(outPath)")
    guard let wavPath = args["wav"], !all.isEmpty else { return }
    if audio.isEmpty { return try writeWav(frames: all, to: wavPath) }
    try writeWav(samples: audio, to: wavPath)
    print(String(format: "wrote %.1fs of audio decoded during generation -> %@", Double(audio.count) / 44_100, wavPath))
}

/// drama profile [--rows 128]: GPU ms per kernel group for one backbone forward.
func profile(_ args: Args) throws {
    let model = try loadModel(args)
    let rows = args.int("rows", 128)
    let groups = try model.profileBackbone(rows: rows)
    let total = groups.reduce(0) { $0 + $1.1 }
    print("backbone forward, \(rows) rows: \(String(format: "%.1f", total)) ms summed over groups")
    for (name, ms) in groups.sorted(by: { $0.1 > $1.1 }) {
        print("  " + name.padding(toLength: 22, withPad: " ", startingAt: 0) + String(format: " %7.2f ms  %4.1f%%", ms, ms / total * 100))
    }
}

func printTuneTable(_ report: TuningReport) {
    func pad(_ s: String, _ n: Int) -> String { s.padding(toLength: max(n, s.count), withPad: " ", startingAt: 0) }
    print(pad("entry", 18) + pad("default", 10) + pad("chosen", 10) + pad("default ms", 12) + pad("chosen ms", 11) + pad("gain %", 8) + "relL2")
    for e in report.entries {
        let ms = e.rounds == 0 ? ("not measured", "") : (String(format: "%.3f", e.defaultMs), String(format: "%.3f", e.chosenMs))
        print(pad(e.name, 18) + pad(e.defaultLabel, 10) + pad(e.chosenLabel, 10) + pad(ms.0, 12) + pad(ms.1, 11) + pad(String(format: "%.1f", e.gainPercent), 8) + String(format: "%.1e", e.relL2))
        for r in e.rejected { print("  rejected \(r.label): \(r.reason)") }
        if let note = e.note { print("  \(note)") }
    }
    for f in report.forwardChecks {
        print(String(format: "forward %d rows: default %.1f ms, tuned %.1f ms, %@", f.rows, f.defaultMs, f.tunedMs, f.kept ? "kept" : "reverted to default"))
    }
    print(String(format: "tuned in %.1f s wall", report.wallSeconds))
}

/// drama tune [--force] [--rounds N] [--budget S]: tunes the kernel schedule
/// for this Mac and model, caches it, and compares a forward on both schedules.
func tune(_ args: Args) throws {
    let model = try loadModel(args)
    model.warmUp()
    let key = model.tuningKey
    print("chip \(key.chip); OS \(key.os); kernels \(key.kernels.prefix(12)); model \(key.model)")
    if case .cache(let url) = model.scheduleSource, !args.flags.contains("force") {
        guard let hit = TuningCache.load(key: key) else { fail("tuning cache vanished: \(url.path)") }
        print("tuning cache hit, nothing measured: \(url.path)")
        printTuneTable(hit.report)
        return
    }
    var options = Autotune.Options()
    options.rounds = args.int("rounds", options.rounds)
    options.budgetSeconds = args.double("budget", options.budgetSeconds)
    options.progress = { FileHandle.standardError.write(($0 + "\n").data(using: .utf8)!) }
    let report = try Autotune.tune(model: model, options: options)
    printTuneTable(report)
    for rows in [128, 1] {
        try model.apply(.default)
        let before = try model.probeHidden(rows: rows)
        try model.apply(report.schedule)
        let after = try model.probeHidden(rows: rows)
        print(String(format: "hiddenOut relative L2, default against tuned, %d rows: %.2e", rows, Autotune.relL2(before, after)))
    }
    print("cache: \(TuningCache.url(for: key).path)")
}

/// drama schedule-ab --schedules a.json,b.json[,...] [--rows 128,300,1] [--rounds 7]:
/// GPU ms of one backbone forward per schedule and row count, the schedules
/// applied in turn inside one process and one round, so every side sees the
/// same machine state. The one instrument that isolates a single schedule
/// field: write the default schedule with one field changed and compare.
func scheduleAB(_ args: Args) throws {
    guard let list = args["schedules"] else { fail("usage: drama schedule-ab --schedules a.json,b.json [--rows 128,300,1] [--rounds 7]") }
    let paths = list.split(separator: ",").map(String.init)
    let schedules = try paths.map { try JSONDecoder().decode(KernelSchedule.self, from: Data(contentsOf: URL(fileURLWithPath: $0))) }
    let rows = (args["rows"] ?? "128,300,1").split(separator: ",").compactMap { Int($0) }
    let rounds = args.int("rounds", 7)
    let model = try loadModel(args)
    model.warmUp()
    for (p, s) in zip(paths, schedules) { try s.validate(); print("\(p): \(String(data: try JSONEncoder().encode(s), encoding: .utf8)!)") }
    var ms = [[[Double]]](repeating: [[Double]](repeating: [], count: schedules.count), count: rows.count)
    for _ in 0..<rounds {
        for (ri, r) in rows.enumerated() {
            for (si, s) in schedules.enumerated() {
                try model.apply(s)
                ms[ri][si].append(try model.benchBackbone(rows: r, skip: []) * 1000)
            }
        }
    }
    func median(_ v: [Double]) -> Double { let s = v.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }
    for (ri, r) in rows.enumerated() {
        var line = "rows \(r):"
        for (si, p) in paths.enumerated() {
            let v = ms[ri][si]
            line += String(format: "  %@ %.1f [%.1f..%.1f]", (p as NSString).lastPathComponent, median(v), v.min()!, v.max()!)
        }
        print(line)
        let base = ms[ri][0]
        var wins = ""
        for si in 1..<paths.count {
            let w = zip(ms[ri][si], base).filter { $0 < $1 }.count
            wins += "  \((paths[si] as NSString).lastPathComponent) faster than \((paths[0] as NSString).lastPathComponent) in \(w) of \(rounds) rounds"
        }
        print(wins)
    }
}

/// drama kbench --n 8192 --k 2048 [--ms 2,8,32,128]
func kbench(_ args: Args) throws {
    let ms = (args["ms"] ?? "2,4,8,16,32,64,128").split(separator: ",").compactMap { Int($0) }
    if args.flags.contains("moe") { try KernelBench.moe(rowCounts: ms, iterations: args.int("iterations", 5)); return }
    try KernelBench.denseMatmul(n: args.int("n", 8192), k: args.int("k", 2048), ms: ms, iterations: args.int("iterations", 5))
}

/// drama serve [--port 8780] [--no-token]: the app's HTTP API, headless.
/// DRAMA_PREVIEW_ENGINE=1 serves the mock engine, as in the app.
func serve(_ args: Args) throws -> Never {
    guard let port = UInt16(exactly: args.int("port", Int(APIService.defaultPort))) else { fail("--port must be between 0 and 65535") }
    let synthesizer: any SpeechSynthesizer
    if ProcessInfo.processInfo.environment["DRAMA_PREVIEW_ENGINE"] == "1" {
        synthesizer = MockSynthesizer()
    } else {
        guard let location = args["model"].map({ ModelLocation(directory: URL(fileURLWithPath: $0)) }) ?? ModelLocation.huggingFaceCache(), location.isComplete else {
            fail("no complete model found; pass --model <dir> or run: hf download fishaudio/drama-3")
        }
        guard DramaCodec.modelsExist() else { fail("the Core ML codec models are missing; run tools/convert_codec.py") }
        synthesizer = DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory)
    }
    let token = args.flags.contains("no-token") ? nil : try APIToken.loadOrCreate()
    let service = APIService(synthesizer: synthesizer, voices: VoiceDirectory(), token: token, version: "cli")
    let server = APIServer(service: service, port: port) { state in
        switch state {
        case .listening(let port):
            print("Drama API on http://127.0.0.1:\(port) (\(service.info.kind.rawValue) engine)")
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

let args = Args(Array(CommandLine.arguments.dropFirst()))
switch args.positional.first {
case "dump": try dump(args)
case "bench": try bench(args)
case "kbench": try kbench(args)
case "profile": try profile(args)
case "tune": try tune(args)
case "schedule-ab": try scheduleAB(args)
case "generate": try generate(args)
case "decode": try decodeCommand(args)
case "serve": try serve(args)
default: fail("usage: drama <dump|bench|kbench|profile|tune|schedule-ab|generate|decode|serve> [options]")
}
