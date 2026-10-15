import DramaEngine
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
    let model = try DramaModel(location: loc, precision: precision, capacity: args.int("capacity", DramaModel.defaultCapacity))
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

func bench(_ args: Args) throws {
    let model = try loadModel(args)
    if args.flags.contains("warm") { print(String(format: "warm-up walk: %.2fs", model.benchWarmUp())) }
    let t0 = Date(); _ = try model.benchBackbone(rows: 1, skip: []); print(String(format: "first forward (page faults): %.2fs", Date().timeIntervalSince(t0)))
    for rows in [1, 8, 32, 128] {
        var line = "rows \(rows):"
        for skip in [Set<String>(), ["moe"], ["linear"], ["full"], ["moe", "linear", "full"]] {
            let t = try model.benchBackbone(rows: rows, skip: skip)
            line += String(format: "  %@=%.1fms", skip.isEmpty ? "all" : "no-" + skip.sorted().joined(separator: "-"), t * 1000)
        }
        let tt = try model.benchTower(rows: rows)
        line += String(format: "  tower=%.1fms", tt * 1000)
        print(line)
    }
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
        gen.silenceProbe = { frames in
            let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
            return DramaSynthesizer.trailingSilenceSeconds(try DramaSynthesizer.decodeBlocking(codec, codes: codes))
        }
        FileHandle.standardError.write("codec probe: Core ML\n".data(using: .utf8)!)
    } else {
        config.stopAfterSilenceSeconds = nil
    }
    var all: [Frame] = []
    _ = try gen.generate(text: text, references: refs, config: config) { event in
        switch event {
        case .chunkStarted(let i, let n, let t): FileHandle.standardError.write("chunk \(i + 1)/\(n): \(t.prefix(60))\n".data(using: .utf8)!)
        case .retry(_, let attempt, let spoken, let expected):
            FileHandle.standardError.write(String(format: "  retry %d: spoke %.1fs of an expected %.1fs\n", attempt, spoken, expected).data(using: .utf8)!)
        case .chunkFinished(let r):
            all += r.frames
            FileHandle.standardError.write(String(format: "  %d rows prefilled in %.2fs; %d frames in %.2fs (%.1f frames/s) eos=%@ attempts=%d\n", r.promptRows, r.prefillSeconds, r.frames.count, r.decodeSeconds, Double(r.frames.count) / max(r.decodeSeconds, 1e-9), r.finishedByEOS ? "yes" : "no", r.attempts).data(using: .utf8)!)
        default: break
        }
    }
    try NPY.write(.int64(all.flatMap { $0.map(Int64.init) }, shape: [all.count, 11]), to: URL(fileURLWithPath: outPath))
    print("wrote \(all.count) frames (\(String(format: "%.1f", Double(all.count) * 2048 / 44100))s of audio) to \(outPath)")
}

let args = Args(Array(CommandLine.arguments.dropFirst()))
switch args.positional.first {
case "dump": try dump(args)
case "bench": try bench(args)
case "generate": try generate(args)
default: fail("usage: drama <dump> [options]")
}
