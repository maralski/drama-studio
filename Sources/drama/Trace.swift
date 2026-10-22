import DramaEngine
import Foundation

struct TraceRow: Encodable {
    var index: Int
    var kernel: String
    var phase: String
    var shape: String?
    var weightBytes: Int?
    var medianUs: Double
    var minUs: Double
    var maxUs: Double
    var gapBeforeUs: Double
    var durationsUs: [Double]
}

struct TraceReport: Encodable {
    var chip: String
    var schedule: String
    var method: String
    var promptRows: Int
    var rounds: Int
    var heatDispatches: Int
    var pipelinedFrameMs: Double
    var pipelinedWallMs: Double
    var coldFrameMs: [Double]
    var normalFrameMs: [Double]
    var tracedCommandBufferMs: [Double]
    var tracedSpanMs: [Double]
    var dispatchFloorSerialUs: [Double]
    var dispatchFloorSplitUs: [Double]
    var dispatchFloorConcurrentUs: [Double]
    var idleEstimateMs: Double
    var dispatchSumMs: Double
    var gapSumMs: Double
    var smallDispatchMs: Double
    var smallDispatchCount: Int
    var dispatches: [TraceRow]
}

private func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

/// drama trace [--rows 1] [--rounds 5] [--heat 200] [--pipelined 40] [--text "..."] [--seed 0] [--out trace.json]
///
/// GPU time of every dispatch of a real decode frame: samplers, tower, the
/// backbone step for the new row and the lm-head candidates. Each dispatch
/// runs in a compute pass of its own (Apple GPUs sample timestamps only at
/// encoder boundaries), so the frame is also timed untraced, and the cost of
/// a near-empty dispatch is measured in the engine's encoder (a barrier
/// before each), in a concurrent encoder without barriers, and one pass
/// each; barriered minus unordered, times the dispatch count, estimates the
/// untraced frame's idle time between dispatches. A round times a frame on an
/// idle GPU, then an untraced and a traced frame each behind `--heat` GEMVs
/// that bring the GPU to the clock pipelined decode keeps it at; each
/// dispatch reports its median over the rounds. A pipelined run of
/// `--pipelined` frames gives the reference GPU time per frame.
func trace(_ args: Args) throws {
    guard args.int("rows", 1) == 1 else { fail("drama trace times one decode frame; only --rows 1 is supported") }
    let rounds = args.int("rounds", 5)
    let model = try loadModel(args)
    let schedule = scheduleLine(model, args: args)
    print(schedule)
    model.warmUp()
    let tokenizer = try Tokenizer(url: model.location.directory.appendingPathComponent("tokenizer.json"))
    let text = args["text"] ?? "The lighthouse keeper climbed the stairs one last time, counting every step aloud, because the sea had taken everything else and numbers were all he had left."
    let prompt = try Drama3Prompt.build(text: text, tokenizer: tokenizer)
    let session = try GenerationSession(model: model, prompt: prompt, trace: true)
    var sampler = Sampler(seed: UInt64(args.int("seed", 0)))
    func step() throws -> Double {
        let before = session.gpuSeconds
        guard try session.step(temperature: 0.7, topK: 30, topP: 0.7, sampler: &sampler).frame != nil else { fail("the model emitted eos; pass a longer --text") }
        return (session.gpuSeconds - before) * 1000
    }
    for _ in 0..<3 { _ = try step() }
    let heat = args.int("heat", 200)
    var coldMs: [Double] = [], normalMs: [Double] = [], tracedMs: [Double] = [], frames: [[TracedDispatch]] = []
    for _ in 0..<rounds {
        coldMs.append(try step())
        try model.heat(dispatches: heat)
        normalMs.append(try step())
        let tracer = try DispatchTracer(ctx: model.ctx)
        try model.heat(dispatches: heat)
        model.tracer = tracer
        let ms = try step()
        model.tracer = nil
        tracedMs.append(ms)
        frames.append(try tracer.collect())
    }
    var floorSerial: [Double] = [], floorSplit: [Double] = [], floorConcurrent: [Double] = []
    for _ in 0..<3 {
        try model.heat(dispatches: heat)
        floorSerial.append(try model.dispatchFloor(count: 1000))
        try model.heat(dispatches: heat)
        floorConcurrent.append(try model.dispatchFloor(count: 1000, unordered: true))
        try model.heat(dispatches: heat)
        model.tracer = try DispatchTracer(ctx: model.ctx)
        floorSplit.append(try model.dispatchFloor(count: 1000))
        model.tracer = nil
    }
    let pipelined = try GenerationSession(model: model, prompt: prompt)
    var pipelinedSampler = Sampler(seed: UInt64(args.int("seed", 0)))
    var pipelinedFrames = 0
    let wall0 = Date()
    for _ in 0..<args.int("pipelined", 40) {
        guard try pipelined.step(temperature: 0.7, topK: 30, topP: 0.7, sampler: &pipelinedSampler).frame != nil else { break }
        pipelinedFrames += 1
    }
    let pipelinedWallMs = Date().timeIntervalSince(wall0) * 1000 / Double(max(pipelinedFrames, 1))
    let pipelinedMs = pipelined.gpuSeconds * 1000 / Double(max(pipelinedFrames, 1))

    let first = frames[0]
    for f in frames where f.map(\.kernel) != first.map(\.kernel) { fail("the dispatch sequence changed between rounds: \(f.count) against \(first.count) dispatches") }
    let rows: [TraceRow] = first.indices.map { i in
        let d = frames.map { $0[i].durationUs }
        let gaps = frames.map { i == 0 ? 0 : $0[i].startUs - $0[i - 1].endUs }
        return TraceRow(index: i, kernel: first[i].kernel, phase: first[i].phase, shape: first[i].shape, weightBytes: first[i].weightBytes,
                        medianUs: median(d), minUs: d.min()!, maxUs: d.max()!, gapBeforeUs: median(gaps), durationsUs: d)
    }
    let dispatchSum = rows.reduce(0) { $0 + $1.medianUs } / 1000
    let gapSum = rows.reduce(0) { $0 + $1.gapBeforeUs } / 1000
    let small = rows.filter { $0.medianUs < 20 }
    let smallMs = small.reduce(0) { $0 + $1.medianUs } / 1000
    let spans = frames.map { ($0.map(\.endUs).max()! - $0.map(\.startUs).min()!) / 1000 }
    let idleMs = Double(first.count) * (median(floorSerial) - median(floorConcurrent)) / 1000

    func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s + " " : s.padding(toLength: n, withPad: " ", startingAt: 0) }
    func lpad(_ s: String, _ n: Int) -> String { s.count >= n ? " " + s : String(repeating: " ", count: n - s.count) + s }
    func range(_ v: [Double]) -> String { String(format: "%.2f [%.2f..%.2f]", median(v), v.min()!, v.max()!) }
    func gbps(_ bytes: Int, _ us: Double) -> String { us > 0 ? String(format: "%.0f", Double(bytes) / us / 1000) : "-" }

    print("prompt \(prompt.count) rows; \(rounds) rounds; \(first.count) dispatches per frame")
    print("method: one compute pass per dispatch, stage-boundary timestamps (no dispatch-boundary sampling on this GPU)")
    print(String(format: "pipelined decode, %d frames: wall %.2f ms/frame; summed command buffer GPU spans %.2f ms/frame", pipelinedFrames, pipelinedWallMs, pipelinedMs))
    print("frame GPU ms, untraced, GPU idle before: " + range(coldMs))
    print("frame GPU ms, untraced (one encoder):     " + range(normalMs))
    print("frame GPU ms, traced (one pass each):     " + range(tracedMs) + "; first start to last end " + range(spans))
    print(String(format: "sum of median dispatch times:  %.2f ms", dispatchSum))
    print(String(format: "sum of median gaps, traced:    %.2f ms between consecutive passes", gapSum))
    print(String(format: "dispatches under 20 us:        %d, %.2f ms", small.count, smallMs))
    print("near-empty dispatch, us each:  barrier before each " + range(floorSerial) + "; no barriers " + range(floorConcurrent) + "; one pass each " + range(floorSplit))
    print(String(format: "idle between dispatches, untraced, estimated: %.2f ms (%d dispatches x (barriered - unordered near-empty cost))", idleMs, first.count))
    print(String(format: "untraced frame minus dispatch sum: %.2f ms", median(normalMs) - dispatchSum))

    struct Bucket { var count = 0; var us = 0.0; var maxUs = 0.0; var bytes = 0 }
    func buckets(_ key: (TraceRow) -> String?) -> [(String, Bucket)] {
        var b: [String: Bucket] = [:]
        for r in rows {
            guard let k = key(r) else { continue }
            var x = b[k] ?? Bucket()
            x.count += 1; x.us += r.medianUs; x.maxUs = max(x.maxUs, r.medianUs); x.bytes += r.weightBytes ?? 0
            b[k] = x
        }
        return b.sorted { $0.value.us > $1.value.us }
    }
    func printBuckets(_ title: String, _ list: [(String, Bucket)], width: Int) {
        print("\n" + pad(title, width) + lpad("count", 6) + lpad("total ms", 10) + lpad("mean us", 9) + lpad("max us", 9) + lpad("MB/disp", 9) + lpad("GB/s", 6))
        for (k, b) in list {
            let mb = b.bytes > 0 ? String(format: "%.2f", Double(b.bytes) / Double(b.count) / 1e6) : "-"
            print(pad(k, width) + lpad("\(b.count)", 6) + lpad(String(format: "%.3f", b.us / 1000), 10) + lpad(String(format: "%.1f", b.us / Double(b.count)), 9)
                  + lpad(String(format: "%.1f", b.maxUs), 9) + lpad(mb, 9) + lpad(b.bytes > 0 ? gbps(b.bytes, b.us) : "-", 6))
        }
    }
    let finalPhase = "backbone.\(model.config.numLayers)"
    printBuckets("kernel", buckets(\.kernel), width: 28)
    let streaming = buckets { r in r.shape.map { "\(r.kernel) \($0)" } }
    printBuckets("weight-streaming kernel and shape", streaming, width: 40)
    let slow = streaming.filter { Double($0.1.bytes) / $0.1.us / 1000 < 250 }
    print(String(format: "below 250 GB/s: %d shapes, %d dispatches, %.2f ms, %.0f MB", slow.count, slow.reduce(0) { $0 + $1.1.count },
                 slow.reduce(0) { $0 + $1.1.us } / 1000, Double(slow.reduce(0) { $0 + $1.1.bytes }) / 1e6))
    printBuckets("phase", buckets { r in
        if r.phase == finalPhase { return "backbone final norm, lm head" }
        if r.phase.hasPrefix("backbone.") { return "backbone layers" }
        if r.phase.hasPrefix("tower.") { return "tower" }
        return r.phase
    }, width: 28)

    print("\ntop 30 dispatches (phase: tower.0-3 shared layers, tower.4-12 residual layer and head, \(finalPhase) final norm and lm head)")
    print(lpad("#", 5) + "  " + pad("kernel", 26) + pad("phase", 13) + pad("shape", 16) + lpad("us", 8) + lpad("min", 8) + lpad("max", 8) + lpad("GB/s", 6))
    for r in rows.sorted(by: { $0.medianUs > $1.medianUs }).prefix(30) {
        print(lpad("\(r.index)", 5) + "  " + pad(r.kernel, 26) + pad(r.phase, 13) + pad(r.shape ?? "-", 16) + lpad(String(format: "%.1f", r.medianUs), 8)
              + lpad(String(format: "%.1f", r.minUs), 8) + lpad(String(format: "%.1f", r.maxUs), 8) + lpad(r.weightBytes.map { gbps($0, r.medianUs) } ?? "-", 6))
    }

    if let out = args["out"] {
        let report = TraceReport(chip: model.ctx.device.name, schedule: schedule,
                                 method: "one compute pass per dispatch, stage-boundary timestamps; median over rounds",
                                 promptRows: prompt.count, rounds: rounds, heatDispatches: heat, pipelinedFrameMs: pipelinedMs, pipelinedWallMs: pipelinedWallMs, coldFrameMs: coldMs, normalFrameMs: normalMs, tracedCommandBufferMs: tracedMs, tracedSpanMs: spans,
                                 dispatchFloorSerialUs: floorSerial, dispatchFloorSplitUs: floorSplit, dispatchFloorConcurrentUs: floorConcurrent, idleEstimateMs: idleMs, dispatchSumMs: dispatchSum, gapSumMs: gapSum,
                                 smallDispatchMs: smallMs, smallDispatchCount: small.count, dispatches: rows)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(report).write(to: URL(fileURLWithPath: out))
        print("\nwrote \(out)")
    }
}
