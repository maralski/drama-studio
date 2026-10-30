// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import DramaEngine

/// `DRAMA_STREAMING_BENCH=1 swift test -c release -Xswiftc -enable-testing --filter StreamingBenchTests`
/// generates the same lines with the codec idle during generation, with the
/// silence probe, and with low-latency decoding, interleaved, and prints one
/// `BENCH` row per run: frames per second while frames land, end-to-end
/// frames per second, first audio, and process CPU seconds.
final class StreamingBenchTests: XCTestCase {
    private enum Mode: String, CaseIterable {
        case decodeAfter = "decode-after"
        case sentence
        case lowLatency = "low-latency"
        case lowLatency32 = "low-latency-32"

        func config(seed: UInt64) -> GenerationConfig {
            var config = GenerationConfig()
            config.seed = seed
            switch self {
            case .decodeAfter: config.stopAfterSilenceSeconds = nil
            case .sentence: config.stopAfterSilenceSeconds = 2
            case .lowLatency:
                config.stopAfterSilenceSeconds = 2
                config.streaming = .defaultLowLatency
                config.maxRetries = 0
            case .lowLatency32:
                config.stopAfterSilenceSeconds = 2
                config.streaming = .lowLatency(stepFrames: 32)
                config.maxRetries = 0
            }
            return config
        }
    }

    private static func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }

    private static func cpuSeconds() -> Double {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        func seconds(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }
        return seconds(usage.ru_utime) + seconds(usage.ru_stime)
    }

    func testStreamingCostsNoGenerationSpeed() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DRAMA_STREAMING_BENCH"] == "1", "set DRAMA_STREAMING_BENCH=1")
        guard let location = ModelLocation.huggingFaceCache(), location.isComplete, DramaCodec.modelsExist() else { throw XCTSkip("model or codec missing") }
        let trace = ProcessInfo.processInfo.environment["DRAMA_STREAMING_BENCH_TRACE"] == "1"
        let runs = Int(ProcessInfo.processInfo.environment["DRAMA_STREAMING_BENCH_RUNS"] ?? "") ?? 5
        let text = ProcessInfo.processInfo.environment["DRAMA_STREAMING_BENCH_TEXT"]
            ?? "The storm had passed by morning, leaving the harbor strangely quiet. Old Tom walked the pier, counting the boats that had survived the night."
        let engine = DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory)
        try await engine.load()
        for try await _ in engine.synthesize(SpeechRequest(text: "Warm up.", config: Mode.sentence.config(seed: 1))) {}
        print("BENCH\tts_utc\trun\tmode\tframes\tframe_fps\tend_to_end_fps\tfirst_audio_s\twall_s\tcpu_s\tcpu_cores\tcodec_s\tretries")
        for run in 1...runs {
            let order = run % 2 == 1 ? Mode.allCases : Mode.allCases.reversed()
            for mode in order {
                let start = ContinuousClock.now
                let cpu0 = Self.cpuSeconds()
                var firstAudio: Double?
                var segment: (first: ContinuousClock.Instant, last: ContinuousClock.Instant, frames: Int)?
                var loopSeconds = 0.0
                var loopFrames = 0
                var retries = 0
                var partials = 0
                var result: SpeechResult?
                func closeSegment() {
                    if let s = segment, s.frames > 1 { loopSeconds += Self.seconds(s.last - s.first); loopFrames += s.frames - 1 }
                    segment = nil
                }
                for try await event in engine.synthesize(SpeechRequest(text: text, config: mode.config(seed: 7))) {
                    let now = ContinuousClock.now
                    if trace {
                        switch event {
                        case .frame(let i, _) where i % 16 != 0: break
                        case .partialAudio(_, let c, let f): print(String(format: "TRACE %@ %.2f partial %d %d..<%d", mode.rawValue, Self.seconds(now - start), c, f.lowerBound, f.upperBound))
                        case .audio(_, let c): print(String(format: "TRACE %@ %.2f audio %d", mode.rawValue, Self.seconds(now - start), c))
                        case .frame(let i, _): print(String(format: "TRACE %@ %.2f frame %d", mode.rawValue, Self.seconds(now - start), i))
                        case .chunkStarted(let c, _, _): print(String(format: "TRACE %@ %.2f chunk %d", mode.rawValue, Self.seconds(now - start), c))
                        case .prefill: print(String(format: "TRACE %@ %.2f prefilled", mode.rawValue, Self.seconds(now - start)))
                        default: break
                        }
                    }
                    switch event {
                    case .frame:
                        segment = segment.map { ($0.first, now, $0.frames + 1) } ?? (now, now, 1)
                    case .chunkStarted:
                        closeSegment()
                    case .retry:
                        closeSegment()
                        retries += 1
                    case .partialAudio:
                        partials += 1
                        if firstAudio == nil { firstAudio = Self.seconds(now - start) }
                    case .audio:
                        if firstAudio == nil { firstAudio = Self.seconds(now - start) }
                    case .finished(let r):
                        result = r
                    default:
                        break
                    }
                }
                closeSegment()
                let wall = Self.seconds(ContinuousClock.now - start)
                let cpu = Self.cpuSeconds() - cpu0
                let r = try XCTUnwrap(result)
                if mode == .lowLatency || mode == .lowLatency32 {
                    XCTAssertGreaterThan(partials, 0, "\(mode.rawValue) must deliver audio before the chunk completes")
                }
                XCTAssertEqual(r.samples.count, r.take.frameCount * ModelConstants.samplesPerFrame, "\(mode.rawValue): every frame decoded")
                let timestamp = ISO8601DateFormatter().string(from: Date())
                print(String(format: "BENCH\t%@\t%d\t%@\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%.1f\t%.2f\t%.2f\t%d", timestamp, run, mode.rawValue, r.take.frameCount,
                             Double(loopFrames) / max(loopSeconds, 1e-9), Double(r.take.frameCount) / wall, firstAudio ?? -1, wall, cpu, cpu / wall,
                             r.timings.codecSeconds, retries))
            }
        }
    }
}
