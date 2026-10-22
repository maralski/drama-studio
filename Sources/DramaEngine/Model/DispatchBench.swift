import Foundation
import Metal

/// Fixed cost of a dispatch and of the gap between dependent dispatches on
/// this GPU, from command buffer timestamps. No model needed.
package enum DispatchBench {
    static let taxSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void tax_one(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) x[0] += 1u; }
    kernel void tax_touch(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.999f + 1.0f; }
    """

    enum Mode: String, CaseIterable {
        case serial
        case concurrent
        case concurrentFenced = "concurrent+barrier"
    }

    static func encoder(_ cb: MTLCommandBuffer, _ mode: Mode) -> MTLComputeCommandEncoder {
        mode == .serial ? cb.makeComputeCommandEncoder()! : cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
    }

    static func gpuSeconds(_ ctx: MetalContext, _ mode: Mode, _ body: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = encoder(cb, mode)
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw EngineError.badModel("\(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    static func summary(_ xs: [Double]) -> (min: Double, median: Double, max: Double) {
        let s = xs.sorted()
        return (s.first!, s[s.count / 2], s.last!)
    }

    static func line(_ label: String, _ perDispatch: [Double]) -> String {
        let s = summary(perDispatch.map { $0 * 1e6 })
        return label.padding(toLength: 52, withPad: " ", startingAt: 0) + String(format: "median %6.2f us  [%6.2f .. %6.2f]", s.median, s.min, s.max)
    }

    /// GPU microseconds per dispatch for chains of trivial kernels, and for
    /// chains of real decode GEMVs over distinct weights, in each mode.
    package static func tax(counts: [Int], repeats: Int) throws {
        let ctx = try MetalContext()
        let d = ctx.device
        let lib = try d.makeLibrary(source: taxSource, options: nil)
        let one = try d.makeComputePipelineState(function: lib.makeFunction(name: "tax_one")!)
        let touch = try d.makeComputePipelineState(function: lib.makeFunction(name: "tax_touch")!)
        let buf = d.makeBuffer(length: 2048 * 4, options: .storageModeShared)!
        memset(buf.contents(), 0, 2048 * 4)
        print("device \(d.name); per-dispatch GPU time = command buffer GPU time / N, \(repeats) repeats after a warm-up")
        for _ in 0..<20 {
            _ = try gpuSeconds(ctx, .serial) { enc in
                enc.setComputePipelineState(touch); enc.setBuffer(buf, offset: 0, index: 0)
                for _ in 0..<2000 { enc.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
            }
        }
        for (name, pipe, threads, group) in [("one threadgroup (32 threads)", one, 32, 32), ("2048 floats (8 x 256 threads)", touch, 2048, 256)] {
            for n in counts {
                for mode in Mode.allCases {
                    _ = try gpuSeconds(ctx, mode) { enc in
                        enc.setComputePipelineState(pipe); enc.setBuffer(buf, offset: 0, index: 0)
                        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
                    }
                    var per: [Double] = []
                    for _ in 0..<repeats {
                        let t = try gpuSeconds(ctx, mode) { enc in
                            enc.setComputePipelineState(pipe)
                            enc.setBuffer(buf, offset: 0, index: 0)
                            for _ in 0..<n {
                                if mode == .concurrentFenced { enc.memoryBarrier(scope: .buffers) }
                                enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
                            }
                        }
                        per.append(t / Double(n))
                    }
                    print(line("\(name) N=\(n) \(mode.rawValue)", per))
                }
            }
        }
        try gemvChains(ctx, repeats: repeats)
    }

    /// Chains of decode GEMVs over 48 distinct matrices (more than the system
    /// cache holds), serial against concurrent without barriers: the
    /// difference per dispatch is the drain-and-refill gap a dependency costs.
    static func gemvChains(_ ctx: MetalContext, repeats: Int) throws {
        let k = Kernels(ctx: ctx, precision: .exact)
        for (n, kDim) in [(2048, 2048), (512, 2048), (8192, 2048), (2048, 4096)] {
            let mats = (0..<48).map { _ in KernelBench.bf16Weights(ctx.device, rows: n, cols: kDim) }
            let x = Buf(ctx.device, floats: kDim, label: "x")
            for i in 0..<kDim { x.floats[i] = Float(i % 97) / 97 - 0.5 }
            let ys = (0..<2).map { Buf(ctx.device, floats: n, label: "y\($0)") }
            let chain = 96
            var results: [Mode: [Double]] = [:]
            for _ in 0..<repeats {
                for mode in Mode.allCases {
                    let t = try gpuSeconds(ctx, mode == .concurrentFenced ? .concurrent : mode) { enc in
                        if mode == .concurrent { try k.parallel(enc) { for i in 0..<chain { try k.gemv(enc, mats[i % mats.count], x: x, y: ys[i % 2], m: 1) } } }
                        else { for i in 0..<chain { try k.gemv(enc, mats[i % mats.count], x: x, y: ys[i % 2], m: 1) } }
                    }
                    results[mode, default: []].append(t / Double(chain))
                }
            }
            let bytes = Double(n * kDim * 2)
            for mode in Mode.allCases {
                let r = results[mode]!
                print(line("gemv \(n)x\(kDim) \(mode.rawValue)", r) + String(format: "  %.0f GB/s at median", bytes / summary(r).median / 1e9))
            }
        }
    }

    /// GPU microseconds per instance of each replica block, separate against
    /// fused, on a serial encoder and on a concurrent one with the engine's
    /// barriers, over `copies` instances with distinct buffers per command buffer.
    package static func blocks(copies: Int, repeats: Int, schedule: KernelSchedule) throws {
        let ctx = try MetalContext()
        let replicas = (0..<copies).map { _ in FusionReplica(ctx: ctx, schedule: schedule).allBlocks() }
        print("decode GEMV \(schedule.gemvDecode.label); per instance: command buffer GPU time / \(copies) instances with distinct buffers; \(repeats) repeats, variants interleaved")
        let variants: [(String, Mode, Bool)] = [("serial separate", .serial, false), ("serial fused", .serial, true),
                                                ("concurrent separate", .concurrent, false), ("concurrent fused", .concurrent, true)]
        var frameDelta: [String: Double] = [:]
        for b in replicas[0].indices {
            let name = replicas[0][b].name, perFrame = replicas[0][b].perFrame
            let applicable = variants.filter { !$0.2 || replicas[0][b].fused != nil }
            var times: [String: [Double]] = [:]
            for r in 0...repeats {
                for (label, mode, fused) in applicable {
                    let t = try gpuSeconds(ctx, mode) { enc in
                        for c in 0..<copies { try replicas[c][b].encode(enc, fused: fused) }
                    }
                    if r > 0 { times[label, default: []].append(t / Double(copies)) }
                }
            }
            print("\(name), \(perFrame) per frame")
            let base = summary(times["serial separate"]!).median
            for (label, _, _) in applicable {
                let m = summary(times[label]!).median
                print("  " + line(label, times[label]!) + String(format: "  frame delta %+.3f ms", (m - base) * Double(perFrame) * 1000))
                frameDelta[label, default: 0] += (m - base) * Double(perFrame)
            }
        }
        for (label, _, _) in variants {
            print(String(format: "all blocks, %@: %+.3f ms per frame against serial separate", label, frameDelta[label, default: 0] * 1000))
        }
    }
}
