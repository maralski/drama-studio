// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// One dispatch of a traced command buffer. Times are microseconds from the
/// start of the first dispatch of the command buffer.
package struct TracedDispatch: Codable, Sendable {
    package var kernel: String
    /// Where in the frame the dispatch sits, inferred from the kernel sequence.
    package var phase: String
    /// N x K for the GEMV and GEMM kernels, experts x N x K for the expert kernels.
    package var shape: String?
    /// Weight bytes the dispatch streams, for the GEMV, GEMM and expert kernels.
    package var weightBytes: Int?
    package var startUs: Double
    package var endUs: Double
    package var durationUs: Double { endUs - startUs }
}

/// Times every dispatch of the command buffers made while it is installed.
///
/// Apple GPUs sample counters only at stage boundaries, so each dispatch runs
/// in a compute pass of its own with start and end timestamps. The encoder the
/// engine sees is a proxy: state calls go to the open pass, and each dispatch
/// closes it, so the next kernel's state lands in a fresh pass.
package final class DispatchTracer {
    let ctx: MetalContext
    private let counterSet: MTLCounterSet
    private var encoders: [SplitPassEncoder] = []

    package static let samplesPerBuffer = 4096

    package init(ctx: MetalContext) throws {
        guard ctx.device.supportsCounterSampling(.atStageBoundary),
              let set = ctx.device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            throw EngineError.badModel("this GPU cannot sample timestamps at stage boundaries")
        }
        self.ctx = ctx
        counterSet = set
        _ = try makeSampleBuffer()
    }

    func makeSampleBuffer() throws -> MTLCounterSampleBuffer {
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = counterSet
        d.sampleCount = Self.samplesPerBuffer
        d.storageMode = .shared
        return try ctx.device.makeCounterSampleBuffer(descriptor: d)
    }

    func encoder(for cb: MTLCommandBuffer) -> MTLComputeCommandEncoder {
        let e = SplitPassEncoder(commandBuffer: cb, tracer: self)
        encoders.append(e)
        // A proxy that forwards to a real encoder: @objc protocol existentials
        // are a bare object reference and every call is an Objective-C message.
        return unsafeBitCast(e, to: MTLComputeCommandEncoder.self)
    }

    /// GPU timestamp units per nanosecond, from two CPU and GPU timestamp pairs.
    package func ticksPerNanosecond() -> Double {
        let a = ctx.device.sampleTimestamps()
        usleep(20_000)
        let b = ctx.device.sampleTimestamps()
        let cpu = Double(b.cpu &- a.cpu), gpu = Double(b.gpu &- a.gpu)
        return cpu > 0 && gpu > 0 ? gpu / cpu : 1
    }

    /// The dispatches of the completed command buffers, in encode order.
    package func collect() throws -> [TracedDispatch] {
        let perNs = ticksPerNanosecond()
        var raw: [(SplitPassEncoder.Pass, UInt64, UInt64)] = []
        for e in encoders {
            var resolved: [ObjectIdentifier: [UInt64]] = [:]
            for p in e.passes {
                let key = ObjectIdentifier(p.sampleBuffer)
                if resolved[key] == nil {
                    guard let data = try p.sampleBuffer.resolveCounterRange(0..<Self.samplesPerBuffer) else {
                        throw EngineError.badModel("counter samples did not resolve")
                    }
                    resolved[key] = data.withUnsafeBytes { Array($0.bindMemory(to: MTLCounterResultTimestamp.self).map(\.timestamp)) }
                }
                let s = resolved[key]!
                let start = s[2 * p.slot], end = s[2 * p.slot + 1]
                guard start != MTLCounterErrorValue, end != MTLCounterErrorValue, end >= start else {
                    throw EngineError.badModel("invalid timestamp for \(p.kernel)")
                }
                raw.append((p, start, end))
            }
        }
        guard let t0 = raw.map(\.1).min() else { return [] }
        let phases = Self.phases(raw.map(\.0.kernel))
        return raw.enumerated().map { i, r in
            TracedDispatch(kernel: r.0.kernel, phase: phases[i], shape: r.0.shape, weightBytes: r.0.weightBytes,
                           startUs: Double(r.1 - t0) / perNs / 1000, endUs: Double(r.2 - t0) / perNs / 1000)
        }
    }

    /// Labels a decode frame's dispatches: the samplers, tower layer n (each
    /// starts with tower_cat_norm at one row, tower_embed otherwise), and
    /// backbone layer n (each ends with moe_post_fp8 at one row, moe_combine otherwise).
    static func phases(_ kernels: [String]) -> [String] {
        var phase = "frame", tower = -1, layer = 0, inBackbone = false
        return kernels.map { k in
            if k.hasPrefix("sample_") { return "sampler" }
            if k == "tower_embed" || k == "tower_cat_norm" { tower += 1; phase = "tower.\(tower)" }
            if k == "embed_mix" { inBackbone = true; layer = 0; phase = "backbone.0" }
            let label = phase
            if inBackbone && (k == "moe_combine" || k == "moe_post_fp8") { layer += 1; phase = "backbone.\(layer)" }
            return label
        }
    }
}

/// The encoder `DispatchTracer` hands out. Calls it does not implement are
/// forwarded to the open pass; dispatches are recorded and close the pass.
final class SplitPassEncoder: NSObject {
    struct Pass {
        var kernel: String
        var shape: String?
        var weightBytes: Int?
        var sampleBuffer: MTLCounterSampleBuffer
        var slot: Int
    }

    private let commandBuffer: MTLCommandBuffer
    private unowned let tracer: DispatchTracer
    private var open: MTLComputeCommandEncoder?
    private var residency: [([MTLResource], MTLResourceUsage)] = []
    private var kernel = "?"
    private var params: [Int: [UInt32]] = [:]
    private var sampleBuffer: MTLCounterSampleBuffer?
    private var slot = 0
    private(set) var passes: [Pass] = []

    init(commandBuffer: MTLCommandBuffer, tracer: DispatchTracer) {
        self.commandBuffer = commandBuffer
        self.tracer = tracer
    }

    private func current() -> MTLComputeCommandEncoder {
        if let open { return open }
        if sampleBuffer == nil || 2 * (slot + 1) > DispatchTracer.samplesPerBuffer {
            sampleBuffer = try! tracer.makeSampleBuffer()
            slot = 0
        }
        let d = MTLComputePassDescriptor()
        d.dispatchType = .serial
        let a = d.sampleBufferAttachments[0]!
        a.sampleBuffer = sampleBuffer
        a.startOfEncoderSampleIndex = 2 * slot
        a.endOfEncoderSampleIndex = 2 * slot + 1
        let enc = commandBuffer.makeComputeCommandEncoder(descriptor: d)!
        for (r, u) in residency { enc.useResources(r, usage: u) }
        open = enc
        return enc
    }

    override func forwardingTarget(for aSelector: Selector!) -> Any? { current() }

    @objc(setComputePipelineState:)
    func setComputePipelineState(_ state: MTLComputePipelineState) {
        kernel = tracer.ctx.name(of: state) ?? "?"
        current().setComputePipelineState(state)
    }

    @objc(setBytes:length:atIndex:)
    func setBytes(_ bytes: UnsafeRawPointer, length: Int, index: Int) {
        params[index] = Array(UnsafeBufferPointer(start: bytes.assumingMemoryBound(to: UInt32.self), count: length / 4))
        current().setBytes(bytes, length: length, index: index)
    }

    @objc(useResources:count:usage:)
    func useResources(_ resources: UnsafePointer<MTLResource>, count: Int, usage: MTLResourceUsage) {
        let list = Array(UnsafeBufferPointer(start: resources, count: count))
        residency.append((list, usage))
        open?.useResources(list, usage: usage)
    }

    @objc(dispatchThreadgroups:threadsPerThreadgroup:)
    func dispatchThreadgroups(_ groups: MTLSize, threadsPerThreadgroup: MTLSize) {
        current().dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerThreadgroup)
        close(groups: groups)
    }

    @objc(dispatchThreads:threadsPerThreadgroup:)
    func dispatchThreads(_ threads: MTLSize, threadsPerThreadgroup: MTLSize) {
        current().dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        close(groups: threads)
    }

    @objc func endEncoding() {
        open?.endEncoding()
        open = nil
    }

    private func close(groups: MTLSize) {
        let (shape, bytes) = weightTraffic(groups: groups)
        passes.append(Pass(kernel: kernel, shape: shape, weightBytes: bytes, sampleBuffer: sampleBuffer!, slot: slot))
        open?.endEncoding()
        open = nil
        params = [:]
        slot += 1
    }

    /// Reads the shapes from each kernel's parameter block, laid out as
    /// Kernels.swift binds it: GEMV, GEMM and the fused tower decode kernels
    /// {N, K, ...} at index 3; the packed kernels {N, K, M, round, escaped}
    /// at 3, their bytes the packed sections plus one per escaped exponent;
    /// the expert kernels {hidden, intermediate} at 5 (gate/up) or 6 (down)
    /// with one threadgroup column per active expert.
    private func weightTraffic(groups: MTLSize) -> (String?, Int?) {
        if kernel.hasPrefix("gemv_packed") || kernel == "swiglu_packed_dec", let p = params[3], p.count >= 5 {
            let n = Int(p[0]), k = Int(p[1]), escaped = Int(p[4]), matrices = kernel == "swiglu_packed_dec" ? 2 : 1
            return (matrices == 2 ? "2x\(n)x\(k)" : "\(n)x\(k)", matrices * PackedBF16.sectionOffsets(tiles: n * k / PackedBF16.tile).escape + escaped)
        }
        if kernel == "swiglu_dec" || kernel == "gemv_residual", let p = params[3], p.count >= 2 {
            let n = Int(p[0]), k = Int(p[1]), matrices = kernel == "swiglu_dec" ? 2 : 1
            return (matrices == 2 ? "2x\(n)x\(k)" : "\(n)x\(k)", matrices * n * k * 2)
        }
        if kernel.hasPrefix("gemv_bf16") || kernel.hasPrefix("gemm_bf16"), let p = params[3], p.count >= 2 {
            let n = Int(p[0]), k = Int(p[1])
            return ("\(n)x\(k)", n * k * 2)
        }
        let gateUp = kernel.hasPrefix("moe_gate_up_fp8"), down = kernel.hasPrefix("moe_down_fp8")
        if gateUp || down, let p = params[gateUp ? 5 : 6], p.count >= 2 {
            let h = Int(p[0]), i = Int(p[1]), experts = groups.width
            let scales = (h / 128) * (i / 128) * 4
            return gateUp ? ("\(experts)x\(2 * i)x\(h)", experts * 2 * (i * h + scales)) : ("\(experts)x\(h)x\(i)", experts * (h * i + scales))
        }
        return (nil, nil)
    }
}
