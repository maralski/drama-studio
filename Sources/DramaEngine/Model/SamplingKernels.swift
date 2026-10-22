import Foundation
import Metal

/// The settings of one `Sampler.sample` call.
struct SamplingParams: Equatable {
    var temperature: Double
    var topK: Int
    var topP: Double
}

/// What the GPU sampler kernels take; mirrors `SamplerParams` in Sampling.metal.
/// Doubles travel as double-float pairs so the kernel can match the CPU's arithmetic.
struct SamplerKernelParams {
    var n: UInt32
    var topK: UInt32
    var tHi: Float, tLo: Float
    var topPHi: Float, topPLo: Float
    var rngIn: UInt32, rngOut: UInt32
    var frameRow: UInt32
    var position: UInt32
    var head: UInt32
    var eos: Int32
    var semanticStart: Int32
    var codebookSize: UInt32
    var retryTHi: Float, retryTLo: Float
    var retryTopK: UInt32
    var retryTopPHi: Float, retryTopPLo: Float
    var windowFrames: UInt32

    static func split(_ x: Double) -> (Float, Float) {
        let hi = Float(x)
        return (hi, Float(x - Double(hi)))
    }

    /// `rngIn` is the RNG slot the kernel starts from, `rngOut` the slot it leaves its
    /// state in; `position` is the row the frame is sampled for, so the frame lands in
    /// row `position + 1`.
    init(_ p: SamplingParams, n: Int, rngIn: Int, rngOut: Int, position: Int, head: Int,
         eos: Int32, semanticStart: Int32, codebookSize: Int) {
        let t = Self.split(max(p.temperature, 1e-5)), topP = Self.split(p.topP)
        let rt = Self.split(MainTokenSampler.retryTemperature), rtopP = Self.split(MainTokenSampler.retryTopP)
        self.n = UInt32(n)
        topK = UInt32(max(p.topK, 0))
        tHi = t.0; tLo = t.1
        topPHi = topP.0; topPLo = topP.1
        self.rngIn = UInt32(rngIn); self.rngOut = UInt32(rngOut)
        frameRow = UInt32(position + 1)
        self.position = UInt32(position)
        self.head = UInt32(head)
        self.eos = eos
        self.semanticStart = semanticStart
        self.codebookSize = UInt32(codebookSize)
        retryTHi = rt.0; retryTLo = rt.1
        retryTopK = UInt32(MainTokenSampler.retryTopK)
        retryTopPHi = rtopP.0; retryTopPLo = rtopP.1
        windowFrames = UInt32(MainTokenSampler.windowFrames)
    }
}

extension Kernels {
    static let samplerThreads = 256

    /// Samples code `head + 1` from `logits` (4096 floats at `logitsOffset`) and writes
    /// it to `codeOut[0]` and to the frame row's slot.
    func sampleTower(_ enc: MTLComputeCommandEncoder, logits: Buf, logitsOffset: Int, rng: Buf, frames: Buf, codeOut: Buf, status: Buf,
                     params: SamplerKernelParams) throws {
        enc.setBuffer(logits.buffer, offset: logits.byteOffset + logitsOffset * 4, index: 0)
        enc.setBuffer(rng.buffer, offset: rng.byteOffset, index: 1)
        enc.setBuffer(frames.buffer, offset: frames.byteOffset, index: 2)
        enc.setBuffer(codeOut.buffer, offset: codeOut.byteOffset, index: 3)
        enc.setBuffer(status.buffer, offset: status.byteOffset, index: 4)
        set(enc, params, index: 5)
        try dispatchGroups(enc, "sample_tower", groups: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: Self.samplerThreads, height: 1, depth: 1))
    }

    /// Samples the main token from the 4097 candidate logits with the repetition-aware
    /// retry, and writes the frame row's main token and code 0, the tower's code 0 and
    /// semantic flag.
    func sampleMain(_ enc: MTLComputeCommandEncoder, logits: Buf, rng: Buf, frames: Buf, code0: Buf, semantic: Buf, status: Buf,
                    params: SamplerKernelParams) throws {
        enc.setBuffer(logits.buffer, offset: logits.byteOffset, index: 0)
        enc.setBuffer(rng.buffer, offset: rng.byteOffset, index: 1)
        enc.setBuffer(frames.buffer, offset: frames.byteOffset, index: 2)
        enc.setBuffer(code0.buffer, offset: code0.byteOffset, index: 3)
        enc.setBuffer(status.buffer, offset: status.byteOffset, index: 4)
        set(enc, params, index: 5)
        enc.setBuffer(semantic.buffer, offset: semantic.byteOffset, index: 6)
        try dispatchGroups(enc, "sample_main", groups: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: Self.samplerThreads, height: 1, depth: 1))
    }
}

/// The two RNG slots and the per-slot status word the sampler kernels chain
/// through. Frame job k starts from slot k % 2 and leaves its state in slot
/// (k + 1) % 2, so the slot a finished job wrote is only read by the job in
/// flight and the CPU can copy it without a race.
final class SamplerState {
    let rng: Buf
    let status: Buf

    init(device: MTLDevice) {
        rng = Buf(device, floats: 2 * 8, label: "sampler.rng")
        status = Buf(device, floats: 2, label: "sampler.status")
    }

    func state(slot: Int) -> SIMD4<UInt64> {
        let p = (rng.buffer.contents() + slot * 32).assumingMemoryBound(to: UInt64.self)
        return SIMD4(p[0], p[1], p[2], p[3])
    }

    func setState(_ s: SIMD4<UInt64>, slot: Int) {
        let p = (rng.buffer.contents() + slot * 32).assumingMemoryBound(to: UInt64.self)
        for i in 0..<4 { p[i] = s[i] }
    }

    /// Nonzero when a sampler of the job that wrote `slot` saw a non-finite logit.
    func statusWord(slot: Int) -> UInt32 { status.buffer.contents().assumingMemoryBound(to: UInt32.self)[slot] }
}
