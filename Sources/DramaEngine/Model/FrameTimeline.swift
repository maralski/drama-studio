// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Darwin
import Foundation
import Metal

/// Per-frame timestamps of the decode loop, for `drama generate --trace-frames`.
/// Every time is host seconds on the clock Metal reports command buffer times
/// in (`mach_absolute_time`), so CPU and GPU events share one axis.
package final class FrameTimeline {
    package struct Row {
        package var session: Int
        package var position: Int
        /// The `step` call that returned this frame: entry and exit.
        package var stepEnter: Double
        package var stepExit: Double
        /// When this frame's command buffer was encoded and committed.
        package var encodeStart: Double
        package var commit: Double
        /// When the CPU began waiting for it and when the wait returned.
        package var waitStart: Double
        package var ready: Double
        package var kernelStart: Double
        package var kernelEnd: Double
        package var gpuStart: Double
        package var gpuEnd: Double
        /// With `stages`, GPU milliseconds of each encoder of the buffer.
        package var stages: [(name: String, ms: Double)]
    }

    package private(set) var rows: [Row] = []
    package private(set) var sessions = 0
    /// Set, each frame is encoded as one compute encoder per stage with GPU
    /// timestamps at every encoder boundary. The split adds encoder overhead,
    /// so the buffer times of such a run are not the production numbers.
    let stages: StageCounters?

    package init(device: MTLDevice? = nil, stages: Bool = false) throws {
        self.stages = stages ? try StageCounters(device: device ?? MTLCreateSystemDefaultDevice()!) : nil
    }

    private static let ticksPerSecond: Double = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return 1e9 * Double(info.denom) / Double(info.numer)
    }()

    package static func now() -> Double { Double(mach_absolute_time()) / ticksPerSecond }

    func beginSession() -> Int {
        sessions += 1
        return sessions - 1
    }

    func append(_ row: Row) { rows.append(row) }

    package func csv() -> String {
        guard let t0 = rows.first?.encodeStart else { return "" }
        var names: [String] = []
        for r in rows { for s in r.stages where !names.contains(s.name) { names.append(s.name) } }
        var out = "frame,session,position,step_enter,step_exit,encode_start,commit,wait_start,ready,kernel_start,kernel_end,gpu_start,gpu_end"
        out += names.map { ",stage:\($0)" }.joined() + (names.isEmpty ? "" : ",stage_encoders") + "\n"
        for (i, r) in rows.enumerated() {
            let times = [r.stepEnter, r.stepExit, r.encodeStart, r.commit, r.waitStart, r.ready, r.kernelStart, r.kernelEnd, r.gpuStart, r.gpuEnd]
            out += "\(i),\(r.session),\(r.position)," + times.map { String(format: "%.6f", ($0 - t0) * 1000) }.joined(separator: ",")
            for n in names { out += String(format: ",%.6f", r.stages.filter { $0.name == n }.reduce(0) { $0 + $1.ms }) }
            if !names.isEmpty { out += ",\(r.stages.count)" }
            out += "\n"
        }
        return out
    }
}

/// GPU timestamps at compute encoder boundaries.
final class StageCounters {
    let device: MTLDevice
    let counterSet: MTLCounterSet
    static let maxStages = 64
    private var calibration: (cpu: MTLTimestamp, gpu: MTLTimestamp)

    init(device: MTLDevice) throws {
        guard device.supportsCounterSampling(.atStageBoundary),
              let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            throw EngineError.badModel("this GPU has no timestamp counters at encoder boundaries")
        }
        self.device = device
        counterSet = set
        let (cpu, gpu) = device.sampleTimestamps()
        calibration = (cpu, gpu)
    }

    func makeBuffer() throws -> MTLCounterSampleBuffer {
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = counterSet
        d.storageMode = .shared
        d.sampleCount = 2 * Self.maxStages
        return try device.makeCounterSampleBuffer(descriptor: d)
    }

    /// Nanoseconds per GPU timestamp tick, from two calibration samples far
    /// apart; `sampleTimestamps` gives the CPU side in nanoseconds.
    private func nanosecondsPerTick() -> Double {
        let (cpu, gpu) = device.sampleTimestamps()
        guard gpu > calibration.gpu, cpu > calibration.cpu else { return 1 }
        return Double(cpu - calibration.cpu) / Double(gpu - calibration.gpu)
    }

    func resolve(_ buffer: MTLCounterSampleBuffer?, names: [String]) -> [(name: String, ms: Double)] {
        guard let buffer, !names.isEmpty, let data = try? buffer.resolveCounterRange(0 ..< 2 * names.count) else { return [] }
        let scale = nanosecondsPerTick() / 1e6
        return data.withUnsafeBytes { raw in
            let t = raw.bindMemory(to: MTLCounterResultTimestamp.self)
            return names.enumerated().map { i, n in
                let a = t[2 * i].timestamp, b = t[2 * i + 1].timestamp
                let ok = a != MTLCounterErrorValue && b != MTLCounterErrorValue && b >= a
                return (n, ok ? Double(b - a) * scale : .nan)
            }
        }
    }
}

/// The compute encoders of one frame command buffer. Without counters every
/// stage shares one encoder; with them each stage gets its own concurrent
/// encoder (the production kind, so `Kernels.parallel` groups still overlap),
/// timestamped at its start and end.
final class FrameEncoder {
    let commandBuffer: MTLCommandBuffer
    private(set) var encoder: MTLComputeCommandEncoder
    let samples: MTLCounterSampleBuffer?
    private(set) var names: [String] = []
    private let resident: [MTLBuffer]

    init(model: DramaModel, stages: StageCounters?, first: String) throws {
        resident = model.residentWeights
        guard let samples = try stages?.makeBuffer() else {
            (commandBuffer, encoder) = model.commandBuffer()
            self.samples = nil
            return
        }
        self.samples = samples
        commandBuffer = model.ctx.queue.makeCommandBuffer()!
        encoder = Self.open(commandBuffer, samples: samples, index: 0, resident: resident)
        names = [first]
    }

    private static func open(_ cb: MTLCommandBuffer, samples: MTLCounterSampleBuffer, index: Int, resident: [MTLBuffer]) -> MTLComputeCommandEncoder {
        let d = MTLComputePassDescriptor()
        d.dispatchType = .concurrent
        let a = d.sampleBufferAttachments[0]!
        a.sampleBuffer = samples
        a.startOfEncoderSampleIndex = 2 * index
        a.endOfEncoderSampleIndex = 2 * index + 1
        let enc = cb.makeComputeCommandEncoder(descriptor: d)!
        enc.useResources(resident, usage: .read)
        return enc
    }

    /// The encoder for the next stage, `name`.
    func stage(_ name: String) -> MTLComputeCommandEncoder {
        guard let samples, names.count < StageCounters.maxStages else { return encoder }
        encoder.endEncoding()
        encoder = Self.open(commandBuffer, samples: samples, index: names.count, resident: resident)
        names.append(name)
        return encoder
    }
}
