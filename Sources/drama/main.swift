import DramaEngine
import Foundation
import Metal

let ctx = try MetalContext()
print("Metal device:", ctx.device.name)
let pipe = try ctx.pipeline("probe_addr")

// Table of four bf16 values 1,2,3,4 starting 8 bytes into a buffer (8-aligned, not 16-aligned).
let bf16: [UInt16] = [0, 0, 0, 0, 0x3F80, 0x4000, 0x4040, 0x4080]
let weights = ctx.device.makeBuffer(bytes: bf16, length: bf16.count * 2, options: .storageModeShared)!
var addr: UInt64 = weights.gpuAddress + 8
let table = ctx.device.makeBuffer(bytes: &addr, length: 8, options: .storageModeShared)!
let out = ctx.device.makeBuffer(length: 8 * 4, options: .storageModeShared)!
let cb = ctx.queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipe)
enc.setBuffer(table, offset: 0, index: 0)
enc.setBuffer(weights, offset: 8, index: 1)
enc.setBuffer(out, offset: 0, index: 2)
enc.useResource(weights, usage: .read)
enc.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
enc.endEncoding()
cb.commit(); cb.waitUntilCompleted()
let res = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: 8))
print("via address table:", res[0..<4], "via 8-byte offset:", res[4..<8])
