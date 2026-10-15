import Foundation
import Metal

/// One Metal device, queue, and the kernel library shared by the engine.
///
/// The kernels ship as `.metal` sources in the bundle and are compiled once at
/// startup; SwiftPM does not build a metallib from the command line.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError.noMetalDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw EngineError.noMetalDevice }
        self.queue = queue
        self.library = try Self.compileKernels(device: device)
    }

    static func kernelSources() throws -> String {
        guard let dir = Bundle.module.url(forResource: "Kernels", withExtension: nil) else {
            throw EngineError.missingKernel("Kernels directory")
        }
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        // Common.metal carries the shared declarations and must come first.
        let ordered = files.filter { $0.lastPathComponent == "Common.metal" } + files.filter { $0.lastPathComponent != "Common.metal" }
        let sources = try ordered.map { try String(contentsOf: $0, encoding: .utf8) }
        return sources[0] + "\n" + fp8Table() + "\n" + sources.dropFirst().joined(separator: "\n")
    }

    /// FP8 E4M3FN byte to float, as a constant table the kernels index.
    static func fp8Table() -> String {
        let values = (0..<256).map { b -> Float in
            let s = b >> 7, e = (b >> 3) & 0xf, m = b & 7
            var v: Float
            if e == 0 { v = Float(m) * Float(pow(2.0, -9.0)) }
            else if e == 15 && m == 7 { v = .nan }
            else { v = (1 + Float(m) / 8) * Float(pow(2.0, Double(e) - 7)) }
            return s == 1 ? -v : v
        }
        let body = values.map { $0.isNaN ? "NAN" : String(format: "%.9ef", $0) }.joined(separator: ", ")
        return "constant float FP8_LUT[256] = {" + body + "};"
    }

    static func compileKernels(device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version3_1
        return try device.makeLibrary(source: try kernelSources(), options: options)
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw EngineError.missingKernel(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }
}

public enum EngineError: Error, CustomStringConvertible {
    case noMetalDevice
    case missingKernel(String)
    case badModel(String)

    public var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        case .missingKernel(let n): return "kernel \(n) missing from the Metal library"
        case .badModel(let m): return "bad model: \(m)"
        }
    }
}
