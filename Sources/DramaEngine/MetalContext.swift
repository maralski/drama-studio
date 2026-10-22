import CryptoKit
import Foundation
import Metal

/// One Metal device, queue, and the kernel library shared by the engine.
///
/// The kernels ship as `.metal` sources in the bundle and are compiled once at
/// startup; SwiftPM does not build a metallib from the command line.
package final class MetalContext {
    package let device: MTLDevice
    package let queue: MTLCommandQueue
    package let library: MTLLibrary
    /// SHA-256 hex of the exact source string the library was compiled from.
    package let kernelDigest: String
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    package init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError.noMetalDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw EngineError.noMetalDevice }
        self.queue = queue
        let source = try Self.kernelSources()
        kernelDigest = Self.digest(source)
        self.library = try Self.compileKernels(source: source, device: device)
    }

    /// The resource bundle: inside a packaged .app it sits in Contents/Resources,
    /// where SwiftPM's generated accessor does not look.
    static let resourceBundleName = "DramaStudio_DramaEngine.bundle"

    static func resourceBundle() -> Bundle {
        if let url = Bundle.main.resourceURL?.appendingPathComponent(resourceBundleName), let b = Bundle(url: url) { return b }
        return Bundle.module
    }

    static func digest(_ source: String) -> String {
        SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func kernelSources() throws -> String {
        guard let dir = resourceBundle().url(forResource: "Kernels", withExtension: nil) else {
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

    static func compileKernels(source: String, device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version3_1
        return try device.makeLibrary(source: source, options: options)
    }

    /// Builds every kernel's pipeline state up front, in parallel, so the first
    /// request does not pay for them one dispatch at a time.
    package func warmPipelines() {
        let names = library.functionNames
        DispatchQueue.concurrentPerform(iterations: names.count) { i in
            guard let fn = library.makeFunction(name: names[i]), let p = try? device.makeComputePipelineState(function: fn) else { return }
            lock.lock(); pipelines[names[i]] = p; lock.unlock()
        }
    }

    package func name(of pipeline: MTLComputePipelineState) -> String? {
        lock.lock(); defer { lock.unlock() }
        return pipelines.first { $0.value === pipeline }?.key
    }

    package func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw EngineError.missingKernel(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }
}

/// Why the engine could not load or run.
public enum EngineError: Error, CustomStringConvertible {
    /// The Mac has no Metal device the engine can use.
    case noMetalDevice
    /// A kernel, or the resource bundle that holds the kernel sources, is missing.
    case missingKernel(String)
    /// The snapshot's files do not match what the engine expects.
    case badModel(String)
    /// A cached kernel schedule failed validation.
    case badSchedule(String)

    /// A message for a person.
    public var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        case .missingKernel(let n): return "kernel \(n) missing from the Metal library"
        case .badModel(let m): return "bad model: \(m)"
        case .badSchedule(let m): return "bad kernel schedule: \(m)"
        }
    }
}
