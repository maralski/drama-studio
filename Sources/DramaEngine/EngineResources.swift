// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
internal import DramaTTS

/// SwiftPM's generated accessor does not search an app's Contents/Resources.
package enum EngineResources {
    package static let bundleName = "DramaStudio_DramaEngine.bundle"
    package static func bundle() -> Bundle {
        if let url = Bundle.main.resourceURL?.appendingPathComponent(bundleName), let bundle = Bundle(url: url) { return bundle }
        return Bundle.module
    }

    /// Compiles the Metal kernels DramaTTS ships and returns the GPU's name,
    /// to prove the app holds DramaTTS's resource bundle where the engine finds it.
    package static func compileKernels() throws -> String {
        try Drama3Errors.translating { try DramaResources.compileKernels() }
    }
}
