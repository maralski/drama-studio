// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// SwiftPM's generated accessor does not search an app's Contents/Resources.
package enum EngineResources {
    package static let bundleName = "DramaStudio_DramaEngine.bundle"
    package static func bundle() -> Bundle {
        if let url = Bundle.main.resourceURL?.appendingPathComponent(bundleName), let bundle = Bundle(url: url) { return bundle }
        return Bundle.module
    }
}
