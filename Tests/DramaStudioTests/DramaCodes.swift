// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine

extension ReferenceVoice {
    var dramaCodes: [[Int32]]? {
        guard case .drama3(let codes)? = encodings[.drama3] else { return nil }
        return codes
    }
}

extension Take {
    var dramaCodes: [[Int32]] {
        switch context {
        case .drama3(let codes): codes
        }
    }
}
