// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import SwiftUI

extension View {
    func dialogActions() -> some View {
        padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 20)
            .fixedSize(horizontal: false, vertical: true)
            .layoutPriority(1)
            .background(.bar)
            .overlay(alignment: .top) { Divider() }
    }
}
