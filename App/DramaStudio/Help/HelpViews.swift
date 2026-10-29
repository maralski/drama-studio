// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct HelpButton: View {
    let topic: SettingHelp
    @State private var isShown = false

    init(_ topic: SettingHelp) {
        self.topic = topic
    }

    var body: some View {
        Button("About \(topic.title)", systemImage: "questionmark.circle") { isShown.toggle() }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("About \(topic.title)")
            .popover(isPresented: $isShown, arrowEdge: .trailing) {
                HelpEntryView(topic: topic, titleFont: .headline)
                    .padding(14)
                    .frame(width: 300)
            }
    }
}

/// A label with its help button, for controls that take a label view.
struct HelpLabel: View {
    let title: String
    let topic: SettingHelp

    init(_ title: String, help topic: SettingHelp) {
        self.title = title
        self.topic = topic
    }

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
            HelpButton(topic)
        }
    }
}

struct HelpEntryView: View {
    let topic: SettingHelp
    var titleFont: Font = .title2.bold()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(topic.title).font(titleFont)
            Text(topic.text).fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct HelpWindowView: View {
    static let id = "help"
    @State private var selection: SettingHelp? = .gettingStarted

    var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                ForEach(SettingHelp.Group.allCases, id: \.self) { group in
                    Section(group.rawValue) {
                        ForEach(SettingHelp.allCases.filter { $0.group == group }) { Text($0.title).tag($0) }
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 210)
        } detail: {
            ScrollView {
                if let selection {
                    HelpEntryView(topic: selection)
                        .font(.body)
                        .lineSpacing(3)
                        .frame(maxWidth: 520, alignment: .leading)
                        .padding(28)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .navigationTitle("Drama Studio Help")
    }
}
