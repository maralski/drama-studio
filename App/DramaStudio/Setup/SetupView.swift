// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI

/// The first-run window: one card per ``SetupStage``.
struct SetupView: View {
    let setup: SetupCoordinator
    let model: SessionModel
    @State private var contentHeight: CGFloat = 360
    /// The engine picked on the engine step; nil for the recommended one.
    @State private var pickedEngine: SpeechEngineID?

    /// The engine the window is about: the one being set up, else the one picked so far.
    private var engine: SpeechEngineID { setup.facts.engine ?? pickedEngine ?? setup.facts.offer.recommended }

    private var subtitle: String {
        guard let engine = setup.facts.engine else { return "Expressive speech, generated on this Mac." }
        return "Expressive speech from \(engine.install.publisher)'s \(engine.capabilities.name), running on this Mac."
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label { Text("Drama Studio") } icon: { Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 32, height: 32) }
                            .font(.largeTitle.weight(.semibold))
                        Text(subtitle)
                            .foregroundStyle(.secondary)
                    }
                    StepStrip(stage: setup.stage, steps: setup.steps)
                    Divider()
                    card
                }
                .frame(maxWidth: 660, alignment: .leading)
                .padding(.horizontal, 40)
                .padding(.vertical, 36)
                .frame(maxWidth: .infinity)
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { contentHeight = $0 }
            }
            .frame(height: min(contentHeight, 520))
            .scrollDisabled(contentHeight <= 520)
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle("Set Up Drama Studio")
        .frame(width: 740)
    }

    @ViewBuilder
    private var footer: some View {
        switch setup.stage {
        case .ready, .tuning, .codec(.unpacking):
            EmptyView()
        default:
            HStack {
                switch setup.stage {
                case .engine:
                    Button("Quit") { NSApp.terminate(nil) }
                    Spacer()
                    Button("Continue") { setup.chooseEngine(engine) }
                        .keyboardShortcut(.defaultAction)
                        .help("Set up the selected engine")
                case .requirements:
                    Button("Quit") { NSApp.terminate(nil) }
                    anotherEngine
                    Spacer()
                    Button("Continue Anyway") { setup.acknowledgeRequirements() }
                        .keyboardShortcut(.defaultAction)
                case .terms:
                    Button("Quit") { NSApp.terminate(nil) }
                    anotherEngine
                    Spacer()
                    Button("I Agree") { setup.acceptTerms() }
                        .keyboardShortcut(.defaultAction)
                case .usage:
                    Button("Don't Share") { setup.chooseUsage(enabled: false) }
                        .keyboardShortcut(.defaultAction)
                    Spacer()
                    Button("Share Usage Information") { setup.chooseUsage(enabled: true) }
                case .weights(let transfer), .codec(.downloading(let transfer)):
                    anotherEngine
                    Spacer()
                    if transfer.paused {
                        Button("Resume") { setup.resume() }.keyboardShortcut(.defaultAction)
                    } else {
                        Button("Pause") { setup.pause() }
                    }
                case .codec(.needsConversion(let command)):
                    Button("Copy Command") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(command, forType: .string)
                    }
                    anotherEngine
                    Spacer()
                    Button("Check Again") { setup.resume() }.keyboardShortcut(.defaultAction)
                case .firstLine:
                    Button("Open the Studio") { setup.finish() }
                    Spacer()
                    Button("Generate", systemImage: "play.fill") {
                        setup.finish()
                        model.generate()
                    }
                    .keyboardShortcut(.defaultAction)
                case .failed:
                    anotherEngine
                    Spacer()
                    Button("Try Again") { setup.resume() }.keyboardShortcut(.defaultAction)
                case .ready, .tuning, .codec(.unpacking):
                    EmptyView()
                }
            }
            .frame(maxWidth: 660)
            .frame(maxWidth: .infinity)
            .dialogActions()
        }
    }

    /// Back to the engine step. A download under way stops and keeps its files.
    private var anotherEngine: some View {
        Button("Choose Another Engine") {
            pickedEngine = nil
            setup.chooseAnotherEngine()
        }
        .help("Go back and choose another speech engine")
    }

    @ViewBuilder
    private var card: some View {
        switch setup.stage {
        case .engine(let offer):
            EngineCard(offer: offer, selection: Binding(get: { engine }, set: { pickedEngine = $0 }))
        case .requirements(let gaps): RequirementsCard(engine: engine, gaps: gaps)
        case .terms: TermsCard()
        case .usage: UsageCard()
        case .weights(let transfer):
            let install = engine.install
            TransferCard(title: "Downloading \(engine.capabilities.name)",
                         detail: "\(ModelSetup.gigabytes(install.downloadBytes)) of model files from Hugging Face (\(install.repo) at revision \(install.plan.revision.prefix(7))), into \(AppHome.models(for: engine).path). Each file is checked against its SHA-256 before it is kept. \(install.publisher) publishes them under the \(install.license.name).",
                         transfer: transfer)
        case .codec(.downloading(let transfer)):
            TransferCard(title: "Downloading the voice codec",
                         detail: "The three Core ML models that turn Drama 3's codes into audio, from this release on GitHub.",
                         transfer: transfer)
        case .codec(.unpacking):
            WorkingCard(title: "Unpacking the voice codec", detail: "Checking the three Core ML models and moving them into place.", line: nil)
        case .codec(.needsConversion(let command)): ConversionCard(command: command)
        case .tuning(let detail):
            WorkingCard(title: "Getting \(engine.capabilities.name) ready for this Mac",
                        detail: "Preparing speech generation for this Mac usually takes 30 seconds to a minute. Later launches skip this preparation.",
                        line: detail)
        case .firstLine: FirstLineCard(model: model)
        case .ready: EmptyView()
        case .failed(let step, let reason): FailedCard(title: step.title(for: engine), step: step, reason: reason)
        }
    }
}

private struct StepStrip: View {
    let stage: SetupStage
    /// The steps of the engine being set up.
    let steps: [SetupStep]

    /// The requirements step shows only while it is the one being asked.
    private var shown: [SetupStep] { steps.filter { $0 != .requirements || stage.step == .requirements } }

    var body: some View {
        HStack(spacing: 14) {
            ForEach(shown, id: \.self) { step in
                let state = state(of: step)
                Label(step.label, systemImage: state == .done ? "checkmark.circle.fill" : state == .current ? "circle.inset.filled" : "circle")
                    .foregroundStyle(state == .upcoming ? .secondary : state == .done ? Color.green : Color.accentColor)
                    .fontWeight(state == .current ? .semibold : .regular)
            }
        }
        .font(.callout)
        .labelStyle(.titleAndIcon)
    }

    private enum StepState { case done, current, upcoming }

    private func state(of step: SetupStep) -> StepState {
        guard let current = stage.step else { return .done }
        let all = SetupStep.allCases
        guard let mine = all.firstIndex(of: step), let now = all.firstIndex(of: current) else { return .upcoming }
        return mine < now ? .done : mine == now ? .current : .upcoming
    }
}

private struct Bullets: View {
    let items: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("\u{2022}")
                    Text(item).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

private struct CardTitle: View {
    let text: String
    var body: some View { Text(text).font(.title2.weight(.semibold)) }
}

/// The engine step: one row per engine, the recommended one preselected,
/// with what this Mac lacks for an engine under its row.
private struct EngineCard: View {
    let offer: EngineOffer
    @Binding var selection: SpeechEngineID

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardTitle(text: "Choose a speech engine")
            Text("Drama Studio speaks with one of these engines. It downloads the one you choose now, and you can add the other later in Settings.")
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 8) {
                ForEach(SpeechEngineID.allCases, id: \.self) { engine in
                    EngineRow(engine: engine, recommended: engine == offer.recommended, downloaded: offer.downloaded.contains(engine),
                              gaps: offer.gaps[engine] ?? [], selected: engine == selection) { selection = engine }
                }
            }
        }
    }
}

private struct EngineRow: View {
    let engine: SpeechEngineID
    let recommended: Bool
    let downloaded: Bool
    let gaps: [String]
    let selected: Bool
    let select: () -> Void

    private var install: EngineInstall { engine.install }

    private var facts: String {
        let size = downloaded ? "already on this Mac" : "\(ModelSetup.gigabytes(install.downloadBytes)) download"
        let terms = install.license.requiresAgreement ? ", which asks you to agree to its terms" : ""
        return "By \(install.publisher), \(size). \(install.license.name)\(terms)."
    }

    var body: some View {
        Button(action: select) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(engine.capabilities.name).font(.headline)
                        if recommended {
                            Text("Recommended for this Mac")
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.18), in: Capsule())
                        }
                    }
                    Text(facts).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if let limits = engine.limits {
                        Text(limits).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(gaps, id: \.self) { gap in
                        Label(gap, systemImage: "exclamationmark.triangle.fill")
                            .font(.callout)
                            .symbolRenderingMode(.multicolor)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(selected ? 0.6 : 0.25), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 1.5))
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .help("Choose \(engine.capabilities.name)")
        .accessibilityLabel(engine.capabilities.name)
        .accessibilityValue(selected ? "Selected" : "Not selected")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

private struct RequirementsCard: View {
    let engine: SpeechEngineID
    let gaps: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label { CardTitle(text: "This Mac is below what \(engine.capabilities.name) was tested on") } icon: {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
            Bullets(items: gaps)
            Text("Continuing downloads \(engine.capabilities.name), \(ModelSetup.gigabytes(engine.install.downloadBytes)), unless it is already on this Mac. You can continue at your own risk, or choose another engine.")
                .fixedSize(horizontal: false, vertical: true)
                .foregroundStyle(.secondary)
        }
    }
}

private struct TermsCard: View {

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardTitle(text: "Before you begin")
            Text("Drama Studio runs Fish Audio's Drama 3 speech model. Fish Audio licenses the model under the Fish Audio Community License, and Drama Studio may only make it available to you on those terms. By agreeing, you accept that license and agree not to use the model or the audio it makes to:")
                .fixedSize(horizontal: false, vertical: true)
            Bullets(items: TermsAgreement.restrictions)
            Text("You also accept its limits:")
            Bullets(items: TermsAgreement.limits)
            Link("Read the full Fish Audio Community License", destination: TermsAgreement.licenseURL)
            Text("Drama 3 does not download and does not speak until you agree. Next, Drama Studio downloads any of its \(ModelSetup.gigabytes(ModelDownloadPlan.drama3.totalBytes)) of model files that this Mac lacks from Hugging Face.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }
}

private struct TransferCard: View {
    let title: String
    let detail: String
    let transfer: Transfer

    private var progress: DownloadProgress { transfer.progress }

    private var summary: String {
        guard progress.totalBytes > 0 else { return transfer.paused ? "Paused" : "Connecting\u{2026}" }
        let amount = "\(ModelSetup.gigabytes(progress.doneBytes)) of \(ModelSetup.gigabytes(progress.totalBytes))"
        if transfer.paused { return "\(amount), paused" }
        guard let left = transfer.secondsLeft else { return "\(amount), estimating time left" }
        return "\(amount), \(Self.timeLeft(left))"
    }

    static func timeLeft(_ seconds: Double) -> String {
        if seconds < 60 { return "less than a minute left" }
        let minutes = Int((seconds / 60).rounded(.up))
        if minutes < 60 { return "about \(minutes) min left" }
        return "about \(minutes / 60) h \(minutes % 60) min left"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardTitle(text: title)
            DisclosureGroup("Download details") {
                Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ProgressView(value: progress.fraction)
            HStack {
                Text(summary).monospacedDigit()
                Spacer()
                if progress.fileCount > 1 { Text("File \(min(progress.filesDone + 1, progress.fileCount)) of \(progress.fileCount)").foregroundStyle(.secondary) }
            }
            if let file = progress.file, progress.fileCount > 1 {
                VStack(alignment: .leading, spacing: 4) {
                    Text(file).font(.callout.monospaced())
                    ProgressView(value: progress.fileFraction)
                    Text("\(progress.fileBytes.formatted(.byteCount(style: .file))) of \(progress.fileSize.formatted(.byteCount(style: .file)))")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                .padding(10)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
            Text("You can pause, or quit Drama Studio; the download picks up where it stopped. It uses your normal connection, so on a mobile hotspot or a metered plan, pause until you are on Wi-Fi or Ethernet.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct WorkingCard: View {
    let title: String
    let detail: String
    let line: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                CardTitle(text: title)
            }
            Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if let line { DisclosureGroup("Preparation details") { Text(line).font(.callout.monospaced()).foregroundStyle(.secondary) } }
        }
    }
}

private struct ConversionCard: View {
    let command: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardTitle(text: "Build the voice codec")
            Text("This build of Drama Studio has no codec download, so build the three Core ML codec models once with Python. Run this in Terminal (it needs uv and takes a few minutes), then click Check Again.")
                .fixedSize(horizontal: false, vertical: true)
            Text(command)
                .font(.callout.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}

private struct FirstLineCard: View {
    let model: SessionModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label { CardTitle(text: "Drama Studio is ready") } icon: {
                Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
            }
            Text("Speak your first line. The words in brackets direct the performance; you can change any of it in the editor.")
            Text(model.script)
                .font(.title3)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct FailedCard: View {
    let title: String
    let step: SetupStep
    let reason: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label { CardTitle(text: "\(title) didn't finish") } icon: {
                Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
            }
            Text(reason).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            if step == .weights || step == .codec {
                Text("Files that finished downloading are kept, and a partial file resumes.").foregroundStyle(.secondary)
            }
        }
    }
}

private struct UsageCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardTitle(text: "Help improve Drama Studio")
            Text(UsageMonitoring.explanation)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
