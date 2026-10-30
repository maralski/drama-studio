// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI

/// The first-run window: one card per ``SetupStage``.
struct SetupView: View {
    let setup: SetupCoordinator
    let model: SessionModel

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label { Text("Drama Studio") } icon: { Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 32, height: 32) }
                            .font(.largeTitle.weight(.semibold))
                        Text("Expressive speech from Fish Audio's Drama 3, running on this Mac.")
                            .foregroundStyle(.secondary)
                    }
                    StepStrip(stage: setup.stage)
                    Divider()
                    card
                }
                .frame(maxWidth: 660, alignment: .leading)
                .padding(.horizontal, 40)
                .padding(.vertical, 36)
                .frame(maxWidth: .infinity)
            }
            footer
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle("Set Up Drama Studio")
        .frame(minWidth: 600, minHeight: 480)
    }

    @ViewBuilder
    private var footer: some View {
        switch setup.stage {
        case .ready, .tuning, .codec(.unpacking):
            EmptyView()
        default:
            HStack {
                switch setup.stage {
                case .requirements:
                    Button("Quit") { NSApp.terminate(nil) }
                    Spacer()
                    Button("Continue Anyway") { setup.acknowledgeRequirements() }
                        .keyboardShortcut(.defaultAction)
                case .terms:
                    Button("Quit") { NSApp.terminate(nil) }
                    Spacer()
                    Button("I Agree") { setup.acceptTerms() }
                        .keyboardShortcut(.defaultAction)
                case .weights(let transfer), .codec(.downloading(let transfer)):
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

    @ViewBuilder
    private var card: some View {
        switch setup.stage {
        case .requirements(let gaps): RequirementsCard(gaps: gaps)
        case .terms: TermsCard()
        case .weights(let transfer):
            TransferCard(title: "Downloading Drama 3",
                         detail: "\(ModelSetup.gigabytes(ModelDownloadPlan.drama3.totalBytes)) of model files from Hugging Face (fishaudio/drama-3 at revision e0a9218), into \(AppHome.models.path). Each file is checked against its SHA-256 before it is kept.",
                         transfer: transfer)
        case .codec(.downloading(let transfer)):
            TransferCard(title: "Downloading the voice codec",
                         detail: "The three Core ML models that turn Drama 3's codes into audio, from this release on GitHub.",
                         transfer: transfer)
        case .codec(.unpacking):
            WorkingCard(title: "Unpacking the voice codec", detail: "Checking the three Core ML models and moving them into place.", line: nil)
        case .codec(.needsConversion(let command)): ConversionCard(command: command)
        case .tuning(let detail):
            WorkingCard(title: "Getting Drama 3 ready for this Mac",
                        detail: "Preparing speech generation for this Mac usually takes 30 seconds to a minute. Later launches skip this preparation.",
                        line: detail)
        case .firstLine: FirstLineCard(model: model)
        case .ready: EmptyView()
        case .failed(let step, let reason): FailedCard(step: step, reason: reason)
        }
    }
}

private struct StepStrip: View {
    let stage: SetupStage

    private var steps: [SetupStep] {
        if case .requirements = stage { return SetupStep.allCases }
        if stage.step == .requirements { return SetupStep.allCases }
        return Array(SetupStep.allCases.dropFirst())
    }

    var body: some View {
        HStack(spacing: 14) {
            ForEach(steps, id: \.self) { step in
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
        let mine = all.firstIndex(of: step)!, now = all.firstIndex(of: current)!
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

private struct RequirementsCard: View {
    let gaps: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label { CardTitle(text: "This Mac is below what Drama Studio was tested on") } icon: {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
            Bullets(items: gaps)
            Text("Nothing has been downloaded yet. You can continue at your own risk; the download is 41.5 GB.")
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
            Text("Nothing downloads and nothing is spoken until you agree. Next, Drama Studio downloads \(ModelSetup.gigabytes(ModelDownloadPlan.drama3.totalBytes)) of model files from Hugging Face.")
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
    let step: SetupStep
    let reason: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label { CardTitle(text: "\(step.title) didn't finish") } icon: {
                Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
            }
            Text(reason).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            if step == .weights || step == .codec {
                Text("Files that finished downloading are kept, and a partial file resumes.").foregroundStyle(.secondary)
            }
        }
    }
}
