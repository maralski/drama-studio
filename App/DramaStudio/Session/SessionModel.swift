import DramaEngine
import SwiftUI

@MainActor @Observable
final class SessionModel {
    var script = ""
    var screenplayMode = false
    var config = GenerationConfig()
    var speakerVoices: [Int: ReferenceVoice.ID] = [:]
    var selection: TextSelection?
    var notice: String?

    private(set) var state: PlaybackState = .idle
    private(set) var telemetry = GenerationTelemetry()
    private(set) var audio: [Float] = []
    private(set) var result: SpeechResult?
    private(set) var playhead = 0
    /// Wall time of the current or last generation, from Generate to the end of the run.
    private(set) var generationClock: GenerationClock?
    private(set) var previewingVoice: ReferenceVoice.ID?
    @ObservationIgnored var playsWhileGenerating = true

    let library: VoiceLibrary
    let synthesizer: any SpeechSynthesizer
    @ObservationIgnored private let player = StreamingPlayer()
    @ObservationIgnored private let previewPlayer = StreamingPlayer()
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var run = 0

    init(synthesizer: any SpeechSynthesizer, library: VoiceLibrary) {
        self.synthesizer = synthesizer
        self.library = library
        player.onDrained = { [weak self] in self?.playbackDrained() }
        previewPlayer.onDrained = { [weak self] in self?.stopPreview() }
    }

    var prepared: Result<PreparedScript, ScriptError> {
        Screenplay.prepare(script, asScreenplay: screenplayMode)
    }

    var casting: Casting {
        Casting(speakers: (try? prepared.get())?.speakers ?? [], assignments: speakerVoices, library: library.voices)
    }

    var canGenerate: Bool { !state.isGenerating && (try? prepared.get()) != nil }
    var canPlay: Bool { !audio.isEmpty }
    var canExport: Bool { !audio.isEmpty && !state.isGenerating }

    /// Not observable while playing; poll it from a TimelineView.
    var position: Int { player.position ?? playhead }

    var duration: TimeInterval { Double(audio.count) / Double(ModelConstants.sampleRate) }

    func generate() {
        guard !state.isGenerating, case .success(let script) = prepared else { return }
        stopPlayback(resetTo: 0)
        audio = []
        result = nil
        telemetry = GenerationTelemetry()
        generationClock = GenerationClock(start: .now)
        state = .loading(progress: 0, detail: "Starting engine")
        run += 1
        let current = run
        let request = SpeechRequest(text: script.text, references: casting.references, config: config)
        let stream = synthesizer.synthesize(request)
        generation = Task { [weak self] in
            do {
                for try await event in stream {
                    guard self?.run == current else { return }
                    self?.apply(event)
                }
                self?.generationEnded(run: current, error: nil)
            } catch {
                self?.generationEnded(run: current, error: error)
            }
        }
    }

    func stopGeneration() {
        guard state.isGenerating else { return }
        run += 1
        generation?.cancel()
        generation = nil
        settleAfterGeneration(audio.isEmpty ? .idle : state.afterGeneration)
    }

    func apply(_ event: SpeechEvent) {
        switch event {
        case .loadingModel(let progress, let detail):
            state = .loading(progress: progress, detail: detail)
        case .chunkStarted(let index, let count, _):
            state = .generating(chunk: index + 1, of: count, playing: state.isPlaying)
            telemetry.frame = 0
            telemetry.chunkStart = nil
        case .prefill:
            break
        case .frame(let index, let maxFrames):
            let now = ContinuousClock.now
            let start = telemetry.chunkStart ?? now
            telemetry.chunkStart = start
            telemetry.frame = index + 1
            telemetry.maxFrames = maxFrames
            let elapsed = (now - start).seconds
            if elapsed > 0.25 { telemetry.framesPerSecond = Double(index) / elapsed }
        case .audio(let samples, let chunkIndex):
            audio += samples
            if state.isPlaying {
                player.feed(audio)
            } else if chunkIndex == 0, playsWhileGenerating {
                startPlayback(from: 0)
            }
        case .retry:
            telemetry.retries += 1
            telemetry.frame = 0
        case .finished(let result):
            self.result = result
            if result.timings.framesPerSecond > 0 { telemetry.framesPerSecond = result.timings.framesPerSecond }
            settleAfterGeneration(state.afterGeneration)
        }
    }

    private func generationEnded(run ended: Int, error: Error?) {
        guard ended == run else { return }
        generation = nil
        if let error {
            stopPlayback(resetTo: nil)
            settleAfterGeneration(.failed(Self.describe(error)))
        } else if state.isGenerating {
            settleAfterGeneration(audio.isEmpty ? .failed("The engine stopped without producing audio.") : state.afterGeneration)
        }
    }

    private func settleAfterGeneration(_ next: PlaybackState) {
        state = next
        if generationClock?.end == nil { generationClock?.end = .now }
        if player.isStarved { stopPlayback(resetTo: 0) }
    }

    func togglePlayback() {
        if state.isPlaying {
            stopPlayback(resetTo: nil)
        } else if canPlay {
            startPlayback(from: playhead >= audio.count ? 0 : playhead)
        }
    }

    func seek(toFraction fraction: Double) {
        guard canPlay else { return }
        let sample = min(audio.count - 1, max(0, Int(fraction * Double(audio.count))))
        if state.isPlaying {
            startPlayback(from: sample)
        } else {
            playhead = sample
        }
    }

    private func startPlayback(from sample: Int) {
        do {
            try player.play(audio, from: sample)
            playhead = sample
            state = state.withPlayback(true)
        } catch {
            state = .failed("Couldn't start audio output: \(error.localizedDescription)")
        }
    }

    private func stopPlayback(resetTo sample: Int?) {
        let reached = player.stop()
        playhead = sample ?? reached ?? playhead
        if state.isPlaying { state = state.withPlayback(false) }
    }

    private func playbackDrained() {
        guard !state.isGenerating else { return }
        stopPlayback(resetTo: 0)
    }

    func importScript(from url: URL) {
        do {
            script = try ScriptImporter.importFile(at: url)
            screenplayMode = Screenplay.looksLikeScreenplay(script)
            selection = nil
            notice = nil
        } catch {
            notice = error.localizedDescription
        }
    }

    func open(_ url: URL) {
        guard url.scheme == "dramastudio" else { return importScript(from: url) }
        switch url.host() {
        case "generate": generate()
        case "stop": stopGeneration()
        case "play": togglePlayback()
        default: notice = "Drama Studio doesn't understand the link \(url.absoluteString)."
        }
    }

    func insert(_ cue: PerformanceCue) {
        let range = selectedCharacterRange ?? script.count..<script.count
        let edit = cue.insert(into: script, at: range)
        script = edit.text
        let start = script.index(script.startIndex, offsetBy: edit.selection.lowerBound)
        let end = script.index(script.startIndex, offsetBy: edit.selection.upperBound)
        selection = TextSelection(range: start..<end)
    }

    private var selectedCharacterRange: Range<Int>? {
        guard let selection, case .selection(let range) = selection.indices,
              range.lowerBound >= script.startIndex, range.upperBound <= script.endIndex else { return nil }
        let lower = script.distance(from: script.startIndex, to: range.lowerBound)
        return lower..<(lower + script.distance(from: range.lowerBound, to: range.upperBound))
    }

    func voice(for speaker: Speaker) -> ReferenceVoice.ID? {
        library.voice(speakerVoices[speaker.index])?.id
    }

    func setVoice(_ id: ReferenceVoice.ID?, for speaker: Speaker) {
        speakerVoices[speaker.index] = id
    }

    func export(to url: URL, as format: ExportFormat) {
        do {
            try AudioExporter.write(audio, to: url, as: format)
            notice = nil
        } catch {
            notice = "Export failed: \(error.localizedDescription)"
        }
    }

    func encodeVoice(clip: URL, transcript: String, name: String) async throws {
        let voice = try await synthesizer.encodeReference(audioURL: clip, transcript: transcript, name: name)
        try library.add(voice)
    }

    func togglePreview(_ voice: ReferenceVoice) {
        let wasPreviewing = previewingVoice == voice.id
        stopPreview()
        guard !wasPreviewing else { return }
        previewingVoice = voice.id
        Task {
            do {
                let samples = try await synthesizer.decode(frames: voice.frames)
                guard previewingVoice == voice.id else { return }
                try previewPlayer.play(samples, from: 0)
            } catch {
                previewingVoice = nil
                notice = "Couldn't preview \(voice.name): \(Self.describe(error))"
            }
        }
    }

    func stopPreview() {
        previewPlayer.stop()
        previewingVoice = nil
    }

    func removeVoice(_ id: ReferenceVoice.ID) {
        do {
            try library.remove(id)
            speakerVoices = speakerVoices.filter { $0.value != id }
        } catch {
            notice = "Couldn't remove the voice: \(error.localizedDescription)"
        }
    }

    static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}

struct GenerationClock: Equatable {
    let start: Date
    var end: Date?

    func elapsed(at now: Date) -> TimeInterval {
        (end ?? now).timeIntervalSince(start)
    }

    /// Nil while the run is still going.
    var duration: TimeInterval? { end.map { $0.timeIntervalSince(start) } }
}

/// Which references a request carries. The engine contract has no explicit
/// speaker field on a reference, so reference i speaks for speaker i and the
/// assigned voices must form an unbroken run from speaker 0.
struct Casting: Equatable {
    var references: [ReferenceVoice] = []
    /// Speakers whose assigned voice is dropped because an earlier speaker has none.
    var ignored: [Speaker] = []

    init(speakers: [Speaker], assignments: [Int: ReferenceVoice.ID], library: [ReferenceVoice]) {
        var gap = false
        for speaker in speakers.sorted(by: { $0.index < $1.index }) {
            let voice = library.first { $0.id == assignments[speaker.index] }
            if let voice, !gap, speaker.index == references.count {
                references.append(voice)
            } else if voice != nil {
                ignored.append(speaker)
            } else {
                gap = true
            }
        }
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
