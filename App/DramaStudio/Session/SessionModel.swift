import DramaEngine
import SwiftUI

@MainActor @Observable
final class SessionModel {
    var script = ""
    var screenplayMode = false
    var expressiveness = GenerationConfig.defaultExpressiveness
    var quickPreview = false
    /// Absent means `.modelDefault`.
    private(set) var speakerVoices: [Int: VoiceChoice] = [:]
    /// The first take made with a surprise or designed voice, reused so later takes keep that voice.
    private(set) var sessionVoices: [VoiceChoice: Take] = [:]
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
    private(set) var modelLoaded = false
    /// Every finished read of `takesBasis`, oldest first.
    private(set) var takes: [SessionTake] = []
    /// The script and voices the takes were read with. A retake clones the
    /// first take, so it is only offered while both still hold.
    private(set) var takesBasis: TakesBasis?
    /// The take Export writes.
    private(set) var keptTakeID: UUID?
    private(set) var selectedTakeID: UUID?
    /// The last line tried in the design sheet.
    private(set) var audition: Audition?
    @ObservationIgnored var playsWhileGenerating = true

    let library: VoiceLibrary
    let synthesizer: any SpeechSynthesizer
    @ObservationIgnored private let player = StreamingPlayer()
    @ObservationIgnored private let previewPlayer = StreamingPlayer()
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var run = 0
    @ObservationIgnored private var job: Job?
    @ObservationIgnored private var streamed = StreamedAudio()

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
        Casting(speakers: (try? prepared.get())?.speakers ?? [], assignments: speakerVoices, library: library.voices, sessionVoices: sessionVoices)
    }

    /// No seed, so every generation draws a fresh one.
    var config: GenerationConfig { .make(expressiveness: expressiveness, quickPreview: quickPreview) }

    var canGenerate: Bool { !state.isGenerating && (try? prepared.get()) != nil }
    var canTakeAgain: Bool { canGenerate && !audio.isEmpty }
    var canPlay: Bool { !audio.isEmpty }
    var canExport: Bool { !exportSamples.isEmpty && !state.isGenerating }
    var canRetake: Bool { canGenerate && !takes.isEmpty && retakeBlocker == nil }

    /// Why the takes can't be retaken with a note, when that is not obvious from the controls.
    var retakeBlocker: String? {
        guard case .success(let script) = prepared else { return nil }
        if script.speakers.count > 1 { return "Retakes work on scripts with one speaker, because the note rewrites the line without speaker tags." }
        guard !takes.isEmpty, let takesBasis else { return nil }
        if script.text != takesBasis.text { return "The script changed since these takes. Generate to start a new set." }
        if speakerVoices != takesBasis.voices { return "The voice changed since these takes. Generate to start a new set." }
        return nil
    }

    var keptTake: SessionTake? { takes.first { $0.id == keptTakeID } }
    var exportSamples: [Float] { keptTake?.samples ?? audio }

    /// Not observable while playing; poll it from a TimelineView.
    var position: Int { player.position ?? playhead }

    var duration: TimeInterval { Double(audio.count) / Double(ModelConstants.sampleRate) }

    func generate() {
        guard !state.isGenerating, case .success(let script) = prepared else { return }
        let basis = TakesBasis(text: script.text, voices: speakerVoices)
        if basis != takesBasis {
            takes = []
            keptTakeID = nil
            selectedTakeID = nil
            takesBasis = basis
        }
        let casting = casting
        start(SpeechRequest(text: script.text, references: casting.references, character: casting.character,
                            previousTakes: casting.voiceTake.map { [$0] } ?? [], config: config),
              as: .take(note: nil, recordsVoice: casting.recordsVoice))
    }

    /// Reads the line again with a director's note, after every take so far.
    func retake(note: String) {
        guard canRetake, let note = DirectorsNote.clean(note), case .success(let script) = prepared else { return }
        let casting = casting
        let voice = casting.voiceTake.flatMap { $0 == takes.first?.take ? nil : $0 }
        start(SpeechRequest(text: script.text, references: casting.references, character: casting.character, note: note,
                            previousTakes: (voice.map { [$0] } ?? []) + takes.map(\.take), config: config),
              as: .take(note: note, recordsVoice: casting.recordsVoice))
    }

    /// Speaks a line in a designed voice through the preview player, leaving
    /// the script, the takes, and the transport's audio alone.
    func tryLine(_ line: String, as design: VoiceDesign) {
        guard !state.isGenerating, !DirectorsNote.spokenWords(line).isEmpty else { return }
        audition = nil
        start(SpeechRequest(text: line, character: design, config: config), as: .audition(design))
    }

    var isAuditioning: Bool {
        if state.isGenerating, case .audition? = job { true } else { false }
    }

    func cancelAudition() {
        if isAuditioning { stopGeneration() }
        stopPreview()
        audition = nil
    }

    /// Casts the design; when it is the one just tried, the voice heard becomes its session voice.
    func useDesign(_ design: VoiceDesign, for speaker: Speaker) {
        setVoice(.designed(design), for: speaker)
        if let audition, audition.design == design { sessionVoices[.designed(design)] = audition.take }
        cancelAudition()
    }

    func cast(_ preset: CastPreset) {
        script = preset.line
        screenplayMode = false
        selection = nil
        notice = nil
        setVoice(.designed(preset.design), forSpeaker: 0)
    }

    private func start(_ request: SpeechRequest, as job: Job) {
        stopPlayback(resetTo: 0)
        stopPreview()
        if case .take = job {
            audio = []
            result = nil
            generationClock = GenerationClock(start: .now)
        }
        telemetry = GenerationTelemetry()
        streamed = StreamedAudio()
        state = .loading(progress: 0, detail: "Starting engine")
        run += 1
        self.job = job
        let current = run
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
        case .partialAudio, .audio:
            receive(streamed.newSamples(from: event))
        case .retry:
            telemetry.retries += 1
            telemetry.frame = 0
        case .finished(let result):
            if result.timings.framesPerSecond > 0 { telemetry.framesPerSecond = result.timings.framesPerSecond }
            switch job {
            case .take(let note, let voice)?:
                self.result = result
                addTake(result, note: note, recordsVoice: voice)
                settleAfterGeneration(state.afterGeneration)
            case .audition(let design)?:
                audition = Audition(design: design, take: result.take)
                settleAfterGeneration(audio.isEmpty ? .idle : .ready)
                playAudition(result.samples)
            case nil:
                break
            }
        }
    }

    private func receive(_ samples: [Float]) {
        guard case .take? = job, !samples.isEmpty else { return }
        let first = audio.isEmpty
        audio += samples
        if state.isPlaying {
            player.feed(audio)
        } else if first, playsWhileGenerating {
            startPlayback(from: 0)
        }
    }

    private func playAudition(_ samples: [Float]) {
        do {
            try previewPlayer.play(samples, from: 0)
        } catch {
            notice = "Couldn't start audio output: \(error.localizedDescription)"
        }
    }

    private func addTake(_ result: SpeechResult, note: String?, recordsVoice: VoiceChoice?) {
        if let recordsVoice, sessionVoices[recordsVoice] == nil { sessionVoices[recordsVoice] = result.take }
        let take = SessionTake(number: (takes.last?.number ?? 0) + 1, note: note, take: result.take, samples: result.samples)
        takes.append(take)
        selectedTakeID = take.id
    }

    /// Plays the take, which becomes the transport's audio.
    func selectTake(_ id: SessionTake.ID) {
        guard !state.isGenerating, let take = takes.first(where: { $0.id == id }) else { return }
        stopPlayback(resetTo: 0)
        selectedTakeID = id
        audio = take.samples
        startPlayback(from: 0)
    }

    func toggleKeepSelectedTake() {
        keptTakeID = keptTakeID == selectedTakeID ? nil : selectedTakeID
    }

    private func generationEnded(run ended: Int, error: Error?) {
        guard ended == run else { return }
        generation = nil
        refreshEngineState()
        if let error {
            stopPlayback(resetTo: nil)
            settleAfterGeneration(.failed(Self.describe(error)))
        } else if state.isGenerating {
            settleAfterGeneration(audio.isEmpty ? .failed("The engine stopped without producing audio.") : state.afterGeneration)
        }
    }

    /// The engine also loads for API requests, so ask it rather than infer from this session's runs.
    func refreshEngineState() {
        Task { modelLoaded = await synthesizer.isLoaded }
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
        apply(cue.insert(into: script, at: selectedCharacterRange ?? script.count..<script.count))
    }

    func insert(_ example: ExampleLine) {
        apply(example.insert(into: script, at: selectedCharacterRange ?? script.count..<script.count))
    }

    private func apply(_ edit: (text: String, selection: Range<Int>)) {
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

    /// A library voice that has since been removed reads as the model default.
    func voice(for speaker: Speaker) -> VoiceChoice {
        let choice = speakerVoices[speaker.index] ?? .modelDefault
        if case .library(let id) = choice, library.voice(id) == nil { return .modelDefault }
        return choice
    }

    /// Choosing a different voice forgets the old one's session voice, so
    /// picking Surprise me again gives a fresh surprise.
    func setVoice(_ choice: VoiceChoice, forSpeaker index: Int) {
        let old = speakerVoices[index] ?? .modelDefault
        guard old != choice else { return }
        speakerVoices[index] = choice == .modelDefault ? nil : choice
        if !speakerVoices.values.contains(old) { sessionVoices[old] = nil }
    }

    func setVoice(_ choice: VoiceChoice, for speaker: Speaker) {
        setVoice(choice, forSpeaker: speaker.index)
    }

    /// Designed voices currently cast, for the voice pickers.
    var castDesigns: [VoiceDesign] {
        var seen = Set<VoiceDesign>()
        return speakerVoices.sorted { $0.key < $1.key }.compactMap(\.value.design).filter { seen.insert($0).inserted }
    }

    func canKeepVoice(for speaker: Speaker) -> Bool {
        sessionVoices[voice(for: speaker)] != nil
    }

    /// Saves the speaker's session voice to the library and casts it from there.
    func keepVoice(for speaker: Speaker) {
        let choice = voice(for: speaker)
        guard let take = sessionVoices[choice] else { return }
        let design = choice.design
        let voice = ReferenceVoice(name: design?.name ?? "Surprise voice",
                                   transcript: DirectorsNote.spokenText(ofTurn: take.text, character: design), codes: take.codes)
        do {
            try library.add(voice)
            setVoice(.library(voice.id), for: speaker)
        } catch {
            notice = "Couldn't keep the voice: \(error.localizedDescription)"
        }
    }

    func export(to url: URL, as format: ExportFormat) {
        do {
            try AudioExporter.write(exportSamples, to: url, as: format)
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
            speakerVoices = speakerVoices.filter { $0.value != .library(id) }
        } catch {
            notice = "Couldn't remove the voice: \(error.localizedDescription)"
        }
    }

    static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}

/// What a generation run is for.
private enum Job {
    case take(note: String?, recordsVoice: VoiceChoice?)
    case audition(VoiceDesign)
}

struct TakesBasis: Equatable {
    var text: String
    var voices: [Int: VoiceChoice]
}

/// A line spoken in a designed voice before it is cast.
struct Audition: Equatable {
    let design: VoiceDesign
    let take: Take
}

/// One finished read of the script, kept so it can be replayed, retaken, or exported.
struct SessionTake: Identifiable, Equatable {
    let id = UUID()
    let number: Int
    let note: String?
    let take: Take
    let samples: [Float]
    var duration: TimeInterval { Double(samples.count) / Double(ModelConstants.sampleRate) }
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

/// What a request carries for its voices. The engine contract has no
/// explicit speaker field on a reference, so reference i speaks for speaker i
/// and library voices must form an unbroken run from speaker 0. A surprise or
/// designed voice has no reference to place, so it only casts a script with
/// one speaker.
struct Casting: Equatable {
    struct Ignored: Equatable, Identifiable {
        enum Reason: Equatable { case gap, needsLibrary }
        var speaker: Speaker
        var reason: Reason
        var id: Int { speaker.index }

        var message: String {
            switch reason {
            case .gap: "\(speaker.name)'s voice is not used: voices apply in speaker order, so every earlier speaker needs one too."
            case .needsLibrary: "\(speaker.name)'s voice is not used: a script with several speakers takes library voices only. Cast it alone and click Keep This Voice to add it to the library first."
            }
        }
    }

    var references: [ReferenceVoice] = []
    var character: VoiceDesign?
    /// The session voice the request reuses as its voice reference.
    var voiceTake: Take?
    /// The choice whose session voice this request's take becomes.
    var recordsVoice: VoiceChoice?
    var ignored: [Ignored] = []

    init(speakers: [Speaker], assignments: [Int: VoiceChoice], library: [ReferenceVoice], sessionVoices: [VoiceChoice: Take] = [:]) {
        var gap = false
        for speaker in speakers.sorted(by: { $0.index < $1.index }) {
            switch assignments[speaker.index] ?? .modelDefault {
            case .library(let id):
                guard let voice = library.first(where: { $0.id == id }) else {
                    gap = true
                    continue
                }
                if !gap, speaker.index == references.count {
                    references.append(voice)
                } else {
                    ignored.append(Ignored(speaker: speaker, reason: .gap))
                }
            case .modelDefault:
                gap = true
            case let choice where speakers.count == 1:
                character = choice.design
                voiceTake = sessionVoices[choice]
                if voiceTake == nil { recordsVoice = choice }
            case .surprise, .designed:
                ignored.append(Ignored(speaker: speaker, reason: .needsLibrary))
                gap = true
            }
        }
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
