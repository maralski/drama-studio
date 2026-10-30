// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import DramaEngine
import DramaServer
import SwiftUI

@MainActor @Observable
final class SessionModel {
    /// What is narrated: a plain script or a document read section by section.
    var source: ScriptSource = .plain("")
    /// The plain script; setting it makes the source plain.
    var script: String {
        get { source.plainText }
        set { source = .plain(newValue) }
    }
    var document: NarrationDocument? { source.document }
    var selectedSegmentID: NarrationDocument.Segment.ID?
    var screenplayMode = false
    var expressiveness = GenerationConfig.defaultExpressiveness
    var quickPreview = false
    /// Absent means `.modelDefault`.
    private(set) var speakerVoices: [Int: VoiceChoice] = [:]
    /// The first take made with a surprise or designed voice, reused so later takes keep that voice.
    private(set) var sessionVoices: [VoiceChoice: Take] = [:] {
        didSet {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            sessionVoiceFingerprints = sessionVoices.compactMapValues { take in
                guard let data = try? encoder.encode(take) else { return nil }
                return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            }
        }
    }
    private var sessionVoiceFingerprints: [VoiceChoice: String] = [:]
    private(set) var projectURL: URL?
    private var savedRevision: ProjectRevision?
    @ObservationIgnored var confirmReplacement: (String) -> Bool = { _ in true }
    @ObservationIgnored private var generationBasis: TakesBasis?
    private(set) var sectionSelections: [UUID: UUID] = [:]
    private(set) var previewingTakeID: UUID?

    var hasUnsavedChanges: Bool { savedRevision != projectRevision }
    var projectName: String { projectURL?.deletingPathExtension().lastPathComponent ?? title }

    private var projectRevision: ProjectRevision {
        ProjectRevision(source: source, voices: speakerVoices, sessionVoices: sessionVoices, screenplay: screenplayMode,
                        expressiveness: expressiveness, quickPreview: quickPreview,
                        takes: takes.map(\.id), sections: sectionTakes.mapValues { $0.takes.map(\.id) },
                        exportTake: keptTakeID, sectionExports: sectionTakes.mapValues(\.chosenID))
    }

    var selection: TextSelection?
    var notice: String?
    var exportedURL: URL?

    private(set) var state: PlaybackState = .idle
    private(set) var telemetry = GenerationTelemetry()
    private(set) var audio: [Float] = []
    private(set) var result: SpeechResult?
    private(set) var playhead = 0
    /// Wall time of the current or last generation, from Generate to the end of the run.
    private(set) var generationClock: GenerationClock?
    /// Time left in the current run, from its events.
    private(set) var estimate: GenerationEstimate?
    @ObservationIgnored private var estimateStart = Date()
    private(set) var previewingVoice: ReferenceVoice.ID?
    private(set) var modelLoaded = false
    /// All recorded takes, with their individual source text and voices.
    private(set) var takes: [SessionTake] = []
    /// The take Export writes.
    private(set) var keptTakeID: UUID?
    private(set) var selectedTakeID: UUID?
    /// A document's takes, per section.
    private(set) var sectionTakes: [NarrationDocument.Segment.ID: SectionTakes] = [:]
    /// The last line tried in the design sheet.
    private(set) var audition: Audition?
    private(set) var recentProjects = RecentProjects.load()
    var playbackRate: Double {
        didSet {
            player.rate = Float(playbackRate)
            AppHome.defaults.set(playbackRate, forKey: Self.playbackRateKey)
        }
    }
    @ObservationIgnored var playsWhileGenerating = true

    let library: VoiceLibrary
    private(set) var synthesizer: any SpeechSynthesizer
    /// Why generation is refused right now; the app asks its setup coordinator.
    @ObservationIgnored var generationRefusal: @MainActor () -> String? = { nil }
    @ObservationIgnored private let pronunciationURL: URL
    @ObservationIgnored private let player = StreamingPlayer()
    @ObservationIgnored private let previewPlayer = StreamingPlayer()
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var run = 0
    @ObservationIgnored private var job: Job?
    @ObservationIgnored private var streamed = StreamedAudio()
    /// The sentences of the request under way, placed as their audio arrives.
    @ObservationIgnored private var liveSpans: [SentenceSpan] = []
    @ObservationIgnored private var liveChunks: [Int: Int] = [:]
    /// The script's own lines of the request under way, and how the dictionary rewrote them.
    @ObservationIgnored private var scriptLines: [String] = []
    @ObservationIgnored private var pronunciation = PronunciationDictionary()

    static let streamsAudioKey = "streamsAudio"
    static let playbackRateKey = "playbackRate"
    static let playbackRates: [Double] = [0.75, 1, 1.25, 1.5]
    /// The one segment a plain script's take stands for in captions and projects.
    static let plainSegmentID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    init(synthesizer: any SpeechSynthesizer, library: VoiceLibrary,
         pronunciationURL: URL = PronunciationDictionary.defaultURL) {
        self.synthesizer = synthesizer
        self.library = library
        self.pronunciationURL = pronunciationURL
        let rate = AppHome.defaults.double(forKey: Self.playbackRateKey)
        playbackRate = Self.playbackRates.contains(rate) ? rate : 1
        player.rate = Float(playbackRate)
        player.onDrained = { [weak self] in self?.playbackDrained() }
        previewPlayer.onDrained = { [weak self] in self?.stopPreview() }
        savedRevision = projectRevision
    }

    var prepared: Result<PreparedScript, ScriptError> {
        Screenplay.prepare(script, asScreenplay: screenplayMode)
    }

    /// The text in the editor: the plain script, or the selected section of a document.
    var editorText: String {
        get { document.map { $0.segments.first { $0.id == selectedSegmentID }?.text ?? "" } ?? script }
        set {
            switch source {
            case .plain: source = .plain(newValue)
            case .document(var document):
                guard let index = document.segments.firstIndex(where: { $0.id == selectedSegmentID }) else { return }
                document.segments[index].text = newValue
                source = .document(document)
            }
        }
    }

    var selectedSegment: NarrationDocument.Segment? {
        document?.segments.first { $0.id == selectedSegmentID }
    }

    /// The speakers the Cast section lists.
    var castSpeakers: Result<[Speaker], ScriptError> {
        guard let document else { return prepared.map(\.speakers) }
        return SectionRequests.queue(document).isEmpty ? .failure(.empty) : .success(SectionRequests.speakers(of: document))
    }

    var casting: Casting {
        Casting(speakers: (try? prepared.get())?.speakers ?? [], assignments: speakerVoices, library: library.voices, sessionVoices: sessionVoices)
    }

    /// No seed, so every generation draws a fresh one.
    var config: GenerationConfig {
        var config = GenerationConfig.make(expressiveness: expressiveness, quickPreview: quickPreview)
        if AppHome.defaults.object(forKey: Self.streamsAudioKey) as? Bool ?? true { config.streaming = .defaultLowLatency }
        return config
    }

    var title: String {
        document?.title ?? Self.suggestedName(for: script)
    }

    static func suggestedName(for script: String) -> String {
        let words = script.replacingOccurrences(of: #"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]"#, with: " ", options: .regularExpression)
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .prefix(5)
        return words.isEmpty ? "Drama Studio Take" : words.joined(separator: " ")
    }

    func selectSegment(_ id: NarrationDocument.Segment.ID?) {
        stopPreview()
        selectedSegmentID = id
        selection = nil
        guard let id, let take = narrationTakes.first(where: { $0.segmentID == id }) else { return }
        seek(toSample: take.sampleRange.lowerBound)
    }

    func setSkip(_ skip: Bool, for id: NarrationDocument.Segment.ID) {
        guard !state.isGenerating else { return }
        updateSegment(id) { $0.skip = skip }
        stopPlayback(resetTo: 0)
        recompose()
    }

    func setSpeaker(_ tag: Int, for id: NarrationDocument.Segment.ID) {
        guard !state.isGenerating else { return }
        updateSegment(id) { $0.speakerTag = tag }
    }

    private func updateSegment(_ id: NarrationDocument.Segment.ID, _ change: (inout NarrationDocument.Segment) -> Void) {
        guard var document, let index = document.segments.firstIndex(where: { $0.id == id }) else { return }
        change(&document.segments[index])
        source = .document(document)
    }

    /// Starts a document: its first section is selected, nothing narrated yet.
    func open(_ document: NarrationDocument) {
        guard confirmReplacement("Open another document?") else { return }
        projectURL = nil
        replaceSource(with: .document(document))
    }

    /// Turns the document back into a plain script of its narrated sections.
    func flattenToScript() {
        guard let document, confirmReplacement("Flattening removes the sections and their takes. Save a project copy before continuing?") else { return }
        projectURL = nil
        replaceSource(with: .plain(document.flattened()))
        screenplayMode = false
    }

    /// A confirmed import starts a new project; editing text keeps its history.
    private func replaceSource(with new: ScriptSource) {
        clearNarration()
        projectURL = nil
        savedRevision = nil
        exportedURL = nil
        source = new
        selectedSegmentID = new.document?.segments.first?.id
        selection = nil
    }

    private func clearNarration() {
        stopGeneration()
        stopPreview()
        stopPlayback(resetTo: 0)
        audio = []
        result = nil
        takes = []
        keptTakeID = nil
        selectedTakeID = nil
        sectionSelections = [:]
        sectionTakes = [:]
        liveSpans = []
        generationClock = nil
        state = .idle
    }

    /// The narrated sections in document order with their chosen takes.
    private var chosenSections: [(segment: NarrationDocument.Segment, take: SessionTake)] {
        document?.segments.compactMap { segment in
            guard !segment.skip, let take = sectionTakes[segment.id]?.chosen else { return nil }
            return (segment, take)
        } ?? []
    }

    /// The document's narration timeline: consecutive ranges over the chosen takes.
    var narrationTakes: [NarrationTake] {
        var cursor = 0
        return chosenSections.map { segment, take in
            defer { cursor += take.samples.count }
            return NarrationTake(segmentID: segment.id, sampleRange: cursor..<(cursor + take.samples.count), sentences: take.sentences, take: take.take)
        }
    }

    func chosenTake(for id: NarrationDocument.Segment.ID) -> SessionTake? {
        sectionTakes[id]?.chosen
    }

    /// The takes strip: a plain script's takes, or the selected section's.
    var visibleTakes: [SessionTake] {
        document == nil ? takes : selectedSegmentID.flatMap { sectionTakes[$0]?.takes } ?? []
    }

    var visibleSelectedTakeID: SessionTake.ID? {
        document == nil ? selectedTakeID : selectedSegmentID.flatMap { sectionSelections[$0] ?? sectionTakes[$0]?.chosenID }
    }

    /// Rebuilds the transport's audio from the chosen takes. Callers that
    /// change audio before the playhead stop playback first.
    private func recompose() {
        guard document != nil else { return }
        audio = chosenSections.flatMap(\.take.samples)
        playhead = min(playhead, audio.count)
        if state.isPlaying { player.feed(audio) }
    }

    /// Where each sentence sits in the transport's audio.
    var sentenceSpans: [SentenceSpan] {
        if state.isGenerating { return liveSpans }
        if document != nil {
            return narrationTakes.flatMap { SentenceSpan.spans(of: $0.sentences, in: $0.sampleRange, segmentID: $0.segmentID) }
        }
        guard let take = takes.first(where: { $0.id == selectedTakeID }), take.samples.count == audio.count else { return [] }
        return SentenceSpan.spans(of: take.sentences, in: 0..<audio.count, segmentID: nil)
    }

    /// The sentence playing at `position`, with its take's sentences in order.
    func highlight(at position: Int) -> SpokenHighlight? {
        let spans = sentenceSpans
        guard let current = spans.lastIndex(where: { $0.range.contains(position) }) else { return nil }
        let segment = spans[current].segmentID
        let own = spans.indices.filter { spans[$0].segmentID == segment }
        guard let index = own.firstIndex(of: current) else { return nil }
        return SpokenHighlight(segmentID: segment, index: index, texts: own.map { spans[$0].text })
    }

    /// The section playing at `position`.
    func playingSegment(at position: Int) -> NarrationDocument.Segment.ID? {
        if case .sections(let run)? = job, state.isGenerating {
            return position >= run.audioStart ? run.current : narrationTakes.first { $0.sampleRange.contains(position) }?.segmentID
        }
        return narrationTakes.first { $0.sampleRange.contains(position) }?.segmentID
    }

    var canGenerate: Bool {
        guard !state.isGenerating else { return false }
        if let document { return !SectionRequests.queue(document).isEmpty }
        return (try? prepared.get()) != nil
    }
    var canTakeAgain: Bool { canGenerate && !audio.isEmpty }
    var canNarrateSection: Bool { !state.isGenerating && selectedSegment.map { !SectionRequests.spokenText(of: $0).isEmpty } ?? false }
    var canPlay: Bool { !audio.isEmpty }
    var canExport: Bool { !exportSamples.isEmpty && !state.isGenerating }
    var canExportSections: Bool { document != nil && !narrationTakes.isEmpty && !state.isGenerating }
    var canExportCaptions: Bool { !captionTakes.isEmpty && !state.isGenerating }
    var canExportSlidePackage: Bool { document?.source == .pptx && canExportSections }
    var canRetake: Bool { document == nil && canGenerate && !takes.isEmpty && retakeBlocker == nil }

    /// Why the takes can't be retaken with a note, when that is not obvious from the controls.
    var retakeBlocker: String? {
        guard document == nil, case .success(let script) = prepared else { return nil }
        if script.speakers.count > 1 { return "Retakes work on scripts with one speaker, because the note rewrites the line without speaker tags." }
        guard !takes.isEmpty else { return nil }
        guard let selected = takes.first(where: { $0.id == selectedTakeID }), !selected.isIncomplete,
              let takesBasis = selected.basis else { return "Generate a complete take before adding a direction note." }
        if script.text != takesBasis.text { return "The script changed since these takes. Generate to make an updated take." }
        if speakerVoices != takesBasis.voices || currentPlainBasis?.voiceAnchors != takesBasis.voiceAnchors { return "The voice changed since these takes. Generate to make an updated take." }
        return nil
    }

    var keptTake: SessionTake? { takes.first { $0.id == keptTakeID } }
    var exportSamples: [Float] { document == nil ? (keptTake?.samples ?? audio) : chosenSections.flatMap(\.take.samples) }
    var visibleExportTakeID: UUID? {
        document == nil ? (keptTakeID ?? selectedTakeID) : selectedSegmentID.flatMap { sectionTakes[$0]?.chosenID }
    }
    var exportSummary: String {
        if document != nil { return "Exporting \(narrationTakes.count) sections" }
        return exportTake.map { "Exporting Take \($0.number)" } ?? "Exporting partial audio"
    }
    var currentPlainBasis: TakesBasis? {
        (try? prepared.get()).map { makeBasis(text: $0.text, voices: speakerVoices) }
    }
    func basis(for segment: NarrationDocument.Segment) -> TakesBasis {
        let tag = segment.speakerTag ?? 0
        return makeBasis(text: SectionRequests.spokenText(of: segment), voices: [tag: speakerVoices[tag] ?? .modelDefault])
    }
    private func makeBasis(text: String, voices: [Int: VoiceChoice]) -> TakesBasis {
        let anchors = sessionVoiceFingerprints.filter { voices.values.contains($0.key) }
        return TakesBasis(text: text, voices: voices, voiceAnchors: anchors.isEmpty ? nil : anchors)
    }

    private func recordedBasis(for voice: VoiceChoice?) -> TakesBasis? {
        guard var basis = generationBasis else { return nil }
        if let voice, let fingerprint = sessionVoiceFingerprints[voice] {
            var anchors = basis.voiceAnchors ?? [:]
            anchors[voice] = fingerprint
            basis.voiceAnchors = anchors
        }
        return basis
    }

    func isOutdated(_ take: SessionTake, in segment: NarrationDocument.Segment? = nil) -> Bool {
        guard !take.isIncomplete, let recorded = take.basis else { return true }
        return recorded != (segment.map { basis(for: $0) } ?? currentPlainBasis)
    }
    func sectionStatus(_ segment: NarrationDocument.Segment) -> String {
        if segment.skip { return "Skipped" }
        if state.isGenerating && currentSection == segment.id { return "Generating…" }
        guard let take = chosenTake(for: segment.id) else { return "Not generated" }
        if take.isIncomplete { return "Incomplete" }
        return isOutdated(take, in: segment) ? "Needs regeneration" : "Ready"
    }
    var exportWarning: String? {
        if let document {
            let included = document.segments.filter { !$0.skip }
            let missing = included.filter { chosenTake(for: $0.id) == nil }.count
            let outdated = included.filter { segment in chosenTake(for: segment.id).map { isOutdated($0, in: segment) } ?? false }.count
            guard missing > 0 || outdated > 0 else { return nil }
            var parts: [String] = []
            if outdated > 0 { parts.append("\(outdated) \(outdated == 1 ? "section has" : "sections have") outdated or incomplete audio.") }
            if missing > 0 { parts.append("\(missing) \(missing == 1 ? "section has" : "sections have") no audio and will be omitted.") }
            parts.append("Generate changed sections, or export the existing audio.")
            return parts.joined(separator: " ")
        }
        guard let take = exportTake else { return canExport ? "This audio is incomplete because generation stopped early." : nil }
        if take.isIncomplete { return "This take is incomplete because generation stopped early. Regenerate it or export the existing audio." }
        return isOutdated(take) ? "The export take uses earlier text or voices, or comes from a project without generation history. Regenerate it or export the existing audio." : nil
    }
    var canGenerateChangedSections: Bool {
        !state.isGenerating && !changedSections.isEmpty
    }
    private var changedSections: [UUID] {
        guard let document else { return [] }
        return SectionRequests.queue(document).filter { id in
            guard let segment = document.segments.first(where: { $0.id == id }), let take = chosenTake(for: id) else { return true }
            return isOutdated(take, in: segment)
        }
    }
    func generateChangedSections() {
        guard canGenerateChangedSections, let first = changedSections.first else { return }
        let queue = changedSections
        beginRun()
        narrate(SectionRun(current: first, remaining: Array(queue.dropFirst()), audioStart: 0, startsPlayback: true))
    }


    /// The take Export writes, as a plain script's one-section timeline.
    private var exportTake: SessionTake? {
        if let keptTake { return keptTake }
        guard let take = takes.first(where: { $0.id == selectedTakeID }), take.samples.count == audio.count else { return nil }
        return take
    }

    /// The timeline captions and projects describe: the document's narration,
    /// or the exported take of a plain script.
    var captionTakes: [NarrationTake] {
        if document != nil { return narrationTakes }
        guard let take = exportTake else { return [] }
        return [NarrationTake(segmentID: Self.plainSegmentID, sampleRange: 0..<take.samples.count, sentences: take.sentences, take: take.take)]
    }

    func estimateLabel(at date: Date) -> String? {
        estimate?.label(at: date.timeIntervalSince(estimateStart))
    }

    /// Not observable while playing; poll it from a TimelineView.
    var position: Int { player.position ?? playhead }

    var duration: TimeInterval { Double(audio.count) / Double(ModelConstants.sampleRate) }

    func generate() {
        guard !state.isGenerating else { return }
        if let document {
            let queue = SectionRequests.queue(document)
            guard let first = queue.first else { return }
            beginRun()
            narrate(SectionRun(current: first, remaining: Array(queue.dropFirst()), audioStart: 0, startsPlayback: true))
            return
        }
        guard case .success(let script) = prepared else { return }
        let casting = casting
        start(SpeechRequest(text: script.text, references: casting.references, character: casting.character,
                            previousTakes: casting.voiceTake.map { [$0] } ?? [], config: config),
              as: .take(note: nil, recordsVoice: casting.recordsVoice))
    }

    /// Narrates one section again; its new take becomes the chosen one.
    func narrateSection(_ id: NarrationDocument.Segment.ID) {
        guard !state.isGenerating, let document, let segment = document.segments.first(where: { $0.id == id }),
              !SectionRequests.spokenText(of: segment).isEmpty else { return }
        if segment.skip { updateSegment(id) { $0.skip = false } }
        beginRun()
        let before = Set(document.segments.prefix { $0.id != id }.map(\.id))
        audio = chosenSections.filter { before.contains($0.segment.id) }.flatMap(\.take.samples)
        narrate(SectionRun(current: id, remaining: [], audioStart: audio.count, startsPlayback: true))
    }

    func narrateSelectedSection() {
        selectedSegmentID.map(narrateSection)
    }

    private func beginRun() {
        stopPlayback(resetTo: 0)
        audio = []
        result = nil
        liveSpans = []
        generationClock = GenerationClock(start: .now)
    }

    private func narrate(_ run: SectionRun) {
        guard let document, let segment = document.segments.first(where: { $0.id == run.current }) else { return }
        let casting = SectionRequests.casting(forSpeaker: segment.speakerTag ?? 0, assignments: speakerVoices,
                                              library: library.voices, sessionVoices: sessionVoices)
        var run = run
        run.recordsVoice = casting.recordsVoice
        let request = SectionRequests.request(for: segment, in: document, casting: casting,
                                              takes: sectionTakes.mapValues(\.chosen.take), config: config)
        start(request, as: .sections(run))
    }

    /// Reads the line again with a director's note, after every take so far.
    func retake(note: String) {
        guard canRetake, let note = DirectorsNote.clean(note), case .success(let script) = prepared else { return }
        let casting = casting
        let matching = takes.filter { !$0.isIncomplete && $0.basis == currentPlainBasis }
        let voice = casting.voiceTake.flatMap { $0 == matching.first?.take ? nil : $0 }
        start(SpeechRequest(text: script.text, references: casting.references, character: casting.character, note: note,
                            previousTakes: (voice.map { [$0] } ?? []) + matching.map(\.take), config: config),
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
        guard confirmReplacement("Replace the script with this character example?") else { return }
        projectURL = nil
        replaceSource(with: .plain(preset.line))
        screenplayMode = false
        notice = nil
        setVoice(.designed(preset.design), forSpeaker: 0)
    }

    /// Every request leaves through here, so the pronunciation dictionary
    /// rewrites every text.
    private func start(_ request: SpeechRequest, as job: Job) {
        stopPreview()
        if let refusal = generationRefusal() {
            settleAfterGeneration(.failed(refusal))
            return
        }
        var request = request
        switch job {
        case .take:
            beginRun()
        case .audition:
            stopPlayback(resetTo: 0)
        case .sections:
            break
        }
        switch job {
        case .take: generationBasis = currentPlainBasis
        case .sections(let run): generationBasis = document?.segments.first { $0.id == run.current }.map { basis(for: $0) }
        case .audition: generationBasis = nil
        }
        pronunciation = (try? PronunciationDictionary.load(from: pronunciationURL)) ?? PronunciationDictionary()
        scriptLines = (try? GenerationPlan(request))?.chunks.map(\.line) ?? []
        request.text = pronunciation.apply(to: request.text)
        telemetry = GenerationTelemetry()
        streamed = StreamedAudio()
        liveChunks = [:]
        estimate = (try? GenerationPlan(request)).map(GenerationEstimate.init(plan:))
        estimateStart = .now
        state = .loading(progress: 0, detail: "Starting engine", playing: state.isPlaying)
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
        preservePartialTake()
        run += 1
        generation?.cancel()
        generation = nil
        if case .sections(let sectionRun)? = job {
            stopPlayback(resetTo: sectionRun.audioStart)
            result = nil
            recompose()
        }
        settleAfterGeneration(audio.isEmpty ? .idle : state.afterGeneration)
    }

    private func preservePartialTake() {
        guard state.isGenerating else { return }
        let start: Int
        let section: UUID?
        switch job {
        case .take?: start = 0; section = nil
        case .sections(let run)?: start = run.audioStart; section = run.current
        default: return
        }
        guard start < audio.count else { return }
        let number = section.flatMap { sectionTakes[$0]?.nextNumber } ?? ((takes.last?.number ?? 0) + 1)
        let partial = SessionTake(number: number, note: "Stopped early", take: Take(text: generationBasis?.text ?? "", codes: []),
                                  samples: Array(audio[start...]), basis: generationBasis, isIncomplete: true)
        if let section {
            if sectionTakes[section] == nil { sectionTakes[section] = SectionTakes(first: partial) }
            else { sectionTakes[section]?.add(partial) }
            sectionSelections[section] = partial.id
        } else {
            takes.append(partial)
            selectedTakeID = partial.id
            if keptTakeID == nil { keptTakeID = partial.id }
        }
    }

    func apply(_ event: SpeechEvent) {
        estimate?.apply(event, at: Date().timeIntervalSince(estimateStart))
        switch event {
        case .loadingModel(let progress, let detail):
            state = .loading(progress: progress, detail: detail, playing: state.isPlaying)
        case .chunkStarted(let index, let count, let text):
            state = .generating(chunk: index + 1, of: count, playing: state.isPlaying)
            telemetry.frame = 0
            telemetry.chunkStart = nil
            let scripted = index < scriptLines.count && pronunciation.apply(to: scriptLines[index]) == text ? scriptLines[index] : text
            liveChunks[index] = liveSpans.count
            liveSpans.append(SentenceSpan(segmentID: currentSection, text: scripted, range: audio.count..<audio.count))
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
        case .partialAudio(_, let chunk, _), .audio(_, let chunk):
            receive(streamed.newSamples(from: event), chunk: chunk)
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
            case .sections(var sectionRun)?:
                self.result = result
                addSectionTake(result, run: sectionRun)
                if let next = sectionRun.remaining.first {
                    sectionRun.remaining.removeFirst()
                    sectionRun.current = next
                    sectionRun.audioStart = audio.count
                    sectionRun.startsPlayback = false
                    narrate(sectionRun)
                } else {
                    settleAfterGeneration(state.afterGeneration)
                    recompose()
                }
            case .audition(let design)?:
                audition = Audition(design: design, take: result.take)
                settleAfterGeneration(audio.isEmpty ? .idle : .ready)
                playAudition(result.samples)
            case nil:
                break
            }
        }
    }

    private var currentSection: NarrationDocument.Segment.ID? {
        if case .sections(let run)? = job { run.current } else { nil }
    }

    private func receive(_ samples: [Float], chunk: Int) {
        guard !samples.isEmpty else { return }
        let autoplayFrom: Int?
        switch job {
        case .take?: autoplayFrom = audio.isEmpty ? 0 : nil
        case .sections(let run)?: autoplayFrom = run.startsPlayback && audio.count == run.audioStart ? run.audioStart : nil
        case .audition?, nil: return
        }
        if let index = liveChunks[chunk] {
            let span = liveSpans[index]
            let start = span.range.isEmpty ? audio.count : span.range.lowerBound
            liveSpans[index].range = start..<(audio.count + samples.count)
        }
        audio += samples
        if state.isPlaying {
            player.feed(audio)
        } else if let autoplayFrom, playsWhileGenerating {
            startPlayback(from: autoplayFrom)
        }
    }

    private func playAudition(_ samples: [Float]) {
        do {
            try previewPlayer.play(samples, from: 0)
        } catch {
            notice = "Couldn't start audio output: \(error.localizedDescription)"
        }
    }

    /// The take's sentences in the script's own words where the dictionary rewrote them.
    private func sentences(of result: SpeechResult) -> [NarrationTake.Sentence] {
        let texts = SentenceLocator.scriptTexts(spoken: result.sentences.map(\.text), original: scriptLines, rewrite: pronunciation.apply)
        return zip(texts, result.sentences).map { NarrationTake.Sentence(text: $0, frames: $1.frames) }
    }

    private func addTake(_ result: SpeechResult, note: String?, recordsVoice: VoiceChoice?) {
        if let recordsVoice, sessionVoices[recordsVoice] == nil { sessionVoices[recordsVoice] = result.take }
        let take = SessionTake(number: (takes.last?.number ?? 0) + 1, note: note, take: result.take, samples: result.samples, sentences: sentences(of: result), basis: recordedBasis(for: recordsVoice))
        takes.append(take)
        selectedTakeID = take.id
        if keptTakeID == nil { keptTakeID = take.id }
    }

    private func addSectionTake(_ result: SpeechResult, run: SectionRun) {
        if let choice = run.recordsVoice, sessionVoices[choice] == nil { sessionVoices[choice] = result.take }
        let take = SessionTake(number: sectionTakes[run.current]?.nextNumber ?? 1, note: nil, take: result.take, samples: result.samples, sentences: sentences(of: result), basis: recordedBasis(for: run.recordsVoice))
        sectionSelections[run.current] = take.id
        if sectionTakes[run.current] == nil {
            sectionTakes[run.current] = SectionTakes(first: take)
        } else {
            sectionTakes[run.current]?.add(take)
        }
    }

    /// Auditions a take without changing the export selection.
    func selectTake(_ id: SessionTake.ID) {
        guard !state.isGenerating else { return }
        if document != nil {
            guard let section = sectionTakes.first(where: { $0.value.takes.contains { $0.id == id } })?.key else { return }
            stopPlayback(resetTo: nil)
            stopPreview()
            sectionSelections[section] = id
            if let take = sectionTakes[section]?.takes.first(where: { $0.id == id }) {
                previewingTakeID = id
                playAudition(take.samples)
            }
            return
        }
        guard let take = takes.first(where: { $0.id == id }) else { return }
        stopPreview()
        stopPlayback(resetTo: 0)
        selectedTakeID = id
        audio = take.samples
        startPlayback(from: 0)
    }

    func useSelectedTakeForExport() {
        guard !state.isGenerating, let selected = visibleSelectedTakeID else { return }
        if let section = selectedSegmentID, document != nil {
            sectionTakes[section]?.chosenID = selected
            stopPreview()
            stopPlayback(resetTo: 0)
            recompose()
        } else {
            keptTakeID = selected
        }
    }



    private func generationEnded(run ended: Int, error: Error?) {
        guard ended == run else { return }
        generation = nil
        refreshEngineState()
        if let error {
            preservePartialTake()
            stopPlayback(resetTo: nil)
            settleAfterGeneration(.failed(Self.describe(error)))
            recompose()
        } else if state.isGenerating {
            settleAfterGeneration(audio.isEmpty ? .failed("The engine stopped without producing audio.") : state.afterGeneration)
            recompose()
        }
    }

    /// Swaps in the engine once setup has put its files in place; a run under way stops first.
    func replaceSynthesizer(_ next: any SpeechSynthesizer) {
        stopGeneration()
        synthesizer = next
        refreshEngineState()
    }

    /// The setup window's last step: a short line in the editor, ready for Generate.
    func prepareFirstLine() {
        replaceSource(with: .plain(Examples.firstLine.text))
        screenplayMode = false
        notice = nil
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
        stopPreview()
        if state.isPlaying {
            stopPlayback(resetTo: nil)
        } else if canPlay {
            startPlayback(from: playhead >= audio.count ? 0 : playhead)
        }
    }

    func seek(toFraction fraction: Double) {
        guard canPlay else { return }
        seek(toSample: Int(fraction * Double(audio.count)))
    }

    func seek(toSample sample: Int) {
        guard canPlay else { return }
        let sample = min(audio.count - 1, max(0, sample))
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
            if ImportFormat(url: url)?.opensAsDocument == true {
                open(try NarrationDocument.importFile(at: url))
            } else {
                let text = try ScriptImporter.importFile(at: url)
                guard confirmReplacement("Import another script?") else { return }
                projectURL = nil
                replaceSource(with: .plain(text))
                screenplayMode = Screenplay.looksLikeScreenplay(script)
            }
            notice = nil
        } catch {
            notice = error.localizedDescription
        }
    }

    func open(_ url: URL) {
        guard url.scheme == "dramastudio" else {
            if url.pathExtension.lowercased() == StudioProject.fileExtension {
                openProject(from: url)
            } else {
                importScript(from: url)
            }
            return
        }
        switch url.host() {
        case "generate": generate()
        case "stop": stopGeneration()
        case "play": togglePlayback()
        default: notice = "Drama Studio doesn't understand the link \(url.absoluteString)."
        }
    }

    func insert(_ cue: PerformanceCue) {
        apply(cue.insert(into: editorText, at: selectedCharacterRange ?? editorText.count..<editorText.count))
    }

    func insert(_ example: ExampleLine) {
        apply(example.insert(into: editorText, at: selectedCharacterRange ?? editorText.count..<editorText.count))
    }

    private func apply(_ edit: (text: String, selection: Range<Int>)) {
        editorText = edit.text
        let text = editorText
        let start = text.index(text.startIndex, offsetBy: edit.selection.lowerBound)
        let end = text.index(text.startIndex, offsetBy: edit.selection.upperBound)
        selection = TextSelection(range: start..<end)
    }

    private var selectedCharacterRange: Range<Int>? {
        let text = editorText
        guard let selection, case .selection(let range) = selection.indices,
              range.lowerBound >= text.startIndex, range.upperBound <= text.endIndex else { return nil }
        let lower = text.distance(from: text.startIndex, to: range.lowerBound)
        return lower..<(lower + text.distance(from: range.lowerBound, to: range.upperBound))
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

    @discardableResult
    func encodeVoice(clip: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        if let refusal = generationRefusal() { throw GenerationRefused(message: refusal) }
        let voice = try await synthesizer.encodeReference(audioURL: clip, transcript: transcript, name: name)
        try Task.checkCancellation()
        try library.add(voice)
        return voice
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
        previewingTakeID = nil
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

    func export(to url: URL, as format: ExportFormat) {
        report(to: url) { try AudioExporter.write(exportSamples, to: url, as: format) }
    }

    func exportSections(to folder: URL, as format: ExportFormat) {
        guard let document else { return }
        report(to: folder) { try SegmentExport.export(takes: narrationTakes, document: document, samples: audio, to: folder, format: format) }
    }

    func exportCaptions(to url: URL, as format: Captions.Format) {
        report(to: url) { try Data(Captions.render(captionTakes, as: format).utf8).write(to: url, options: .atomic) }
    }

    func exportSlidePackage(to folder: URL, as format: ExportFormat) {
        guard let document else { return }
        report(to: folder) { try SlidePackage.export(takes: narrationTakes, document: document, samples: audio, to: folder, format: format) }
    }

    private func report(to url: URL, _ export: () throws -> Void) {
        do {
            try export()
            exportedURL = url
            notice = nil
        } catch {
            exportedURL = nil
            notice = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Library voices by speaker; surprise and designed voices live only in the session.
    private var libraryCasting: [Int: UUID] {
        speakerVoices.compactMapValues { if case .library(let id) = $0 { id } else { nil } }
    }

    var recoveryRevision: ProjectRevision { projectRevision }

    var projectSnapshot: ProjectSnapshot {
        var project = StudioProject(title: title, script: script, document: document, casting: libraryCasting,
                                    expressiveness: expressiveness, quickPreview: quickPreview, takes: captionTakes)
        project.archive = ProjectArchive(plainTakes: takes.map(ArchivedTake.init),
                                         sectionTakes: sectionTakes.mapValues { $0.takes.map(ArchivedTake.init) },
                                         sectionExports: sectionTakes.mapValues(\.chosenID),
                                         selectedTake: selectedTakeID, exportTake: keptTakeID,
                                         sectionSelections: sectionSelections, selectedSegment: selectedSegmentID, originalProjectURL: projectURL, voices: speakerVoices,
                                         sessionVoices: sessionVoices, screenplayMode: screenplayMode)
        let all = takes + sectionTakes.values.flatMap(\.takes)
        return ProjectSnapshot(project: project, audio: exportSamples,
                               archivedAudio: Dictionary(uniqueKeysWithValues: all.map { ($0.id, $0.samples) }))
    }

    @discardableResult
    func saveProject(to url: URL) -> Bool {
        guard !state.isGenerating else {
            notice = "Stop generation before saving the project."
            return false
        }
        do {
            try projectSnapshot.save(to: url)
            projectURL = url
            savedRevision = projectRevision
            rememberProject(url)
            notice = nil
            return true
        } catch {
            notice = "Couldn't save the project: \(error.localizedDescription)"
            return false
        }
    }

    @discardableResult
    func openProject(from url: URL, recovering: Bool = false) -> Bool {
        do {
            let project = try StudioProject.load(from: url)
            let audio = try project.audio(in: url)
            let restored = try project.archive?.restoreTakes(in: url) ?? [:]
            guard recovering || confirmReplacement("Open another project?") else { return false }
            load(project, audio: audio)
            if let archive = project.archive {
                takes = archive.plainTakes.compactMap { restored[$0.id] }
                sectionTakes = archive.sectionTakes.compactMapValues { entries in
                    let values = entries.compactMap { restored[$0.id] }
                    guard let first = values.first else { return nil }
                    var group = SectionTakes(first: first)
                    for take in values.dropFirst() { group.add(take) }
                    return group
                }
                for (id, chosen) in archive.sectionExports where sectionTakes[id]?.takes.contains(where: { $0.id == chosen }) == true {
                    sectionTakes[id]?.chosenID = chosen
                }
                selectedTakeID = archive.selectedTake
                keptTakeID = archive.exportTake
                sectionSelections = archive.sectionSelections
                if let selected = archive.selectedSegment, document?.segments.contains(where: { $0.id == selected }) == true { selectedSegmentID = selected }
                speakerVoices = archive.voices
                sessionVoices = archive.sessionVoices
                screenplayMode = archive.screenplayMode
                if document != nil { recompose() }
                else { self.audio = takes.first { $0.id == selectedTakeID }?.samples ?? keptTake?.samples ?? [] }
                state = self.audio.isEmpty ? .idle : .ready
            }
            projectURL = recovering ? project.archive?.originalProjectURL : url
            savedRevision = recovering ? nil : projectRevision
            if !recovering { rememberProject(url) }
            notice = recovering ? "Recovered unsaved work. Save the project to keep it." : nil
            return true
        } catch {
            notice = "Couldn't open the project: \(Self.describe(error))"
            return false
        }
    }

    private func load(_ project: StudioProject, audio: [Float]) {
        replaceSource(with: project.document.map(ScriptSource.document) ?? .plain(project.script))
        clearNarration()
        speakerVoices = project.casting.mapValues { .library($0) }
        sessionVoices = [:]
        expressiveness = project.expressiveness
        quickPreview = project.quickPreview
        screenplayMode = project.document == nil && Screenplay.looksLikeScreenplay(project.script)
        for stored in project.takes where stored.sampleRange.upperBound <= audio.count && !stored.sampleRange.isEmpty {
            let take = SessionTake(number: 1, note: nil, take: stored.take ?? Take(text: "", codes: []),
                                   samples: Array(audio[stored.sampleRange]), sentences: stored.sentences)
            if project.document != nil {
                sectionTakes[stored.segmentID] = SectionTakes(first: take)
            } else {
                takes = [take]
                selectedTakeID = take.id
                self.audio = take.samples
            }
        }
        recompose()
        state = self.audio.isEmpty ? .idle : .ready
    }

    private func rememberProject(_ url: URL) {
        recentProjects = RecentProjects.updated(recentProjects, adding: url)
        RecentProjects.save(recentProjects)
    }

    static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}

/// What a generation run is for.
private enum Job {
    case take(note: String?, recordsVoice: VoiceChoice?)
    case audition(VoiceDesign)
    case sections(SectionRun)
}

/// A document run: the section being narrated and the ones still to come.
private struct SectionRun {
    var current: NarrationDocument.Segment.ID
    var remaining: [NarrationDocument.Segment.ID]
    /// Where the current section's audio begins in the transport.
    var audioStart: Int
    /// Only the run's first section starts playback by itself; a later one
    /// continues it, or stays quiet when the listener paused.
    var startsPlayback: Bool
    var recordsVoice: VoiceChoice?
}

struct TakesBasis: Equatable, Codable, Sendable {
    var text: String
    var voices: [Int: VoiceChoice]
    var voiceAnchors: [VoiceChoice: String]? = nil
}

/// A line spoken in a designed voice before it is cast.
struct Audition: Equatable {
    let design: VoiceDesign
    let take: Take
}

/// One finished read of the script, kept so it can be replayed, retaken, or exported.
struct SessionTake: Identifiable, Equatable {
    var id = UUID()
    let number: Int
    let note: String?
    let take: Take
    let samples: [Float]
    var sentences: [NarrationTake.Sentence] = []
    var basis: TakesBasis? = nil
    var isIncomplete = false
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
