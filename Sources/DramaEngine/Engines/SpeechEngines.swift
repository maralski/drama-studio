// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Synchronization

/// Where reference clips live, so an engine can encode a voice another
/// engine cloned. A library of voice files conforms.
public protocol VoiceEncodingStore: Sendable {
    /// The voice's clip as mono 44.1 kHz samples, or nil when the store has none.
    /// A store may recreate a missing clip, for example from the voice's Drama 3 codes.
    func clip(for id: ReferenceVoice.ID) async throws -> [Float]?
    /// Adds `encoding` to the stored voice, leaving its clip and other encodings alone.
    func storeEncoding(_ encoding: VoiceEncoding, for id: ReferenceVoice.ID) async throws
}

/// Every engine the app or server can run, with at most one resident at a
/// time. It is a ``SpeechSynthesizer``: every call goes to the selected
/// engine, so callers keep the one path they had with one engine. Engines
/// are built from factories when first needed and dropped before another
/// loads; no call returns an engine, so nothing else can keep dropped
/// weights alive. Calls run one at a time, so a run on one engine finishes
/// before the other loads.
public actor SpeechEngines: SpeechSynthesizer {
    /// How to make an engine, with what it reports before it is made.
    public struct Factory: Sendable {
        public var description: EngineDescription
        public var make: @Sendable () -> any SpeechSynthesizer
        public init(description: EngineDescription, make: @escaping @Sendable () -> any SpeechSynthesizer) {
            self.description = description
            self.make = make
        }

        /// A factory for a Drama 3 snapshot.
        public static func drama3(location: ModelLocation, codecDirectory: URL, tuningDirectory: URL = DramaSynthesizer.defaultTuningDirectory) -> Factory {
            let engine = { DramaSynthesizer(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory) }
            return Factory(description: engine().engineDescription, make: engine)
        }

        /// A factory for a Ming checkpoint.
        public static func ming(location: MingLocation) -> Factory {
            let engine = { MingSynthesizer(location: location) }
            return Factory(description: engine().engineDescription, make: engine)
        }

        /// A mock standing in for `engine`.
        public static func mock(_ engine: SpeechEngineID, pace: MockSynthesizer.Pace = .realistic) -> Factory {
            let make = { MockSynthesizer(engine: engine, pace: pace) }
            return Factory(description: make().engineDescription, make: make)
        }
    }

    private struct State {
        var factories: [SpeechEngineID: Factory]
        var selected: SpeechEngineID
        var resident: (engine: SpeechEngineID, synthesizer: any SpeechSynthesizer)?
    }

    private let state: Mutex<State>
    private let voices: (any VoiceEncodingStore)?
    private var tail: Task<Void, Never>?

    /// - Parameters:
    ///   - selected: The engine new requests use. It needs a factory.
    ///   - factories: One per installed engine.
    ///   - voices: Where clips live, so a voice cloned by one engine can be encoded for another and the encoding kept.
    public init(selected: SpeechEngineID, factories: [SpeechEngineID: Factory], voices: (any VoiceEncodingStore)? = nil) throws {
        guard factories[selected] != nil else { throw EngineError.engineUnavailable(selected) }
        state = Mutex(State(factories: factories, selected: selected))
        self.voices = voices
    }

    /// A registry of one engine that is already built, for tests and hosts with one engine.
    public init(only synthesizer: any SpeechSynthesizer) {
        let engine = synthesizer.engine
        state = Mutex(State(factories: [engine: Factory(description: synthesizer.engineDescription) { synthesizer }], selected: engine,
                            resident: (engine, synthesizer)))
        voices = nil
    }

    /// Engines with a factory.
    public nonisolated var available: Set<SpeechEngineID> { state.withLock { Set($0.factories.keys) } }
    /// What `engine` reports, without building it; nil without a factory.
    public nonisolated func description(of engine: SpeechEngineID) -> EngineDescription? { state.withLock { $0.factories[engine]?.description } }
    /// The engine new requests use.
    public nonisolated var selected: SpeechEngineID { state.withLock { $0.selected } }

    /// Changes the engine new requests use. Nothing loads or unloads until a request needs it.
    /// - Throws: ``EngineError/engineUnavailable(_:)`` when `engine` has no factory.
    public nonisolated func select(_ engine: SpeechEngineID) throws {
        try state.withLock {
            guard $0.factories[engine] != nil else { throw EngineError.engineUnavailable(engine) }
            $0.selected = engine
        }
    }

    /// Adds or replaces an engine's factory, as setup does once its files are in place.
    /// A resident engine of that kind is dropped once its current run ends.
    public func register(_ engine: SpeechEngineID, _ factory: Factory) async {
        await serial {
            self.state.withLock {
                $0.factories[engine] = factory
                if $0.resident?.engine == engine { $0.resident = nil }
            }
        }
    }

    public nonisolated var engineDescription: EngineDescription {
        state.withLock { $0.factories[$0.selected]!.description }
    }

    public nonisolated var isTuned: Bool? {
        state.withLock { $0.resident?.engine == $0.selected ? $0.resident?.synthesizer.isTuned : nil }
    }

    public var isLoaded: Bool {
        get async {
            guard let resident = state.withLock({ $0.resident }), resident.engine == selected else { return false }
            return await resident.synthesizer.isLoaded
        }
    }

    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        try await load(selected, onPhase: onPhase)
    }

    /// Loads `engine` and makes it the resident one, whatever is selected.
    public func load(_ engine: SpeechEngineID, onPhase: @escaping @Sendable (String) -> Void = { _ in }) async throws {
        try await serial { try await self.resident(engine).load(onPhase: onPhase) }
    }

    public func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding {
        try await serial { try await self.resident(self.selected).encodeVoice(samples: samples, transcript: transcript) }
    }

    /// Runs the request on the selected engine. First, every reference voice
    /// without an encoding for that engine is encoded from its stored clip
    /// and the encoding is written back, so it is computed once. A voice
    /// with no clip fails with ``EngineError/noClip(voiceName:)`` before
    /// anything loads.
    public nonisolated func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.serial {
                        let engine = self.selected
                        var request = request
                        try await self.encodeMissingReferences(in: &request, for: engine)
                        for try await event in try self.resident(engine).synthesize(request) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private nonisolated func encodeMissingReferences(in request: inout SpeechRequest, for engine: SpeechEngineID) async throws {
        for index in request.references.indices where request.references[index].encodings[engine] == nil {
            var voice = request.references[index]
            guard let clip = try await voices?.clip(for: voice.id) else { throw EngineError.noClip(voiceName: voice.name) }
            let encoding = try await resident(engine).encodeVoice(samples: clip, transcript: voice.transcript)
            voice.store(encoding)
            try await voices?.storeEncoding(encoding, for: voice.id)
            request.references[index] = voice
        }
    }

    /// The one resident engine, made from its factory after any other is dropped.
    private nonisolated func resident(_ engine: SpeechEngineID) throws -> any SpeechSynthesizer {
        try state.withLock { state in
            if let resident = state.resident, resident.engine == engine { return resident.synthesizer }
            guard let factory = state.factories[engine] else { throw EngineError.engineUnavailable(engine) }
            state.resident = nil
            let synthesizer = factory.make()
            state.resident = (engine, synthesizer)
            return synthesizer
        }
    }

    /// Runs `work` after every earlier call has finished.
    private func serial<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tail
        let task = Task {
            await previous?.value
            return try await work()
        }
        tail = Task { _ = try? await task.value }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }

    private func serial(_ work: @escaping @Sendable () async -> Void) async {
        _ = try? await serial { () async throws -> Void in await work() }
    }
}
