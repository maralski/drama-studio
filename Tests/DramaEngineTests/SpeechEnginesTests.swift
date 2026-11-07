// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Synchronization
import XCTest
@testable import DramaEngine

/// A voice store with one clip and a log of stored encodings.
private actor ClipStore: VoiceEncodingStore {
    let clips: [UUID: [Float]]
    private(set) var stored: [(VoiceEncoding, UUID)] = []
    init(clips: [UUID: [Float]]) { self.clips = clips }
    func clip(for id: UUID) async throws -> [Float]? { clips[id] }
    func storeEncoding(_ encoding: VoiceEncoding, for id: UUID) async throws { stored.append((encoding, id)) }
}

final class SpeechEnginesTests: XCTestCase {
    /// Mock factories that count how many engines they made and let the test watch the instances die.
    private final class Watched: Sendable {
        let made = Mutex<[SpeechEngineID: Int]>([:])
        let live = Mutex<[SpeechEngineID: WeakBox]>([:])
        struct WeakBox: Sendable { weak var engine: MockSynthesizer? }
        func factory(_ engine: SpeechEngineID) -> SpeechEngines.Factory {
            SpeechEngines.Factory(description: MockSynthesizer(engine: engine, pace: .instant).engineDescription) { [self] in
                let mock = MockSynthesizer(engine: engine, pace: .instant)
                made.withLock { $0[engine, default: 0] += 1 }
                live.withLock { $0[engine] = WeakBox(engine: mock) }
                return mock
            }
        }
        func isAlive(_ engine: SpeechEngineID) -> Bool { live.withLock { $0[engine]?.engine != nil } }
    }

    private func result(of engines: SpeechEngines, _ request: SpeechRequest) async throws -> SpeechResult {
        for try await event in engines.synthesize(request) {
            if case .finished(let result) = event { return result }
        }
        throw XCTSkip("no result")
    }

    func testSelectionPicksTheEngineAndOnlyOneStaysResident() async throws {
        let watched = Watched()
        let engines = try SpeechEngines(selected: .drama3, factories: [.drama3: watched.factory(.drama3), .ming: watched.factory(.ming)])
        XCTAssertEqual(engines.available, [.drama3, .ming])
        XCTAssertEqual(engines.engineDescription.engine, .drama3)

        let first = try await result(of: engines, SpeechRequest(text: "One line.", config: .unchecked))
        XCTAssertEqual(first.take.context.engine, .drama3)
        XCTAssertEqual(watched.made.withLock { $0 }, [.drama3: 1])

        try engines.select(.ming)
        XCTAssertEqual(engines.engineDescription.engine, .ming)
        let second = try await result(of: engines, SpeechRequest(text: "Another line.", config: .unchecked))
        XCTAssertEqual(second.take.context.engine, .ming)
        XCTAssertEqual(watched.made.withLock { $0 }, [.drama3: 1, .ming: 1])
        XCTAssertFalse(watched.isAlive(.drama3), "the Drama engine was dropped before Ming loaded")
        XCTAssertTrue(watched.isAlive(.ming))

        try engines.select(.drama3)
        _ = try await result(of: engines, SpeechRequest(text: "Back again.", config: .unchecked))
        XCTAssertEqual(watched.made.withLock { $0 }, [.drama3: 2, .ming: 1], "switching back builds Drama again")
        XCTAssertFalse(watched.isAlive(.ming))
    }

    func testSelectingAnEngineWithoutAFactoryFails() throws {
        let watched = Watched()
        XCTAssertThrowsError(try SpeechEngines(selected: .ming, factories: [.drama3: watched.factory(.drama3)]))
        let engines = try SpeechEngines(selected: .drama3, factories: [.drama3: watched.factory(.drama3)])
        XCTAssertThrowsError(try engines.select(.ming)) { error in
            XCTAssertEqual((error as? EngineError)?.description, "Ming isn't installed")
        }
        XCTAssertEqual(engines.selected, .drama3)
    }

    func testAnOldVoiceIsEncodedForTheNewEngineFromItsClipOnceAndStoredBack() async throws {
        let watched = Watched()
        let voice = ReferenceVoice(name: "Old", transcript: "The clip says this.", encodings: [.drama3(codes: MockSynthesizer.codes(frameCount: 20))], clipSampleCount: 44_100)
        let store = ClipStore(clips: [voice.id: (0..<44_100).map { Float(sin(Double($0) * 0.03)) }])
        let engines = try SpeechEngines(selected: .ming, factories: [.ming: watched.factory(.ming)], voices: store)

        let first = try await result(of: engines, SpeechRequest(text: "Say it.", references: [voice], config: .unchecked))
        XCTAssertEqual(first.take.context.engine, .ming)
        let stored = await store.stored
        XCTAssertEqual(stored.count, 1)
        XCTAssertEqual(stored.first?.1, voice.id)
        guard case .ming(let reference)? = stored.first?.0 else { return XCTFail("a Ming encoding was stored, got \(String(describing: stored.first?.0))") }
        XCTAssertEqual(reference.sampleCount, 44_100)
        XCTAssertEqual(reference.transcript, voice.transcript)

        var updated = voice
        updated.store(stored[0].0)
        _ = try await result(of: engines, SpeechRequest(text: "Say it again.", references: [updated], config: .unchecked))
        let after = await store.stored.count
        XCTAssertEqual(after, 1, "a voice that already has the encoding is not encoded again")
    }

    func testAVoiceWithoutAClipCannotBeEncodedForAnotherEngine() async throws {
        let watched = Watched()
        let voice = ReferenceVoice(name: "Clipless", transcript: "Hi.", encodings: [.drama3(codes: MockSynthesizer.codes(frameCount: 5))])
        let engines = try SpeechEngines(selected: .ming, factories: [.ming: watched.factory(.ming)], voices: ClipStore(clips: [:]))
        do {
            _ = try await result(of: engines, SpeechRequest(text: "Say it.", references: [voice], config: .unchecked))
            XCTFail("expected noClip")
        } catch {
            XCTAssertEqual((error as? EngineError)?.description, "the voice Clipless has no stored clip to encode for this engine")
        }
        XCTAssertEqual(watched.made.withLock { $0 }, [:], "nothing loads for a request that cannot run")
    }

    func testEarlierTakesFromTheOtherEngineAreReportedNotUsed() async throws {
        let watched = Watched()
        let engines = try SpeechEngines(selected: .ming, factories: [.ming: watched.factory(.ming)])
        let drama = Take(text: "Earlier.", context: .drama3(codes: MockSynthesizer.codes(frameCount: 8)))
        let result = try await result(of: engines, SpeechRequest(text: "Now this.", previousTakes: [drama], config: .unchecked))
        XCTAssertEqual(result.notices, [.otherEngineTakes(.drama3, count: 1)])
        XCTAssertEqual(result.notices.map(\.message), ["1 earlier take from Drama 3 can't carry the voice into this one."])
    }

    func testTheOutlineAndEstimateCountTheEnginesOwnSteps() throws {
        let request = SpeechRequest(text: "This sentence is about forty bytes long. And this one is a little shorter.")
        let drama = try SpeechOutline(request, engine: .drama3)
        let ming = try SpeechOutline(request, engine: .ming)
        XCTAssertEqual(drama.lines, ming.lines)
        XCTAssertEqual(drama.lines.count, 2)
        XCTAssertEqual(drama.expectedSeconds, ming.expectedSeconds)
        XCTAssertEqual(GenerationEstimate(outline: drama).expectedFrames, GenerationEstimate(lines: drama.lines).expectedFrames, "Drama's estimate is unchanged")
        let steps = GenerationEstimate(outline: ming)
        XCTAssertEqual(steps.expectedFrames, ming.expectedSeconds.map { Int(($0 * 44_100 / 3528).rounded()) })
        XCTAssertEqual(steps.framesPerSecond, 60)
        XCTAssertEqual(GenerationEstimate(outline: drama).framesPerSecond, 35)
    }

    func testTheOutlineReportsWhatMingCannotPerform() throws {
        let outline = try SpeechOutline(SpeechRequest(text: "<whisper>Run.</whisper> [laughs] Now!"), engine: .ming)
        XCTAssertEqual(outline.lines, ["<whisper>Run.</whisper>", "[laughs] Now!"])
        XCTAssertEqual(outline.notices, [.unsupportedCue("<whisper>")])
        XCTAssertEqual(try SpeechOutline(SpeechRequest(text: "<whisper>Run.</whisper> [laughs] Now!"), engine: .drama3).notices, [])
    }
}

private extension GenerationConfig {
    static var unchecked: GenerationConfig {
        var config = GenerationConfig()
        config.wordVerification = .uncheckedPreview
        config.maxRetries = 0
        return config
    }
}
