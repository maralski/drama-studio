// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

final class AppCastingTests: XCTestCase {
    private let voice = ReferenceVoice(name: "Ann", transcript: "a", codes: [[1]])
    private let design = VoiceDesign(name: "Nana", description: "warm elderly woman")
    private let take = Take(text: "Hello.", codes: [[7]])
    private let solo = [Speaker(index: 0, name: "Narrator")]
    private let duo = [Speaker(index: 0, name: "A"), Speaker(index: 1, name: "B")]

    func testSoloLibraryVoiceIsAReference() {
        let casting = Casting(speakers: solo, assignments: [0: .library(voice.id)], library: [voice])
        XCTAssertEqual(casting.references, [voice])
        XCTAssertNil(casting.character)
        XCTAssertNil(casting.voiceTake)
        XCTAssertNil(casting.recordsVoice)
    }

    func testSoloModelDefaultCarriesNothing() {
        for assignments: [Int: VoiceChoice] in [[:], [0: .modelDefault], [0: .library(UUID())]] {
            XCTAssertEqual(Casting(speakers: solo, assignments: assignments, library: [voice]), Casting(speakers: [], assignments: [:], library: []))
        }
    }

    func testSoloSurpriseRecordsItsFirstTakeThenReusesIt() {
        let fresh = Casting(speakers: solo, assignments: [0: .surprise], library: [])
        XCTAssertEqual(fresh.recordsVoice, .surprise)
        XCTAssertNil(fresh.voiceTake)
        XCTAssertNil(fresh.character)
        XCTAssertEqual(fresh.references, [])

        let recorded = Casting(speakers: solo, assignments: [0: .surprise], library: [], sessionVoices: [.surprise: take])
        XCTAssertEqual(recorded.voiceTake, take)
        XCTAssertNil(recorded.recordsVoice)
    }

    func testSoloDesignedVoiceIsTheCharacterPlusItsSessionVoice() {
        let fresh = Casting(speakers: solo, assignments: [0: .designed(design)], library: [], sessionVoices: [.surprise: take])
        XCTAssertEqual(fresh.character, design)
        XCTAssertNil(fresh.voiceTake, "another choice's session voice is not this one")
        XCTAssertEqual(fresh.recordsVoice, .designed(design))

        let recorded = Casting(speakers: solo, assignments: [0: .designed(design)], library: [], sessionVoices: [.designed(design): take])
        XCTAssertEqual(recorded.character, design)
        XCTAssertEqual(recorded.voiceTake, take)
        XCTAssertNil(recorded.recordsVoice)
    }

    func testSeveralSpeakersIgnoreSurpriseAndDesignedVoices() {
        let casting = Casting(speakers: duo, assignments: [0: .designed(design), 1: .surprise], library: [], sessionVoices: [.surprise: take])
        XCTAssertEqual(casting.ignored.map(\.speaker.index), [0, 1])
        XCTAssertEqual(casting.ignored.map(\.reason), [.needsLibrary, .needsLibrary])
        XCTAssertTrue(casting.ignored[0].message.contains("library"))
        XCTAssertNil(casting.character)
        XCTAssertNil(casting.voiceTake)
        XCTAssertNil(casting.recordsVoice)

        let mixed = Casting(speakers: duo, assignments: [0: .library(voice.id), 1: .designed(design)], library: [voice])
        XCTAssertEqual(mixed.references, [voice])
        XCTAssertEqual(mixed.ignored.map(\.reason), [.needsLibrary])
    }

    func testAnUnusedDesignedVoiceBreaksTheLibraryRun() {
        let casting = Casting(speakers: duo, assignments: [0: .surprise, 1: .library(voice.id)], library: [voice])
        XCTAssertEqual(casting.references, [])
        XCTAssertEqual(casting.ignored.map(\.reason), [.needsLibrary, .gap])
    }
}
