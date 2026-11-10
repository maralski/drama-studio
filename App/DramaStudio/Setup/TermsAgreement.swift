// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// The user's agreement to the Fish Audio Community License's use
/// restrictions. Section IV(b) of that license requires anyone who makes
/// speech generation or voice cloning available to others to bind them by
/// enforceable terms, so until the user agrees no Fish Audio file downloads
/// and neither Drama 3 nor its voice codec runs. An engine whose license
/// asks for no agreement can be set up and used without it.
struct TermsAgreement {
    /// Raise it when the terms or their summary change; everyone is asked again.
    static let version = "fish-audio-community-license-1.0"
    static let versionKey = "termsAcceptedVersion"
    static let dateKey = "termsAcceptedDate"

    static let refusal = "Drama Studio won't generate speech until you agree to the Fish Audio Community License terms. Open Drama Studio and click I Agree in its setup window."

    /// From NOTICE: what the license forbids doing with the model or its output.
    static let restrictions = [
        "Imitate the voice of an identifiable person without that person's consent (for a minor, a parent's or guardian's; for someone who has died, the rights holder's), or another legal right to do so.",
        "Impersonate a person or organization to deceive, defraud or harass anyone, or place calls that break telemarketing or robocall laws.",
        "Generate sexual content involving a minor or a voice presented as a minor's, or use any real person's voice without their consent.",
        "Remove or obscure any watermark or provenance signal the model applies.",
        "Create, train, distill or improve another speech, audio or voice generation model, or a foundational generative model.",
    ]

    /// From NOTICE: who may use it commercially, and the hosted-service limit.
    static let limits = [
        "Commercial use is licensed only while your annual revenue is below USD 500,000.",
        "Offering text to speech or voice cloning to other people as a hosted service always needs a separate license from Fish Audio. Drama Studio's HTTP API and MCP server listen only on this Mac; exposing them to others would make one.",
    ]

    static let licenseWebURL = SpeechEngineID.drama3.install.license.url

    /// The full license text inside the app bundle; the web copy when running from a build folder.
    static var licenseURL: URL { Bundle.main.url(forResource: "LICENSE-FISH-AUDIO", withExtension: nil) ?? licenseWebURL }

    let defaults: UserDefaults

    var isAccepted: Bool { defaults.string(forKey: Self.versionKey) == Self.version }

    var acceptedDate: Date? { isAccepted ? defaults.object(forKey: Self.dateKey) as? Date : nil }

    func accept(at date: Date = .now) {
        defaults.set(Self.version, forKey: Self.versionKey)
        defaults.set(date, forKey: Self.dateKey)
    }

    /// Whether `engine` may download and run: always, unless its license asks for this agreement.
    func allows(_ engine: SpeechEngineID) -> Bool {
        !engine.install.license.requiresAgreement || isAccepted
    }
}

/// Which engine the setup window installs. Settings keep the choice between
/// launches; within one, this object holds it, so a choice just made never
/// depends on reading a setting back.
@MainActor
final class SetupEngineChoice {
    nonisolated static let key = "setupEngine"

    private let defaults: UserDefaults
    /// Nil until the user chooses.
    private(set) var engine: SpeechEngineID?
    /// The choice was made this launch, so its engine becomes the one new takes use once it is in place.
    private(set) var madeThisLaunch = false

    init(defaults: UserDefaults) {
        self.defaults = defaults
        engine = Self.stored(in: defaults)
    }

    /// The choice the settings hold. An install from before there was a
    /// choice set up Drama 3 and recorded that by agreeing to its terms, so
    /// it counts as having chosen Drama 3.
    nonisolated static func stored(in defaults: UserDefaults) -> SpeechEngineID? {
        if let stored = defaults.string(forKey: key) { return SpeechEngineID(rawValue: stored) }
        return defaults.string(forKey: TermsAgreement.versionKey) == nil ? nil : .drama3
    }

    /// Records the choice. Nil records that the user is choosing again, which
    /// an earlier agreement to the terms must not answer for them.
    func set(_ engine: SpeechEngineID?) {
        self.engine = engine
        madeThisLaunch = engine != nil
        defaults.set(engine?.rawValue ?? "", forKey: Self.key)
    }
}

/// A generation the setup state does not allow yet.
struct GenerationRefused: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}
