// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The user's agreement to the Fish Audio Community License's use
/// restrictions. Section IV(b) of that license requires anyone who makes
/// speech generation or voice cloning available to others to bind them by
/// enforceable terms, so nothing downloads or speaks until the user agrees.
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

    static let licenseWebURL = URL(string: "https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE")!

    /// The full license text inside the app bundle; the web copy when running from a build folder.
    static var licenseURL: URL { Bundle.main.url(forResource: "LICENSE-FISH-AUDIO", withExtension: nil) ?? licenseWebURL }

    let defaults: UserDefaults

    var isAccepted: Bool { defaults.string(forKey: Self.versionKey) == Self.version }

    var acceptedDate: Date? { isAccepted ? defaults.object(forKey: Self.dateKey) as? Date : nil }

    func accept(at date: Date = .now) {
        defaults.set(Self.version, forKey: Self.versionKey)
        defaults.set(date, forKey: Self.dateKey)
    }
}

/// A generation the setup state does not allow yet.
struct GenerationRefused: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}
