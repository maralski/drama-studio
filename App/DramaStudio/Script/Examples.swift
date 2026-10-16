import DramaEngine
import Foundation

/// A line that shows off one opening style.
struct ExampleLine: Identifiable, Hashable {
    let title: String
    let text: String
    var id: String { title }

    /// Puts the line on a line of its own at the cursor, replacing any selection.
    func insert(into script: String, at range: Range<Int>) -> (text: String, selection: Range<Int>) {
        let chars = Array(script)
        let before = String(chars[..<range.lowerBound])
        let after = String(chars[range.upperBound...])
        let lead = before.isEmpty || before.hasSuffix("\n") ? "" : "\n"
        let trail = after.isEmpty || after.hasPrefix("\n") ? "" : "\n"
        let end = before.count + lead.count + text.count
        return (before + lead + text + trail + after, end..<end)
    }
}

/// A character from the Drama 3 Space's cast list, with a line to try.
struct CastPreset: Identifiable, Hashable {
    let design: VoiceDesign
    let line: String
    var id: String { design.name }
}

enum Examples {
    static let lines: [ExampleLine] = [
        ExampleLine(title: "Good News", text: "[with bright eager excitement] They said yes! [gasps] They actually said yes, and we start on Monday!"),
        ExampleLine(title: "The Ultimatum", text: "[icy calm] Sit down. [pause] You have one minute to explain where the money went."),
        ExampleLine(title: "The Empty Chair", text: "[heavy, deflated sadness] I kept his chair by the window. [sighs] I just can't bring myself to move it."),
        ExampleLine(title: "Another Meeting", text: "[sarcastic and deadpan] Oh, wonderful. Another meeting that could have been an email. [clears throat] I'm thrilled."),
        ExampleLine(title: "In the Hallway", text: "[breathless and terrified] Something's in the hallway. <whisper>Don't move, it can hear us.</whisper>"),
        ExampleLine(title: "One Thing", text: "[furious, barely holding it together] I asked you for one thing. [pause] <emphasis>One</emphasis> thing."),
        ExampleLine(title: "Goodnight", text: "[warm, gentle bedtime voice] And the little fox curled up under the stars. [chuckles] Time to close your eyes, sleepyhead."),
        ExampleLine(title: "The Contract", text: "[low and menacing] You should have read the contract more carefully. [chuckles] The house belongs to me now."),
    ]

    static let cast: [CastPreset] = [
        CastPreset(
            design: VoiceDesign(name: "Captain", description: "gravelly old male voice, about seventy, weathered and slow, salt in every word",
                                scene: "a storm-battered tavern at the edge of the harbor"),
            line: "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out."
        ),
        CastPreset(
            design: VoiceDesign(name: "Night Owl", description: "smooth warm baritone, a late-night radio host, relaxed and intimate, close to the mic"),
            line: "It's two in the morning, and if you're still awake, [chuckles] you're in good company. <slow>This next one goes out to the night drivers.</slow>"
        ),
        CastPreset(
            design: VoiceDesign(name: "Mia", description: "excited little girl, about seven, talks fast, giggly and breathless"),
            line: "[with bright eager excitement] Guess what, guess what! [laughs] I found a frog in the garden and his name is Mister Hops!"
        ),
        CastPreset(
            design: VoiceDesign(name: "Detective", description: "world-weary man in his forties, low smoky voice, clipped and dry",
                                scene: "a rain-soaked office, one desk lamp still on"),
            line: "[sighs] The rain hadn't stopped in three days. [pause] Neither had the phone."
        ),
        CastPreset(
            design: VoiceDesign(name: "Herald", description: "booming, pompous royal herald, theatrical and grandly formal"),
            line: "[clears throat] Hear ye, hear ye! By order of the crown, <emphasis>all</emphasis> shall gather in the square at noon!"
        ),
        CastPreset(
            design: VoiceDesign(name: "Emma", description: "young female voice, sleepy, half talking to herself",
                                scene: "a small office on a Monday morning; the weekly meeting is about to start"),
            line: "Okay. Coffee first. [sighs] Then I'll pretend I read the agenda."
        ),
        CastPreset(
            design: VoiceDesign(name: "The Duke", description: "silky, theatrical villain with a rich low voice, amused and unhurried"),
            line: "Oh, you came all this way to stop me? [laughs] How delightfully brave of you."
        ),
        CastPreset(
            design: VoiceDesign(name: "Nana", description: "warm elderly woman, soft and slightly shaky, a gentle storyteller",
                                scene: "a quiet bedroom, the lights already dimmed"),
            line: "[warm, gentle bedtime voice] Once there was a lighthouse keeper who talked to the moon. [pause] And one night, the moon talked back."
        ),
    ]

    /// Director's notes offered as one-click fills for a retake.
    static let notes = [
        "softer and warmer", "more excited", "more amused than before", "drop the hurt",
        "slower, with more pauses", "angrier", "whispered", "faster", "more sarcastic",
    ]
}
