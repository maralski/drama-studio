# Examples

## SpeakCLI

`SpeakCLI` is a separate Swift package that depends on this repository by path
and uses only DramaEngine's public API. It clones a voice from a clip, speaks
a text with it, prints progress as the audio streams, and writes a 16-bit WAV.

```bash
cd Examples/SpeakCLI
swift build
.build/debug/SpeakCLI --check
.build/debug/SpeakCLI clip.wav "What the clip says." "The text to speak." out.wav
```

`--check` prints `DramaSetup.check` for the Hugging Face cache and exits with
status 1 when something the engine needs is missing. `--mock` runs the same
code on `MockSynthesizer`, which needs no model files, and writes a sine
sweep instead of speech.

## Add DramaEngine to an Xcode project

1. In Xcode, choose **File > Add Package Dependencies**.
2. Enter the repository URL and choose a version rule from 1.0.0.
3. Add the **DramaEngine** library to the app target. Leave **DramaServer**
   out unless the app serves the HTTP API.
4. Get the model and the codec models as "Getting the model" in the
   DramaEngine documentation describes.
5. `import DramaEngine` and create one `DramaSynthesizer` for the app.

In a `Package.swift`, depend on the package and the product:

```swift
dependencies: [.package(url: "<repository URL>", from: "1.0.0")],
targets: [
	.target(name: "MyApp", dependencies: [.product(name: "DramaEngine", package: "<repository name>")]),
]
```

SwiftPM names a URL dependency after the last path component of its URL, so
`package:` is that name. `SpeakCLI` depends by path and names the package
`DramaStudio` with `.package(name:path:)`.

The package's minimum platform is macOS 15. DramaEngine has no dependencies
beyond the system frameworks: Metal, Core ML, AVFoundation, Foundation and
CryptoKit.

## Entitlements

DramaEngine needs no entitlement of its own. The app's existing ones are
enough to load the model, synthesize, and decode.

- Recording a clip for cloning needs the microphone: an
  `NSMicrophoneUsageDescription` string in `Info.plist` and, under the App
  Sandbox or the Hardened Runtime, `com.apple.security.device.audio-input`.
- Under the App Sandbox the app cannot read `~/.cache/huggingface` or start
  Python. Ask the person for the snapshot folder with an open panel, which
  needs `com.apple.security.files.user-selected.read-only`, and keep a
  security-scoped bookmark. Build the codec models outside the app.

## Documentation

Xcode builds the DocC catalog without any package plugin:

```bash
xcodebuild docbuild -scheme DramaEngine -destination 'generic/platform=macOS' -derivedDataPath .build/docbuild
open .build/docbuild/Build/Products/Debug/DramaEngine.doccarchive
```

`swift package generate-documentation` needs the swift-docc-plugin package,
which this repository does not depend on.

## A binary framework

`tools/make_xcframework.sh` builds `dist/DramaEngine.xcframework` for macOS on
Apple silicon. The Metal kernel sources and the codec conversion script travel
inside the framework, where the engine finds them at run time.
