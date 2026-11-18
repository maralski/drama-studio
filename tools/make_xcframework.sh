#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Builds dist/DramaEngine.xcframework for macOS on Apple silicon, then proves a
# consumer can use it: Examples/SpeakCLI is built against the framework instead
# of the source package and run with --check and --mock.
#
# SwiftPM cannot emit a framework, so this archives the DramaEngine product as a
# dynamic library through xcodebuild (DRAMA_ENGINE_DYNAMIC=1 in Package.swift)
# and assembles the framework from the archive. Xcode leaves two things out of a
# package framework, and both are added here: the Swift module interface, and the
# resource bundles: DramaEngine's with the word-check data, and DramaTTS's with
# the Metal kernel sources and convert_codec.py. The bundles go in the
# framework's Resources, where SwiftPM's generated Bundle.module looks
# (Bundle(for:).resourceURL).
set -euo pipefail

root=${0:A:h:h}
cd "$root"
work="$root/.build/xcframework"
bundles=(DramaStudio_DramaEngine.bundle DramaTTS_DramaTTS.bundle)
out="$root/dist/DramaEngine.xcframework"

rm -rf "$work" "$out"
DRAMA_ENGINE_DYNAMIC=1 xcodebuild archive \
    -scheme DramaEngine \
    -destination 'generic/platform=macOS' \
    -archivePath "$work/DramaEngine.xcarchive" \
    -derivedDataPath "$work/dd" \
    ARCHS=arm64 SKIP_INSTALL=NO BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    -quiet

products="$work/dd/Build/Intermediates.noindex/ArchiveIntermediates/DramaEngine/BuildProductsPath/Release"
fw="$work/DramaEngine.framework"
cp -R "$work/DramaEngine.xcarchive/Products/usr/local/lib/DramaEngine.framework" "$fw"

mkdir -p "$fw/Versions/A/Modules/DramaEngine.swiftmodule"
for f in "$products/DramaEngine.swiftmodule"/*; do
    # The package and private interfaces would publish the 'package' internals.
    case $f in *.package.swiftinterface|*.private.swiftinterface) continue ;; esac
    cp -R "$f" "$fw/Versions/A/Modules/DramaEngine.swiftmodule/"
done
ln -sfh Versions/Current/Modules "$fw/Modules"
# A consumer has only this binary, so the public interface must not import the
# engine packages DramaEngine links.
if grep -nE '\b(DramaTTS|MingTTS)\b' "$fw/Versions/A/Modules/DramaEngine.swiftmodule"/*.swiftinterface; then
    print -u2 "ERROR: DramaEngine's public interface names DramaTTS or MingTTS."
    exit 1
fi
# The archive holds the bundles as symlinks into the build folder; copy what they point at.
for bundle in $bundles; do cp -RL "$products/$bundle" "$fw/Versions/A/Resources/$bundle"; done
codesign --force --sign - "$fw"

mkdir -p "$root/dist"
xcodebuild -create-xcframework -framework "$fw" -output "$out" >/dev/null
print "Built $out"

# The consumer must find the resources inside the framework, not in a build folder.
rm -rf "$work/dd"
consumer="$work/consumer"
mkdir -p "$consumer/Sources/SpeakCLI"
cp "$root/Examples/SpeakCLI/Sources/SpeakCLI/main.swift" "$consumer/Sources/SpeakCLI/"
cp -R "$out" "$consumer/DramaEngine.xcframework"
cat > "$consumer/Package.swift" <<'MANIFEST'
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SpeakCLI",
    platforms: [.macOS(.v15)],
    targets: [
        .binaryTarget(name: "DramaEngine", path: "DramaEngine.xcframework"),
        .executableTarget(name: "SpeakCLI", dependencies: ["DramaEngine"]),
    ]
)
MANIFEST
# SwiftPM copies the framework beside the executable but, from Swift 6.4, adds
# no rpath that finds it there.
swift build --package-path "$consumer" -c release -Xlinker -rpath -Xlinker @executable_path
bin=$(swift build --package-path "$consumer" -c release --show-bin-path)
check=$("$bin/SpeakCLI" --check || true)
print -r -- "$check"
if [[ $check != *"conversion script: "*"/DramaEngine.framework/"*"/DramaTTS_DramaTTS.bundle/"*"/CodecConversion/convert_codec.py"* ]]; then
    print -u2 "ERROR: convert_codec.py does not resolve inside the framework's DramaTTS bundle."
    exit 1
fi
say -o "$work/clip.aiff" "The quick brown fox jumps over the lazy dog."
"$bin/SpeakCLI" "$work/clip.aiff" "The quick brown fox jumps over the lazy dog." \
    "A binary framework speaks. The kernels came from inside it." "$work/mock.wav" --mock
print "Consumer built against $out and ran."
