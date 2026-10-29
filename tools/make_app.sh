#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Builds dist/DramaStudio.app from the SwiftPM products and signs it.
# The MCP server and the say command ship beside the app binary as
# Contents/MacOS/drama-mcp and Contents/MacOS/drama-say.
#
#   SIGN_IDENTITY           codesign identity; "-" (the default) signs ad hoc,
#                           anything else signs with the hardened runtime
#   DS_CODEC_BUNDLE_URL     with DS_CODEC_BUNDLE_SHA256, the Core ML codec zip
#   DS_CODEC_BUNDLE_SHA256  the setup window downloads; unset in development
#                           builds, which convert the codec with Python instead
set -euo pipefail

root=${0:A:h:h}
cd "$root"

swift build -c release --product DramaStudio
swift build -c release --product drama-mcp
swift build -c release --product drama-say
bin=$(swift build -c release --show-bin-path)
resources="DramaStudio_DramaEngine.bundle"

app="$root/dist/DramaStudio.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin/DramaStudio" "$app/Contents/MacOS/DramaStudio"
cp "$bin/drama-mcp" "$app/Contents/MacOS/drama-mcp"
cp "$bin/drama-say" "$app/Contents/MacOS/drama-say"

cp -R "$bin/$resources" "$app/Contents/Resources/$resources"
cp LICENSE NOTICE LICENSE-FISH-AUDIO THIRD_PARTY_NOTICES "$app/Contents/Resources/"
cp App/Icon/AppIcon.icns "$app/Contents/Resources/AppIcon.icns"

short_version=$(<"$root/VERSION")
version=$(git -C "$root" rev-list --count HEAD 2>/dev/null || echo 1)

doc_type() {
    local name=$1; shift
    print "        <dict>
            <key>CFBundleTypeName</key><string>$name</string>
            <key>CFBundleTypeRole</key><string>Viewer</string>
            <key>LSHandlerRank</key><string>Alternate</string>
            <key>LSItemContentTypes</key>
            <array>"
    for uti in "$@"; do print "                <string>$uti</string>"; done
    print "            </array>
        </dict>"
}

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>io.github.maralski.DramaStudio</string>
    <key>CFBundleName</key><string>Drama Studio</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleDisplayName</key><string>Drama Studio</string>
    <key>CFBundleExecutable</key><string>DramaStudio</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$short_version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSMicrophoneUsageDescription</key><string>Drama Studio records reference clips for voice cloning.</string>
    <key>NSHumanReadableCopyright</key><string>Copyright 2026 Peter Marelas. Apache License 2.0. Built with Fish Audio; Fish Audio Drama 3 is licensed under the Fish Audio Community License.</string>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key><string>io.github.maralski.DramaStudio</string>
            <key>CFBundleURLSchemes</key><array><string>dramastudio</string></array>
        </dict>
    </array>
    <key>CFBundleDocumentTypes</key>
    <array>
$(doc_type "Plain Text Script" public.plain-text)
$(doc_type "Markdown Script" net.daringfireball.markdown)
$(doc_type "Rich Text Script" public.rtf com.apple.rtfd)
$(doc_type "HTML Script" public.html)
$(doc_type "Word Document" org.openxmlformats.wordprocessingml.document)
$(doc_type "Subtitles" io.github.maralski.DramaStudio.subrip org.w3.webvtt)
$(doc_type "PowerPoint Presentation" org.openxmlformats.presentationml.presentation)
$(doc_type "PDF Document" com.adobe.pdf)
$(doc_type "EPUB Book" org.idpf.epub-container)
        <dict>
            <key>CFBundleTypeName</key><string>Drama Studio Project</string>
            <key>CFBundleTypeRole</key><string>Editor</string>
            <key>LSHandlerRank</key><string>Owner</string>
            <key>LSTypeIsPackage</key><true/>
            <key>LSItemContentTypes</key>
            <array>
                <string>io.github.maralski.DramaStudio.project</string>
            </array>
        </dict>
    </array>
    <key>UTExportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key><string>io.github.maralski.DramaStudio.project</string>
            <key>UTTypeDescription</key><string>Drama Studio Project</string>
            <key>UTTypeConformsTo</key><array><string>com.apple.package</string></array>
            <key>UTTypeTagSpecification</key>
            <dict><key>public.filename-extension</key><array><string>dramaproj</string></array></dict>
        </dict>
    </array>
    <key>UTImportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key><string>io.github.maralski.DramaStudio.subrip</string>
            <key>UTTypeDescription</key><string>SubRip Subtitles</string>
            <key>UTTypeConformsTo</key><array><string>public.plain-text</string></array>
            <key>UTTypeTagSpecification</key>
            <dict><key>public.filename-extension</key><array><string>srt</string></array></dict>
        </dict>
    </array>
</dict>
</plist>
PLIST
codec_url=${DS_CODEC_BUNDLE_URL:-}
codec_sha=${DS_CODEC_BUNDLE_SHA256:-}
if [[ -n $codec_url || -n $codec_sha ]]; then
    if [[ ! $codec_url == https://* || ! $codec_sha =~ '^[0-9a-f]{64}$' ]]; then
        print -u2 "DS_CODEC_BUNDLE_URL must be an https URL and DS_CODEC_BUNDLE_SHA256 a lowercase sha256; set both or neither."
        exit 1
    fi
    plutil -insert DSCodecBundleURL -string "$codec_url" "$app/Contents/Info.plist"
    plutil -insert DSCodecBundleSHA256 -string "$codec_sha" "$app/Contents/Info.plist"
fi
plutil -lint "$app/Contents/Info.plist" >/dev/null

identity=${SIGN_IDENTITY:--}
if [[ $identity == - ]]; then
    codesign --force --deep --sign - "$app"
else
    # The app records reference clips from the microphone. Metal, Core ML and
    # the Python conversion child process need no hardened-runtime exception.
    entitlements="$root/dist/DramaStudio.entitlements"
    cat > "$entitlements" <<'ENTITLEMENTS'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.audio-input</key><true/>
</dict>
</plist>
ENTITLEMENTS
    for tool in drama-mcp drama-say; do
        codesign --force --options runtime --timestamp --sign "$identity" "$app/Contents/MacOS/$tool"
    done
    codesign --force --options runtime --timestamp --entitlements "$entitlements" --sign "$identity" "$app"
fi
codesign --verify --deep --strict "$app"

# SwiftPM's generated Bundle.module looks only at the .app root, where
# codesign forbids content, and then at the absolute build directory. Hide the
# build copy so this check proves resolution from inside the app.
mv "$bin/$resources" "$bin/$resources.hidden"
trap 'mv "$bin/$resources.hidden" "$bin/$resources"' EXIT
if "$app/Contents/MacOS/DramaStudio" --check-resources 2>/dev/null; then
    print "Built $app"
else
    print -u2 "WARNING: DramaEngine's Bundle.module does not resolve inside the signed app."
    print -u2 "         The Metal kernels ship in Contents/Resources/$resources; the engine must look"
    print -u2 "         there (Bundle.main.resourceURL) before falling back to Bundle.module."
    print -u2 "         The preview engine does not need them, so the app still runs."
    print "Built $app (engine resources unresolved)"
fi
