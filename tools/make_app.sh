#!/bin/zsh
# Builds dist/DramaStudio.app from the SwiftPM products and ad-hoc signs it.
# The MCP server and the say command ship beside the app binary as
# Contents/MacOS/drama-mcp and Contents/MacOS/drama-say.
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
    <key>CFBundleIdentifier</key><string>audio.fish.community.DramaStudio</string>
    <key>CFBundleName</key><string>Drama Studio</string>
    <key>CFBundleDisplayName</key><string>Drama Studio</string>
    <key>CFBundleExecutable</key><string>DramaStudio</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSMicrophoneUsageDescription</key><string>Drama Studio records reference clips for voice cloning.</string>
    <key>NSHumanReadableCopyright</key><string>Copyright © 2026 Peter Marelas. Built with Fish Audio. Fish Audio Drama 3 is licensed under the Fish Audio Community License.</string>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key><string>audio.fish.community.DramaStudio</string>
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
$(doc_type "Subtitles" audio.fish.community.DramaStudio.subrip org.w3.webvtt)
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
                <string>audio.fish.community.DramaStudio.project</string>
            </array>
        </dict>
    </array>
    <key>UTExportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key><string>audio.fish.community.DramaStudio.project</string>
            <key>UTTypeDescription</key><string>Drama Studio Project</string>
            <key>UTTypeConformsTo</key><array><string>com.apple.package</string></array>
            <key>UTTypeTagSpecification</key>
            <dict><key>public.filename-extension</key><array><string>dramaproj</string></array></dict>
        </dict>
    </array>
    <key>UTImportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key><string>audio.fish.community.DramaStudio.subrip</string>
            <key>UTTypeDescription</key><string>SubRip Subtitles</string>
            <key>UTTypeConformsTo</key><array><string>public.plain-text</string></array>
            <key>UTTypeTagSpecification</key>
            <dict><key>public.filename-extension</key><array><string>srt</string></array></dict>
        </dict>
    </array>
</dict>
</plist>
PLIST
plutil -lint "$app/Contents/Info.plist" >/dev/null

codesign --force --deep --sign - "$app"
codesign --verify --strict "$app"

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
