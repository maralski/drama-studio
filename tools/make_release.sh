#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Builds the release assets in dist/release/<version>/: the app DMG, the
# Core ML codec zip the setup window downloads, SHA256SUMS and the release
# notes, then prints the gh command that publishes them.
#
#   tools/make_release.sh [version]   version defaults to the VERSION file
#
#   REPO            GitHub repository the assets are published to
#   NOTARY_PROFILE  notarytool keychain profile; with a Developer ID
#                   Application identity it signs, notarizes and staples
set -euo pipefail

root=${0:A:h:h}
cd "$root"

REPO=${REPO:-maralski/drama-studio}
version=${1:-$(<VERSION)}
if [[ $version != $(<VERSION) ]]; then
    print -u2 "Version $version differs from VERSION ($(<VERSION)); update VERSION so the app reports the release it ships in."
    exit 1
fi

codec_dir="$HOME/Library/Application Support/DramaStudio/Codec"
packages=(DramaCodecEncoder.mlpackage DramaCodecDecoder.mlpackage DramaCodecProbe.mlpackage)
licenses=(LICENSE NOTICE LICENSE-FISH-AUDIO THIRD_PARTY_NOTICES)
# GitHub documents a 2 GiB (2,147,483,648 byte) cap per release asset; stay
# clear of it rather than at it.
asset_limit=2000000000

release="$root/dist/release/$version"
zip_name="DramaStudio-Codec-$version.zip"
dmg_name="DramaStudio-$version.dmg"
zip="$release/$zip_name"
dmg="$release/$dmg_name"
codec_url="https://github.com/$REPO/releases/download/v$version/$zip_name"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
rm -rf "$release"
mkdir -p "$release"

missing=()
for package in $packages; do
    [[ -d "$codec_dir/$package" ]] || missing+=$package
done
if (( $#missing )); then
    print -u2 "Missing from $codec_dir: ${(j:, :)missing}."
    print -u2 "Build them with Sources/DramaEngine/CodecConversion/convert_codec.py (see the README's codec section)."
    exit 1
fi

# The compiled .mlmodelc caches beside the packages are machine-specific and stay out.
codec_stage="$work/codec"
mkdir "$codec_stage"
for package in $packages; do
    ditto --norsrc --noextattr "$codec_dir/$package" "$codec_stage/$package"
done
cp -p LICENSE-FISH-AUDIO NOTICE "$codec_stage/"
print "Zipping the codec models"
ditto -c -k --norsrc --noextattr "$codec_stage" "$zip"

roundtrip="$work/roundtrip"
mkdir "$roundtrip"
/usr/bin/ditto -x -k "$zip" "$roundtrip"
expected=(${(o)packages} LICENSE-FISH-AUDIO NOTICE)
found=("$roundtrip"/*(DN:t))
if [[ ${(oj: :)found} != ${(oj: :)expected} ]]; then
    print -u2 "The codec zip unpacks to [${(j:, :)found}], expected [${(j:, :)expected}]."
    exit 1
fi
for package in $packages; do
    [[ -f "$roundtrip/$package/Manifest.json" ]] || { print -u2 "$package unpacked without its Manifest.json."; exit 1; }
done

zip_bytes=$(stat -f %z "$zip")
if (( zip_bytes > asset_limit )); then
    print -u2 "$zip_name is $zip_bytes bytes, over the $asset_limit byte margin under GitHub's 2 GiB asset cap."
    exit 1
fi
zip_sha=$(shasum -a 256 "$zip" | awk '{print $1}')

# The SHA-1 column, so two certificates with one name stay unambiguous.
identity=$(security find-identity -v -p codesigning | awk '/"Developer ID Application: / {print $2; exit}')
notarized=false
if [[ -n $identity && -n ${NOTARY_PROFILE:-} ]] && xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    notarized=true
else
    identity=-
fi

DS_CODEC_BUNDLE_URL=$codec_url DS_CODEC_BUNDLE_SHA256=$zip_sha SIGN_IDENTITY=$identity tools/make_app.sh

notarize() {
    local submission=$1 log="$work/notary.json"
    xcrun notarytool submit "$submission" --keychain-profile "$NOTARY_PROFILE" --wait --output-format json > "$log"
    if [[ $(plutil -extract status raw -o - "$log") != Accepted ]]; then
        print -u2 "Notarization of ${submission:t} was not accepted:"
        cat "$log" >&2
        exit 1
    fi
}

dmg_stage="$work/dmg"
mkdir "$dmg_stage"
ditto dist/DramaStudio.app "$dmg_stage/DramaStudio.app"
if $notarized; then
    ditto -c -k --keepParent "$dmg_stage/DramaStudio.app" "$work/DramaStudio.zip"
    notarize "$work/DramaStudio.zip"
    xcrun stapler staple "$dmg_stage/DramaStudio.app"
fi
ln -s /Applications "$dmg_stage/Applications"
cp $licenses "$dmg_stage/"
hdiutil create -quiet -ov -format UDZO -fs HFS+ -volname "Drama Studio $version" -srcfolder "$dmg_stage" "$dmg"
# One build on macOS 27.0.1 wrote a DMG that failed its own checksum and
# would not attach; a rerun was fine.
if ! hdiutil verify -quiet "$dmg" 2>/dev/null; then
    print -u2 "$dmg_name does not pass hdiutil verify; rerun tools/make_release.sh."
    exit 1
fi
if $notarized; then
    codesign --force --timestamp --sign "$identity" "$dmg"
    notarize "$dmg"
    xcrun stapler staple "$dmg"
fi

(cd "$release" && shasum -a 256 "$dmg_name" "$zip_name" > SHA256SUMS)

zip_gb=$(printf '%.1f' $(( zip_bytes / 1e9 )))
{
    print "Drama Studio is a Mac app that runs Fish Audio Drama 3 text-to-speech locally on Apple silicon. It has a script editor, voice cloning, voice design, and an HTTP API and MCP server on 127.0.0.1.

## Requirements

- macOS 15 or later on Apple silicon. Verified only on an M3 Max with 128 GB.
- 64 GB of unified memory or more recommended: the engine maps about 41 GB of weights.
- About 47 GB of free disk for the weights download, the codec models and the packed-weight cache.

## Install

1. Download $dmg_name and open it.
2. Drag Drama Studio to Applications.
3. Open Drama Studio from Applications."
    if ! $notarized; then
        print "
This build is not notarized, so on current macOS the first open is blocked. To allow it:

1. Open System Settings, then Privacy & Security.
2. Scroll to Security and click Open Anyway for Drama Studio.
3. Confirm.

From Terminal you can instead run \`xattr -dr com.apple.quarantine /Applications/DramaStudio.app\`."
    fi
    print "
## First run

1. The setup window asks you to agree to the Fish Audio Community License terms.
2. It downloads the Drama 3 weights from Hugging Face (41.5 GB, pinned to revision e0a9218) and the Core ML codec models from this release ($zip_gb GB).
3. It tunes the kernels for about 30 seconds.
4. You speak your first line.

## License

- Drama Studio's code: Apache License 2.0.
- The Drama 3 weights and the codec models: Fish Audio Community License v1.0. Commercial use is allowed only under USD 500,000 annual revenue. A hosted TTS or voice-cloning service needs a separate license from Fish Audio. Do not imitate identifiable people without their consent.

Built with Fish Audio.

## Checksums

\`\`\`
$(<"$release/SHA256SUMS")
\`\`\`"
} > "$release/RELEASE_NOTES.md"

if $notarized; then
    print "Signed with $identity, notarized and stapled."
else
    print "Release is ad-hoc signed and not notarized: users need the Gatekeeper Open Anyway step (the release notes say how)."
fi
print "Assets in $release:"
ls -l "$release"
print "\nTo publish:"
print -r -- "cd ${(q)release} && gh release create v$version $dmg_name $zip_name SHA256SUMS --repo $REPO --title \"Drama Studio $version\" --notes-file RELEASE_NOTES.md"
