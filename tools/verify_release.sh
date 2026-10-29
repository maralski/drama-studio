#!/bin/zsh
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
# Checks the assets tools/make_release.sh wrote, on the real files: the
# checksums, the DMG's contents and signature, the codec keys in the app's
# Info.plist, and that the codec zip unpacks to the three models.
#
#   tools/verify_release.sh [version]   version defaults to the VERSION file
set -euo pipefail

root=${0:A:h:h}
version=${1:-$(<"$root/VERSION")}
release="$root/dist/release/$version"
dmg="$release/DramaStudio-$version.dmg"
zip="$release/DramaStudio-Codec-$version.zip"
work=$(mktemp -d)
mount="$work/mount"
trap 'hdiutil detach -quiet "$mount" 2>/dev/null || true; rm -rf "$work"' EXIT

fail() { print -u2 "FAIL: $*"; exit 1; }

print "== checksums"
(cd "$release" && shasum -a 256 -c SHA256SUMS)

print "== DMG"
mkdir "$mount"
hdiutil attach -quiet -nobrowse -readonly -mountpoint "$mount" "$dmg"
ls -l "$mount"
[[ -L "$mount/Applications" && $(readlink "$mount/Applications") == /Applications ]] || fail "no Applications symlink"
for file in DramaStudio.app LICENSE NOTICE LICENSE-FISH-AUDIO THIRD_PARTY_NOTICES; do
    [[ -e "$mount/$file" ]] || fail "$file missing from the DMG"
done
app="$mount/DramaStudio.app"
for file in LICENSE NOTICE LICENSE-FISH-AUDIO THIRD_PARTY_NOTICES; do
    [[ -f "$app/Contents/Resources/$file" ]] || fail "$file missing from Contents/Resources"
done

print "== Info.plist"
plist="$app/Contents/Info.plist"
short=$(plutil -extract CFBundleShortVersionString raw -o - "$plist")
url=$(plutil -extract DSCodecBundleURL raw -o - "$plist")
sha=$(plutil -extract DSCodecBundleSHA256 raw -o - "$plist")
print "CFBundleShortVersionString $short\nDSCodecBundleURL $url\nDSCodecBundleSHA256 $sha"
[[ $short == $version ]] || fail "the app says $short"
[[ $url == https://github.com/*/releases/download/v$version/${zip:t} ]] || fail "unexpected codec URL $url"
[[ $sha == $(shasum -a 256 "$zip" | awk '{print $1}') ]] || fail "DSCodecBundleSHA256 is not the zip's sha256"

print "== codesign"
codesign --verify --deep --strict --verbose=2 "$app"
codesign -dv "$app" 2>&1 | grep -E "^(Signature|Format|Identifier|TeamIdentifier|flags|CodeDirectory)" || true

print "== codec zip"
unpacked="$work/codec"
mkdir "$unpacked"
/usr/bin/ditto -x -k "$zip" "$unpacked"
ls -l "$unpacked"
for package in DramaCodecEncoder DramaCodecDecoder DramaCodecProbe; do
    [[ -f "$unpacked/$package.mlpackage/Manifest.json" ]] || fail "$package.mlpackage missing from the zip"
done
du -sh "$unpacked"/*.mlpackage
print "release $version verified"
