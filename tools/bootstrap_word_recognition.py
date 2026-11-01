#!/usr/bin/env python3
"""Provision pinned local recognition assets before SwiftPM reads its binary target."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'Sources/WordRecognitionProtocol/Assets.json'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def verified(path, item):
    return path.is_file() and ('bytes' not in item or path.stat().st_size == item['bytes']) and digest(path) == item['sha256']


def provision(item, target, seeds):
    if verified(target, item):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        candidate = seed / target.name
        if verified(candidate, item):
            temporary = target.with_suffix(target.suffix + '.part')
            shutil.copyfile(candidate, temporary)
            os.replace(temporary, target)
            print('Verified cached', target.name, flush=True)
            return
    temporary = target.with_suffix(target.suffix + '.part')
    offset = temporary.stat().st_size if temporary.exists() else 0
    if item.get('bytes') is not None and offset > item['bytes']:
        temporary.unlink(); offset = 0
    request = urllib.request.Request(item['url'], headers={'Range': 'bytes=' + str(offset) + '-'} if offset else {})
    print('Downloading pinned', target.name, flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0
        with temporary.open('ab' if offset else 'wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                if item.get('bytes') is not None and output.tell() > item['bytes']:
                    raise RuntimeError('Download exceeds declared size: ' + target.name)
    if not verified(temporary, item):
        temporary.unlink(missing_ok=True)
        raise RuntimeError('Checksum or size mismatch: ' + target.name)
    os.replace(temporary, target)


def generated_source(manifest):
    quote = json.dumps
    lines = ['// Generated from Assets.json by tools/bootstrap_word_recognition.py.',
             '// Copyright 2026 Peter Marelas', '// SPDX-License-Identifier: Apache-2.0',
             'package enum WordRecognitionAssets {']
    for name in ['policyRevision', 'protocolVersion']:
        lines.append(f'    package static let {name} = {manifest[name]}')
    for name in ['policyDefinition', 'configurationID', 'helperName', 'modelDirectory']:
        lines.append(f'    package static let {name} = {quote(manifest[name])}')
    for model in manifest['models']:
        route = 'englishModel' if model['englishOnly'] else 'multilingualModel'
        lines.append(f'    package static let {route} = RecognitionModelAsset(id: {quote(model["id"])}, fileName: {quote(model["fileName"])}, sha256: {quote(model["sha256"])}, bytes: {model["bytes"]}, englishOnly: {str(model["englishOnly"]).lower()})')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def framework_valid(path):
    if not (path / 'Info.plist').is_file():
        return False
    candidates = list(path.glob('macos-*/whisper.framework'))
    return len(candidates) == 1 and all((candidates[0] / name).is_symlink() for name in ['Headers', 'Modules']) and (candidates[0] / 'Headers/whisper.h').is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework-only', action='store_true')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--check-generated', action='store_true')
    parser.add_argument('--seed-directory', type=Path, action='append', default=[])
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest['models']) == 2 and {model['englishOnly'] for model in manifest['models']} == {True, False}
    source = generated_source(manifest)
    generated = MANIFEST.with_name('WordRecognitionAssets.generated.swift')
    if args.check_generated:
        if not generated.exists() or generated.read_text() != source:
            raise SystemExit('Generated recognition assets differ; run tools/bootstrap_word_recognition.py --generate-only')
        return
    if not generated.exists() or generated.read_text() != source:
        generated.write_text(source)
    if args.generate_only:
        return
    vendor = ROOT / 'Vendor/WordRecognition'
    vendor.mkdir(parents=True, exist_ok=True)
    with (vendor / '.bootstrap.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        framework = manifest['framework']
        archive = vendor / framework['url'].rsplit('/', 1)[-1]
        target = ROOT / framework['localPath']
        marker = vendor / 'framework.sha256'
        if not framework_valid(target) or not marker.exists() or marker.read_text().strip() != framework['sha256']:
            provision(framework, archive, args.seed_directory)
            with zipfile.ZipFile(archive) as zipped:
                if any(Path(name).is_absolute() or '..' in Path(name).parts for name in zipped.namelist()):
                    raise RuntimeError('Unsafe framework archive path')
            with tempfile.TemporaryDirectory(dir=vendor) as staging:
                subprocess.run(['/usr/bin/ditto', '-x', '-k', str(archive), staging], check=True)
                unpacked = Path(staging) / framework['archivePath']
                if not framework_valid(unpacked):
                    raise RuntimeError('Framework extraction lost required symlinks')
                if target.exists():
                    shutil.rmtree(target)
                os.replace(unpacked, target)
            marker.write_text(framework['sha256'] + '\n')
        print('Pinned recognition framework ready', flush=True)
        if not args.framework_only:
            for model in manifest['models']:
                provision(model, vendor / 'Models' / model['fileName'], args.seed_directory)
            print('Pinned English and multilingual recognition models ready', flush=True)


if __name__ == '__main__':
    main()
