#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Downloads the pinned word-check audio, verifies it, and unpacks it.

    python3 -I tools/word_check_eval/fetch_audio.py [--archive <local zip>]

The zip named in audio.json lands in tools/word_check_eval/cache and unpacks to
cache/<zip stem>/<set>/<clip>.flac. --archive uses a local copy instead of the
download. Either way the zip must match the pinned size and SHA-256.
"""
import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'

spec = importlib.util.spec_from_file_location('bootstrap', HERE.parent / 'bootstrap_word_recognition.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def fetch(archive=None):
    pin = json.loads((HERE / 'audio.json').read_text())
    target = CACHE / pin['fileName']
    unpacked = CACHE / Path(pin['fileName']).stem
    marker = unpacked / '.sha256'
    if marker.is_file() and marker.read_text().strip() == pin['sha256']:
        return unpacked
    seeds = []
    if archive:
        if archive.name != pin['fileName']:
            raise SystemExit(f'--archive must be named {pin["fileName"]}')
        if not bootstrap.verified(archive, pin):
            raise SystemExit(f'{archive} does not match the size and SHA-256 in audio.json')
        seeds.append(archive.resolve().parent)
    bootstrap.provision(pin, target, seeds)
    with zipfile.ZipFile(target) as zipped:
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in zipped.namelist()):
            raise RuntimeError('Unsafe audio archive path')
        staging = Path(tempfile.mkdtemp(dir=CACHE))
        try:
            zipped.extractall(staging)
            (staging / '.sha256').write_text(pin['sha256'] + '\n')
            shutil.rmtree(unpacked, ignore_errors=True)
            os.replace(staging, unpacked)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return unpacked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, help='local copy of the pinned zip')
    print(fetch(parser.parse_args().archive))


if __name__ == '__main__':
    main()
