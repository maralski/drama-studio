#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Packs each set's WAV clips into the FLAC zip published as a release asset.

    python3 -I tools/word_check_eval/pack_audio.py <out.zip> <set>=<wav dir> ...

The zip holds one folder per set and one FLAC per clip the set's items.json
generates. Every FLAC is decoded again and compared with its WAV, and the zip
has fixed timestamps, so the same WAVs and ffmpeg give the same bytes.
"""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SETS = Path(__file__).resolve().parent / 'sets'


def pcm(path):
    return subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(path), '-f', 's16le', '-'],
                          capture_output=True, check=True).stdout


def encode(wav, flac):
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(wav), '-map_metadata', '-1',
                    '-fflags', '+bitexact', '-flags:a', '+bitexact', '-compression_level', '12', str(flac)], check=True)
    if pcm(wav) != pcm(flac):
        raise RuntimeError(f'FLAC samples differ from {wav}')


def main():
    output, pairs = Path(sys.argv[1]), [argument.split('=', 1) for argument in sys.argv[2:]]
    if not pairs:
        raise SystemExit(__doc__)
    jobs = []
    for name, directory in pairs:
        items = json.loads((SETS / name / 'items.json').read_text())
        for item in items:
            if 'audio' not in item:
                wav = Path(directory) / (item['id'] + '.wav')
                if not wav.is_file():
                    raise SystemExit(f'missing clip {wav}')
                jobs.append((wav, f'{name}/{item["id"]}.flac'))
    with tempfile.TemporaryDirectory() as staging, concurrent.futures.ThreadPoolExecutor() as pool:
        for future in [pool.submit(encode, wav, Path(staging) / member.replace('/', '__')) for wav, member in jobs]:
            future.result()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as archive:
            for wav, member in sorted(jobs, key=lambda job: job[1]):
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                archive.writestr(info, (Path(staging) / member.replace('/', '__')).read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f'{output}: {len(jobs)} clips, {output.stat().st_size} bytes, sha256 {digest}')


if __name__ == '__main__':
    main()
