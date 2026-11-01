#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Run source-derived cue/voice cases against the local app and verify their words.

Only one speech request runs at a time. Audio, exact requests, recognizer output
and failures are retained. A recognizer disagreement is not counted as a pass.
Requires the evaluation environment's cached Whisper weights and Python packages.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

from verify_voice_design import words, differences, has_reaction_cue

BASE = os.environ.get('DRAMA_API_BASE', 'http://127.0.0.1:8780').rstrip('/')
_base = urllib.parse.urlsplit(BASE)
if (_base.scheme != 'http' or _base.hostname != '127.0.0.1' or _base.username or _base.password
        or _base.path or _base.query or _base.fragment):
    raise ValueError('DRAMA_API_BASE must be a loopback HTTP origin, such as http://127.0.0.1:8780')


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')




def build_cases(catalogue, library):
    voices = [{'id': 'default', 'label': 'Model default', 'parameters': {}}] + catalogue['designs']
    voices += [{'id': voice['id'], 'label': voice['name'], 'parameters': {'voice': voice['id']}}
               for voice in library['voices']]
    cases = []
    for voice in voices:
        for cue in catalogue['cases']:
            case = dict(cue, voice=voice['label'], parameters=voice['parameters'])
            case['id'] = '--'.join(map(slug, [voice['label'], cue['cue'], cue['position']]))
            cases.append(case)
    return cases


def run(args):
    import torch
    import whisper
    torch.set_num_threads(2)
    weights = Path.home() / '.cache/whisper/small.en.pt'
    if not weights.is_file():
        raise RuntimeError('Cached small.en speech recognizer is required')
    recognizer = whisper.load_model(str(weights), device='cpu')
    second = None
    token = os.environ.get('DRAMA_API_TOKEN')
    if token is None:
        token = (Path.home() / 'Library/Application Support/DramaStudio/api-token').read_text().strip()
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    cases = json.loads(args.manifest.read_text())
    if args.voice:
        cases = [case for case in cases if case['voice'] in args.voice]
    if args.cue:
        cases = [case for case in cases if case['cue'] in args.cue]
    if args.limit:
        cases = cases[:args.limit]
    if args.resume and any('reuse_of' in case for case in cases):
        raise ValueError('Dependent retake cases require a fresh run because cached take IDs expire')
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / 'results.jsonl'
    completed = set()
    if args.resume and results.exists():
        completed = {json.loads(line)['key'] for line in results.read_text().splitlines()}
    failures = 0
    take_ids = {}
    for case in cases:
        for seed in args.seeds:
            key = case['id'] + '--s' + str(seed)
            if key in completed:
                continue
            pause = args.pause_file or args.out / 'PAUSE'
            paused = False
            while pause.exists():
                if not paused:
                    print('PAUSED before', key, flush=True)
                    paused = True
                time.sleep(1)
            if paused:
                print('RESUMED', key, flush=True)
            body = {'input': case['input'], 'seed': seed, 'response_format': 'pcm', **case['parameters']}
            body.update(args.overrides)
            row = {'key': key, 'case': case, 'seed': seed, 'request': body, 'api_base': BASE,
                   'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            prefix = args.out / key
            print('START', key, flush=True)
            done = threading.Event()
            health = []

            def observe():
                while not done.wait(0.5):
                    try:
                        request = urllib.request.Request(BASE + '/v1/health', headers=headers)
                        with urllib.request.urlopen(request, timeout=3) as response:
                            health.append(json.loads(response.read()))
                    except Exception:
                        pass

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            start = time.monotonic()
            try:
                if 'reuse_of' in case:
                    source = (case['reuse_of'], seed)
                    if source not in take_ids:
                        raise RuntimeError('The prerequisite take did not pass independent verification')
                    body['previous_takes'] = [{'take_id': take_ids[source]}]
                request = urllib.request.Request(BASE + '/v1/audio/speech', data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    audio = response.read()
                    row['http_status'] = response.status
                    row['headers'] = {k: v for k, v in response.headers.items() if k.lower().startswith('x-drama-')}
                    if args.require_coverage and response.headers.get('X-Drama-Word-Coverage') != 'recognized':
                        raise RuntimeError('The installed engine did not report recognized word coverage')
                done.set()
                row['generation_wall_seconds'] = time.monotonic() - start
                if not audio or len(audio) % (2048 * 2):
                    raise RuntimeError('Invalid or incomplete PCM audio')
                with wave.open(str(prefix.with_suffix('.wav')), 'wb') as wav:
                    wav.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                    wav.writeframes(audio)
                row['audio'] = str(prefix.with_suffix('.wav'))
                row['audio_seconds'] = len(audio) / 88200
                row['audio_sha256'] = hashlib.sha256(audio).hexdigest()
                row['health'] = health
                row['frames_observed'] = max((h.get('estimate', {}).get('frames_done', 0) for h in health), default=0)
                row['frames_kept'] = len(audio) // (2048 * 2)
                expected = words(case['expected'])
                transcript = recognizer.transcribe(row['audio'], fp16=False, language='en', temperature=0,
                                                   condition_on_previous_text=False)['text'].strip()
                reaction = has_reaction_cue(case['input'])
                row['transcript'] = transcript
                row['differences'] = differences(expected, transcript, reaction, line=case['input'])
                row['verdict'] = 'PASS' if not row['differences'] else 'FAIL'
                if row['differences'] and args.confirm_failures:
                    if second is None:
                        second = whisper.load_model(str(Path.home() / '.cache/whisper/base.pt'), device='cpu')
                    second_text = second.transcribe(row['audio'], fp16=False, language='en', temperature=0,
                                                    condition_on_previous_text=False)['text'].strip()
                    row['second_transcript'] = second_text
                    row['second_differences'] = differences(expected, second_text, reaction, line=case['input'])
                    if not row['second_differences']:
                        row['verdict'] = 'INCONCLUSIVE_ASR'
            except urllib.error.HTTPError as error:
                row['http_status'] = error.code
                row['error'] = error.read().decode('utf-8', errors='replace').strip()
                row['verdict'] = 'REJECTED' if 'spoken-word check' in row['error'] else 'ERROR'
            except Exception as error:
                row['verdict'] = 'ERROR'
                row['error'] = str(error)
            finally:
                row.setdefault('generation_wall_seconds', time.monotonic() - start)
                row.setdefault('health', health)
                done.set()
                observer.join(timeout=4)
            if row['verdict'] != 'PASS':
                failures += 1
            elif row.get('headers', {}).get('X-Drama-Take-Id'):
                take_ids[(case['id'], seed)] = row['headers']['X-Drama-Take-Id']
            prefix.with_suffix('.json').write_text(json.dumps(row, indent=2) + '\n')
            with results.open('a') as output:
                output.write(json.dumps(row) + '\n')
            print(row['verdict'], key, row.get('transcript', row.get('error', '')), flush=True)
            if args.stop_after_failures and failures >= args.stop_after_failures:
                return 1
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('matrix')
    make.add_argument('--catalogue', type=Path, required=True)
    make.add_argument('--voices', type=Path, required=True)
    make.add_argument('--out', type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--manifest', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--seeds', nargs='+', type=int, default=[1])
    execute.add_argument('--voice', action='append')
    execute.add_argument('--cue', action='append')
    execute.add_argument('--limit', type=int)
    execute.add_argument('--timeout', type=int, default=120)
    execute.add_argument('--confirm-failures', action='store_true')
    execute.add_argument('--require-coverage', action='store_true')
    execute.add_argument('--pause-file', type=Path, help='Pause between requests while this file exists; defaults to OUT/PAUSE')
    execute.add_argument('--resume', action='store_true')
    execute.add_argument('--stop-after-failures', type=int, default=0)
    execute.add_argument('--overrides', type=json.loads, default={})
    args = parser.parse_args()
    if args.command == 'matrix':
        cases = build_cases(json.loads(args.catalogue.read_text()), json.loads(args.voices.read_text()))
        args.out.write_text(json.dumps(cases, indent=2) + '\n')
        print(len(cases), 'cases')
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
