#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Records one set's recognizer output as the replay fixture and the baseline.

    python3 -I tools/word_check_eval/record.py <set> <recog.json>

Writes Tests/DramaEngineTests/Fixtures/word-check-<set>.json, one clip per
line, and tools/word_check_eval/sets/<set>/baseline.json, the clips the
check accepted when the audio was recognized.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    name, source = sys.argv[1], Path(sys.argv[2])
    if not (ROOT / 'tools/word_check_eval/sets' / name / 'items.json').is_file():
        raise SystemExit(f'unknown set {name}: no tools/word_check_eval/sets/{name}/items.json')
    records = sorted(json.loads(source.read_text()), key=lambda record: record['id'])
    clips = [{'id': record['id'], 'line': record['line'], 'native': record['native'],
              'nativeSpans': [{'text': span['text'], 'confidence': span['confidence']} for span in record['native_spans']],
              'independent': record['whisper_small_en']} for record in records]
    fixture = ROOT / f'Tests/DramaEngineTests/Fixtures/word-check-{name}.json'
    fixture.write_text('[\n' + ',\n'.join(json.dumps(clip, ensure_ascii=False) for clip in clips) + '\n]\n')
    accepted = [record['id'] for record in records if record['production_accepted']]
    baseline = ROOT / 'tools/word_check_eval/sets' / name / 'baseline.json'
    baseline.write_text(json.dumps(accepted, indent=1) + '\n')
    print(f'{name}: {len(clips)} clips to {fixture.relative_to(ROOT)}, {len(accepted)} accepted to {baseline.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
