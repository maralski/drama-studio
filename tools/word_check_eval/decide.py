#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Runs the full word check over every set's recorded audio and scores it.

    python3 -I tools/word_check_eval/decide.py --bin <swift bin dir> --audio <unpacked dir>

Runs `word-check-lab decide` on all sets at once and writes each set's
decisions to tools/word_check_eval/cache/decisions/<set>.json. Exits 1 when
any set accepts a wrong clip, falls below its audioFloor in
Tests/DramaEngineTests/Fixtures/word-check-sets.json, or has a clip with no
decision. Clips the baseline accepted and the check now rejects are listed
but do not fail the run.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parents[1] / 'Tests/DramaEngineTests/Fixtures/word-check-sets.json'
KINDS = {'pos': 'correct', 'spell': 'spelled', 'neg': 'wrong', 'negspell': 'wrong', 'drop': 'wrong'}


def score(entry, decisions):
    name = entry['name']
    items = json.loads((HERE / 'sets' / name / 'items.json').read_text())
    baseline = set(json.loads((HERE / 'sets' / name / 'baseline.json').read_text()))
    tally = {kind: [0, 0] for kind in ('correct', 'spelled', 'wrong')}
    for item in items:
        kind = KINDS[item['id'].split('.')[1]]
        tally[kind][0] += decisions.get(item['id'], False)
        tally[kind][1] += 1
    wrong = sorted(i for i, ok in decisions.items() if ok and KINDS[i.split('.')[1]] == 'wrong')
    missing = sorted({item['id'] for item in items} - decisions.keys())
    lost = sorted(i for i in baseline if not decisions.get(i) and KINDS[i.split('.')[1]] != 'wrong')
    accepted = tally['correct'][0] + tally['spelled'][0]
    counts = ' '.join(f'{kind} {got}/{total}' for kind, (got, total) in tally.items() if total)
    print(f'{name}: {counts}; correct accepted {accepted}, floor {entry["audioFloor"]}; '
          f'baseline accepted {len(baseline)}, lost vs baseline {lost}')
    failures = []
    if wrong:
        failures.append(f'{name} accepted wrong clips {wrong}')
    if accepted < entry['audioFloor']:
        failures.append(f'{name} accepted {accepted} correct clips, below its floor of {entry["audioFloor"]}')
    if missing:
        failures.append(f'{name} has no decision for {missing}')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bin', type=Path, required=True, help='directory holding word-check-lab and the staged recognizer')
    parser.add_argument('--audio', type=Path, required=True, help='directory with one folder of clips per set')
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    out = HERE / 'cache/decisions'
    out.mkdir(parents=True, exist_ok=True)
    runs = {entry['name']: subprocess.Popen([str(args.bin / 'word-check-lab'), 'decide',
                                             str(HERE / 'sets' / entry['name'] / 'items.json'),
                                             str(args.audio / entry['name']), str(out / f'{entry["name"]}.json')],
                                            stderr=subprocess.PIPE, text=True)
            for entry in registry}
    failures = []
    for entry in registry:
        name = entry['name']
        _, errors = runs[name].communicate()
        if runs[name].returncode:
            failures.append(f'{name}: word-check-lab decide exited {runs[name].returncode}: {errors.strip()}')
            continue
        decisions = {row['id']: row['accepted'] for row in json.loads((out / f'{name}.json').read_text())}
        failures += score(entry, decisions)
    for failure in failures:
        print('FAIL', failure)
    print('audio: FAIL' if failures else 'audio: PASS')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
