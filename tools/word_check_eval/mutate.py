#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Checks that the word-check tests kill every source mutation in mutations.toml.

    python3 -I tools/word_check_eval/mutate.py [--workers N] [--check] [name ...]

Each worker clones the checkout with APFS copy-on-write, runs the tests once
unmutated, then takes mutations from a shared queue: apply one, run the tests,
restore the file. The checkout itself is never edited. Names limit the run to
those mutations. --check only confirms every `old` string occurs once.

Exits 1 when a mutation survives, fails to build, or is marked equivalent and
is killed anyway. Needs Python 3.11 or later for tomllib.
"""
import argparse
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = 'WordCoverageTests|BoundaryVerificationTests|WordCheckCorpusTests|LocalWordVerifierTests'
TIMEOUT = 1800


def load(names):
    mutations = tomllib.loads((HERE / 'mutations.toml').read_text())['mutation']
    unknown = set(names) - {mutation['name'] for mutation in mutations}
    if unknown:
        raise SystemExit(f'unknown mutations: {sorted(unknown)}')
    return [mutation for mutation in mutations if not names or mutation['name'] in names]


def misplaced(mutations, root):
    return [f'{mutation["name"]}: old text occurs {count} times in {mutation["file"]}'
            for mutation in mutations
            if (count := (root / mutation['file']).read_text().count(mutation['old'])) != 1]


def run_tests(clone):
    try:
        run = subprocess.run(['swift', 'test', '-c', 'release', '--filter', TESTS],
                             cwd=clone, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {'failed': ['timed out'], 'build error': False, 'sets': []}
    out = run.stdout + run.stderr
    failed = sorted(set(re.findall(r'error: -\[DramaEngineTests\.(\w+ \w+)\]', out)))
    return {'failed': failed, 'build error': run.returncode != 0 and not failed,
            'sets': re.findall(r'word-check (\w+: \d+/\d+) correct', out)}


def clone(index, parent):
    path = parent / f'w{index}'
    subprocess.run(['cp', '-c', '-R', str(ROOT), str(path)], check=True)
    # Clang's module cache records absolute paths, so a moved copy must rebuild it.
    for cache in (path / '.build').glob('*/*/ModuleCache'):
        shutil.rmtree(cache)
    return path


def worker(index, parent, jobs, results, broken):
    path = clone(index, parent)
    baseline = run_tests(path)
    if baseline['failed'] or baseline['build error']:
        broken.append(f'worker {index}: unmutated tests fail: {baseline["failed"] or "build error"}')
        return
    while not broken:
        try:
            mutation = jobs.get_nowait()
        except queue.Empty:
            return
        source_file = path / mutation['file']
        source = source_file.read_text()
        source_file.write_text(source.replace(mutation['old'], mutation['new']))
        try:
            result = run_tests(path)
        finally:
            source_file.write_text(source)
        results[mutation['name']] = result
        print(f'{mutation["name"]}: {verdict(mutation, result)} {result["failed"]} {result["sets"]}', flush=True)


def verdict(mutation, result):
    if result['build error']:
        return 'BUILD ERROR'
    if mutation.get('equivalent'):
        return 'EQUIVALENT KILLED' if result['failed'] else 'equivalent'
    return 'killed' if result['failed'] else 'SURVIVED'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check', action='store_true', help='only check that every mutation applies')
    parser.add_argument('names', nargs='*')
    args = parser.parse_args()
    mutations = load(args.names)
    problems = misplaced(mutations, ROOT)
    if problems or args.check:
        print(*problems, f'{len(mutations)} mutations, {len(problems)} do not apply', sep='\n')
        sys.exit(1 if problems else 0)
    workers = min(args.workers, len(mutations))
    print(f'{len(mutations)} mutations across {workers} workers', flush=True)
    jobs = queue.Queue()
    for mutation in mutations:
        jobs.put(mutation)
    results, broken = {}, []
    start = time.time()
    parent = Path(tempfile.mkdtemp(prefix='word-check-mutations-'))
    try:
        threads = [threading.Thread(target=worker, args=(index, parent, jobs, results, broken)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        shutil.rmtree(parent, ignore_errors=True)
    failures = broken + [f'{mutation["name"]}: {verdict(mutation, results[mutation["name"]])}'
                         for mutation in mutations
                         if mutation['name'] in results and verdict(mutation, results[mutation['name']]) not in ('killed', 'equivalent')]
    failures += [f'{mutation["name"]}: not run' for mutation in mutations if mutation['name'] not in results and not broken]
    print(f'{len(results)}/{len(mutations)} mutations run in {time.time() - start:.0f}s')
    for failure in failures:
        print('FAIL', failure)
    print('mutations: FAIL' if failures else 'mutations: PASS')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
