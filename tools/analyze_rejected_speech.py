#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Cross-check opt-in rejected-attempt recordings with cached local Whisper models."""
import argparse
import json
from pathlib import Path
import re
import time
import wave

from cue_reliability import differences
from verify_voice_design import words, has_reaction_cue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    import torch
    import whisper
    torch.set_num_threads(2)
    models = {name: whisper.load_model(str(Path.home() / '.cache/whisper' / (name + '.pt')), device='cpu')
              for name in ['small.en', 'base']}
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.watch_seconds
    while True:
        for file in sorted(args.input.glob('*.json')):
            output = args.out / file.name
            audio = file.with_suffix('.wav')
            if output.exists() or not audio.exists():
                continue
            try:
                native = json.loads(file.read_text())
                with wave.open(str(audio)) as recording:
                    if len(recording.readframes(recording.getnframes())) != recording.getnframes() * 2:
                        continue
            except (ValueError, OSError, EOFError, wave.Error):
                continue
            expected = re.sub(r'<\|[^|>]*\|>|\[[^\[\]]*\]|</?[A-Za-z][^<>]*>', ' ', native['line'])
            row = {'audit': str(file), 'audio': str(audio), 'native': native, 'recognizers': {}}
            for name, model in models.items():
                transcript = model.transcribe(str(audio), fp16=False, language='en', temperature=0,
                                               condition_on_previous_text=False)['text'].strip()
                row['recognizers'][name] = {'transcript': transcript,
                                            'differences': differences(words(expected), transcript, has_reaction_cue(native['line']), line=native['line'])}
            output.write_text(json.dumps(row, indent=2) + '\n')
            print(file.stem, native['line'], row['recognizers'], flush=True)
        if time.monotonic() >= deadline:
            return
        time.sleep(1)


if __name__ == '__main__':
    main()
