#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Generate a designed voice through the running app and check an English script.

Requires the evaluation Python environment's torch and openai-whisper packages.
Uses cached Whisper weights only; it does not download models or change voices.

    python3 tools/verify_voice_design.py --out /tmp/duke-verification

Defaults to the Duke regression sample. --name, --description, --scene and --text
can check any preset or custom design. Writes recordings and results for five
fresh designs and two uses of a cached voice. Laughter is permitted; missing,
replaced or additional spoken words fail.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import re
import urllib.request
import wave
import unicodedata

TEXT = "Oh, you came all this way to stop me? [laughs] How delightfully brave of you."
CHARACTER = {
    "name": "The Duke",
    "description": "silky, theatrical villain with a rich low voice, amused and unhurried",
}
BASE = "http://127.0.0.1:8780"
NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    map(str, range(21)),
))


ENGLISH_FORMS = json.loads((Path(__file__).resolve().parents[1] / "Sources/DramaEngine/Verification/EnglishSpokenForms.json").read_text())


def without_reaction_annotations(text, line):
    for annotation in ENGLISH_FORMS['reaction_annotations']:
        if re.search(annotation['cue_pattern'], line, flags=re.IGNORECASE):
            label = '(?:' + annotation['label_pattern'] + ')'
            pattern = r'\(\s*' + label + r'\s*\)|\[\s*' + label + r'\s*\]|\*\s*' + label + r'\s*\*'
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def words(text):
    # Canonical spoken forms preserve every lexical word while allowing written
    # abbreviations, contractions and small integers from speech recognizers.
    for title, spoken in ENGLISH_FORMS['titles'].items():
        text = re.sub(r"\b" + re.escape(title) + r"\.?(?=\s+[^\W\d_])", spoken, text)
    text = text.replace("’", "'").lower()
    text = ''.join(char for char in unicodedata.normalize('NFKD', text) if not unicodedata.combining(char))
    text = re.sub(r"\bcannot\b", 'can not', text)
    for short, long in ENGLISH_FORMS['contractions'].items():
        text = re.sub(r"\b" + re.escape(short) + r"\b", long, text)
    text = re.sub(r"(?<![a-z0-9.])[-−+](?=\d|\.\d)", lambda m: ' plus ' if m[0] == '+' else ' minus ', text)
    text = text.replace('%', ' percent ')
    text = re.sub(r"(?<![a-z0-9.])(\d*)\.(\d+)(?![a-z0-9]|\.\d)",
                  lambda m: (m[1] or '0') + ' point ' + ' '.join(m[2]), text)
    return [NUMBER_WORDS.get(word, word) for word in re.findall(r"[a-z']+|\d+", text)]


def has_reaction_cue(line):
    return re.search(r'\[(?:laugh[^\]]*|chuckle[^\]]*|sigh[^\]]*|gasp[^\]]*|cough[^\]]*|clears throat|hesitates|pause)\]', line, flags=re.IGNORECASE) is not None


def differences(expected, transcript, reactions, *, line=''):
    if line:
        reactions = has_reaction_cue(line)
    actual = words(without_reaction_annotations(transcript, line))
    changes = []
    for op, a, b, c, d in difflib.SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        # Only extra nonverbal sounds are permitted. Never remove expected
        # words such as "he" or excuse a deletion/substitution as a reaction.
        if reactions and op == 'insert' and all(
                re.fullmatch(ENGLISH_FORMS['reaction_pattern'], word)
                or word in ENGLISH_FORMS['reaction_words']
                # Audio-only recognizers spell the same nonlexical hum as
                # hmm, mm or mmm. This permits insertions, never missing words.
                or re.fullmatch(r'h?m{2,}', word)
                for word in actual[c:d]):
            continue
        changes.append({'operation': op, 'expected': expected[a:b], 'heard': actual[c:d]})
    return changes



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--reuse-seeds", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--name", default=CHARACTER["name"])
    parser.add_argument("--description", default=CHARACTER["description"])
    parser.add_argument("--scene")
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args()
    character = {"name": args.name, "description": args.description}
    if args.scene:
        character["scene"] = args.scene
    import torch
    import whisper

    weights = Path.home() / ".cache/whisper" / (args.asr_model + ".pt")
    if not weights.is_file():
        parser.error(f"Cached recognition model is missing: {weights}")
    token = os.environ.get("DRAMA_API_TOKEN")
    if token is None:
        token = (Path.home() / "Library/Application Support/DramaStudio/api-token").read_text().strip()
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def save():
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")

    def generate(kind, seed, reference=None):
        label = f"{kind}-{seed}"
        body = {"input": args.text, "character": character, "seed": seed,
                "response_format": "pcm", "stream": True}
        if reference:
            body["previous_takes"] = [{"take_id": reference}]
        print("Generating", label, flush=True)
        request = urllib.request.Request(BASE + "/v1/audio/speech", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            audio = response.read()
            take_id = response.headers["X-Drama-Take-Id"]
            if response.status != 200 or not audio or len(audio) % 2:
                raise RuntimeError(f"{label}: invalid PCM response")
        path = args.out / (label + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes(audio)
        row = {"label": label, "seed": seed, "input": args.text, "character": character,
               "reference": reference, "take_id": take_id, "audio": str(path),
               "audio_seconds": len(audio) / 88200}
        rows.append(row)
        save()
        print("Recorded", label, flush=True)
        return take_id

    try:
        anchor = None
        for seed in args.seeds:
            take_id = generate("fresh", seed)
            if anchor is None:
                anchor = take_id
        for seed in args.reuse_seeds:
            generate("reused", seed, anchor)
    except Exception as error:
        rows.append({"error": str(error)})
        save()
        raise

    # Recognition runs after generation so it does not contend with the engine.
    torch.set_num_threads(2)
    recognizer = whisper.load_model(str(weights), device="cpu")
    spoken = re.sub(r"\[[^]]*\]|<[^>]*>", " ", args.text)
    expected = words(spoken)
    for row in rows:
        result = recognizer.transcribe(row["audio"], fp16=False, language="en", temperature=0,
                                        condition_on_previous_text=False)
        row["transcript"] = result["text"].strip()
        row["differences"] = differences(expected, result["text"], has_reaction_cue(args.text), line=args.text)
        row["passed"] = not row["differences"]
        save()
        print("PASS" if row["passed"] else "FAIL", row["label"], row["transcript"], flush=True)
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)


if __name__ == "__main__":
    main()
