# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Rescores non-English reliability rows with Whisper's multilingual `small`
model (the study runs small.en, which cannot transcribe French or German) and
writes the corrected rows to <out>/rescored/*.jsonl, leaving the originals."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e2e import wer
from reliability import spoken_text
import whisper

out = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/out/reliability")
slices = {item["id"]: item for p in (Path(__file__).parent / "reliability/slices").glob("*.json") for item in json.loads(p.read_text())}
asr = whisper.load_model("small")
(out / "rescored").mkdir(exist_ok=True)
changed = 0
for log in sorted(out.glob("*.jsonl")):
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    for r in rows:
        item = slices.get(r["id"])
        if not item or item.get("language", "en") == "en" or "wer" not in r:
            continue
        hyp = asr.transcribe(str(out / f"{r['name']}.wav"), fp16=False, language=item["language"])["text"].strip()
        r["hyp_small_en"], r["hyp"] = r["hyp"], hyp
        r["wer"] = round(wer(spoken_text(item["text"]), hyp), 3)
        threshold = item.get("wer_threshold", 0.15)
        if r["verdict"] in ("garbled", "ok", "dropped"):
            ratio = r["spoken_seconds"] / max(r["expected_seconds"], 0.1)
            r["verdict"] = "ok" if r["wer"] <= threshold else ("dropped" if ratio < 0.6 else "garbled")
        changed += 1
        print(f"{r['name']:40s} {r['verdict']:8s} wer={r['wer']}  {hyp[:70]}")
    (out / "rescored" / log.name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(f"rescored {changed} non-English rows into {out / 'rescored'}")
