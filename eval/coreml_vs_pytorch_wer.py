"""Decodes the same frames with the shipped Core ML codec and with the PyTorch codec, transcribes both, and compares word error rates."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e2e import wer
import whisper

run = Path(sys.argv[1]); drama = Path(sys.argv[2])
sentences = {s["id"]: s["text"] for s in json.loads((Path(__file__).parent / "sentences.json").read_text())}
report = json.loads((run / "report.json").read_text())
asr = whisper.load_model("small.en")
out = run / "coreml"; out.mkdir(exist_ok=True)
rows = []
for r in report["rows"]:
    if "wer" not in r: continue
    wav = out / f"{r['name']}.wav"
    subprocess.run([str(drama), "decode", "--frames", str(run / f"{r['name']}.npy"), "--wav", str(wav)], check=True, capture_output=True)
    hyp = asr.transcribe(str(wav), fp16=False, language="en")["text"]
    rows.append({"name": r["name"], "wer_pytorch": r["wer"], "wer_coreml": wer(sentences[r["name"].split("_")[0]], hyp), "hyp_coreml": hyp.strip()})
    print(f"{r['name']:22s} pytorch {r['wer']:.2f}  coreml {rows[-1]['wer_coreml']:.2f}")
summary = {"mean_wer_pytorch": float(np.mean([x["wer_pytorch"] for x in rows])), "mean_wer_coreml": float(np.mean([x["wer_coreml"] for x in rows])),
           "runs": len(rows), "runs_where_coreml_worse": int(sum(x["wer_coreml"] > x["wer_pytorch"] + 1e-9 for x in rows))}
(out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary))
