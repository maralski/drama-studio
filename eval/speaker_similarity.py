"""Speaker similarity (resemblyzer embedding cosine) between the reference clip and each cloned e2e output; writes it into report.json."""
import json, sys
from pathlib import Path
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

def main(run_dir, ref_wav):
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    enc = VoiceEncoder()
    ref = enc.embed_utterance(preprocess_wav(Path(ref_wav)))
    sims, plain = [], []
    for row in report["rows"]:
        wav = run_dir / f"{row['name']}.wav"
        if "wer" not in row or not wav.exists():
            continue
        sim = float(np.dot(ref, enc.embed_utterance(preprocess_wav(wav))))
        row["speaker_similarity"] = sim
        (sims if row["cloned"] else plain).append(sim)
    report["summary"]["mean_speaker_similarity"] = float(np.mean(sims)) if sims else None
    report["summary"]["mean_speaker_similarity_uncloned"] = float(np.mean(plain)) if plain else None
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"cloned: mean {np.mean(sims):.3f} min {np.min(sims):.3f} | uncloned (random voice): mean {np.mean(plain):.3f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/clip.wav")
