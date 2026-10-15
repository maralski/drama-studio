"""End-to-end speech evaluation of the Metal engine.

For each sentence in sentences.json, with and without the reference voice:
generate frames with the drama CLI, decode to audio, transcribe with Whisper,
and score the word error rate against the input text. Cloned items also get
the speaker similarity between the reference clip and the output when
resemblyzer is installed. Results land in out/e2e/<tag>/report.json.
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def normalize(text):
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.lower().replace("seashells", "sea shells").replace("seashore", "sea shore")
    return re.findall(r"[a-z0-9']+", text)


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--sentences", type=Path, default=HERE / "sentences.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--drama", type=Path, default=ROOT / ".build/release/drama")
    ap.add_argument("--ref-codes", type=Path, default=HERE / "data/clip_codes.npy")
    ap.add_argument("--ref-wav", type=Path, default=HERE / "data/clip.wav")
    ap.add_argument("--ref-text", default="<|speaker:0|> Every morning I walk to the little bakery at the end of our street.")
    ap.add_argument("--extra", nargs="*", default=[], help="extra drama generate flags")
    ap.add_argument("--whisper", default="small.en")
    a = ap.parse_args()
    out = HERE / "out/e2e" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    sentences = json.loads(a.sentences.read_text())
    import whisper
    from decode_codes import load_codec
    asr = whisper.load_model(a.whisper)
    codec = load_codec(next(Path.home().glob(".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/*/")) / "codec.pth")
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        spk = VoiceEncoder()
        ref_embed = spk.embed_utterance(preprocess_wav(a.ref_wav))
    except Exception as e:
        print("speaker similarity disabled:", e)
        spk = None
    import torch
    rows = []
    for item in sentences:
        for cloned in (False, True):
            for seed in a.seeds:
                name = f"{item['id']}_{'clone' if cloned else 'plain'}_s{seed}"
                cmd = [str(a.drama), "generate", "--seed", str(seed), "--text", item["text"], "--out", str(out / f"{name}.npy"), *a.extra]
                if cloned:
                    cmd += ["--ref", str(a.ref_codes), "--ref-text", a.ref_text]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                gen_seconds = time.time() - t0
                if proc.returncode != 0:
                    rows.append({"name": name, "error": proc.stderr[-500:]}); print(name, "FAILED"); continue
                frames = np.load(out / f"{name}.npy")
                eos = "eos=yes" in proc.stderr
                codes = torch.from_numpy(np.ascontiguousarray(frames[:, 1:].T)).long()[None]
                with torch.inference_mode():
                    audio = codec.from_indices(codes)[0, 0].float().numpy()
                sf.write(out / f"{name}.wav", audio, 44100)
                frame_rms = np.sqrt((audio[: len(audio) // 2048 * 2048].reshape(-1, 2048) ** 2).mean(1))
                quiet_tail = int(len(frame_rms) - (np.where(frame_rms >= 0.01)[0].max() + 1)) if (frame_rms >= 0.01).any() else len(frame_rms)
                hyp = asr.transcribe(str(out / f"{name}.wav"), fp16=False, language="en")["text"]
                row = {"name": name, "seed": seed, "cloned": cloned, "frames": int(len(frames)), "seconds": len(audio) / 44100, "eos": eos,
                       "quiet_frames": int((frame_rms < 0.01).sum()), "quiet_tail_frames": quiet_tail, "wer": wer(item["text"], hyp), "hyp": hyp.strip(),
                       "gen_seconds": gen_seconds}
                if cloned and spk is not None:
                    row["speaker_similarity"] = float(np.dot(ref_embed, spk.embed_utterance(preprocess_wav(out / f"{name}.wav"))))
                rows.append(row)
                print(f"{name}: {row['frames']} frames eos={eos} wer={row['wer']:.2f} quiet_tail={quiet_tail}" + (f" spk={row['speaker_similarity']:.2f}" if 'speaker_similarity' in row else "") + f"  -> {hyp.strip()[:80]}")
    ok = [r for r in rows if "wer" in r]
    summary = {
        "runs": len(rows), "failed": len(rows) - len(ok),
        "mean_wer": float(np.mean([r["wer"] for r in ok])) if ok else None,
        "wer_le_0.1": float(np.mean([r["wer"] <= 0.1 for r in ok])) if ok else None,
        "eos_rate": float(np.mean([r["eos"] for r in ok])) if ok else None,
        "mean_quiet_tail_frames": float(np.mean([r["quiet_tail_frames"] for r in ok])) if ok else None,
        "mean_speaker_similarity": float(np.mean([r["speaker_similarity"] for r in ok if "speaker_similarity" in r])) if any("speaker_similarity" in r for r in ok) else None,
        "extra": a.extra,
    }
    (out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
