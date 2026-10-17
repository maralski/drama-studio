"""Reliability study: does every sentence come out as speech, every time?

One row per (condition, mode, seed): the drama CLI generates frames, the
reference PyTorch codec decodes them, Whisper small.en transcribes, and the row
records word error rate, whether eos was reached, how many retries ran, the
spoken and silent durations, and a verdict. Rows append to a JSONL file so
several workers can run disjoint slices in parallel and the summary joins them.

  eval/.venv/bin/python -u eval/reliability.py --slice eval/reliability/slices/a.json --mode full --seeds 1 2 3 4 5
  eval/.venv/bin/python eval/reliability.py --summary eval/out/reliability
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from e2e import wer, normalize  # noqa: E402

FRAME_SECONDS = 2048 / 44100
REF_TEXT = "<|speaker:0|> Every morning I walk to the little bakery at the end of our street."


def spoken_text(text):
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run_one(drama, item, mode, seed, out_dir, codec, asr):
    name = f"{item['id']}__{mode}__s{seed}"
    frames_path = out_dir / f"{name}.npy"
    cmd = [str(drama), "generate", "--precision-mode", mode, "--seed", str(seed), "--text", item["text"],
           "--chunking", item.get("chunking", "sentences"), "--out", str(frames_path)]
    for flag, value in item.get("flags", {}).items():
        cmd += [f"--{flag}"] + ([str(value)] if value != "" else [])
    if item.get("ref"):
        cmd += ["--ref", str(HERE / "data/clip_codes.npy"), "--ref-text", REF_TEXT]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    row = {"name": name, "id": item["id"], "category": item.get("category", ""), "mode": mode, "seed": seed, "wall_seconds": round(wall, 2)}
    if proc.returncode != 0 or not frames_path.exists():
        row.update(verdict="error", error=proc.stderr[-400:])
        return row
    stderr = proc.stderr
    row["retries"] = stderr.count("retry ")
    row["eos"] = all("eos=yes" in line for line in stderr.splitlines() if "eos=" in line)
    row["chunks"] = sum(1 for line in stderr.splitlines() if "eos=" in line)
    frames = np.load(frames_path)
    import torch
    codes = torch.from_numpy(np.ascontiguousarray(frames[:, 1:].T)).long()[None]
    with torch.inference_mode():
        audio = codec.from_indices(codes)[0, 0].float().numpy()
    wav = out_dir / f"{name}.wav"
    sf.write(wav, audio, 44100)
    frame_rms = np.sqrt((audio[: len(audio) // 2048 * 2048].reshape(-1, 2048) ** 2).mean(1))
    loud = np.where(frame_rms >= 0.01)[0]
    quiet_tail = int(len(frame_rms) - (loud.max() + 1)) if len(loud) else len(frame_rms)
    hyp = asr.transcribe(str(wav), fp16=False, language=item.get("language", "en"))["text"].strip()
    target = spoken_text(item["text"])
    row.update(frames=int(len(frames)), seconds=round(len(audio) / 44100, 2), quiet_tail_seconds=round(quiet_tail * FRAME_SECONDS, 2),
               spoken_seconds=round((len(frame_rms) - quiet_tail) * FRAME_SECONDS, 2), wer=round(wer(target, hyp), 3), hyp=hyp,
               expected_seconds=round(len(target) / 16, 2))
    ratio = row["spoken_seconds"] / max(row["expected_seconds"], 0.1)
    threshold = item.get("wer_threshold", 0.15)
    if not row["eos"]:
        row["verdict"] = "runaway"
    elif row["wer"] > threshold and ratio < 0.6:
        row["verdict"] = "dropped"
    elif row["wer"] > threshold:
        row["verdict"] = "garbled"
    else:
        row["verdict"] = "ok"
    return row


def summary(out_dir):
    rows = [json.loads(line) for p in Path(out_dir).glob("*.jsonl") for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        print("no rows"); return
    modes = sorted({r["mode"] for r in rows})
    print(f"{len(rows)} rows, modes {modes}\n")
    print(f"{'category':14s} {'mode':9s} {'n':>3s} {'ok':>5s} {'dropped':>7s} {'garbled':>7s} {'runaway':>7s} {'error':>5s} {'meanWER':>8s} {'retries':>7s}")
    cats = sorted({r["category"] for r in rows})
    for cat in cats + ["ALL"]:
        for mode in modes:
            sel = [r for r in rows if r["mode"] == mode and (cat == "ALL" or r["category"] == cat)]
            if not sel: continue
            v = [r.get("verdict") for r in sel]
            wers = [r["wer"] for r in sel if "wer" in r]
            print(f"{cat:14s} {mode:9s} {len(sel):3d} {v.count('ok')/len(sel):5.0%} {v.count('dropped'):7d} {v.count('garbled'):7d} {v.count('runaway'):7d} {v.count('error'):5d} {np.mean(wers) if wers else float('nan'):8.3f} {sum(r.get('retries', 0) for r in sel):7d}")
    if len(modes) == 2:
        a, b = modes
        by = {}
        for r in rows: by.setdefault((r["id"], r["seed"]), {})[r["mode"]] = r
        pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v and "wer" in v[a] and "wer" in v[b]]
        if pairs:
            d = np.array([pb["wer"] - pa["wer"] for pa, pb in pairs])
            worse = sum((pb["verdict"] != "ok") and (pa["verdict"] == "ok") for pa, pb in pairs)
            better = sum((pa["verdict"] != "ok") and (pb["verdict"] == "ok") for pa, pb in pairs)
            print(f"\npaired {len(pairs)}: mean WER({b}) - WER({a}) = {d.mean():+.3f} (standard error {d.std(ddof=1)/np.sqrt(len(d)):.3f}); "
                  f"{b} fails where {a} is ok in {worse} pairs, the reverse in {better}")
    bad = [r for r in rows if r.get("verdict") != "ok"]
    if bad:
        print("\nfailures:")
        for r in bad: print(f"  {r['name']:40s} {r.get('verdict'):8s} wer={r.get('wer', 'n/a')} retries={r.get('retries', 0)} spoken={r.get('spoken_seconds')}s/{r.get('expected_seconds')}s  {str(r.get('hyp', r.get('error', '')))[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", type=Path, default=HERE / "out/reliability")
    ap.add_argument("--drama", type=Path, default=ROOT / ".build/release/drama")
    ap.add_argument("--summary", type=Path)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary); return
    a.out.mkdir(parents=True, exist_ok=True)
    items = json.loads(a.slice.read_text())
    import whisper
    from decode_codes import load_codec
    asr = whisper.load_model("small.en")
    codec = load_codec(next(Path.home().glob(".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/*/")) / "codec.pth")
    log = a.out / f"{a.slice.stem}__{a.mode}.jsonl"
    done = {json.loads(l)["name"] for l in log.read_text().splitlines()} if log.exists() else set()
    with log.open("a") as f:
        for item in items:
            for seed in a.seeds:
                if f"{item['id']}__{a.mode}__s{seed}" in done: continue
                row = run_one(a.drama, item, a.mode, seed, a.out, codec, asr)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{row['name']:40s} {row.get('verdict'):8s} wer={row.get('wer', 'n/a')} retries={row.get('retries', 0)} {str(row.get('hyp', ''))[:60]}", flush=True)


if __name__ == "__main__":
    main()
