"""Performance-cue experiments: which chunking, retry, loop-guard and sampling
policy keeps every sentence of a cue line?

Every variant in conditions.json runs every line in lines.json through the
drama CLI under the app's configuration (--text-cap, codec probe, silence
stop), scored as eval/reliability.py scores (PyTorch codec, Whisper small.en,
word error rate, eos) plus one more column: whether every sentence of the
line is present in the transcript, because the duration rule misses a short
dropped clause.

  eval/.venv/bin/python -u eval/cues/run.py --variants a e --seeds 1 2 3 4 5
  eval/.venv/bin/python eval/cues/run.py --summary
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
from reliability import run_one, spoken_text  # noqa: E402
from e2e import normalize  # noqa: E402

OUT = EVAL / "out/cues"
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
BRACKET_CUE = re.compile(r"\[([^\[\]]*)\]")
# Opening styles from the app's palette that are not official tags, mapped to the closest official one.
STYLE_TO_TAG = {
    "with bright eager excitement": "excited",
    "icy calm": "calm",
    "heavy, deflated sadness": "sad",
    "sarcastic and deadpan": "sarcastic",
    "breathless and terrified": "scared",
    "furious, barely holding it together": "angry",
    "warm, gentle bedtime voice": "gentle",
    "low and menacing": "menacing",
}


def normalise_cues(text):
    def one(m):
        cue = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        return "[" + STYLE_TO_TAG.get(cue, cue) + "]"
    return BRACKET_CUE.sub(one, text)


def edit_distance(a, b):
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    d[:, 0] = np.arange(len(a) + 1); d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[len(a), len(b)])


def sentences_present(text, hyp):
    """One flag per spoken sentence of `text`: present when some window of the
    transcript's words is within a third of the sentence's words in edits."""
    words = normalize(hyp)
    flags = []
    for sentence in SENTENCE_BREAK.split(spoken_text(text)):
        ref = normalize(sentence)
        if not ref:
            continue
        n = len(ref)
        allowed = max(1, n // 3)
        best = min((edit_distance(ref, words[i:i + m]) for m in (n - 1, n, n + 1) if m >= 1 for i in range(0, max(1, len(words) - m + 1))), default=n)
        flags.append(best <= allowed)
    return flags


def item_for(variant, condition, line):
    item = dict(line)
    item["id"] = f"{variant}--{line['id']}"
    item["flags"] = {"text-cap": "", **condition.get("flags", {})}
    if "chunking" in condition:
        item["chunking"] = condition["chunking"]
    if condition.get("normalise"):
        item["text"] = normalise_cues(line["text"])
    return item


def summary(out_dir, lines_path):
    rows = [json.loads(l) for p in sorted(Path(out_dir).glob("*.jsonl")) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        print("no rows"); return
    lines = {l["id"]: l for l in json.loads(lines_path.read_text())}
    variants = sorted({r["variant"] for r in rows})
    print(f"{'variant':10s} {'runs':>4s} {'all sentences':>13s} {'WER<0.15':>9s} {'eos':>5s} {'runaways':>8s} {'loops':>5s} {'capped':>6s} {'retries':>7s} {'mean s':>7s} {'wall s':>7s}")
    for v in variants:
        sel = [r for r in rows if r["variant"] == v and "wer" in r]
        if not sel: continue
        n = len(sel)
        alls = sum(1 for r in sel if r["all_sentences"])
        good = sum(1 for r in sel if r["wer"] < 0.15)
        eos = sum(1 for r in sel if r["eos"])
        runaway = sum(1 for r in sel if r["verdict"] == "runaway")
        loops = sum(r.get("loops", 0) for r in sel)
        capped = sum(r.get("capped", 0) for r in sel)
        retries = sum(r.get("retries", 0) for r in sel)
        print(f"{v:10s} {n:4d} {alls:6d} ({alls / n:4.0%}) {good:4d} ({good / n:4.0%}) {eos:5d} {runaway:8d} {loops:5d} {capped:6d} {retries:7d} {np.mean([r['seconds'] for r in sel]):7.2f} {np.mean([r['wall_seconds'] for r in sel]):7.1f}")
    print("\nall sentences present, per line:")
    ids = [i for i in lines if any(r["line"] == i for r in rows)]
    print(f"{'line':12s} " + " ".join(f"{v:>7s}" for v in variants))
    for i in ids:
        cells = []
        for v in variants:
            sel = [r for r in rows if r["variant"] == v and r["line"] == i and "wer" in r]
            cells.append(f"{sum(1 for r in sel if r['all_sentences']):3d}/{len(sel):<3d}" if sel else "      -")
        print(f"{i:12s} " + " ".join(cells))
    bad = [r for r in rows if "wer" in r and not r["all_sentences"]]
    if bad:
        print("\nruns missing a sentence:")
        for r in sorted(bad, key=lambda r: r["name"]):
            print(f"  {r['name']:34s} {r['verdict']:8s} wer={r['wer']:.3f} retries={r.get('retries', 0)} spoken={r['spoken_seconds']}s/{r['expected_seconds']}s present={''.join('1' if f else '0' for f in r['sentences'])} {r['hyp'][:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--lines", type=Path, default=HERE / "lines.json")
    ap.add_argument("--conditions", type=Path, default=HERE / "conditions.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--drama", type=Path, default=EVAL.parent / ".build/release/drama")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a.out, a.lines); return
    a.out.mkdir(parents=True, exist_ok=True)
    conditions = json.loads(a.conditions.read_text())
    lines = json.loads(a.lines.read_text())
    import whisper
    from decode_codes import load_codec
    asr = whisper.load_model("small.en")
    codec = load_codec(next(Path.home().glob(".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/*/")) / "codec.pth")
    done = {json.loads(l)["name"] for p in a.out.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip()}
    log = a.out / f"{'-'.join(a.variants)}__s{min(a.seeds)}-{max(a.seeds)}.jsonl"
    with log.open("a") as f:
        for variant in a.variants:
            for line in lines:
                item = item_for(variant, conditions[variant], line)
                if item["text"] == line["text"] and conditions[variant].get("normalise"):
                    continue
                for seed in a.seeds:
                    name = f"{item['id']}__full__s{seed}"
                    if name in done: continue
                    row = run_one(a.drama, item, "full", seed, a.out, codec, asr)
                    row.update(variant=variant, line=line["id"], text=item["text"])
                    if "hyp" in row:
                        row["sentences"] = sentences_present(item["text"], row["hyp"])
                        row["all_sentences"] = all(row["sentences"])
                        row["loops"] = sum("loop=yes" in l for l in row.get("chunk_log", []))
                        row["capped"] = sum(1 for l in row.get("chunk_log", []) if "eos=no" in l and (m := re.search(r"(\d+) frames in.*cap=(\d+)", l)) and m.group(1) == m.group(2))
                    f.write(json.dumps(row) + "\n"); f.flush()
                    print(f"{name:36s} {row.get('verdict'):8s} wer={row.get('wer', 'n/a')} all={row.get('all_sentences')} retries={row.get('retries', 0)} {str(row.get('hyp', row.get('error', '')))[:60]}", flush=True)


if __name__ == "__main__":
    main()
