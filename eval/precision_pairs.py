"""Paired comparison of two speech-suite reports (same names, same seeds):
mean difference of WER and of speaker similarity per run, with the standard
error over the pairs, and the runs whose transcripts differ.
usage: precision_pairs.py <reference report.json> <candidate report.json>"""
import json, sys
import numpy as np

a = {r["name"]: r for r in json.load(open(sys.argv[1]))["rows"] if "wer" in r}
b = {r["name"]: r for r in json.load(open(sys.argv[2]))["rows"] if "wer" in r}
names = sorted(set(a) & set(b))
for key, label in [("wer", "WER"), ("speaker_similarity", "speaker similarity (cloned runs)")]:
    pairs = [(a[n][key], b[n][key]) for n in names if key in a[n] and key in b[n] and (key != "speaker_similarity" or a[n]["cloned"])]
    d = np.array([y - x for x, y in pairs])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"{label}: n={len(d)} mean difference {d.mean():+.4f} (candidate minus reference), standard error {se:.4f}, "
          f"candidate better in {(d < 0).sum() if key == 'wer' else (d > 0).sum()}, worse in {(d > 0).sum() if key == 'wer' else (d < 0).sum()}, equal in {(d == 0).sum()}")
diff = [n for n in names if a[n]["hyp"] != b[n]["hyp"]]
print(f"transcripts differ in {len(diff)} of {len(names)} runs: {', '.join(diff)}")
for n in names:
    if a[n]["wer"] != b[n]["wer"]:
        print(f"  {n}: WER {a[n]['wer']:.2f} -> {b[n]['wer']:.2f}: {a[n]['hyp']!r} -> {b[n]['hyp']!r}")
