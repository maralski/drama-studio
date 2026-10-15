"""Checks a sampled engine trace: every pick must lie inside the top-k / top-p
set of the logits the engine itself produced, and the rank histogram must look
like draws from the truncated softmax rather than uniform noise."""
import argparse, json
import numpy as np

SEMANTIC_START = 248099
EOS = 248046


def truncated_probs(logits, temperature, top_k, top_p):
    order = np.argsort(-logits, kind="stable")
    v = logits[order] / max(temperature, 1e-5)
    p = np.exp(v - v.max()); p /= p.sum()
    cum = np.cumsum(p)
    dropped = (cum > top_p) | (np.arange(len(v)) >= top_k)
    dropped[0] = False
    kept = np.where(~dropped)[0]
    q = p[kept] / p[kept].sum()
    return order[kept], q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    a = ap.parse_args()
    d = np.load(a.trace)
    meta = json.loads(d["meta"].tobytes().decode())
    t, k, p = meta["temperature"], meta["top_k"], meta["top_p"]
    frames, cand, tower = d["frames"], d["cand_logits"], d["tower_logits"]
    candidates = np.concatenate(([EOS], np.arange(SEMANTIC_START, SEMANTIC_START + 4096)))
    outside = 0
    outside_main = 0
    ranks, expected_rank1, observed_rank1, total_logp = [], 0.0, 0, 0.0
    mains = list(d["prompt_rows"][:, 0])
    for i, f in enumerate(frames):
        picks = [(cand[i], int(np.where(candidates == f[0])[0][0]))] + [(tower[i, j], int(f[2 + j])) for j in range(9)]
        for j, (logits, pick) in enumerate(picks):
            ids, q = truncated_probs(logits.astype(np.float64), t, k, p)
            where = np.where(ids == pick)[0]
            if len(where) == 0 and j == 0:
                # A main token may come from the repetition-aware retry draw
                # (temperature 1, top-k 30, top-p 0.95) when the first draw repeated.
                retry_ids, _ = truncated_probs(logits.astype(np.float64), 1.0, 30, 0.95)
                repeat_possible = any(int(candidates[m]) in mains[-16:] for m in ids)
                if pick in retry_ids and repeat_possible:
                    mains.append(int(f[0]))
                    continue
                outside_main += 1
            if j == 0:
                mains.append(int(f[0]))
            if len(where) == 0:
                outside += 1
                continue
            r = int(where[0])
            ranks.append(r)
            expected_rank1 += q[0]
            observed_rank1 += r == 0
            total_logp += np.log(q[r])
    n = len(ranks) + outside
    print(f"{len(frames)} frames, {n} picks: {outside} outside the top-k/top-p set ({outside_main} main tokens not explained by a retry)")
    print(f"rank-0 picks: observed {observed_rank1} vs expected {expected_rank1:.1f} (binomial sd ~{np.sqrt(expected_rank1):.1f})")
    print(f"mean log-prob per pick {total_logp / max(len(ranks), 1):.3f}; rank histogram {np.bincount(ranks, minlength=8)[:8].tolist()} ...")
    mains = frames[:, 0]
    print("eos frames:", int((mains == EOS).sum()), "distinct c0:", len(set(frames[:, 1].tolist())))
    return 1 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())
