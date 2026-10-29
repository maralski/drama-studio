#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Information content of the bf16 weights a decode frame streams, per matrix
family, read with numpy over the memory-mapped safetensors shards (CPU only).

For every family: the sign bit, the 7 mantissa bits (entropy and per-bit zero
rate) and the 8 exponent bits (entropy, alphabet, mode), then the exponent
range inside blocks of 16 to 256 consecutive weights along K, from which the
cost of each exact packing scheme follows.

  eval/.venv/bin/python -I eval/weight_entropy.py [--layers N] [--out FILE.json]
"""
import argparse, glob, json, os, re, struct, sys, time
import numpy as np

SNAPSHOT = os.path.expanduser("~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots")

# (family, regex over the tensor name, rows used per frame or None for all).
FAMILIES = [
    ("gdn.qkv", r"linear_attn\.in_proj_qkv\.weight$", None),
    ("gdn.z", r"linear_attn\.in_proj_z\.weight$", None),
    ("gdn.a", r"linear_attn\.in_proj_a\.weight$", None),
    ("gdn.b", r"linear_attn\.in_proj_b\.weight$", None),
    ("gdn.out", r"linear_attn\.out_proj\.weight$", None),
    ("attn.q", r"self_attn\.q_proj\.weight$", None),
    ("attn.k", r"self_attn\.k_proj\.weight$", None),
    ("attn.v", r"self_attn\.v_proj\.weight$", None),
    ("attn.o", r"self_attn\.o_proj\.weight$", None),
    ("moe.router", r"mlp\.gate\.weight$", None),
    ("moe.shared_gate", r"mlp\.shared_expert\.gate_proj\.weight$", None),
    ("moe.shared_up", r"mlp\.shared_expert\.up_proj\.weight$", None),
    ("moe.shared_down", r"mlp\.shared_expert\.down_proj\.weight$", None),
    ("moe.expert_gate", r"mlp\.shared_expert_gate\.weight$", None),
    ("lm_head.semantic", r"^lm_head\.weight$", "semantic"),
    ("tower.eh", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.eh_proj\.weight$", None),
    ("tower.wqkv", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.transformer_layer\.attention\.wqkv\.weight$", None),
    ("tower.wo", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.transformer_layer\.attention\.wo\.weight$", None),
    ("tower.w1", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.transformer_layer\.feed_forward\.w1\.weight$", None),
    ("tower.w3", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.transformer_layer\.feed_forward\.w3\.weight$", None),
    ("tower.w2", r"audio_decoder_mtp(_shared)?\.layers\.\d+\.transformer_layer\.feed_forward\.w2\.weight$", None),
    ("tower.head", r"audio_decoder_mtp\.heads\.\d+\.weight$", None),
]

BLOCKS = [16, 32, 64, 128, 256]
OFFSET_BITS = [2, 3, 4]
ALPHABET_BITS = [3, 4]
CODE_BITS = [2, 3, 4]
TILE = 256


def snapshot_dir(path):
    if path:
        return path
    dirs = sorted(glob.glob(os.path.join(SNAPSHOT, "*")))
    if not dirs:
        sys.exit("no drama-3 snapshot in the Hugging Face cache")
    return dirs[-1]


class Shards:
    """Memory-mapped safetensors shards; tensor(name) returns a numpy view."""

    def __init__(self, directory):
        index = json.load(open(os.path.join(directory, "model.safetensors.index.json")))
        self.weight_map = index["weight_map"]
        self.maps = {}
        self.headers = {}
        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(directory, shard)
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            n = struct.unpack("<Q", bytes(mm[:8]))[0]
            self.headers[shard] = (json.loads(bytes(mm[8:8 + n]).decode()), 8 + n)
            self.maps[shard] = mm

    def names(self):
        return sorted(self.weight_map)

    def tensor(self, name):
        shard = self.weight_map[name]
        header, base = self.headers[shard]
        entry = header[name]
        a, b = entry["data_offsets"]
        raw = self.maps[shard][base + a:base + b]
        dtype = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8}[entry["dtype"]]
        return entry["dtype"], raw.view(dtype).reshape(entry["shape"])


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def analyze_tensor(bits):
    """bits: uint16 [N, K] bf16 patterns. Returns the per-tensor counts."""
    n, k = bits.shape
    sign = (bits >> 15).astype(np.uint8)
    exp = ((bits >> 7) & 0xFF).astype(np.uint8)
    man = (bits & 0x7F).astype(np.uint8)
    r = {
        "count": int(bits.size),
        "neg": int(sign.sum()),
        "exp_hist": np.bincount(exp.ravel(), minlength=256),
        "man_hist": np.bincount(man.ravel(), minlength=128),
        "man_bit_zero": [int((man & (1 << b) == 0).sum()) for b in range(7)],
        "zero": int((bits & 0x7FFF == 0).sum()),
        "blocks": {},
    }
    for B in BLOCKS:
        if k % B:
            continue
        e = exp.reshape(n, k // B, B)
        rng = e.max(axis=2).astype(np.int16) - e.min(axis=2).astype(np.int16)
        fit = {kb: (rng <= (1 << kb) - 1) for kb in OFFSET_BITS}
        tiles_per_row = k // TILE
        bpt = TILE // B
        r["blocks"][B] = {
            "blocks": int(rng.size),
            "fit": {kb: int(fit[kb].sum()) for kb in OFFSET_BITS},
            "tile_fit": {kb: int(fit[kb].reshape(n, tiles_per_row, bpt).all(axis=2).sum()) for kb in OFFSET_BITS},
            "tiles": n * tiles_per_row,
        }
    # Per-tile window of 2^c - 1 exponents ending d below the tile's maximum,
    # with the last code an escape to a raw exponent byte; each tile takes the
    # d (0 to 7) that escapes least. Escaped weights, not tiles, are counted.
    e = exp.reshape(n, k // TILE, TILE)
    top = e.max(axis=2, keepdims=True).astype(np.int16)
    e16 = e.astype(np.int16)
    r["codes"] = {}
    for c in CODE_BITS:
        width = (1 << c) - 1
        best = None
        for d in range(8):
            inside = ((e16 <= top - d) & (e16 > top - d - width)).sum(axis=2)
            best = inside if best is None else np.maximum(best, inside)
        r["codes"][c] = {"escaped": int((TILE - best).sum()), "tiles": n * (k // TILE)}
    # Per-tensor alphabet: the 2^j most frequent exponents; a tile escapes
    # when any of its 256 exponents lies outside the alphabet.
    order = np.argsort(-r["exp_hist"])
    r["alphabet"] = {}
    for j in ALPHABET_BITS:
        table = np.zeros(256, dtype=bool)
        table[order[: 1 << j]] = True
        inside = table[exp]
        r["alphabet"][j] = {
            "weights_in": int(inside.sum()),
            "tiles_in": int(inside.reshape(n, k // TILE, TILE).all(axis=2).sum()),
            "tiles": n * (k // TILE),
        }
    return r


def merge(into, r):
    if not into:
        into.update({"count": 0, "neg": 0, "exp_hist": np.zeros(256, np.int64), "man_hist": np.zeros(128, np.int64),
                     "man_bit_zero": [0] * 7, "zero": 0, "blocks": {}, "alphabet": {}, "codes": {}, "tensors": 0,
                     "exp_entropy_sum": 0.0, "distinct_sum": 0})
    into["count"] += r["count"]
    into["neg"] += r["neg"]
    into["exp_hist"] += r["exp_hist"]
    into["man_hist"] += r["man_hist"]
    into["man_bit_zero"] = [a + b for a, b in zip(into["man_bit_zero"], r["man_bit_zero"])]
    into["zero"] += r["zero"]
    into["tensors"] += 1
    into["exp_entropy_sum"] += entropy(r["exp_hist"]) * r["count"]
    into["distinct_sum"] += int((r["exp_hist"] > 0).sum())
    for B, b in r["blocks"].items():
        d = into["blocks"].setdefault(B, {"blocks": 0, "tiles": 0, "fit": {kb: 0 for kb in OFFSET_BITS}, "tile_fit": {kb: 0 for kb in OFFSET_BITS}})
        d["blocks"] += b["blocks"]
        d["tiles"] += b["tiles"]
        for kb in OFFSET_BITS:
            d["fit"][kb] += b["fit"][kb]
            d["tile_fit"][kb] += b["tile_fit"][kb]
    for j, a in r["alphabet"].items():
        d = into["alphabet"].setdefault(j, {"weights_in": 0, "tiles_in": 0, "tiles": 0})
        for key in d:
            d[key] += a[key]
    for c, a in r["codes"].items():
        d = into["codes"].setdefault(c, {"escaped": 0, "tiles": 0})
        for key in d:
            d[key] += a[key]


def scheme_costs(f):
    """Bits per weight of each exact scheme on the family, with its escape rate.

    Layout per 256-weight tile: 256 sign+mantissa bytes, k bit-planes of 32
    bytes, one base byte per block, a 4-byte tile header; an escaped tile keeps
    its slot and spills 512 raw bytes."""
    out = {}
    for B, b in f["blocks"].items():
        for kb in OFFSET_BITS:
            esc = 1 - b["tile_fit"][kb] / b["tiles"]
            bytes_per_tile = TILE + 32 * kb + TILE // B + 4 + esc * 512
            out[f"base{B}+off{kb}"] = {"bits": bytes_per_tile * 8 / TILE, "escape": esc}
    for j, a in f["alphabet"].items():
        esc = 1 - a["tiles_in"] / a["tiles"]
        bytes_per_tile = TILE + 32 * j + 4 + esc * 512
        out[f"alphabet{j}"] = {"bits": bytes_per_tile * 8 / TILE, "escape": esc}
    # Escape codes: c bit-planes of 32 bytes, a 4-byte header holding the base
    # and the tile's offset into the escape bytes, one byte per escaped weight.
    for c, a in f["codes"].items():
        esc = a["escaped"] / (a["tiles"] * TILE)
        bytes_per_tile = TILE + 32 * c + 4 + esc * TILE
        out[f"code{c}+esc"] = {"bits": bytes_per_tile * 8 / TILE, "escape": esc}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--layers", type=int, default=None, help="only the first N tensors of each family")
    ap.add_argument("--family", default=None, help="regex over family names")
    ap.add_argument("--out", default=None, help="write the per-family results as JSON")
    args = ap.parse_args()
    shards = Shards(snapshot_dir(args.checkpoint))
    config = json.load(open(os.path.join(snapshot_dir(args.checkpoint), "config.json")))
    sem0 = config["semantic_start_token_id"]
    families = {}
    t0 = time.time()
    for fam, pattern, rows in FAMILIES:
        if args.family and not re.search(args.family, fam):
            continue
        names = [n for n in shards.names() if re.search(pattern, n)]
        if args.layers:
            names = names[: args.layers]
        acc = {}
        for name in names:
            dtype, t = shards.tensor(name)
            if dtype != "BF16":
                print(f"{fam}: {name} is {dtype} {t.shape}, skipped", file=sys.stderr)
                continue
            if rows == "semantic":
                t = t[sem0:sem0 + 4096]
            merge(acc, analyze_tensor(np.ascontiguousarray(t)))
            print(f"{fam}: {name} {list(t.shape)} ({time.time() - t0:.0f} s)", file=sys.stderr)
        if acc:
            acc["shape"] = list(t.shape)
            families[fam] = acc
    report(families, args.out)


def report(families, out):
    rows = []
    print("family            tensors  shape          Mweights  neg    zero      H(exp) distinct  mode  H(man)  man bit0..2 zero")
    for fam, f in families.items():
        c = f["count"]
        hexp = entropy(f["exp_hist"])
        hman = entropy(f["man_hist"])
        mode = int(np.argmax(f["exp_hist"]))
        distinct = int((f["exp_hist"] > 0).sum())
        bz = " ".join(f"{z / c:.3f}" for z in f["man_bit_zero"][:3])
        print(f"{fam:17s} {f['tensors']:5d}  {str(f['shape']):14s} {c / 1e6:8.1f}  {f['neg'] / c:.3f}  {f['zero'] / c:.2e}  {hexp:5.3f}  {distinct:4d}   {mode - 127:4d}  {hman:5.3f}  {bz}")
        rows.append((fam, f))
    print()
    print("exponent range fits k offset bits: fraction of blocks (tiles of 256 with every block fitting)")
    print("family            " + "  ".join(f"B={B:3d} k2/k3/k4             " for B in BLOCKS))
    for fam, f in rows:
        cells = []
        for B in BLOCKS:
            b = f["blocks"].get(B)
            if not b:
                cells.append(" " * 30)
                continue
            cells.append("/".join(f"{b['fit'][kb] / b['blocks']:.4f}({b['tile_fit'][kb] / b['tiles']:.3f})" for kb in OFFSET_BITS))
        print(f"{fam:17s} " + "  ".join(cells))
    print()
    print("alphabet of the 2^j most frequent exponents: fraction of weights inside (tiles fully inside)")
    for fam, f in rows:
        cells = [f"j={j}: {a['weights_in'] / f['count']:.5f} ({a['tiles_in'] / a['tiles']:.4f})" for j, a in f["alphabet"].items()]
        print(f"{fam:17s} " + "  ".join(cells))
    print()
    print("entropy floor, bits per weight: H(sign) + H(exp) + H(man)")
    for fam, f in rows:
        c = f["count"]
        hs = entropy(np.array([f["neg"], c - f["neg"]]))
        print(f"{fam:17s} {hs + entropy(f['exp_hist']) + entropy(f['man_hist']):6.3f}")
    print()
    schemes = None
    totals = {}
    frame_bytes = 0
    for fam, f in rows:
        costs = scheme_costs(f)
        if schemes is None:
            schemes = list(costs)
            print("bits per weight achieved (tile escape rate)")
            print("family            " + " ".join(f"{s:>14s}" for s in schemes))
        frame_bytes += f["count"] * 2
        for s in schemes:
            totals.setdefault(s, 0.0)
            totals[s] += costs[s]["bits"] / 8 * f["count"]
        print(f"{fam:17s} " + " ".join(f"{costs[s]['bits']:6.2f}({costs[s]['escape']:.4f})" for s in schemes))
    print()
    print(f"bf16 bytes per frame over these families: {frame_bytes / 1e9:.3f} GB")
    for s in schemes:
        print(f"  {s:14s} {totals[s] / 1e9:.3f} GB  ({totals[s] / frame_bytes * 100:.1f} % of bf16, saves {(frame_bytes - totals[s]) / 1e9:.3f} GB)")
    if out:
        ser = {}
        for fam, f in families.items():
            g = {k: v for k, v in f.items() if k not in ("exp_hist", "man_hist")}
            g["exp_hist"] = f["exp_hist"].tolist()
            g["man_hist"] = f["man_hist"].tolist()
            g["schemes"] = scheme_costs(f)
            ser[fam] = g
        json.dump(ser, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
