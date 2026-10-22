#!/usr/bin/env python3
"""Packs bf16 matrices from the drama-3 shards into the exponent-coded layout
gemv_packed_t reads (Sources/DramaEngine/Kernels/PackedLinear.metal), unpacks
them again and checks every bit pattern came back, then reports bits per
weight, the escape rate and the packing time. CPU only, numpy over the
memory-mapped shards.

Layout, per tile of 256 consecutive weights of a row (slot = row * K/256 + tile):
  headers[slot]  uint32: bits 0-7 base exponent, bits 8-31 byte offset of the
                 tile's escape list in the escape area
  mant[slot]     256 bytes: sign in bit 7, the 7 mantissa bits below
  codes[slot]    96 bytes: 32 lanes x 24 bits, lane l weights 8l..8l+7, 3 bits
                 each, little endian; code c < 7 is exponent base + c, code 7
                 escapes to the next byte of the tile's escape list
  escapes        one byte per escaped weight, tiles in slot order
The sections follow each other in that order, each 16-byte aligned.

  eval/.venv/bin/python -I eval/pack_bf16.py [--tensor REGEX] [--count N] [--write DIR]
"""
import argparse, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_entropy import Shards, snapshot_dir

TILE = 256
WINDOW = 7
CODE_BYTES = 96


def align16(x):
    return (x + 15) // 16 * 16


def pack(bits):
    """bits: uint16 [N, K], K a multiple of 256. Returns one bytes object and
    the section offsets."""
    n, k = bits.shape
    assert k % TILE == 0
    t = n * k // TILE
    e = ((bits >> 7) & 0xFF).astype(np.int16).reshape(t, TILE)
    hist = np.bincount((e + (np.arange(t, dtype=np.int64) * 256)[:, None]).ravel(), minlength=t * 256).reshape(t, 256)
    cs = np.zeros((t, 257), dtype=np.int32)
    np.cumsum(hist, axis=1, out=cs[:, 1:])
    base = np.argmax(cs[:, WINDOW:] - cs[:, :-WINDOW], axis=1).astype(np.int16)
    rel = e - base[:, None]
    escaped = (rel < 0) | (rel >= WINDOW)
    code = np.where(escaped, 7, rel).astype(np.uint32)
    q = np.zeros((t, 32), dtype=np.uint32)
    lanes = code.reshape(t, 32, 8)
    for j in range(8):
        q |= lanes[:, :, j] << (3 * j)
    codes = q.astype("<u4").view(np.uint8).reshape(t, 32, 4)[:, :, :3].reshape(t, CODE_BYTES)
    mant = (((bits >> 8) & 0x80) | (bits & 0x7F)).astype(np.uint8).reshape(t, TILE)
    counts = escaped.sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint32)
    assert offsets[-1] + counts[-1] < 1 << 24
    headers = (base.astype(np.uint32) | (offsets << 8)).astype("<u4")
    escapes = e[escaped].astype(np.uint8)
    h0, m0 = 0, align16(t * 4)
    c0 = align16(m0 + t * TILE)
    x0 = align16(c0 + t * CODE_BYTES)
    blob = np.zeros(x0 + len(escapes), dtype=np.uint8)
    blob[h0:h0 + t * 4] = headers.view(np.uint8)
    blob[m0:m0 + t * TILE] = mant.ravel()
    blob[c0:c0 + t * CODE_BYTES] = codes.ravel()
    blob[x0:] = escapes
    return blob, {"header": h0, "mant": m0, "code": c0, "escape": x0, "tiles": t, "escaped": int(len(escapes))}


def unpack(blob, sections, n, k):
    t = sections["tiles"]
    headers = blob[sections["header"]:sections["header"] + t * 4].view("<u4")
    mant = blob[sections["mant"]:sections["mant"] + t * TILE].reshape(t, TILE).astype(np.uint16)
    codes = blob[sections["code"]:sections["code"] + t * CODE_BYTES].reshape(t, 32, 3)
    escapes = blob[sections["escape"]:]
    q = codes[:, :, 0].astype(np.uint32) | (codes[:, :, 1].astype(np.uint32) << 8) | (codes[:, :, 2].astype(np.uint32) << 16)
    code = np.stack([(q >> (3 * j)) & 7 for j in range(8)], axis=2).reshape(t, TILE)
    base = (headers & 0xFF).astype(np.uint16)
    escaped = code == 7
    index = np.cumsum(escaped.ravel()) - 1
    e = np.where(escaped, escapes[index.reshape(t, TILE)], base[:, None] + code).astype(np.uint16)
    return (((mant & 0x80) << 8) | (e << 7) | (mant & 0x7F)).reshape(n, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tensor", default=r"layers\.0\.linear_attn\.in_proj_qkv\.weight$", help="regex over tensor names")
    ap.add_argument("--count", type=int, default=1, help="how many matching tensors")
    ap.add_argument("--write", default=None, help="directory for <name>.packed blobs")
    args = ap.parse_args()
    shards = Shards(snapshot_dir(args.checkpoint))
    names = [m for m in shards.names() if re.search(args.tensor, m)][: args.count]
    if not names:
        sys.exit(f"no tensor matches {args.tensor}")
    total_logical = total_packed = total_seconds = 0
    for name in names:
        dtype, view = shards.tensor(name)
        if dtype != "BF16" or view.ndim != 2:
            print(f"{name}: {dtype} {view.shape}, skipped")
            continue
        bits = np.ascontiguousarray(view)
        n, k = bits.shape
        t0 = time.time()
        blob, sections = pack(bits)
        seconds = time.time() - t0
        back = unpack(blob, sections, n, k)
        exact = bool(np.array_equal(back, bits))
        logical = n * k * 2
        total_logical += logical
        total_packed += len(blob)
        total_seconds += seconds
        print(f"{name} [{n}, {k}]: {len(blob) * 8 / (n * k):.3f} bits/weight, {sections['escaped'] / (n * k) * 100:.2f}% escaped, "
              f"{logical / 1e6:.1f} MB -> {len(blob) / 1e6:.1f} MB, packed in {seconds:.2f} s, round trip {'exact' if exact else 'MISMATCH'}")
        if not exact:
            sys.exit(1)
        if args.write:
            os.makedirs(args.write, exist_ok=True)
            blob.tofile(os.path.join(args.write, name + ".packed"))
    if len(names) > 1:
        print(f"total: {total_packed / total_logical * 16:.3f} bits/weight, {total_logical / 1e9:.3f} GB -> {total_packed / 1e9:.3f} GB, "
              f"{total_logical / 1e9 / total_seconds:.2f} GB/s packing")


if __name__ == "__main__":
    main()
