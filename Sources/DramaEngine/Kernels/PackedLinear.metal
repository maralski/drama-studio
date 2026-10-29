// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// GEMV over exponent-coded bf16 (eval/weight_entropy.py, eval/pack_bf16.py).
// A row is tiles of 256 weights. Per tile: a header word (bits 0-7 the base
// exponent, bits 8-31 the byte offset of the tile's escape list), 256
// sign+mantissa bytes, and 96 bytes of 3-bit exponent codes, 24 bits per lane
// for its 8 weights. Code c below 7 means exponent base + c; code 7 means the
// exponent is the next byte of the tile's escape list, which holds the
// escaped exponents in weight order. Reconstruction assembles the bf16 bits,
// so every pattern (zero, denormal, inf, nan) comes back exactly, and the
// products and sums are the ones gemv_bf16_t forms, so the outputs are
// bit-identical to it for every token tile and simdgroup count.
//
// The tile loop is a three-stage pipeline: the loads of tile t+2 issue while
// tile t+1's exponents (and their escape bytes) are gathered and tile t is
// multiplied, so neither the header nor the escape gather adds a memory
// latency to each tile. The pipeline state costs registers, so the token
// tile 1 variant streams fastest at decode (PackedGemvTests bench).
struct PackedTile {
    uint h;
    uint2 m;
    uint q;
};

// GemvParams plus the matrix's escaped-exponent count, which the kernel does
// not read: the dispatch trace takes the bytes streamed from this block.
struct PackedGemvParams {
    uint N;
    uint K;
    uint M;
    uint roundOutput;
    uint escaped;
};

inline PackedTile load_packed_tile(device const uint* headers, device const uchar* mant, device const uchar* codes, uint slot, uint lane) {
    PackedTile t;
    t.h = headers[slot];
    t.m = ((device const uint2*)(mant + slot * 256))[lane];
    device const uchar* cp = codes + slot * 96 + lane * 3;
    t.q = uint(cp[0]) | (uint(cp[1]) << 8) | (uint(cp[2]) << 16);
    return t;
}

// The exponents of the lane's 8 weights as four pair words: weight 2a's
// exponent at bits 7-14 of word a and weight 2a+1's at bits 23-30.
inline uint4 packed_exponents(PackedTile t, device const uchar* escapes) {
    uint base = t.h & 0xffu;
    uint q = t.q;
    uint sevens = q & (q >> 1) & (q >> 2) & 0x249249u;
    device const uchar* esc = escapes + (t.h >> 8) + simd_prefix_exclusive_sum(popcount(sevens));
    uint e[8];
#pragma unroll
    for (uint j = 0; j < 8; j++) {
        uint c = (q >> (3 * j)) & 7u;
        e[j] = base + c;
        if (c == 7u) e[j] = esc[popcount(sevens & ((1u << (3 * j)) - 1u))];
    }
    return uint4((e[0] << 7) | (e[1] << 23), (e[2] << 7) | (e[3] << 23), (e[4] << 7) | (e[5] << 23), (e[6] << 7) | (e[7] << 23));
}

// Two bf16 patterns from their sign+mantissa bytes (low byte weight 2a, high
// byte weight 2a+1) and their positioned exponents.
inline uint bf16_pair(uint skel, uint exponents) {
    return (skel & 0x7fu) | ((skel & 0x7f80u) << 8) | ((skel & 0x8000u) << 16) | exponents;
}

// The lane's 8 weights of a tile as floats, from its sign+mantissa words and
// its exponent pair words.
inline void packed_weights(PackedTile t, uint4 e, thread float* w) {
    uint p01 = bf16_pair(t.m.x & 0xffffu, e.x);
    uint p23 = bf16_pair(t.m.x >> 16, e.y);
    uint p45 = bf16_pair(t.m.y & 0xffffu, e.z);
    uint p67 = bf16_pair(t.m.y >> 16, e.w);
    w[0] = bf16_to_float(p01 & 0xffffu); w[1] = bf16_to_float(p01 >> 16);
    w[2] = bf16_to_float(p23 & 0xffffu); w[3] = bf16_to_float(p23 >> 16);
    w[4] = bf16_to_float(p45 & 0xffffu); w[5] = bf16_to_float(p45 >> 16);
    w[6] = bf16_to_float(p67 & 0xffffu); w[7] = bf16_to_float(p67 >> 16);
}

template <uint MTILE>
kernel void gemv_packed_t(device const uint* headers [[buffer(0)]],
                          device const float* x [[buffer(1)]],
                          device float* y [[buffer(2)]],
                          constant PackedGemvParams& p [[buffer(3)]],
                          device const uchar* mant [[buffer(4)]],
                          device const uchar* codes [[buffer(5)]],
                          device const uchar* escapes [[buffer(6)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint n = tg.x * simdgroups + sg;
    if (n >= p.N) return;
    uint tiles = p.K / 256;
    uint m0 = tg.y * MTILE;
    if (m0 < p.M) {
        uint mcount = min(uint(MTILE), p.M - m0);
        float acc[MTILE];
#pragma unroll
        for (uint t = 0; t < MTILE; t++) acc[t] = 0.0f;
        uint slot0 = n * tiles;
        PackedTile cur = load_packed_tile(headers, mant, codes, slot0, lane);
        PackedTile next = load_packed_tile(headers, mant, codes, slot0 + min(1u, tiles - 1), lane);
        uint4 ecur = packed_exponents(cur, escapes);
        for (uint tile = 0; tile < tiles; tile++) {
            PackedTile after = load_packed_tile(headers, mant, codes, slot0 + min(tile + 2, tiles - 1), lane);
            uint4 enext = packed_exponents(next, escapes);
            float w[8];
            packed_weights(cur, ecur, w);
            uint k0 = tile * 256 + lane * 8;
#pragma unroll
            for (uint t = 0; t < MTILE; t++) {
                if (t < mcount) {
                    device const float4* xp = (device const float4*)(x + (m0 + t) * p.K + k0);
                    float4 a = xp[0];
                    float4 b = xp[1];
                    acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
                            + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                }
            }
            cur = next;
            ecur = enext;
            next = after;
        }
#pragma unroll
        for (uint t = 0; t < MTILE; t++) {
            if (t < mcount) {
                float s = simd_sum(acc[t]);
                if (lane == 0) y[(m0 + t) * p.N + n] = p.roundOutput ? round_to_bf16(s) : s;
            }
        }
    }
}

// out[n] = silu(W1[n] . x) * (W3[n] . x) for M = 1 over two packed matrices:
// the sums of gemv_packed_t and the epilogue of swiglu_dec (TowerFused.metal),
// so the output is swiglu_dec's bit for bit. Both rows run through one tile
// loop, each with the three-stage pipeline of gemv_packed_t, so a simdgroup
// keeps both rows' loads in flight as swiglu_dec does.
kernel void swiglu_packed_dec(device const uint* headers1 [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant PackedGemvParams& p [[buffer(3)]],
                              device const uchar* mant1 [[buffer(4)]],
                              device const uchar* codes1 [[buffer(5)]],
                              device const uchar* escapes1 [[buffer(6)]],
                              device const uint* headers3 [[buffer(7)]],
                              device const uchar* mant3 [[buffer(8)]],
                              device const uchar* codes3 [[buffer(9)]],
                              device const uchar* escapes3 [[buffer(10)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint n = tg * simdgroups + sg;
    if (n >= p.N) return;
    uint tiles = p.K / 256;
    uint slot0 = n * tiles;
    PackedTile cur1 = load_packed_tile(headers1, mant1, codes1, slot0, lane);
    PackedTile cur3 = load_packed_tile(headers3, mant3, codes3, slot0, lane);
    PackedTile next1 = load_packed_tile(headers1, mant1, codes1, slot0 + min(1u, tiles - 1), lane);
    PackedTile next3 = load_packed_tile(headers3, mant3, codes3, slot0 + min(1u, tiles - 1), lane);
    uint4 ecur1 = packed_exponents(cur1, escapes1);
    uint4 ecur3 = packed_exponents(cur3, escapes3);
    float acc1 = 0.0f, acc3 = 0.0f;
    for (uint tile = 0; tile < tiles; tile++) {
        uint slot = slot0 + min(tile + 2, tiles - 1);
        PackedTile after1 = load_packed_tile(headers1, mant1, codes1, slot, lane);
        PackedTile after3 = load_packed_tile(headers3, mant3, codes3, slot, lane);
        uint4 enext1 = packed_exponents(next1, escapes1);
        uint4 enext3 = packed_exponents(next3, escapes3);
        float w[8], v[8];
        packed_weights(cur1, ecur1, w);
        packed_weights(cur3, ecur3, v);
        device const float4* xp = (device const float4*)(x + tile * 256 + lane * 8);
        float4 a = xp[0];
        float4 b = xp[1];
        acc1 += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
              + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
        acc3 += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w
              + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
        cur1 = next1; ecur1 = enext1; next1 = after1;
        cur3 = next3; ecur3 = enext3; next3 = after3;
    }
    float s1 = simd_sum(acc1);
    float s3 = simd_sum(acc3);
    if (lane == 0) {
        if (p.roundOutput) { s1 = round_to_bf16(s1); s3 = round_to_bf16(s3); }
        float o = silu_f(s1) * s3;
        y[n] = p.roundOutput ? round_to_bf16(o) : o;
    }
}

#define GEMV_PACKED_VARIANT(NAME, MTILE) \
    template [[host_name(NAME)]] kernel void gemv_packed_t<MTILE>( \
        device const uint*, device const float*, device float*, constant PackedGemvParams&, \
        device const uchar*, device const uchar*, device const uchar*, uint2, uint, uint, uint);
GEMV_PACKED_VARIANT("gemv_packed_t1", 1)
GEMV_PACKED_VARIANT("gemv_packed_t8", 8)
GEMV_PACKED_VARIANT("gemv_packed_t16", 16)
GEMV_PACKED_VARIANT("gemv_packed_t32", 32)
GEMV_PACKED_VARIANT("gemv_packed_t64", 64)
