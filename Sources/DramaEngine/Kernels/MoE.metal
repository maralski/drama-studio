// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Router: softmax over 256 experts, top-8, renormalized. One threadgroup of
// 256 threads per token.
struct RouterParams {
    uint numExperts;
    uint topK;
    uint roundLogits;
};

kernel void moe_router(device int* ids [[buffer(0)]],
                       device float* weights [[buffer(1)]],
                       device const float* logits [[buffer(2)]],
                       constant RouterParams& p [[buffer(3)]],
                       uint m [[threadgroup_position_in_grid]],
                       uint e [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    threadgroup uint redIdx[8];
    threadgroup float probs[256];
    float v = logits[m * p.numExperts + e];
    if (p.roundLogits) v = round_to_bf16(v);
    float mx = simd_max(v);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = red[0];
    for (uint s = 1; s < 8; s++) gmax = max(gmax, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ex = exp(v - gmax);
    float sum = simd_sum(ex);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gsum = 0.0f;
    for (uint s = 0; s < 8; s++) gsum += red[s];
    float prob = ex / gsum;
    probs[e] = prob;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float picked = 0.0f;
    for (uint k = 0; k < p.topK; k++) {
        float cur = probs[e];
        float best = simd_max(cur);
        uint bestIdx = cur == best ? e : 0xffffffffu;
        bestIdx = simd_min(bestIdx);
        if (lane == 0) { red[sg] = best; redIdx[sg] = bestIdx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gb = red[0];
        uint gi = redIdx[0];
        for (uint s = 1; s < 8; s++) {
            if (red[s] > gb || (red[s] == gb && redIdx[s] < gi)) { gb = red[s]; gi = redIdx[s]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == gi) probs[e] = -1.0f;
        if (e == 0) { ids[m * p.topK + k] = int(gi); weights[m * p.topK + k] = gb; }
        picked += gb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e < p.topK) weights[m * p.topK + e] /= picked;
}

// Per-expert lists of (token * topK + slot) pairs, plus the list of experts
// that have any. counts[0..255] and counts[256] (the active count) must be zero.
kernel void moe_bucket(device atomic_uint* counts [[buffer(0)]],
                       device uint* lists [[buffer(1)]],
                       device const int* ids [[buffer(2)]],
                       device uint* active [[buffer(3)]],
                       constant uint& capacity [[buffer(4)]],
                       uint pair [[thread_position_in_grid]]) {
    uint e = uint(ids[pair]);
    uint slot = atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    lists[e * capacity + slot] = pair;
    if (slot == 0) {
        uint a = atomic_fetch_add_explicit(&counts[256], 1u, memory_order_relaxed);
        active[a] = e;
    }
}

// Expert address table: per expert [gateW, gateScale, upW, upScale, downW, downScale].
struct MoEParams {
    uint hidden;        // 2048
    uint intermediate;  // 512
    uint topK;
    uint capacity;
    uint roundWeights;  // dequantized weights rounded to bf16 (CPU reference path)
    uint roundOutput;
};

inline float fp8_weight(uchar b, float scale, uint roundW) {
    float w = FP8_LUT[b] * scale;
    return roundW ? round_to_bf16(w) : w;
}

// act[pair][n] = silu(gate_n . x_m) * (up_n . x_m) for every pair routed to
// expert e. Grid: (experts, intermediate / simdgroups); one output row per
// simdgroup, so the host picks the rows per threadgroup by the thread count.
// Simdgroup domain: 2, 4, 8, 16 (64 to 512 threads); every value divides the
// 512 intermediate and 2048 hidden columns, and a row's sum never depends on
// the count, so the variants are bit-identical.
kernel void moe_gate_up_fp8(device float* act [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device const ulong* table [[buffer(2)]],
                            device const uint* counts [[buffer(3)]],
                            device const uint* lists [[buffer(4)]],
                            constant MoEParams& p [[buffer(5)]],
                            device const uint* active [[buffer(6)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (tg.x >= counts[256]) return;
    uint e = active[tg.x];
    uint count = counts[e];
    uint n = tg.y * simdgroups + sg;
    device const uchar* gateW = (device const uchar*)table[e * 6 + 0] + n * p.hidden;
    device const float* gateS = (device const float*)table[e * 6 + 1];
    device const uchar* upW = (device const uchar*)table[e * 6 + 2] + n * p.hidden;
    device const float* upS = (device const float*)table[e * 6 + 3];
    uint colBlocks = p.hidden / 128;
    uint rowBlock = n / 128;
    uint tiles = p.hidden / 256;
    for (uint c = 0; c < count; c++) {
        uint pair = lists[e * p.capacity + c];
        uint m = pair / p.topK;
        device const float* xr = x + m * p.hidden;
        float g = 0.0f, u = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint k0 = tile * 256 + lane * 8;
            float sg_ = gateS[rowBlock * colBlocks + k0 / 128];
            float su = upS[rowBlock * colBlocks + k0 / 128];
            uint2 gp = *(device const uint2*)(gateW + k0);
            uint2 up_ = *(device const uint2*)(upW + k0);
            float4 a = *(device const float4*)(xr + k0);
            float4 b = *(device const float4*)(xr + k0 + 4);
            float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (uint i = 0; i < 8; i++) {
                uchar gb = uchar((i < 4 ? gp.x : gp.y) >> (8 * (i & 3)));
                uchar ub = uchar((i < 4 ? up_.x : up_.y) >> (8 * (i & 3)));
                g += fp8_weight(gb, sg_, p.roundWeights) * xs[i];
                u += fp8_weight(ub, su, p.roundWeights) * xs[i];
            }
        }
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            if (p.roundOutput) { g = round_to_bf16(g); u = round_to_bf16(u); }
            float h = silu_f(g) * u;
            act[pair * p.intermediate + n] = p.roundOutput ? round_to_bf16(h) : h;
        }
    }
}

// slotOut[pair][n] = weight[pair] * (down_n . act[pair]). Grid: (experts,
// hidden / simdgroups), one row per simdgroup as above.
kernel void moe_down_fp8(device float* slotOut [[buffer(0)]],
                         device const float* act [[buffer(1)]],
                         device const float* weights [[buffer(2)]],
                         device const ulong* table [[buffer(3)]],
                         device const uint* counts [[buffer(4)]],
                         device const uint* lists [[buffer(5)]],
                         constant MoEParams& p [[buffer(6)]],
                         device const uint* active [[buffer(7)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (tg.x >= counts[256]) return;
    uint e = active[tg.x];
    uint count = counts[e];
    uint n = tg.y * simdgroups + sg;
    device const uchar* W = (device const uchar*)table[e * 6 + 4] + n * p.intermediate;
    device const float* S = (device const float*)table[e * 6 + 5];
    uint colBlocks = p.intermediate / 128;
    uint rowBlock = n / 128;
    uint tiles = p.intermediate / 256;
    for (uint c = 0; c < count; c++) {
        uint pair = lists[e * p.capacity + c];
        device const float* ar = act + pair * p.intermediate;
        float y = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint k0 = tile * 256 + lane * 8;
            float s = S[rowBlock * colBlocks + k0 / 128];
            uint2 wp = *(device const uint2*)(W + k0);
            float4 a = *(device const float4*)(ar + k0);
            float4 b = *(device const float4*)(ar + k0 + 4);
            float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (uint i = 0; i < 8; i++) {
                uchar wb = uchar((i < 4 ? wp.x : wp.y) >> (8 * (i & 3)));
                y += fp8_weight(wb, s, p.roundWeights) * xs[i];
            }
        }
        y = simd_sum(y);
        if (lane == 0) {
            if (p.roundOutput) y = round_to_bf16(y);
            slotOut[pair * p.hidden + n] = y * weights[pair];
        }
    }
}

// Prefill variants of the two expert kernels. One threadgroup per (active
// expert, BN output columns); BN / 8 simdgroups own 8 columns each and the
// expert's routed rows go through simdgroup matrices in blocks of 8, up to
// ROWS rows per pass, stepping K by BK. A weight tile is dequantized once per
// pass into threadgroup memory (the same fp32 value the one-row kernels form),
// so the products match those kernels and only the summation order differs.
// Every output element is one simdgroup's 8-wide accumulation over K in order,
// whatever BN, ROWS and BK are, so the tiles are bit-identical with each other
// and a row's result does not depend on the rows sharing its pass.
// Tile domain: BN in {32, 64}, ROWS in {16, 32, 64}, BK in {16, 32}, except
// BN 64 with ROWS 64, whose gate-up output staging (2 x 64 x 64 floats) is the
// whole 32 KB threadgroup budget. BN must divide 128 so a tile shares one
// block scale; ROWS / 8 accumulator matrices per simdgroup bound ROWS.

// Gathers up to ROWS routed activation rows [rows, K] at columns k0..k0+BK.
template <uint ROWS, uint BK, uint THREADS>
inline void moe_stage_rows(threadgroup float* xs, device const float* src, device const uint* list, uint rows, uint K, uint k0, uint topK, uint tid) {
    constexpr uint LD = BK + 4;
    for (uint idx = tid * 4; idx < ROWS * BK; idx += THREADS * 4) {
        uint r = idx / BK, c = idx % BK;
        float4 v = float4(0.0f);
        if (r < rows) {
            uint pair = list[r];
            uint m = topK == 0 ? pair : pair / topK;
            v = *(device const float4*)(src + m * K + k0 + c);
        }
        *(threadgroup float4*)(xs + r * LD + c) = v;
    }
}

// Dequantizes a [BN rows, BK] fp8 tile of W (row stride K) with one block scale.
template <uint BN, uint BK, uint THREADS>
inline void moe_stage_weights(threadgroup float* ws, device const uchar* W, uint K, uint k0, float scale, uint roundW, uint tid) {
    constexpr uint LD = BK + 4;
    for (uint idx = tid * 4; idx < BN * BK; idx += THREADS * 4) {
        uint r = idx / BK, c = idx % BK;
        uint wv = *(device const uint*)(W + r * K + k0 + c);
        *(threadgroup float4*)(ws + r * LD + c) =
            float4(fp8_weight(uchar(wv), scale, roundW), fp8_weight(uchar(wv >> 8), scale, roundW),
                   fp8_weight(uchar(wv >> 16), scale, roundW), fp8_weight(uchar(wv >> 24), scale, roundW));
    }
}

template <uint BN, uint ROWS, uint BK>
kernel void moe_gate_up_fp8_tiled_t(device float* act [[buffer(0)]],
                                    device const float* x [[buffer(1)]],
                                    device const ulong* table [[buffer(2)]],
                                    device const uint* counts [[buffer(3)]],
                                    device const uint* lists [[buffer(4)]],
                                    constant MoEParams& p [[buffer(5)]],
                                    device const uint* active [[buffer(6)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint LD = BK + 4;
    constexpr uint THREADS = 4 * BN;
    constexpr uint MT = ROWS / 8;
    constexpr uint STAGE = ROWS * LD + 2 * BN * LD;
    constexpr uint OUT = 2 * ROWS * BN;
    if (tg.x >= counts[256]) return;
    threadgroup float smem[STAGE > OUT ? STAGE : OUT];
    threadgroup float* xs = smem;
    threadgroup float* wg = smem + ROWS * LD;
    threadgroup float* wu = smem + ROWS * LD + BN * LD;
    uint e = active[tg.x];
    uint count = counts[e];
    uint n0 = tg.y * BN;
    device const uint* list = lists + e * p.capacity;
    device const uchar* gateW = (device const uchar*)table[e * 6 + 0] + n0 * p.hidden;
    device const float* gateS = (device const float*)table[e * 6 + 1] + (n0 / 128) * (p.hidden / 128);
    device const uchar* upW = (device const uchar*)table[e * 6 + 2] + n0 * p.hidden;
    device const float* upS = (device const float*)table[e * 6 + 3] + (n0 / 128) * (p.hidden / 128);
    for (uint c0 = 0; c0 < count; c0 += ROWS) {
        uint rows = min(uint(ROWS), count - c0);
        uint mtiles = (rows + 7) / 8;
        simdgroup_float8x8 ag[MT], au[MT];
        for (uint t = 0; t < MT; t++) { ag[t] = simdgroup_float8x8(0.0f); au[t] = simdgroup_float8x8(0.0f); }
        for (uint k0 = 0; k0 < p.hidden; k0 += BK) {
            moe_stage_rows<ROWS, BK, THREADS>(xs, x, list + c0, rows, p.hidden, k0, p.topK, tid);
            moe_stage_weights<BN, BK, THREADS>(wg, gateW, p.hidden, k0, gateS[k0 / 128], p.roundWeights, tid);
            moe_stage_weights<BN, BK, THREADS>(wu, upW, p.hidden, k0, upS[k0 / 128], p.roundWeights, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 bg, bu;
                simdgroup_load(bg, wg + sg * 8 * LD + kk, LD, ulong2(0, 0), true);
                simdgroup_load(bu, wu + sg * 8 * LD + kk, LD, ulong2(0, 0), true);
                for (uint t = 0; t < MT; t++) {
                    if (t < mtiles) {
                        simdgroup_float8x8 a;
                        simdgroup_load(a, xs + t * 8 * LD + kk, LD);
                        simdgroup_multiply_accumulate(ag[t], a, bg, ag[t]);
                        simdgroup_multiply_accumulate(au[t], a, bu, au[t]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* og = smem;
        threadgroup float* ou = smem + ROWS * BN;
        for (uint t = 0; t < MT; t++) {
            if (t < mtiles) {
                simdgroup_store(ag[t], og + t * 8 * BN + sg * 8, BN);
                simdgroup_store(au[t], ou + t * 8 * BN + sg * 8, BN);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * BN; idx += THREADS) {
            uint r = idx / BN, c = idx % BN;
            float g = og[r * BN + c], u = ou[r * BN + c];
            if (p.roundOutput) { g = round_to_bf16(g); u = round_to_bf16(u); }
            float h = silu_f(g) * u;
            act[list[c0 + r] * p.intermediate + n0 + c] = p.roundOutput ? round_to_bf16(h) : h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint BN, uint ROWS, uint BK>
kernel void moe_down_fp8_tiled_t(device float* slotOut [[buffer(0)]],
                                 device const float* act [[buffer(1)]],
                                 device const float* weights [[buffer(2)]],
                                 device const ulong* table [[buffer(3)]],
                                 device const uint* counts [[buffer(4)]],
                                 device const uint* lists [[buffer(5)]],
                                 constant MoEParams& p [[buffer(6)]],
                                 device const uint* active [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint LD = BK + 4;
    constexpr uint THREADS = 4 * BN;
    constexpr uint MT = ROWS / 8;
    constexpr uint STAGE = ROWS * LD + BN * LD;
    constexpr uint OUT = ROWS * BN;
    if (tg.x >= counts[256]) return;
    threadgroup float smem[STAGE > OUT ? STAGE : OUT];
    threadgroup float* xs = smem;
    threadgroup float* wd = smem + ROWS * LD;
    uint e = active[tg.x];
    uint count = counts[e];
    uint n0 = tg.y * BN;
    device const uint* list = lists + e * p.capacity;
    device const uchar* W = (device const uchar*)table[e * 6 + 4] + n0 * p.intermediate;
    device const float* S = (device const float*)table[e * 6 + 5] + (n0 / 128) * (p.intermediate / 128);
    for (uint c0 = 0; c0 < count; c0 += ROWS) {
        uint rows = min(uint(ROWS), count - c0);
        uint mtiles = (rows + 7) / 8;
        simdgroup_float8x8 acc[MT];
        for (uint t = 0; t < MT; t++) acc[t] = simdgroup_float8x8(0.0f);
        for (uint k0 = 0; k0 < p.intermediate; k0 += BK) {
            moe_stage_rows<ROWS, BK, THREADS>(xs, act, list + c0, rows, p.intermediate, k0, 0, tid);
            moe_stage_weights<BN, BK, THREADS>(wd, W, p.intermediate, k0, S[k0 / 128], p.roundWeights, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 b;
                simdgroup_load(b, wd + sg * 8 * LD + kk, LD, ulong2(0, 0), true);
                for (uint t = 0; t < MT; t++) {
                    if (t < mtiles) {
                        simdgroup_float8x8 a;
                        simdgroup_load(a, xs + t * 8 * LD + kk, LD);
                        simdgroup_multiply_accumulate(acc[t], a, b, acc[t]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint t = 0; t < MT; t++) {
            if (t < mtiles) simdgroup_store(acc[t], smem + t * 8 * BN + sg * 8, BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * BN; idx += THREADS) {
            uint r = idx / BN, c = idx % BN;
            uint pair = list[c0 + r];
            float y = smem[r * BN + c];
            if (p.roundOutput) y = round_to_bf16(y);
            slotOut[pair * p.hidden + n0 + c] = y * weights[pair];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MOE_VARIANTS(SUFFIX, BN, ROWS, BK) \
    template [[host_name("moe_gate_up_fp8_tiled_" SUFFIX)]] kernel void moe_gate_up_fp8_tiled_t<BN, ROWS, BK>( \
        device float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint); \
    template [[host_name("moe_down_fp8_tiled_" SUFFIX)]] kernel void moe_down_fp8_tiled_t<BN, ROWS, BK>( \
        device float*, device const float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint);
MOE_VARIANTS("32x16x16", 32, 16, 16)
MOE_VARIANTS("32x16x32", 32, 16, 32)
MOE_VARIANTS("32x32x16", 32, 32, 16)
MOE_VARIANTS("32x32x32", 32, 32, 32)
MOE_VARIANTS("32x64x16", 32, 64, 16)
MOE_VARIANTS("32x64x32", 32, 64, 32)
MOE_VARIANTS("64x16x16", 64, 16, 16)
MOE_VARIANTS("64x16x32", 64, 16, 32)
MOE_VARIANTS("64x32x16", 64, 32, 16)
MOE_VARIANTS("64x32x32", 64, 32, 32)

// x[m] += sum_slots slotOut[m][slot] + sigmoid(sharedGate[m]) * sharedOut[m]
struct CombineParams {
    uint hidden;
    uint topK;
    uint roundOutput;
};

kernel void moe_combine(device float* x [[buffer(0)]],
                        device const float* slotOut [[buffer(1)]],
                        device const float* sharedOut [[buffer(2)]],
                        device const float* sharedGate [[buffer(3)]],
                        constant CombineParams& p [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint m = gid.y;
    if (d >= p.hidden) return;
    float routed = 0.0f;
    for (uint s = 0; s < p.topK; s++) routed += slotOut[(m * p.topK + s) * p.hidden + d];
    if (p.roundOutput) routed = round_to_bf16(routed);
    float total = routed + sigmoid_f(sharedGate[m]) * sharedOut[m * p.hidden + d];
    if (p.roundOutput) total = round_to_bf16(total);
    float v = x[m * p.hidden + d] + total;
    x[m * p.hidden + d] = p.roundOutput ? round_to_bf16(v) : v;
}
