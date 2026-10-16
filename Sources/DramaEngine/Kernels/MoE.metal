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

// Expert address table: per expert [gateW, gateParams, upW, upParams, downW, downParams].
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

// Expert weight formats, chosen by a format tag. Table slot s holds the
// weights of one matrix and slot s + 1 its params:
//   fp8: [N][K] E4M3 bytes, params [N/128][K/128] fp32 block scales.
//   i4:  [N][K/2] nibbles and [N][K/64] (half scale, half bias) as the dense
//        int4 format, formed with the same dq_i4.
// Each tag gives the one-row loader (8 weights of row n at column k0, a
// multiple of 8) and the tiled loader (4 weights of row r of a BN-row band at
// an absolute column k, a multiple of 4, with a per-tile context the kernel
// hoists once per K step: the fp8 block scale, or the i4 params base).
struct ExpFP8 {
    static void load8(device const ulong* table, uint e, uint slot, uint n, uint K, uint k0, uint roundW, thread float* w) {
        device const uchar* W = (device const uchar*)table[e * 6 + slot] + n * K;
        device const float* S = (device const float*)table[e * 6 + slot + 1];
        float s = S[(n / 128) * (K / 128) + k0 / 128];
        uint2 wp = *(device const uint2*)(W + k0);
        for (uint i = 0; i < 8; i++) w[i] = fp8_weight(uchar((i < 4 ? wp.x : wp.y) >> (8 * (i & 3))), s, roundW);
    }
    static device const uchar* band(device const ulong* table, uint e, uint slot, uint n0, uint K) {
        return (device const uchar*)table[e * 6 + slot] + n0 * K;
    }
    struct Tile { float scale; };
    static Tile tile(device const ulong* table, uint e, uint slot, uint n0, uint K, uint k0) {
        device const float* S = (device const float*)table[e * 6 + slot + 1] + (n0 / 128) * (K / 128);
        return Tile { S[k0 / 128] };
    }
    static float4 load4(device const uchar* band, Tile t, uint r, uint K, uint k, uint roundW) {
        uint wv = *(device const uint*)(band + r * K + k);
        return float4(fp8_weight(uchar(wv), t.scale, roundW), fp8_weight(uchar(wv >> 8), t.scale, roundW),
                      fp8_weight(uchar(wv >> 16), t.scale, roundW), fp8_weight(uchar(wv >> 24), t.scale, roundW));
    }
};

inline float i4_weight(uint nibble, half2 p, uint roundW) {
    float w = dq_i4(nibble, p);
    return roundW ? round_to_bf16(w) : w;
}

struct ExpI4 {
    static void load8(device const ulong* table, uint e, uint slot, uint n, uint K, uint k0, uint roundW, thread float* w) {
        device const uchar* W = (device const uchar*)table[e * 6 + slot] + n * (K / 2);
        device const half2* P = (device const half2*)table[e * 6 + slot + 1] + n * (K / 64);
        uint q = *(device const uint*)(W + k0 / 2);
        half2 p = P[k0 / 64];
        for (uint i = 0; i < 8; i++) w[i] = i4_weight(q >> (4 * i), p, roundW);
    }
    static device const uchar* band(device const ulong* table, uint e, uint slot, uint n0, uint K) {
        return (device const uchar*)table[e * 6 + slot] + n0 * (K / 2);
    }
    struct Tile { device const half2* P; };
    static Tile tile(device const ulong* table, uint e, uint slot, uint n0, uint K, uint k0) {
        return Tile { (device const half2*)table[e * 6 + slot + 1] + n0 * (K / 64) };
    }
    static float4 load4(device const uchar* band, Tile t, uint r, uint K, uint k, uint roundW) {
        uint q = *(device const ushort*)(band + r * (K / 2) + k / 2);
        half2 p = t.P[r * (K / 64) + k / 64];
        return float4(i4_weight(q, p, roundW), i4_weight(q >> 4, p, roundW), i4_weight(q >> 8, p, roundW), i4_weight(q >> 12, p, roundW));
    }
};

// act[pair][n] = silu(gate_n . x_m) * (up_n . x_m) for every pair routed to
// expert e. Grid: (experts, intermediate / simdgroups); one output row per
// simdgroup, so the host picks the rows per threadgroup by the thread count.
// Simdgroup domain: 2, 4, 8, 16 (64 to 512 threads); every value divides the
// 512 intermediate and 2048 hidden columns, and a row's sum never depends on
// the count, so the variants are bit-identical.
template <typename FMT>
kernel void moe_gate_up_t(device float* act [[buffer(0)]],
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
    uint tiles = p.hidden / 256;
    for (uint c = 0; c < count; c++) {
        uint pair = lists[e * p.capacity + c];
        uint m = pair / p.topK;
        device const float* xr = x + m * p.hidden;
        float g = 0.0f, u = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint k0 = tile * 256 + lane * 8;
            float wg[8], wu[8];
            FMT::load8(table, e, 0, n, p.hidden, k0, p.roundWeights, wg);
            FMT::load8(table, e, 2, n, p.hidden, k0, p.roundWeights, wu);
            float4 a = *(device const float4*)(xr + k0);
            float4 b = *(device const float4*)(xr + k0 + 4);
            float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (uint i = 0; i < 8; i++) {
                g += wg[i] * xs[i];
                u += wu[i] * xs[i];
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
template <typename FMT>
kernel void moe_down_t(device float* slotOut [[buffer(0)]],
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
    uint tiles = p.intermediate / 256;
    for (uint c = 0; c < count; c++) {
        uint pair = lists[e * p.capacity + c];
        device const float* ar = act + pair * p.intermediate;
        float y = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint k0 = tile * 256 + lane * 8;
            float w[8];
            FMT::load8(table, e, 4, n, p.intermediate, k0, p.roundWeights, w);
            float4 a = *(device const float4*)(ar + k0);
            float4 b = *(device const float4*)(ar + k0 + 4);
            float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (uint i = 0; i < 8; i++) y += w[i] * xs[i];
        }
        y = simd_sum(y);
        if (lane == 0) {
            if (p.roundOutput) y = round_to_bf16(y);
            slotOut[pair * p.hidden + n] = y * weights[pair];
        }
    }
}

template [[host_name("moe_gate_up_fp8")]] kernel void moe_gate_up_t<ExpFP8>(
    device float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint, uint);
template [[host_name("moe_gate_up_i4")]] kernel void moe_gate_up_t<ExpI4>(
    device float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint, uint);
template [[host_name("moe_down_fp8")]] kernel void moe_down_t<ExpFP8>(
    device float*, device const float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint, uint);
template [[host_name("moe_down_i4")]] kernel void moe_down_t<ExpI4>(
    device float*, device const float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint, uint);

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
// whole 32 KB threadgroup budget. BN must divide 128 so an fp8 tile shares one
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

// Dequantizes the [BN rows, BK] tile at column k0 of a BN-row band of W (row stride K).
template <typename FMT, uint BN, uint BK, uint THREADS>
inline void moe_stage_weights(threadgroup float* ws, device const uchar* band, typename FMT::Tile t, uint K, uint k0, uint roundW, uint tid) {
    constexpr uint LD = BK + 4;
    for (uint idx = tid * 4; idx < BN * BK; idx += THREADS * 4) {
        uint r = idx / BK, c = idx % BK;
        *(threadgroup float4*)(ws + r * LD + c) = FMT::load4(band, t, r, K, k0 + c, roundW);
    }
}

template <typename FMT, uint BN, uint ROWS, uint BK>
kernel void moe_gate_up_tiled_t(device float* act [[buffer(0)]],
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
    device const uchar* gateW = FMT::band(table, e, 0, n0, p.hidden);
    device const uchar* upW = FMT::band(table, e, 2, n0, p.hidden);
    for (uint c0 = 0; c0 < count; c0 += ROWS) {
        uint rows = min(uint(ROWS), count - c0);
        uint mtiles = (rows + 7) / 8;
        simdgroup_float8x8 ag[MT], au[MT];
        for (uint t = 0; t < MT; t++) { ag[t] = simdgroup_float8x8(0.0f); au[t] = simdgroup_float8x8(0.0f); }
        for (uint k0 = 0; k0 < p.hidden; k0 += BK) {
            moe_stage_rows<ROWS, BK, THREADS>(xs, x, list + c0, rows, p.hidden, k0, p.topK, tid);
            moe_stage_weights<FMT, BN, BK, THREADS>(wg, gateW, FMT::tile(table, e, 0, n0, p.hidden, k0), p.hidden, k0, p.roundWeights, tid);
            moe_stage_weights<FMT, BN, BK, THREADS>(wu, upW, FMT::tile(table, e, 2, n0, p.hidden, k0), p.hidden, k0, p.roundWeights, tid);
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

template <typename FMT, uint BN, uint ROWS, uint BK>
kernel void moe_down_tiled_t(device float* slotOut [[buffer(0)]],
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
    device const uchar* W = FMT::band(table, e, 4, n0, p.intermediate);
    for (uint c0 = 0; c0 < count; c0 += ROWS) {
        uint rows = min(uint(ROWS), count - c0);
        uint mtiles = (rows + 7) / 8;
        simdgroup_float8x8 acc[MT];
        for (uint t = 0; t < MT; t++) acc[t] = simdgroup_float8x8(0.0f);
        for (uint k0 = 0; k0 < p.intermediate; k0 += BK) {
            moe_stage_rows<ROWS, BK, THREADS>(xs, act, list + c0, rows, p.intermediate, k0, 0, tid);
            moe_stage_weights<FMT, BN, BK, THREADS>(wd, W, FMT::tile(table, e, 4, n0, p.intermediate, k0), p.intermediate, k0, p.roundWeights, tid);
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

#define MOE_VARIANTS(TAG, FMT, SUFFIX, BN, ROWS, BK) \
    template [[host_name("moe_gate_up_" TAG "_tiled_" SUFFIX)]] kernel void moe_gate_up_tiled_t<FMT, BN, ROWS, BK>( \
        device float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint); \
    template [[host_name("moe_down_" TAG "_tiled_" SUFFIX)]] kernel void moe_down_tiled_t<FMT, BN, ROWS, BK>( \
        device float*, device const float*, device const float*, device const ulong*, device const uint*, device const uint*, constant MoEParams&, device const uint*, uint2, uint, uint);
#define MOE_FORMAT(TAG, FMT) \
    MOE_VARIANTS(TAG, FMT, "32x16x16", 32, 16, 16) \
    MOE_VARIANTS(TAG, FMT, "32x16x32", 32, 16, 32) \
    MOE_VARIANTS(TAG, FMT, "32x32x16", 32, 32, 16) \
    MOE_VARIANTS(TAG, FMT, "32x32x32", 32, 32, 32) \
    MOE_VARIANTS(TAG, FMT, "32x64x16", 32, 64, 16) \
    MOE_VARIANTS(TAG, FMT, "32x64x32", 32, 64, 32) \
    MOE_VARIANTS(TAG, FMT, "64x16x16", 64, 16, 16) \
    MOE_VARIANTS(TAG, FMT, "64x16x32", 64, 16, 32) \
    MOE_VARIANTS(TAG, FMT, "64x32x16", 64, 32, 16) \
    MOE_VARIANTS(TAG, FMT, "64x32x32", 64, 32, 32)
MOE_FORMAT("fp8", ExpFP8)
MOE_FORMAT("i4", ExpI4)

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
