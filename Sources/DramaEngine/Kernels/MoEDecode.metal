// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// The MoE layer of one decode row in three dispatches: moe_pre_bf16 (router,
// shared gate and up, expert gate), moe_mid_fp8 (routing and the routed
// gate-up), moe_post_fp8 (routed down, shared down, combine). Bit-identical to
// the twelve dispatches the M > 1 path would run at M == 1: every output row
// stays on one simdgroup with the same per-lane order over K (lane L owns
// columns tile * 256 + L * 8 + i), the same simd_sum, and the combine adds the
// eight slots in routing order. Shapes are compile-time: hidden 2048, expert
// and shared intermediate 512, 256 experts, top 8 (ModelConfig checks them).
//
// Expert weights are read 16 contiguous bytes per lane covering two tiles of
// a lane pair, as two uint2 because shard tensors are 8-byte aligned, and the
// halves are swapped with the partner lane. FP8 bytes decode through a half:
// the 7 magnitude bits placed in a half are the E4M3 value times 2^-8 exactly,
// so scaling by scale * 256 is the same single rounding of the same product as
// FP8_LUT[b] * scale; the two differ only on the E4M3 NaN pattern, which
// finite weights never hold.

constexpr constant uint MD_HIDDEN = 2048;
constexpr constant uint MD_INTER = 512;
constexpr constant uint MD_SHARED = 512;
constexpr constant uint MD_EXPERTS = 256;
constexpr constant uint MD_TOPK = 8;
// Output rows per simdgroup in moe_mid_fp8 and moe_post_fp8.
constexpr constant uint MD_R = 2;

struct MoEDecodeParams {
    uint roundWeights;
    uint roundOutput;
};

inline float4 md_load4(device const float* p) { return *(device const float4*)p; }
inline float4 md_load4(threadgroup const float* p) { return *(threadgroup const float4*)p; }

inline float4 md_fp8x4_half(uint v) {
    uint lo = v & 0x00ff00ffu;
    uint hi = (v >> 8) & 0x00ff00ffu;
    half2 a = as_type<half2>(((lo & 0x00800080u) << 8) | ((lo & 0x007f007fu) << 7));
    half2 b = as_type<half2>(((hi & 0x00800080u) << 8) | ((hi & 0x007f007fu) << 7));
    return float4(float(a.x), float(b.x), float(a.y), float(b.y));
}

inline void md_decode8(thread float* w, uint2 b, float s, uint roundW) {
    float s256 = s * 256.0f;
    float4 lo = md_fp8x4_half(b.x), hi = md_fp8x4_half(b.y);
    w[0] = lo.x * s256; w[1] = lo.y * s256; w[2] = lo.z * s256; w[3] = lo.w * s256;
    w[4] = hi.x * s256; w[5] = hi.y * s256; w[6] = hi.z * s256; w[7] = hi.w * s256;
    if (roundW) {
        for (uint i = 0; i < 8; i++) w[i] = round_to_bf16(w[i]);
    }
}

// Bytes of tiles `tile` and `tile + 1` that lane `lane` owns in one row.
inline void md_load_pair(thread uint2& t0, thread uint2& t1, device const uchar* row, uint tile, uint lane) {
    uint odd = lane & 1u;
    device const uchar* src = row + (tile + odd) * 256 + (lane >> 1) * 16;
    uint2 a = *(device const uint2*)src;
    uint2 b = *(device const uint2*)(src + 8);
    uint4 v = uint4(a, b);
    uint2 send = odd ? v.xy : v.zw;
    uint2 recv = uint2(simd_shuffle_xor(send.x, 1), simd_shuffle_xor(send.y, 1));
    t0 = odd ? recv : v.xy;
    t1 = odd ? v.zw : recv;
}

struct MdMat {
    device const uchar* w;
    device const float* s;
};

// acc[m * R + r] += lane's share of row n0 + r of matrix m dotted with v, in
// the shipped kernels' order.
template <uint NM, uint R, uint K, typename V>
inline void md_fp8_rows(thread float* acc, thread const MdMat* mats, uint n0, V v, uint lane, uint roundW) {
    constexpr uint TILES = K / 256;
    constexpr uint COLB = K / 128;
    for (uint tile = 0; tile < TILES; tile += 2) {
        uint2 bytes[NM][R][2];
        for (uint m = 0; m < NM; m++) {
            for (uint r = 0; r < R; r++) {
                md_load_pair(bytes[m][r][0], bytes[m][r][1], mats[m].w + (n0 + r) * K, tile, lane);
            }
        }
        for (uint t = 0; t < 2; t++) {
            uint k0 = (tile + t) * 256 + lane * 8;
            float4 a = md_load4(v + k0);
            float4 b = md_load4(v + k0 + 4);
            float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (uint r = 0; r < R; r++) {
                for (uint m = 0; m < NM; m++) {
                    float s = mats[m].s[((n0 + r) / 128) * COLB + k0 / 128];
                    float w[8];
                    md_decode8(w, bytes[m][r][t], s, roundW);
                    for (uint i = 0; i < 8; i++) acc[m * R + r] += w[i] * xs[i];
                }
            }
        }
    }
}

inline float md_act(float g, float u, uint roundOutput) {
    if (roundOutput) { g = round_to_bf16(g); u = round_to_bf16(u); }
    float h = silu_f(g) * u;
    return roundOutput ? round_to_bf16(h) : h;
}

// Gate-up rows n0 .. n0 + R of expert e against xr; lane 0 gets the activations.
template <uint R, typename V>
inline void md_gate_up(thread float* out, device const ulong* table, uint e, uint n0, V xr, uint lane, uint roundW, uint roundOutput) {
    MdMat mats[2] = {{(device const uchar*)table[e * 6 + 0], (device const float*)table[e * 6 + 1]},
                     {(device const uchar*)table[e * 6 + 2], (device const float*)table[e * 6 + 3]}};
    float acc[2 * R];
    for (uint i = 0; i < 2 * R; i++) acc[i] = 0.0f;
    md_fp8_rows<2, R, MD_HIDDEN>(acc, mats, n0, xr, lane, roundW);
    for (uint r = 0; r < R; r++) {
        float g = simd_sum(acc[r]);
        float u = simd_sum(acc[R + r]);
        out[r] = md_act(g, u, roundOutput);
    }
}

// Down row d of expert e against its activation row, unweighted, rounded like moe_down_fp8.
template <typename V>
inline float md_down(device const ulong* table, uint e, uint d, V ar, uint lane, uint roundW, uint roundOutput) {
    MdMat mats[1] = {{(device const uchar*)table[e * 6 + 4], (device const float*)table[e * 6 + 5]}};
    float acc[1] = {0.0f};
    md_fp8_rows<1, 1, MD_INTER>(acc, mats, d, ar, lane, roundW);
    float y = simd_sum(acc[0]);
    return roundOutput ? round_to_bf16(y) : y;
}

// One bf16 row of W dotted with x (2048 wide), as gemv_bf16_t sums one token's row.
template <typename V>
inline float md_bf16_row(device const ushort* W, uint K, V x, uint lane) {
    device const uint2* wrow = (device const uint2*)W;
    float acc = 0.0f;
    for (uint tile = 0; tile < K / 256; tile++) {
        uint2 lo = wrow[(tile * 32 + lane) * 2];
        uint2 hi = wrow[(tile * 32 + lane) * 2 + 1];
        float w[8] = {
            bf16_to_float(lo.x & 0xffffu), bf16_to_float(lo.x >> 16),
            bf16_to_float(lo.y & 0xffffu), bf16_to_float(lo.y >> 16),
            bf16_to_float(hi.x & 0xffffu), bf16_to_float(hi.x >> 16),
            bf16_to_float(hi.y & 0xffffu), bf16_to_float(hi.y >> 16)};
        uint k0 = tile * 256 + lane * 8;
        float4 a = md_load4(x + k0);
        float4 b = md_load4(x + k0 + 4);
        acc += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
             + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    }
    return simd_sum(acc);
}

// pre = [router logits 256 | shared gate 512 | shared up 512 | expert gate 1] . h,
// one row per simdgroup. Grid: ceil(1281 / simdgroups).
kernel void moe_pre_bf16(device float* pre [[buffer(0)]],
                         device const float* h [[buffer(1)]],
                         device const ushort* router [[buffer(2)]],
                         device const ushort* sharedGate [[buffer(3)]],
                         device const ushort* sharedUp [[buffer(4)]],
                         device const ushort* expertGate [[buffer(5)]],
                         constant MoEDecodeParams& p [[buffer(6)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint r = tg * simdgroups + sg;
    if (r > MD_EXPERTS + 2 * MD_SHARED) return;
    device const ushort* W;
    if (r < MD_EXPERTS) W = router + r * MD_HIDDEN;
    else if (r < MD_EXPERTS + MD_SHARED) W = sharedGate + (r - MD_EXPERTS) * MD_HIDDEN;
    else if (r < MD_EXPERTS + 2 * MD_SHARED) W = sharedUp + (r - MD_EXPERTS - MD_SHARED) * MD_HIDDEN;
    else W = expertGate;
    float s = md_bf16_row(W, MD_HIDDEN, h, lane);
    if (lane == 0) pre[r] = p.roundOutput ? round_to_bf16(s) : s;
}

// moe_router's softmax and top-8 for one token, in a 256-thread threadgroup;
// every thread gets the same ids and normalized weights.
inline void md_route(threadgroup int* tIds, threadgroup float* tW, threadgroup float* red, threadgroup uint* redIdx, threadgroup float* probs,
                     device const float* logits, uint roundLogits, uint e, uint lane, uint sg) {
    float v = logits[e];
    if (roundLogits) v = round_to_bf16(v);
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
    for (uint k = 0; k < MD_TOPK; k++) {
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
        if (e == 0) { tIds[k] = int(gi); tW[k] = gb; }
        picked += gb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e < MD_TOPK) tW[e] /= picked;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Routing (redundantly per threadgroup) and the routed gate-up. Grid:
// (8 slots, 512 / (8 * MD_R)) of 256 threads. Threadgroup (0, 0) stores ids and weights.
kernel void moe_mid_fp8(device float* act [[buffer(0)]],
                        device const float* h [[buffer(1)]],
                        device const ulong* table [[buffer(2)]],
                        device const float* pre [[buffer(3)]],
                        device int* ids [[buffer(4)]],
                        device float* rw [[buffer(5)]],
                        constant MoEDecodeParams& p [[buffer(6)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup int tIds[MD_TOPK];
    threadgroup float tW[MD_TOPK];
    threadgroup float red[8];
    threadgroup uint redIdx[8];
    threadgroup float probs[MD_EXPERTS];
    md_route(tIds, tW, red, redIdx, probs, pre, p.roundOutput, tid, lane, sg);
    if (tg.x == 0 && tg.y == 0 && tid < MD_TOPK) { ids[tid] = tIds[tid]; rw[tid] = tW[tid]; }
    uint slot = tg.x;
    uint n0 = (tg.y * simdgroups + sg) * MD_R;
    float a[MD_R];
    md_gate_up<MD_R>(a, table, uint(tIds[slot]), n0, h, lane, p.roundWeights, p.roundOutput);
    if (lane == 0) {
        for (uint r = 0; r < MD_R; r++) act[slot * MD_INTER + n0 + r] = a[r];
    }
}

// Routed down, shared down and the combine for MD_R output columns per
// simdgroup: x[d] += sum_slots w_s * down_s[d] + sigmoid(gate) * sharedDown[d],
// summed in moe_combine's order. Grid: 2048 / (simdgroups * MD_R).
kernel void moe_post_fp8(device float* x [[buffer(0)]],
                         device const float* act [[buffer(1)]],
                         device const ulong* table [[buffer(2)]],
                         device const int* ids [[buffer(3)]],
                         device const float* rw [[buffer(4)]],
                         device const float* pre [[buffer(5)]],
                         device const ushort* sharedDown [[buffer(6)]],
                         constant MoEDecodeParams& p [[buffer(7)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float sAct[MD_TOPK * MD_INTER];
    threadgroup float sShared[MD_SHARED];
    for (uint i = tid; i < MD_TOPK * MD_INTER; i += threads) sAct[i] = act[i];
    for (uint i = tid; i < MD_SHARED; i += threads) {
        float v = silu_f(pre[MD_EXPERTS + i]) * pre[MD_EXPERTS + MD_SHARED + i];
        sShared[i] = p.roundOutput ? round_to_bf16(v) : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e[MD_TOPK];
    float wts[MD_TOPK];
    for (uint s = 0; s < MD_TOPK; s++) { e[s] = uint(ids[s]); wts[s] = rw[s]; }
    float gate = pre[MD_EXPERTS + 2 * MD_SHARED];
    uint d0 = (tg * simdgroups + sg) * MD_R;
    for (uint r = 0; r < MD_R; r++) {
        uint d = d0 + r;
        float routed = 0.0f;
        for (uint s = 0; s < MD_TOPK; s++) {
            float y = md_down(table, e[s], d, (threadgroup const float*)(sAct + s * MD_INTER), lane, p.roundWeights, p.roundOutput);
            // Two statements: the product must round to float before the add,
            // as moe_down_fp8 stores it; in one expression it contracts to an fma.
            float slot = y * wts[s];
            routed += slot;
        }
        float so = md_bf16_row(sharedDown + d * MD_SHARED, MD_SHARED, (threadgroup const float*)sShared, lane);
        if (p.roundOutput) so = round_to_bf16(so);
        if (p.roundOutput) routed = round_to_bf16(routed);
        float total = routed + sigmoid_f(gate) * so;
        if (p.roundOutput) total = round_to_bf16(total);
        if (lane == 0) {
            float v = x[d] + total;
            x[d] = p.roundOutput ? round_to_bf16(v) : v;
        }
    }
}
