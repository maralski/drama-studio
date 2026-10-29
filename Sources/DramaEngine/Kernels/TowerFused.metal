// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// One-row (decode) kernels for the codebook tower. Each reproduces the
// arithmetic of the dispatches it replaces in the same order, so a tower run
// with them is bit-identical to the prefill path's dispatches on one row.

// y[n] += W[n] . x for M = 1: one output row per simdgroup, the lane's sum
// over the K tiles in order with the expression gemv_bf16_t uses, then the
// residual add that followed the GEMV.
struct GemvDecodeParams {
    uint N;
    uint K;
    uint roundOutput;
};

kernel void gemv_residual(device const ushort* W [[buffer(0)]],
                          device const float* x [[buffer(1)]],
                          device float* y [[buffer(2)]],
                          constant GemvDecodeParams& p [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint n = tg * simdgroups + sg;
    if (n >= p.N) return;
    uint tiles = p.K / 256;
    device const uint2* wrow = (device const uint2*)(W + n * p.K);
    float acc = 0.0f;
    for (uint tile = 0; tile < tiles; tile++) {
        uint2 lo = wrow[(tile * 32 + lane) * 2];
        uint2 hi = wrow[(tile * 32 + lane) * 2 + 1];
        device const float4* xp = (device const float4*)(x + tile * 256 + lane * 8);
        float4 a = xp[0];
        float4 b = xp[1];
        float w[8] = {
            bf16_to_float(lo.x & 0xffffu), bf16_to_float(lo.x >> 16),
            bf16_to_float(lo.y & 0xffffu), bf16_to_float(lo.y >> 16),
            bf16_to_float(hi.x & 0xffffu), bf16_to_float(hi.x >> 16),
            bf16_to_float(hi.y & 0xffffu), bf16_to_float(hi.y >> 16)};
        acc += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
             + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    }
    float s = simd_sum(acc);
    if (lane == 0) y[n] += p.roundOutput ? round_to_bf16(s) : s;
}

// out[n] = silu(W1[n] . x) * (W3[n] . x) for M = 1: the w1 and w3 GEMVs, each
// rounded when roundOutput is set, then silu_mul and its rounding. W1 and W3
// rows are summed as gemv_residual sums them.
kernel void swiglu_dec(device const ushort* W1 [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant GemvDecodeParams& p [[buffer(3)]],
                       device const ushort* W3 [[buffer(4)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint n = tg * simdgroups + sg;
    if (n >= p.N) return;
    uint tiles = p.K / 256;
    device const uint2* row1 = (device const uint2*)(W1 + n * p.K);
    device const uint2* row3 = (device const uint2*)(W3 + n * p.K);
    float acc1 = 0.0f, acc3 = 0.0f;
    for (uint tile = 0; tile < tiles; tile++) {
        uint2 lo1 = row1[(tile * 32 + lane) * 2];
        uint2 hi1 = row1[(tile * 32 + lane) * 2 + 1];
        uint2 lo3 = row3[(tile * 32 + lane) * 2];
        uint2 hi3 = row3[(tile * 32 + lane) * 2 + 1];
        device const float4* xp = (device const float4*)(x + tile * 256 + lane * 8);
        float4 a = xp[0];
        float4 b = xp[1];
        float w[8] = {
            bf16_to_float(lo1.x & 0xffffu), bf16_to_float(lo1.x >> 16),
            bf16_to_float(lo1.y & 0xffffu), bf16_to_float(lo1.y >> 16),
            bf16_to_float(hi1.x & 0xffffu), bf16_to_float(hi1.x >> 16),
            bf16_to_float(hi1.y & 0xffffu), bf16_to_float(hi1.y >> 16)};
        acc1 += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
              + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
        float v[8] = {
            bf16_to_float(lo3.x & 0xffffu), bf16_to_float(lo3.x >> 16),
            bf16_to_float(lo3.y & 0xffffu), bf16_to_float(lo3.y >> 16),
            bf16_to_float(hi3.x & 0xffffu), bf16_to_float(hi3.x >> 16),
            bf16_to_float(hi3.y & 0xffffu), bf16_to_float(hi3.y >> 16)};
        acc3 += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w
              + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
    }
    float s1 = simd_sum(acc1);
    float s3 = simd_sum(acc3);
    if (lane == 0) {
        if (p.roundOutput) { s1 = round_to_bf16(s1); s3 = round_to_bf16(s3); }
        float o = silu_f(s1) * s3;
        y[n] = p.roundOutput ? round_to_bf16(o) : o;
    }
}

// The tower layer's concat input for one row: threadgroup 0 computes the layer
// embedding (tower_embed, then round_bf16_inplace when roundOutput is set) and
// normalizes it into out[0, dim); threadgroup 1 normalizes the hidden row into
// out[dim, 2 dim). Reductions follow rmsnorm_zc with 256 threads.
struct TowerCatNormParams {
    uint dim;
    float eps;
    uint roundOutput;
    uint codebookSize;
    uint numCodebooks;
    uint addUnseen;
    int semanticStart;
    int semanticEnd;
};

kernel void tower_cat_norm(device float* out [[buffer(0)]],
                           device const float* hidden [[buffer(1)]],
                           device const ushort* enorm [[buffer(2)]],
                           device const ushort* hnorm [[buffer(3)]],
                           device const int* codes [[buffer(4)]],
                           device const int* semanticNext [[buffer(5)]],
                           device const ushort* table [[buffer(6)]],
                           device const int* frames [[buffer(7)]],
                           device const ushort* codebooks [[buffer(8)]],
                           constant TowerCatNormParams& p [[buffer(9)]],
                           uint which [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup float row[4096];
    device const ushort* w = which == 0 ? enorm : hnorm;
    if (which == 0) {
        bool sem = semanticNext[0] != 0;
        int c = clamp(codes[0], 0, int(p.codebookSize) - 1);
        int mainId = frames[0];
        bool unseen = p.addUnseen != 0 && mainId >= p.semanticStart && mainId <= p.semanticEnd;
        for (uint d = tid; d < p.dim; d += 256) {
            float v = 0.0f;
            if (sem) v = bf16_to_float(table[uint(c) * p.dim + d]);
            if (unseen) {
                for (uint k = 2; k < p.numCodebooks; k++) {
                    uint r = k * p.codebookSize + uint(frames[1 + k]);
                    v += bf16_to_float(codebooks[r * p.dim + d]);
                }
            }
            row[d] = p.roundOutput ? round_to_bf16(v) : v;
        }
    } else {
        for (uint d = tid; d < p.dim; d += 256) row[d] = hidden[d];
    }
    float ss = 0.0f;
    for (uint i = tid; i < p.dim; i += 256) { float v = row[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8; s++) total += partial[s];
    float inv = rsqrt(total / float(p.dim) + p.eps);
    device float* o = out + which * p.dim;
    for (uint i = tid; i < p.dim; i += 256) {
        float v = row[i] * inv * (1.0f + bf16_to_float(w[i]));
        o[i] = p.roundOutput ? round_to_bf16(v) : v;
    }
}

// attention for one query row with its S key residues spread over B
// threadgroups per head (S / B simdgroups each), U keys of a simdgroup loaded
// before they are folded into its online softmax one at a time in key order.
// Every simdgroup's softmax lands in partials: max and sum at [h * S + s], the
// 256 accumulators at [(heads * S * 2) + (h * S + s) * 256].
// attention_combine_gate then folds them in residue order exactly as the
// attention kernel's threadgroup does.
struct AttnDecodeParams {
    uint heads;
    uint kvHeads;
    uint pos0;
    float scale;
    uint roundOutput;
};

template <uint U>
kernel void attention_split_t(device float* partials [[buffer(0)]],
                              device const float* q [[buffer(1)]],
                              device const float* kcache [[buffer(2)]],
                              device const float* vcache [[buffer(3)]],
                              constant AttnDecodeParams& p [[buffer(4)]],
                              constant uint& S [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sgl [[simdgroup_index_in_threadgroup]],
                              uint SB [[simdgroups_per_threadgroup]]) {
    uint h = tg.x;
    uint sg = tg.y * SB + sgl;
    uint kvh = h / (p.heads / p.kvHeads);
    uint pos = p.pos0;
    uint stride = p.kvHeads * 256;
    device const float* qr = q + h * 256 + lane * 8;
    float4 q0 = *(device const float4*)qr;
    float4 q1 = *(device const float4*)(qr + 4);
    float runMax = -INFINITY;
    float runSum = 0.0f;
    float4 acc0 = 0.0f, acc1 = 0.0f;
    device const float* kb = kcache + kvh * 256 + lane * 8;
    device const float* vb = vcache + kvh * 256 + lane * 8;
    uint key = sg;
    for (; key + (U - 1) * S <= pos; key += U * S) {
        float4 k0[U], k1[U], v0[U], v1[U];
#pragma unroll
        for (uint u = 0; u < U; u++) {
            device const float* kr = kb + (key + u * S) * stride;
            device const float* vr = vb + (key + u * S) * stride;
            k0[u] = *(device const float4*)kr;
            k1[u] = *(device const float4*)(kr + 4);
            v0[u] = *(device const float4*)vr;
            v1[u] = *(device const float4*)(vr + 4);
        }
        float sc[U];
#pragma unroll
        for (uint u = 0; u < U; u++) sc[u] = simd_sum(dot(q0, k0[u]) + dot(q1, k1[u])) * p.scale;
#pragma unroll
        for (uint u = 0; u < U; u++) {
            float newMax = max(runMax, sc[u]);
            float corr = exp(runMax - newMax);
            float pw = exp(sc[u] - newMax);
            runSum = runSum * corr + pw;
            acc0 = acc0 * corr + pw * v0[u];
            acc1 = acc1 * corr + pw * v1[u];
            runMax = newMax;
        }
    }
    for (; key <= pos; key += S) {
        device const float* kr = kb + key * stride;
        float4 k0 = *(device const float4*)kr;
        float4 k1 = *(device const float4*)(kr + 4);
        float s = simd_sum(dot(q0, k0) + dot(q1, k1)) * p.scale;
        float newMax = max(runMax, s);
        float corr = exp(runMax - newMax);
        float pw = exp(s - newMax);
        runSum = runSum * corr + pw;
        device const float* vr = vb + key * stride;
        acc0 = acc0 * corr + pw * *(device const float4*)vr;
        acc1 = acc1 * corr + pw * *(device const float4*)(vr + 4);
        runMax = newMax;
    }
    uint slot = h * S + sg;
    if (lane == 0) { partials[2 * slot] = runMax; partials[2 * slot + 1] = runSum; }
    device float* pa = partials + p.heads * S * 2 + slot * 256 + lane * 8;
    *(device float4*)pa = acc0;
    *(device float4*)(pa + 4) = acc1;
}

template [[host_name("attention_split_u2")]] kernel void attention_split_t<2>(
    device float*, device const float*, device const float*, device const float*, constant AttnDecodeParams&, constant uint&,
    uint2, uint, uint, uint);

// One threadgroup of 256 per head: the residues' softmaxes combined in order,
// then sigmoid_gate_mul, out = o * sigmoid(gate), the attention output and the
// product each rounded when roundOutput is set.
kernel void attention_combine_gate(device float* out [[buffer(0)]],
                                   device const float* partials [[buffer(1)]],
                                   constant AttnDecodeParams& p [[buffer(4)]],
                                   constant uint& S [[buffer(5)]],
                                   device const float* gate [[buffer(6)]],
                                   uint h [[threadgroup_position_in_grid]],
                                   uint d [[thread_position_in_threadgroup]]) {
    device const float* st = partials + 2 * h * S;
    device const float* acc = partials + p.heads * S * 2 + h * S * 256;
    float gmax = -INFINITY;
    for (uint s = 0; s < S; s++) gmax = max(gmax, st[2 * s]);
    float total = 0.0f;
    for (uint s = 0; s < S; s++) total += (st[2 * s] == -INFINITY ? 0.0f : exp(st[2 * s] - gmax)) * st[2 * s + 1];
    float o = 0.0f;
    for (uint s = 0; s < S; s++) {
        float w = st[2 * s] == -INFINITY ? 0.0f : exp(st[2 * s] - gmax);
        o += w * acc[s * 256 + d];
    }
    float a = o / total;
    if (p.roundOutput) a = round_to_bf16(a);
    float gv = a * sigmoid_f(gate[h * 256 + d]);
    out[h * 256 + d] = p.roundOutput ? round_to_bf16(gv) : gv;
}
