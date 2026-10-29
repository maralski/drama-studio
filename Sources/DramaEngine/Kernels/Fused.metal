// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Fused decode kernels. Each one performs the fp32 operations of the kernels
// it replaces in the same order, so its outputs are bit-identical to theirs.

struct NormRopeKVParams {
    uint heads;
    uint kvHeads;
    uint pos0;
    uint rotaryDim;
    float theta;
    float eps;
    uint interleaved;
    HeadView q;
    HeadView k;
    HeadView v;
};

// Per-head q and k rmsnorm_zc, RoPE on both, the k write and the v copy into
// the caches: the two norms, two ropes and two kv_writes of an attention
// block. Threadgroup (j, m) of 256 threads: j < heads is q head j of row m,
// then the kv heads' k, then their v.
kernel void norm_rope_kv(device float* qOut [[buffer(0)]],
                         device float* kCache [[buffer(1)]],
                         device float* vCache [[buffer(2)]],
                         device const float* qSrc [[buffer(3)]],
                         device const float* kSrc [[buffer(4)]],
                         device const float* vSrc [[buffer(5)]],
                         device const ushort* qNorm [[buffer(6)]],
                         device const ushort* kNorm [[buffer(7)]],
                         constant NormRopeKVParams& p [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup float normed[256];
    uint j = tg.x, m = tg.y;
    if (j >= p.heads + p.kvHeads) {
        uint h = j - p.heads - p.kvHeads;
        uint row = m * p.kvHeads + h;
        vCache[((p.pos0 + m) * p.kvHeads + h) * 256 + tid] = vSrc[head_index(p.v, row) + tid];
        return;
    }
    bool isQ = j < p.heads;
    uint heads = isQ ? p.heads : p.kvHeads;
    uint h = isQ ? j : j - p.heads;
    uint row = m * heads + h;
    device const float* xr = (isQ ? qSrc + head_index(p.q, row) : kSrc + head_index(p.k, row));
    device const ushort* w = isQ ? qNorm : kNorm;
    float ss = 0.0f;
    float xv = xr[tid];
    ss += xv * xv;
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8; s++) total += partial[s];
    float inv = rsqrt(total / 256.0f + p.eps);
    float val = xr[tid] * inv * (1.0f + bf16_to_float(w[tid]));
    normed[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* o = isQ ? qOut + row * 256 : kCache + ((p.pos0 + m) * p.kvHeads + h) * 256;
    uint halfDim = p.rotaryDim / 2;
    if (tid >= p.rotaryDim) {
        o[tid] = val;
    } else if (tid < halfDim) {
        uint i = tid;
        float pos = float(p.pos0 + row / heads);
        float angle = pos * pow(p.theta, -float(2 * i) / float(p.rotaryDim));
        float c = cos(angle), s = sin(angle);
        if (p.interleaved != 0) {
            float a = normed[2 * i], b = normed[2 * i + 1];
            o[2 * i] = a * c - b * s;
            o[2 * i + 1] = b * c + a * s;
        } else {
            float a = normed[i], b = normed[i + halfDim];
            o[i] = a * c - b * s;
            o[i + halfDim] = b * c + a * s;
        }
    }
}
