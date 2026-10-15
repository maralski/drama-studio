// Partial RoPE on contiguous [rows, 256] with the row's position = pos0 + row / heads.
struct RopeParams {
    uint heads;
    uint pos0;
    uint rotaryDim;
    float theta;
};

// NeoX layout: pairs (i, i + rotaryDim/2).
kernel void rope_neox(device float* x [[buffer(0)]],
                      constant RopeParams& p [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint row = gid.y;
    uint half = p.rotaryDim / 2;
    if (i >= half) return;
    float pos = float(p.pos0 + row / p.heads);
    float angle = pos * pow(p.theta, -float(2 * i) / float(p.rotaryDim));
    float c = cos(angle), s = sin(angle);
    device float* r = x + row * 256;
    float a = r[i], b = r[i + half];
    r[i] = a * c - b * s;
    r[i + half] = b * c + a * s;
}

// Interleaved layout: pairs (2i, 2i + 1).
kernel void rope_interleaved(device float* x [[buffer(0)]],
                             constant RopeParams& p [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint row = gid.y;
    uint half = p.rotaryDim / 2;
    if (i >= half) return;
    float pos = float(p.pos0 + row / p.heads);
    float angle = pos * pow(p.theta, -float(2 * i) / float(p.rotaryDim));
    float c = cos(angle), s = sin(angle);
    device float* r = x + row * 256;
    float a = r[2 * i], b = r[2 * i + 1];
    r[2 * i] = a * c - b * s;
    r[2 * i + 1] = b * c + a * s;
}

// cache[(pos0 + m) * kvHeads + h][256] = src[view(m * kvHeads + h)][...]
struct KVWriteParams {
    uint kvHeads;
    uint pos0;
    HeadView view;
};

kernel void kv_write(device float* cache [[buffer(0)]],
                     device const float* src [[buffer(1)]],
                     constant KVWriteParams& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint row = gid.y;
    uint m = row / p.kvHeads;
    uint h = row % p.kvHeads;
    cache[((p.pos0 + m) * p.kvHeads + h) * 256 + d] = src[head_index(p.view, row) + d];
}

// Causal GQA attention over an fp32 KV cache, head dim 256, flash-decoding
// style: one threadgroup per (query row, head); 8 simdgroups split the keys
// and keep an online softmax; lanes own 8 dims each.
struct AttnParams {
    uint heads;
    uint kvHeads;
    uint pos0;
    float scale;
};

kernel void attention(device float* out [[buffer(0)]],
                      device const float* q [[buffer(1)]],
                      device const float* kcache [[buffer(2)]],
                      device const float* vcache [[buffer(3)]],
                      constant AttnParams& p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float tgMax[8];
    threadgroup float tgSum[8];
    threadgroup float tgAcc[8][256];
    uint h = tg.x;
    uint m = tg.y;
    uint kvh = h / (p.heads / p.kvHeads);
    uint pos = p.pos0 + m;
    device const float* qr = q + (m * p.heads + h) * 256 + lane * 8;
    float4 q0 = *(device const float4*)qr;
    float4 q1 = *(device const float4*)(qr + 4);
    float runMax = -INFINITY;
    float runSum = 0.0f;
    float4 acc0 = 0.0f, acc1 = 0.0f;
    for (uint key = sg; key <= pos; key += 8) {
        device const float* kr = kcache + (key * p.kvHeads + kvh) * 256 + lane * 8;
        float4 k0 = *(device const float4*)kr;
        float4 k1 = *(device const float4*)(kr + 4);
        float s = simd_sum(dot(q0, k0) + dot(q1, k1)) * p.scale;
        float newMax = max(runMax, s);
        float corr = exp(runMax - newMax);
        float pw = exp(s - newMax);
        runSum = runSum * corr + pw;
        device const float* vr = vcache + (key * p.kvHeads + kvh) * 256 + lane * 8;
        acc0 = acc0 * corr + pw * *(device const float4*)vr;
        acc1 = acc1 * corr + pw * *(device const float4*)(vr + 4);
        runMax = newMax;
    }
    if (lane == 0) { tgMax[sg] = runMax; tgSum[sg] = runSum; }
    for (uint i = 0; i < 4; i++) { tgAcc[sg][lane * 8 + i] = acc0[i]; tgAcc[sg][lane * 8 + 4 + i] = acc1[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = -INFINITY;
    for (uint s = 0; s < 8; s++) gmax = max(gmax, tgMax[s]);
    float total = 0.0f;
    float o = 0.0f;
    for (uint s = 0; s < 8; s++) {
        float w = tgMax[s] == -INFINITY ? 0.0f : exp(tgMax[s] - gmax);
        total += w * tgSum[s];
        o += w * tgAcc[s][tid];
    }
    out[(m * p.heads + h) * 256 + tid] = o / total;
}
