// y[m*N + n] = sum_k W[rowStart + n][k] * x[m*K + k] for bf16 W [*, K], fp32 x.
// One simdgroup per output row, each lane reads 8 consecutive bf16 per tile.
// K must be a multiple of 256. Tokens are processed in tiles of 32 so the
// weight tile stays in registers across them.
struct GemvParams {
    uint N;
    uint K;
    uint M;
    uint rowStart;
    uint roundOutput;
};

#define GEMV_MTILE 32

kernel void gemv_bf16(device const ushort* W [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvParams& p [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    uint n = tg * 4 + sg;
    if (n >= p.N) return;
    // Tensor data in the shards is 8-byte aligned, not 16, so load two uint2.
    device const uint2* wrow = (device const uint2*)(W + (p.rowStart + n) * p.K);
    uint tiles = p.K / 256;
    for (uint m0 = 0; m0 < p.M; m0 += GEMV_MTILE) {
        uint mcount = min(uint(GEMV_MTILE), p.M - m0);
        float acc[GEMV_MTILE];
#pragma unroll
        for (uint t = 0; t < GEMV_MTILE; t++) acc[t] = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint2 lo = wrow[(tile * 32 + lane) * 2];
            uint2 hi = wrow[(tile * 32 + lane) * 2 + 1];
            float w[8] = {
                bf16_to_float(lo.x & 0xffffu), bf16_to_float(lo.x >> 16),
                bf16_to_float(lo.y & 0xffffu), bf16_to_float(lo.y >> 16),
                bf16_to_float(hi.x & 0xffffu), bf16_to_float(hi.x >> 16),
                bf16_to_float(hi.y & 0xffffu), bf16_to_float(hi.y >> 16)};
            uint k0 = tile * 256 + lane * 8;
            // Constant trip count keeps acc in registers; the guard predicates.
#pragma unroll
            for (uint t = 0; t < GEMV_MTILE; t++) {
                if (t < mcount) {
                    device const float4* xp = (device const float4*)(x + (m0 + t) * p.K + k0);
                    float4 a = xp[0];
                    float4 b = xp[1];
                    acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
                            + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                }
            }
        }
#pragma unroll
        for (uint t = 0; t < GEMV_MTILE; t++) {
            if (t < mcount) {
                float s = simd_sum(acc[t]);
                if (lane == 0) y[(m0 + t) * p.N + n] = p.roundOutput ? round_to_bf16(s) : s;
            }
        }
    }
}

// Depthwise causal conv (kernel 4) over time with carried state, then SiLU.
// mixed [M, C]; state [C, 3] oldest first; w [C, 4]; out [M, C].
struct ConvParams {
    uint C;
    uint M;
};

kernel void gdn_conv_silu(device float* out [[buffer(0)]],
                          device const float* mixed [[buffer(1)]],
                          device const float* state [[buffer(2)]],
                          device const ushort* w [[buffer(3)]],
                          constant ConvParams& p [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint m = gid.y;
    if (c >= p.C) return;
    float acc = 0.0f;
    for (uint j = 0; j < 4; j++) {
        int t = int(m) - 3 + int(j);
        float v = t >= 0 ? mixed[uint(t) * p.C + c] : state[c * 3 + uint(t + 3)];
        acc += bf16_to_float(w[c * 4 + j]) * v;
    }
    out[m * p.C + c] = silu_f(acc);
}

// New state = the last three pre-activation inputs.
kernel void gdn_conv_state_update(device float* state [[buffer(0)]],
                                  device const float* mixed [[buffer(1)]],
                                  constant ConvParams& p [[buffer(2)]],
                                  uint c [[thread_position_in_grid]]) {
    if (c >= p.C) return;
    float v[3];
    for (uint j = 0; j < 3; j++) {
        int t = int(p.M) - 3 + int(j);
        v[j] = t >= 0 ? mixed[uint(t) * p.C + c] : state[c * 3 + uint(t + 3)];
    }
    for (uint j = 0; j < 3; j++) state[c * 3 + j] = v[j];
}

// Gated delta rule, token by token, in fp32. One threadgroup (128 threads) per
// value head; thread j owns state column j. conv_out rows hold [q | k | v] with
// q, k over 16 heads of 128 and v over 32 heads of 128; value head h uses key
// head h / 2. a and b are [M, 32] each.
struct GDNParams {
    uint M;
    uint rowStride;
    uint numVHeads;
    uint numKHeads;
    float eps;
};

kernel void gdn_recurrent(device float* core [[buffer(0)]],
                          device const float* conv_out [[buffer(1)]],
                          device const float* aIn [[buffer(2)]],
                          device const float* bIn [[buffer(3)]],
                          device float* state [[buffer(4)]],
                          device const ushort* A_log [[buffer(5)]],
                          device const ushort* dt_bias [[buffer(6)]],
                          constant GDNParams& p [[buffer(7)]],
                          uint h [[threadgroup_position_in_grid]],
                          uint j [[thread_position_in_threadgroup]]) {
    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float v[128];
    threadgroup float sums[2];
    uint groups = p.numVHeads / p.numKHeads;
    uint kh = h / groups;
    uint kDim = p.numKHeads * 128;
    device float* S = state + h * 128 * 128;
    float s[128];
    for (uint i = 0; i < 128; i++) s[i] = S[i * 128 + j];
    float aLog = exp(bf16_to_float(A_log[h]));
    float dtb = bf16_to_float(dt_bias[h]);
    float qScale = rsqrt(128.0f);
    for (uint m = 0; m < p.M; m++) {
        device const float* row = conv_out + m * p.rowStride;
        q[j] = row[kh * 128 + j];
        k[j] = row[kDim + kh * 128 + j];
        v[j] = row[2 * kDim + h * 128 + j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j < 2) {
            float ss = 0.0f;
            for (uint i = 0; i < 128; i++) { float t = j == 0 ? q[i] : k[i]; ss += t * t; }
            sums[j] = rsqrt(ss + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qn = sums[0] * qScale;
        float kn = sums[1];
        float a = aIn[m * p.numVHeads + h];
        float b = bIn[m * p.numVHeads + h];
        float beta = sigmoid_f(b);
        float decay = -aLog * softplus_f(a + dtb);
        float g = exp(decay);
        float rem = 0.0f;
        for (uint i = 0; i < 128; i++) { s[i] *= g; rem += s[i] * k[i] * kn; }
        float delta = (v[j] - rem) * beta;
        float o = 0.0f;
        for (uint i = 0; i < 128; i++) { s[i] += k[i] * kn * delta; o += s[i] * q[i] * qn; }
        core[m * (p.numVHeads * 128) + h * 128 + j] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 128; i++) S[i * 128 + j] = s[i];
}
