// y[m*N + n] = sum_k W[rowStart + n][k] * x[m*K + k] for bf16 W [*, K], fp32 x.
// One simdgroup per output row and 32-token tile (the grid's second
// dimension), each lane reads 8 consecutive bf16 per K tile and keeps them in
// registers across the tokens. K must be a multiple of 256. A token's sum
// never depends on the other tokens in its tile.
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
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    uint n = tg.x * 4 + sg;
    if (n >= p.N) return;
    // Tensor data in the shards is 8-byte aligned, not 16, so load two uint2.
    device const uint2* wrow = (device const uint2*)(W + (p.rowStart + n) * p.K);
    uint tiles = p.K / 256;
    uint m0 = tg.y * GEMV_MTILE;
    if (m0 < p.M) {
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

// y[m*N + n] = sum_k W[rowStart + n][k] * x[m*K + k] for M > 1 rows, as a
// BM x BN output tile per threadgroup of WM x WN simdgroups, stepping K by BK.
// The bf16 weights are widened to fp32 in threadgroup memory, so each product
// is the same fp32 product the GEMV forms; only the summation order differs.
// Every output element depends on its own row of x alone, so a row's result
// does not depend on which chunk it is in. K must be a multiple of BK; rows
// past M and columns past N are masked.
template <uint BM, uint BN, uint BK, uint WM, uint WN>
kernel void gemm_bf16_t(device const ushort* W [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant GemvParams& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint LD = BK + 4;
    constexpr uint THREADS = 32 * WM * WN;
    constexpr uint SM = BM / WM, SN = BN / WN;
    constexpr uint TM = SM / 8, TN = SN / 8;
    threadgroup float smem[(BM + BN) * LD > BM * BN ? (BM + BN) * LD : BM * BN];
    threadgroup float* xs = smem;
    threadgroup float* ws = smem + BM * LD;
    const uint n0 = tg.x * BN, m0 = tg.y * BM;
    const uint sm = (sg / WN) * SM, sn = (sg % WN) * SN;
    simdgroup_float8x8 acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    device const ushort* wbase = W + (p.rowStart + n0) * p.K;
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint idx = tid * 4; idx < BM * BK; idx += THREADS * 4) {
            uint row = idx / BK, col = idx % BK;
            float4 xv = float4(0.0f);
            if (m0 + row < p.M) xv = *(device const float4*)(x + (m0 + row) * p.K + k0 + col);
            *(threadgroup float4*)(xs + row * LD + col) = xv;
        }
        for (uint idx = tid * 4; idx < BN * BK; idx += THREADS * 4) {
            uint row = idx / BK, col = idx % BK;
            uint2 wv = uint2(0u);
            if (n0 + row < p.N) wv = *(device const uint2*)(wbase + row * p.K + k0 + col);
            *(threadgroup float4*)(ws + row * LD + col) =
                float4(bf16_to_float(wv.x & 0xffffu), bf16_to_float(wv.x >> 16),
                       bf16_to_float(wv.y & 0xffffu), bf16_to_float(wv.y >> 16));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[TM], bt[TN];
            for (uint i = 0; i < TM; i++) simdgroup_load(a[i], xs + (sm + i * 8) * LD + kk, LD);
            for (uint j = 0; j < TN; j++) simdgroup_load(bt[j], ws + (sn + j * 8) * LD + kk, LD, ulong2(0, 0), true);
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], bt[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++) simdgroup_store(acc[i][j], smem + (sm + i * 8) * BN + sn + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid * 4; idx < BM * BN; idx += THREADS * 4) {
        uint row = idx / BN, col = idx % BN;
        if (m0 + row >= p.M || n0 + col >= p.N) continue;
        float4 v = *(threadgroup float4*)(smem + row * BN + col);
        if (p.roundOutput) v = float4(round_to_bf16(v.x), round_to_bf16(v.y), round_to_bf16(v.z), round_to_bf16(v.w));
        device float* out = y + (m0 + row) * p.N + n0 + col;
        if (n0 + col + 4 <= p.N && (p.N & 3u) == 0u) {
            *(device float4*)out = v;
        } else {
            for (uint c = 0; c < 4 && n0 + col + c < p.N; c++) out[c] = v[c];
        }
    }
}

#define GEMM_VARIANT(NAME, BM, BN, BK, WM, WN) \
    template [[host_name(NAME)]] kernel void gemm_bf16_t<BM, BN, BK, WM, WN>( \
        device const ushort*, device const float*, device float*, constant GemvParams&, uint2, uint, uint);
GEMM_VARIANT("gemm_bf16_32x32x16", 32, 32, 16, 2, 2)
GEMM_VARIANT("gemm_bf16_32x32x32", 32, 32, 32, 2, 2)
GEMM_VARIANT("gemm_bf16_64x32x16", 64, 32, 16, 2, 2)
GEMM_VARIANT("gemm_bf16_32x64x16", 32, 64, 16, 2, 2)
GEMM_VARIANT("gemm_bf16_64x64x16", 64, 64, 16, 2, 2)
GEMM_VARIANT("gemm_bf16_64x64x32", 64, 64, 32, 2, 2)
GEMM_VARIANT("gemm_bf16_16x64x16", 16, 64, 16, 1, 4)
GEMM_VARIANT("gemm_bf16_16x32x16", 16, 32, 16, 1, 4)

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

// Gated delta rule, token by token, in fp32. One threadgroup of 128 * P
// threads per value head; thread (part, j) owns rows [part * 128 / P, +128 / P)
// of state column j, so the per-token dot products over the 128 rows are P
// partial sums combined in a fixed order. conv_out rows hold [q | k | v] with
// q, k over 16 heads of 128 and v over 32 heads of 128; value head h uses key
// head h / 2. a and b are [M, 32] each.
struct GDNParams {
    uint M;
    uint rowStride;
    uint numVHeads;
    uint numKHeads;
    float eps;
};

template <uint P>
kernel void gdn_recurrent_t(device float* core [[buffer(0)]],
                            device const float* conv_out [[buffer(1)]],
                            device const float* aIn [[buffer(2)]],
                            device const float* bIn [[buffer(3)]],
                            device float* state [[buffer(4)]],
                            device const ushort* A_log [[buffer(5)]],
                            device const ushort* dt_bias [[buffer(6)]],
                            constant GDNParams& p [[buffer(7)]],
                            uint h [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint R = 128 / P;
    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float v[128];
    threadgroup float red[8];
    threadgroup float remPart[P][128];
    threadgroup float oPart[P][128];
    uint j = t % 128, part = t / 128;
    uint groups = p.numVHeads / p.numKHeads;
    uint kh = h / groups;
    uint kDim = p.numKHeads * 128;
    device float* S = state + h * 128 * 128 + part * R * 128;
    float s[R];
    for (uint r = 0; r < R; r++) s[r] = S[r * 128 + j];
    float aLog = exp(bf16_to_float(A_log[h]));
    float dtb = bf16_to_float(dt_bias[h]);
    float qScale = rsqrt(128.0f);
    for (uint m = 0; m < p.M; m++) {
        if (part == 0) {
            device const float* row = conv_out + m * p.rowStride;
            float qj = row[kh * 128 + j], kj = row[kDim + kh * 128 + j];
            q[j] = qj;
            k[j] = kj;
            v[j] = row[2 * kDim + h * 128 + j];
            float qq = simd_sum(qj * qj), kk = simd_sum(kj * kj);
            if (lane == 0) { red[sg] = qq; red[4 + sg] = kk; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qn = rsqrt(red[0] + red[1] + red[2] + red[3] + p.eps) * qScale;
        float kn = rsqrt(red[4] + red[5] + red[6] + red[7] + p.eps);
        float a = aIn[m * p.numVHeads + h];
        float b = bIn[m * p.numVHeads + h];
        float beta = sigmoid_f(b);
        float decay = -aLog * softplus_f(a + dtb);
        float g = exp(decay);
        float rem = 0.0f;
        for (uint r = 0; r < R; r++) { s[r] *= g; rem += s[r] * k[part * R + r] * kn; }
        remPart[part][j] = rem;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float remAll = 0.0f;
        for (uint i = 0; i < P; i++) remAll += remPart[i][j];
        float delta = (v[j] - remAll) * beta;
        float o = 0.0f;
        for (uint r = 0; r < R; r++) { s[r] += k[part * R + r] * kn * delta; o += s[r] * q[part * R + r] * qn; }
        oPart[part][j] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0) {
            float oAll = 0.0f;
            for (uint i = 0; i < P; i++) oAll += oPart[i][j];
            core[m * (p.numVHeads * 128) + h * 128 + j] = oAll;
        }
    }
    for (uint r = 0; r < R; r++) S[r * 128 + j] = s[r];
}

#define GDN_VARIANT(NAME, P) \
    template [[host_name(NAME)]] kernel void gdn_recurrent_t<P>( \
        device float*, device const float*, device const float*, device const float*, device float*, \
        device const ushort*, device const ushort*, constant GDNParams&, uint, uint, uint, uint);
GDN_VARIANT("gdn_recurrent_p1", 1)
GDN_VARIANT("gdn_recurrent_p2", 2)
GDN_VARIANT("gdn_recurrent_p4", 4)
GDN_VARIANT("gdn_recurrent_p8", 8)
