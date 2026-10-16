// Dense projections over three weight formats, chosen by a format tag:
//   bf16: W [N][K] ushort, no params.
//   i8:   W [N][K] int8, params [N][K/128] half scales (symmetric, per row per
//         128 columns), w = q * s.
//   i4:   W [N][K/2] bytes, element k in byte k/2 and the even element in the
//         low nibble, params [N][K/64] (half scale, half bias), w = q * s + b
//         with q in 0..15, formed with one fma so every reader of the format
//         dequantizes a weight to the same float.
// Each tag gives the row stride and two loaders: 8 consecutive weights at a
// column that is a multiple of 8 (the GEMV lane load) and 4 at a multiple of 4
// (the GEMM staging load), both as fp32. The bf16 loaders are the original
// code, so the bf16 kernels are unchanged bit for bit.
struct GemvParams {
    uint N;
    uint K;
    uint M;
    uint roundOutput;
};

struct FmtBF16 {
    static device const uchar* row(device const uchar* W, uint n, uint K) { return W + n * K * 2; }
    static void load8(device const uchar* row, device const half* P, uint n, uint K, uint k0, thread float* w) {
        // Tensor data in the shards is 8-byte aligned, not 16, so load two uint2.
        device const uint2* wrow = (device const uint2*)row;
        uint2 lo = wrow[k0 / 4];
        uint2 hi = wrow[k0 / 4 + 1];
        w[0] = bf16_to_float(lo.x & 0xffffu); w[1] = bf16_to_float(lo.x >> 16);
        w[2] = bf16_to_float(lo.y & 0xffffu); w[3] = bf16_to_float(lo.y >> 16);
        w[4] = bf16_to_float(hi.x & 0xffffu); w[5] = bf16_to_float(hi.x >> 16);
        w[6] = bf16_to_float(hi.y & 0xffffu); w[7] = bf16_to_float(hi.y >> 16);
    }
    static float4 load4(device const uchar* row, device const half* P, uint n, uint K, uint k0) {
        uint2 wv = *(device const uint2*)(row + k0 * 2);
        return float4(bf16_to_float(wv.x & 0xffffu), bf16_to_float(wv.x >> 16),
                      bf16_to_float(wv.y & 0xffffu), bf16_to_float(wv.y >> 16));
    }
};

inline float dq_i8(uint byte, float s) { return float(char(byte)) * s; }

struct FmtI8 {
    static device const uchar* row(device const uchar* W, uint n, uint K) { return W + n * K; }
    static void load8(device const uchar* row, device const half* P, uint n, uint K, uint k0, thread float* w) {
        uint2 q = *(device const uint2*)(row + k0);
        float s = float(P[n * (K / 128) + k0 / 128]);
        w[0] = dq_i8(q.x, s); w[1] = dq_i8(q.x >> 8, s); w[2] = dq_i8(q.x >> 16, s); w[3] = dq_i8(q.x >> 24, s);
        w[4] = dq_i8(q.y, s); w[5] = dq_i8(q.y >> 8, s); w[6] = dq_i8(q.y >> 16, s); w[7] = dq_i8(q.y >> 24, s);
    }
    static float4 load4(device const uchar* row, device const half* P, uint n, uint K, uint k0) {
        uint q = *(device const uint*)(row + k0);
        float s = float(P[n * (K / 128) + k0 / 128]);
        return float4(dq_i8(q, s), dq_i8(q >> 8, s), dq_i8(q >> 16, s), dq_i8(q >> 24, s));
    }
};

inline float dq_i4(uint nibble, half2 p) { return fma(float(nibble & 0xfu), float(p.x), float(p.y)); }

struct FmtI4 {
    static device const uchar* row(device const uchar* W, uint n, uint K) { return W + n * (K / 2); }
    static void load8(device const uchar* row, device const half* P, uint n, uint K, uint k0, thread float* w) {
        uint q = *(device const uint*)(row + k0 / 2);
        half2 p = ((device const half2*)P)[n * (K / 64) + k0 / 64];
        for (uint i = 0; i < 8; i++) w[i] = dq_i4(q >> (4 * i), p);
    }
    static float4 load4(device const uchar* row, device const half* P, uint n, uint K, uint k0) {
        uint q = *(device const ushort*)(row + k0 / 2);
        half2 p = ((device const half2*)P)[n * (K / 64) + k0 / 64];
        return float4(dq_i4(q, p), dq_i4(q >> 4, p), dq_i4(q >> 8, p), dq_i4(q >> 12, p));
    }
};

// y[m*N + n] = sum_k W[n][k] * x[m*K + k] for W [N, K], fp32 x.
// One simdgroup per output row and MTILE-token tile (the grid's second
// dimension); a threadgroup holds simdgroups_per_threadgroup rows, so the host
// chooses the rows per threadgroup by the thread count alone. Each lane reads
// 8 consecutive weights per K tile and keeps them in registers across the tokens.
// K must be a multiple of 256. A token's sum never depends on the tile, the
// other tokens in it, or the simdgroup count, so every variant is bit-identical.
// Token tile domain: 1, 8, 16, 32, 64. The accumulators live in registers, one
// per token, and 64 is where the unrolled loop stops fitting; 1 is the plain
// GEMV. Simdgroups per threadgroup: 1, 2, 4, 8 (32 to 256 threads).
template <typename FMT, uint MTILE>
kernel void gemv_t(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant GemvParams& p [[buffer(3)]],
                   device const half* P [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint simdgroups [[simdgroups_per_threadgroup]]) {
    uint n = tg.x * simdgroups + sg;
    if (n >= p.N) return;
    device const uchar* wrow = FMT::row(W, n, p.K);
    uint tiles = p.K / 256;
    uint m0 = tg.y * MTILE;
    if (m0 < p.M) {
        uint mcount = min(uint(MTILE), p.M - m0);
        float acc[MTILE];
#pragma unroll
        for (uint t = 0; t < MTILE; t++) acc[t] = 0.0f;
        for (uint tile = 0; tile < tiles; tile++) {
            uint k0 = tile * 256 + lane * 8;
            float w[8];
            FMT::load8(wrow, P, n, p.K, k0, w);
            // Constant trip count keeps acc in registers; the guard predicates.
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

#define GEMV_VARIANT(NAME, FMT, MTILE) \
    template [[host_name(NAME)]] kernel void gemv_t<FMT, MTILE>( \
        device const uchar*, device const float*, device float*, constant GemvParams&, device const half*, uint2, uint, uint, uint);
#define GEMV_FORMAT(TAG, FMT) \
    GEMV_VARIANT("gemv_" TAG "_t1", FMT, 1) \
    GEMV_VARIANT("gemv_" TAG "_t8", FMT, 8) \
    GEMV_VARIANT("gemv_" TAG "_t16", FMT, 16) \
    GEMV_VARIANT("gemv_" TAG "_t32", FMT, 32) \
    GEMV_VARIANT("gemv_" TAG "_t64", FMT, 64)
GEMV_FORMAT("bf16", FmtBF16)
GEMV_FORMAT("i8", FmtI8)
GEMV_FORMAT("i4", FmtI4)

// y[m*N + n] = sum_k W[n][k] * x[m*K + k] for M > 1 rows, as a
// BM x BN output tile per threadgroup of WM x WN simdgroups, stepping K by BK.
// The weights are widened to fp32 in threadgroup memory, so each product is
// the same fp32 product the GEMV forms; only the summation order differs.
// Every output element depends on its own row of x alone, so a row's result
// does not depend on which chunk it is in. K must be a multiple of BK; rows
// past M and columns past N are masked. Tile domain: the eight shapes
// instantiated below, 16 to 64 rows by 32 or 64 columns by a K step of 16 or
// 32; 64x64x32 is the largest whose staging (two 64 x 36 tiles) fits the
// 32 KB threadgroup budget with a 64 x 64 output tile.
template <typename FMT, uint BM, uint BN, uint BK, uint WM, uint WN>
kernel void gemm_t(device const uchar* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant GemvParams& p [[buffer(3)]],
                   device const half* P [[buffer(4)]],
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
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint idx = tid * 4; idx < BM * BK; idx += THREADS * 4) {
            uint row = idx / BK, col = idx % BK;
            float4 xv = float4(0.0f);
            if (m0 + row < p.M) xv = *(device const float4*)(x + (m0 + row) * p.K + k0 + col);
            *(threadgroup float4*)(xs + row * LD + col) = xv;
        }
        for (uint idx = tid * 4; idx < BN * BK; idx += THREADS * 4) {
            uint row = idx / BK, col = idx % BK;
            float4 wv = float4(0.0f);
            if (n0 + row < p.N) wv = FMT::load4(FMT::row(W, n0 + row, p.K), P, n0 + row, p.K, k0 + col);
            *(threadgroup float4*)(ws + row * LD + col) = wv;
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

#define GEMM_VARIANT(NAME, FMT, BM, BN, BK, WM, WN) \
    template [[host_name(NAME)]] kernel void gemm_t<FMT, BM, BN, BK, WM, WN>( \
        device const uchar*, device const float*, device float*, constant GemvParams&, device const half*, uint2, uint, uint);
#define GEMM_FORMAT(TAG, FMT) \
    GEMM_VARIANT("gemm_" TAG "_32x32x16", FMT, 32, 32, 16, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_32x32x32", FMT, 32, 32, 32, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_64x32x16", FMT, 64, 32, 16, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_32x64x16", FMT, 32, 64, 16, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_64x64x16", FMT, 64, 64, 16, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_64x64x32", FMT, 64, 64, 32, 2, 2) \
    GEMM_VARIANT("gemm_" TAG "_16x64x16", FMT, 16, 64, 16, 1, 4) \
    GEMM_VARIANT("gemm_" TAG "_16x32x16", FMT, 16, 32, 16, 1, 4)
GEMM_FORMAT("bf16", FmtBF16)
GEMM_FORMAT("i8", FmtI8)
GEMM_FORMAT("i4", FmtI4)

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
// partial sums combined in a fixed order. P domain: 1, 2, 4, 8 (128 to 1024
// threads, the Apple threadgroup limit). conv_out rows hold [q | k | v] with
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
