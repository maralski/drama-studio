// Quantizers for the weight cache, one thread per row group. The scales are
// half: their 2^-11 relative rounding is an eighth of the int8 step and half
// the bytes of fp32. A group whose scale rounds to zero quantizes to q = 0.
struct QuantizeParams {
    uint N;          // rows of the matrix being quantized
    uint K;          // columns, a multiple of 128
    uint rowOffset;  // first source row (the lm_head slices)
    uint kind;       // expert matrix: 0 gate, 1 up, 2 down
};

// Q[n][k] int8 and S[n][k/128] = half(absmax / 127); grid (K/128, N).
kernel void quantize_bf16_i8(device const ushort* W [[buffer(0)]],
                             device char* Q [[buffer(1)]],
                             device half* S [[buffer(2)]],
                             constant QuantizeParams& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint g = gid.x, n = gid.y;
    if (g >= p.K / 128 || n >= p.N) return;
    device const ushort* src = W + (p.rowOffset + n) * p.K + g * 128;
    float absmax = 0.0f;
    for (uint i = 0; i < 128; i++) absmax = max(absmax, fabs(bf16_to_float(src[i])));
    half hs = half(absmax / 127.0f);
    float s = float(hs);
    device char* dst = Q + n * p.K + g * 128;
    for (uint i = 0; i < 128; i++) {
        float q = s > 0.0f ? clamp(rint(bf16_to_float(src[i]) / s), -127.0f, 127.0f) : 0.0f;
        dst[i] = char(int(q));
    }
    S[n * (p.K / 128) + g] = hs;
}

// One group of 64 weights to nibbles: hs = half((max - min) / 15), hb =
// half(min), q = clamp(rint((w - hb) / hs), 0, 15), element k in byte k/2
// with the even element in the low nibble, and P[group] = (hs, hb).
inline void quantize_group_i4(thread const float* w, device uchar* dst, device half2* param) {
    float lo = w[0], hi = w[0];
    for (uint i = 1; i < 64; i++) { lo = min(lo, w[i]); hi = max(hi, w[i]); }
    half hs = half((hi - lo) / 15.0f);
    half hb = half(lo);
    float s = float(hs), b = float(hb);
    for (uint i = 0; i < 32; i++) {
        uint q0 = s > 0.0f ? uint(clamp(rint((w[2 * i] - b) / s), 0.0f, 15.0f)) : 0u;
        uint q1 = s > 0.0f ? uint(clamp(rint((w[2 * i + 1] - b) / s), 0.0f, 15.0f)) : 0u;
        dst[i] = uchar(q0 | (q1 << 4));
    }
    *param = half2(hs, hb);
}

// Q[n][k/2] nibbles and P[n][k/64]; grid (K/64, N).
kernel void quantize_bf16_i4(device const ushort* W [[buffer(0)]],
                             device uchar* Q [[buffer(1)]],
                             device half2* P [[buffer(2)]],
                             constant QuantizeParams& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint g = gid.x, n = gid.y;
    if (g >= p.K / 64 || n >= p.N) return;
    device const ushort* src = W + (p.rowOffset + n) * p.K + g * 64;
    float w[64];
    for (uint i = 0; i < 64; i++) w[i] = bf16_to_float(src[i]);
    quantize_group_i4(w, Q + n * (p.K / 2) + g * 32, P + n * (p.K / 64) + g);
}

// All experts of one layer and one matrix kind through the fp8 address table,
// into an expert-major staging: expert e's nibbles at Q + e * N * K / 2 and its
// params at P + e * N * K / 64. The weight is the same float the fp8 kernels
// form, FP8_LUT[byte] * blockScale. Grid (K/64, N, experts).
kernel void quantize_fp8_i4(device const ulong* table [[buffer(0)]],
                            device uchar* Q [[buffer(1)]],
                            device half2* P [[buffer(2)]],
                            constant QuantizeParams& p [[buffer(3)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint g = gid.x, n = gid.y, e = gid.z;
    if (g >= p.K / 64 || n >= p.N) return;
    device const uchar* W = (device const uchar*)table[e * 6 + p.kind * 2] + n * p.K + g * 64;
    device const float* S = (device const float*)table[e * 6 + p.kind * 2 + 1];
    float scale = S[(n / 128) * (p.K / 128) + (g * 64) / 128];
    float w[64];
    for (uint i = 0; i < 64; i++) w[i] = FP8_LUT[W[i]] * scale;
    quantize_group_i4(w, Q + (e * p.N + n) * (p.K / 2) + g * 32, P + (e * p.N + n) * (p.K / 64) + g);
}
