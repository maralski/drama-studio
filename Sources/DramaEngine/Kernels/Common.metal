#include <metal_stdlib>
using namespace metal;

// bf16 stored as ushort; widening is exact.
inline float bf16_to_float(ushort b) { return as_type<float>(uint(b) << 16); }

// Round-to-nearest-even to bf16 and back, mirroring torch's bf16 rounding.
inline float round_to_bf16(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return x;
    uint lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return as_type<float>(u & 0xffff0000u);
}

// FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa, no infinities.
inline float fp8_e4m3_to_float(uchar b) {
    uint s = b >> 7;
    uint e = (b >> 3) & 0xf;
    uint m = b & 0x7;
    float v;
    if (e == 0) {
        v = ldexp(float(m), -9);
    } else if (e == 15 && m == 7) {
        v = NAN;
    } else {
        v = ldexp(1.0f + float(m) / 8.0f, int(e) - 7);
    }
    return s ? -v : v;
}

inline float silu_f(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softplus_f(float x) { return x > 20.0f ? x : log1p(exp(x)); }

// A view of per-head rows inside a wider row: element (row, i) lives at
// base[(row / heads) * groupStride + (row % heads) * headStride + i].
struct HeadView {
    uint heads;
    uint groupStride;
    uint headStride;
};

inline uint head_index(HeadView v, uint row) {
    return (row / v.heads) * v.groupStride + (row % v.heads) * v.headStride;
}
