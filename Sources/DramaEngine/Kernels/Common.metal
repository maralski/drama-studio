#include <metal_stdlib>
using namespace metal;

// bf16 stored as ushort; widening is exact.
inline float bf16_to_float(ushort b) { return as_type<float>(uint(b) << 16); }
