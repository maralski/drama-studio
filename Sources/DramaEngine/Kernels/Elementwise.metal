kernel void residual_add(device float* x [[buffer(0)]],
                         device const float* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    x[i] += y[i];
}

kernel void zero_u32(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 0u; }

kernel void round_bf16_inplace(device float* x [[buffer(0)]],
                               uint i [[thread_position_in_grid]]) {
    x[i] = round_to_bf16(x[i]);
}

kernel void silu_mul(device float* out [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = silu_f(a[i]) * b[i];
}

// Backbone input: text rows take the token embedding; audio rows take
// (token embedding + codebook-1 embedding of c1) / sqrt(2).
struct EmbedMixParams {
    uint dim;
    uint codebookSize;
    int semanticStart;
    int semanticEnd;
};

kernel void embed_mix(device float* out [[buffer(0)]],
                      device const int* frames [[buffer(1)]],
                      device const ushort* embed [[buffer(2)]],
                      device const ushort* codebooks [[buffer(3)]],
                      constant EmbedMixParams& p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint m = gid.y;
    if (d >= p.dim) return;
    int mainId = frames[m * 11];
    float text = bf16_to_float(embed[uint(mainId) * p.dim + d]);
    bool semantic = mainId >= p.semanticStart && mainId <= p.semanticEnd;
    if (semantic) {
        uint row = p.codebookSize + uint(frames[m * 11 + 2]);
        float extra = bf16_to_float(codebooks[row * p.dim + d]);
        out[m * p.dim + d] = (text + extra) * 0.70710678118654752f;
    } else {
        out[m * p.dim + d] = text;
    }
}

// Tower layer input e: the layer's embedding of `codes` for semantic rows, else
// zero; the first shared layer also adds codebooks 2..9 of the current frame.
struct TowerEmbedParams {
    uint dim;
    uint codebookSize;
    uint numCodebooks;
    uint addUnseen;
    int semanticStart;
    int semanticEnd;
};

kernel void tower_embed(device float* out [[buffer(0)]],
                        device const int* codes [[buffer(1)]],
                        device const int* semanticNext [[buffer(2)]],
                        device const ushort* table [[buffer(3)]],
                        device const int* frames [[buffer(4)]],
                        device const ushort* codebooks [[buffer(5)]],
                        constant TowerEmbedParams& p [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint m = gid.y;
    if (d >= p.dim) return;
    float e = 0.0f;
    if (semanticNext[m] != 0) {
        int c = clamp(codes[m], 0, int(p.codebookSize) - 1);
        e = bf16_to_float(table[uint(c) * p.dim + d]);
    }
    if (p.addUnseen != 0) {
        int mainId = frames[m * 11];
        if (mainId >= p.semanticStart && mainId <= p.semanticEnd) {
            for (uint k = 2; k < p.numCodebooks; k++) {
                uint row = k * p.codebookSize + uint(frames[m * 11 + 1 + k]);
                e += bf16_to_float(codebooks[row * p.dim + d]);
            }
        }
    }
    out[m * p.dim + d] = e;
}

// Zero-centered RMSNorm over rows of `dim`: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
// Input rows come through a HeadView; output rows are contiguous with outStride.
struct NormParams {
    uint dim;
    float eps;
    HeadView in;
    uint outStride;
};

kernel void rmsnorm_zc(device float* out [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device const ushort* w [[buffer(2)]],
                       constant NormParams& p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float* xr = x + head_index(p.in, row);
    float ss = 0.0f;
    for (uint i = tid; i < p.dim; i += 256) { float v = xr[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8; s++) total += partial[s];
    float inv = rsqrt(total / float(p.dim) + p.eps);
    device float* o = out + row * p.outStride;
    for (uint i = tid; i < p.dim; i += 256) {
        o[i] = xr[i] * inv * (1.0f + bf16_to_float(w[i]));
    }
}

// Gated DeltaNet output norm over 128-wide heads: w * rmsnorm(x) * silu(z).
// One simdgroup per row, four values per lane.
kernel void gated_rmsnorm(device float* out [[buffer(0)]],
                          device const float* x [[buffer(1)]],
                          device const float* z [[buffer(2)]],
                          device const ushort* w [[buffer(3)]],
                          constant float& eps [[buffer(4)]],
                          uint gid [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    uint base = row * 128 + lane * 4;
    float4 v = *(device const float4*)(x + base);
    float ss = simd_sum(dot(v, v));
    float inv = rsqrt(ss / 128.0f + eps);
    float4 g = *(device const float4*)(z + base);
    float4 ww = float4(bf16_to_float(w[lane * 4]), bf16_to_float(w[lane * 4 + 1]),
                       bf16_to_float(w[lane * 4 + 2]), bf16_to_float(w[lane * 4 + 3]));
    float4 r = ww * (v * inv);
    r *= float4(silu_f(g.x), silu_f(g.y), silu_f(g.z), silu_f(g.w));
    *(device float4*)(out + base) = r;
}

// out[row][i] = attn[row][i] * sigmoid(gate[view(row)][i]) for 256-wide heads.
kernel void sigmoid_gate_mul(device float* out [[buffer(0)]],
                             device const float* attn [[buffer(1)]],
                             device const float* gate [[buffer(2)]],
                             constant HeadView& view [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint row = gid.y;
    out[row * 256 + i] = attn[row * 256 + i] * sigmoid_f(gate[head_index(view, row) + i]);
}
