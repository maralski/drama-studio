kernel void residual_add(device float* x [[buffer(0)]],
                         device const float* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    x[i] += y[i];
}

// Probe: read bf16 through a uint64 address table and through an offset buffer.
kernel void probe_addr(device const ulong* table [[buffer(0)]],
                       device const ushort* viaOffset [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    device const ushort* p = (device const ushort*)(table[0]);
    out[i] = bf16_to_float(p[i]);
    out[i + 4] = bf16_to_float(viaOffset[i]);
}
