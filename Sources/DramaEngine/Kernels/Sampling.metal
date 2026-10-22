// Top-k / top-p sampling on the GPU, reproducing the CPU `Sampler` bit for bit
// in its decisions: the CPU works in double, Metal has no double, so the
// arithmetic here runs in double-float (an unevaluated hi + lo pair of fp32,
// about 48 bits of precision). The sort is replaced by extracting the top
// elements one at a time, since at most topK are ever kept.

struct df { float hi; float lo; };

inline df df_make(float h, float l) { df r; r.hi = h; r.lo = l; return r; }

inline df df_two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return df_make(s, (a - (s - bb)) + (b - bb));
}

inline df df_quick_two_sum(float a, float b) {
    float s = a + b;
    return df_make(s, b - (s - a));
}

inline df df_two_prod(float a, float b) {
    float p = a * b;
    return df_make(p, fma(a, b, -p));
}

inline df df_add(df a, df b) {
    df s = df_two_sum(a.hi, b.hi);
    df t = df_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = df_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return df_quick_two_sum(s.hi, s.lo);
}

inline df df_neg(df a) { return df_make(-a.hi, -a.lo); }
inline df df_sub(df a, df b) { return df_add(a, df_neg(b)); }

inline df df_mul(df a, df b) {
    df p = df_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return df_quick_two_sum(p.hi, p.lo);
}

inline df df_mulf(df a, float b) {
    df p = df_two_prod(a.hi, b);
    p.lo += a.lo * b;
    return df_quick_two_sum(p.hi, p.lo);
}

inline df df_div(df a, df b) {
    if (b.hi == 0.0f) return df_make(a.hi / b.hi, 0.0f);
    float q1 = a.hi / b.hi;
    df r = df_sub(a, df_mulf(b, q1));
    float q2 = r.hi / b.hi;
    r = df_sub(r, df_mulf(b, q2));
    float q3 = r.hi / b.hi;
    df q = df_quick_two_sum(q1, q2);
    return df_add(q, df_make(q3, 0.0f));
}

inline bool df_gt(df a, df b) { return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo); }

// ln 2 in four pieces: the first two have few enough bits that k * piece is
// exact for |k| < 512, the last two carry the rest to about 2^-86.
constant float DF_LN2_HI = 0.693145751953125f;
constant float DF_LN2_MID = 1.4285906217992306e-06f;
constant float DF_LN2_LO = 1.619850954759361e-11f;
constant float DF_LN2_LO2 = 6.390629391522767e-19f;
constant float DF_INV_LN2 = 1.4426950216293335f;

// 1 / j! for j = 0...13 as double-float.
constant float DF_EXP_C_HI[14] = {
    1.0f, 1.0f, 0.5f, 0.1666666716337204f, 0.0416666679084301f, 0.008333333767950535f,
    0.0013888889225199819f, 0.00019841270113829523f, 2.4801587642286904e-05f, 2.7557318844628753e-06f,
    2.755731998149713e-07f, 2.5052107943679403e-08f, 2.0876755879584152e-09f, 1.6059044372074283e-10f};
constant float DF_EXP_C_LO[14] = {
    0.0f, 0.0f, 0.0f, -4.967053879312289e-09f, -1.2417634698280722e-09f, -4.34617203337595e-10f,
    -3.3631094437103215e-11f, -2.725596874933456e-12f, -3.40699609366682e-13f, 3.793571224297229e-14f,
    -7.575112209051195e-15f, 4.4176230446483665e-16f, 1.1082839147459852e-16f, -5.352526511562726e-18f};

// Taylor series of exp(r) - 1 for |r| <= 0.35; the truncation error is below 1e-17.
inline df df_expm1_reduced(df r) {
    df e = df_make(DF_EXP_C_HI[13], DF_EXP_C_LO[13]);
    for (int j = 12; j >= 1; j--) e = df_add(df_mul(e, r), df_make(DF_EXP_C_HI[j], DF_EXP_C_LO[j]));
    return df_mul(e, r);
}

// exp(x) for x <= 88. Results below 2^-126 flush to zero, which is harmless
// for the sampler: such terms cannot move a sum of magnitude 1 or win an argmax.
inline df df_exp(df x) {
    if (x.hi < -87.0f) return df_make(0.0f, 0.0f);
    float k = rint(x.hi * DF_INV_LN2);
    float t = x.hi - k * DF_LN2_HI;
    df r = df_two_sum(t, -k * DF_LN2_MID);
    df kl = df_two_prod(k, DF_LN2_LO);
    kl.lo += k * DF_LN2_LO2;
    r = df_sub(r, kl);
    r = df_add(r, df_make(x.lo, 0.0f));
    df e = df_add(df_expm1_reduced(r), df_make(1.0f, 0.0f));
    int ki = int(k);
    return df_make(ldexp(e.hi, ki), ldexp(e.lo, ki));
}

// log(m * 2^-53) for m in [1, 2^53], matching the CPU's -log(uniformOpenZero())
// to about 1e-13 relative. m has 53 bits and a double-float holds 48, so the
// lowest bits enter as a first-order correction eps / u; near u = 1 the
// computation goes through log1p so the result keeps relative precision.
inline df df_log_unit(ulong m) {
    long dm = long(m) - (long(1) << 53);
    if (dm > -(long(1) << 50)) {
        ulong a = ulong(-dm);
        ulong a1 = a >> 2;
        float a1hi = float(a1);
        float a1lo = float(long(a1) - long(a1hi));
        df d1 = df_make(-ldexp(a1hi, -51), -ldexp(a1lo, -51));
        float eps = -ldexp(float(uint(a & 3)), -53);
        float d = d1.hi;
        float y0 = d * (1.0f - d * (0.5f - d * (1.0f / 3.0f - d * (0.25f - d * 0.2f))));
        df y = df_make(y0, 0.0f);
        for (int step = 0; step < 2; step++) {
            df em = df_expm1_reduced(y);
            df num = df_sub(d1, em);
            df den = df_add(df_make(1.0f, 0.0f), em);
            y = df_add(y, df_div(num, den));
        }
        return df_add(y, df_make(eps / (1.0f + d1.hi), 0.0f));
    }
    df u1;
    float eps;
    if (m < (ulong(1) << 48)) {
        float mh = float(m);
        float ml = float(long(m) - long(mh));
        u1 = df_make(ldexp(mh, -53), ldexp(ml, -53));
        eps = 0.0f;
    } else {
        ulong m1 = m >> 5;
        float mh = float(m1);
        float ml = float(long(m1) - long(mh));
        u1 = df_make(ldexp(mh, -48), ldexp(ml, -48));
        eps = ldexp(float(uint(m & 31)), -53);
    }
    df y = df_make(log(u1.hi), 0.0f);
    for (int step = 0; step < 2; step++) {
        df ex = df_exp(y);
        y = df_add(y, df_div(df_sub(u1, ex), ex));
    }
    return df_add(y, df_make(eps / u1.hi, 0.0f));
}

// xoshiro256**, the CPU SeededGenerator.
inline ulong rotl64(ulong x, uint k) { return (x << k) | (x >> (64 - k)); }

inline ulong xoshiro_next(thread ulong* s) {
    ulong result = rotl64(s[1] * 5, 7) * 9;
    ulong t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

// A sampler is one threadgroup. At 1024 threads it waits for a whole core to
// drain whenever another process uses the GPU, about 2 ms per draw instead of
// 40 us; 256 threads find room on a busy core.
#define SAMPLE_TG 256
#define SAMPLE_SG (SAMPLE_TG / 32)
#define SAMPLE_SLOTS 17
#define SAMPLE_MAX_KEPT 1024
static_assert(SAMPLE_TG * SAMPLE_SLOTS >= 4097 && SAMPLE_SLOTS <= 32, "every candidate needs a slot and a bit of `taken`");

struct SamplerParams {
    uint n;
    uint topK;
    float tHi, tLo;
    float topPHi, topPLo;
    uint rngIn, rngOut;
    uint frameRow;
    uint position;
    uint head;
    int eos;
    int semanticStart;
    uint codebookSize;
    float retryTHi, retryTLo;
    uint retryTopK;
    float retryTopPHi, retryTopPLo;
    uint windowFrames;
};

// Candidate record for the extraction: (value desc, index asc) ordering, carrying
// the element's exp(value - max) so the winner's term is at hand.
struct Cand {
    float val;
    uint idx;
    float ehi;
    float elo;
};

inline bool cand_better(Cand a, Cand b) { return a.val > b.val || (a.val == b.val && a.idx < b.idx); }

inline Cand cand_shuffle_xor(Cand c, uint mask) {
    Cand o;
    o.val = simd_shuffle_xor(c.val, mask);
    o.idx = simd_shuffle_xor(c.idx, mask);
    o.ehi = simd_shuffle_xor(c.ehi, mask);
    o.elo = simd_shuffle_xor(c.elo, mask);
    return o;
}

inline Cand cand_simd_best(Cand c) {
    for (uint mask = 16; mask >= 1; mask >>= 1) {
        Cand o = cand_shuffle_xor(c, mask);
        if (cand_better(o, c)) c = o;
    }
    return c;
}

inline df df_simd_sum(df v) {
    for (uint mask = 16; mask >= 1; mask >>= 1) {
        df o = df_make(simd_shuffle_xor(v.hi, mask), simd_shuffle_xor(v.lo, mask));
        v = df_add(v, o);
    }
    return v;
}

struct SamplerShared {
    Cand partials[32];
    df sums[32];
    float maxVal;
    df expSum;
    uint winner;
    uint stop;
    uint kept;
    df keptSum;
    uint pick;
    uint keptIdx[SAMPLE_MAX_KEPT];
    df keptE[SAMPLE_MAX_KEPT];
    ulong keptU[SAMPLE_MAX_KEPT];
};

// One draw of the CPU `Sampler.sample`: returns the index into `logits`. Every
// thread of the SAMPLE_TG-wide threadgroup must call it; `rng` is advanced on
// thread 0 only. `bad` is set when a logit is not finite.
inline uint sample_one(device const float* logits, uint n, df t, df topP, uint topK,
                       thread ulong* rng, threadgroup SamplerShared& sh,
                       uint tid, uint lane, uint sg, thread bool& bad) {
    float v[SAMPLE_SLOTS];
    df e[SAMPLE_SLOTS];
    float localMax = -INFINITY;
    for (uint s = 0; s < SAMPLE_SLOTS; s++) {
        uint i = tid + s * SAMPLE_TG;
        float x = i < n ? logits[i] : -INFINITY;
        if (i < n && !isfinite(x)) { bad = true; x = -INFINITY; }
        v[s] = x;
        localMax = max(localMax, x);
    }
    localMax = simd_max(localMax);
    if (lane == 0) sh.partials[sg].val = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m = simd_max(lane < SAMPLE_SG ? sh.partials[lane].val : -INFINITY);
        if (lane == 0) sh.maxVal = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxVal = sh.maxVal;

    df local = df_make(0.0f, 0.0f);
    for (uint s = 0; s < SAMPLE_SLOTS; s++) {
        uint i = tid + s * SAMPLE_TG;
        if (i < n && v[s] > -INFINITY) {
            e[s] = df_exp(df_div(df_two_sum(v[s], -maxVal), t));
        } else {
            e[s] = df_make(0.0f, 0.0f);
        }
        local = df_add(local, e[s]);
    }
    local = df_simd_sum(local);
    if (lane == 0) sh.sums[sg] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        df total = df_simd_sum(lane < SAMPLE_SG ? sh.sums[lane] : df_make(0.0f, 0.0f));
        if (lane == 0) {
            sh.expSum = total;
            sh.kept = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    df expSum = sh.expSum;

    uint taken = 0;
    df cumulative = df_make(0.0f, 0.0f);
    df keptSum = df_make(0.0f, 0.0f);
    uint limit = min(clamp(topK, 1u, uint(SAMPLE_MAX_KEPT)), n);
    for (uint i = 0; i < limit; i++) {
        Cand best;
        best.val = -INFINITY; best.idx = 0xffffffffu; best.ehi = 0.0f; best.elo = 0.0f;
        for (uint s = 0; s < SAMPLE_SLOTS; s++) {
            uint idx = tid + s * SAMPLE_TG;
            if (idx < n && !(taken & (1u << s))) {
                Cand c;
                c.val = v[s]; c.idx = idx; c.ehi = e[s].hi; c.elo = e[s].lo;
                if (cand_better(c, best)) best = c;
            }
        }
        best = cand_simd_best(best);
        if (lane == 0) sh.partials[sg] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            Cand none;
            none.val = -INFINITY; none.idx = 0xffffffffu; none.ehi = 0.0f; none.elo = 0.0f;
            Cand w = cand_simd_best(lane < SAMPLE_SG ? sh.partials[lane] : none);
            if (lane == 0) {
                df term = df_make(w.ehi, w.elo);
                cumulative = df_add(cumulative, df_div(term, expSum));
                bool dropped = i > 0 && df_gt(cumulative, topP);
                if (!dropped) {
                    sh.keptIdx[i] = w.idx;
                    sh.keptE[i] = term;
                    keptSum = df_add(keptSum, term);
                    sh.kept = i + 1;
                }
                sh.winner = w.idx;
                sh.stop = dropped ? 1u : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sh.stop != 0u) break;
        uint w = sh.winner;
        if (w % SAMPLE_TG == tid) taken |= 1u << (w / SAMPLE_TG);
    }
    uint kept = sh.kept;

    if (tid == 0) {
        sh.keptSum = keptSum;
        for (uint j = 0; j < kept; j++) sh.keptU[j] = (xoshiro_next(rng) >> 11) + 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < kept; j += SAMPLE_TG) {
        ulong m = sh.keptU[j];
        df exponential = df_neg(df_log_unit(m));
        df p = df_div(sh.keptE[j], sh.keptSum);
        df score = df_div(p, exponential);
        sh.keptE[j] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint best = 0;
        df bestScore = df_make(-INFINITY, 0.0f);
        for (uint j = 0; j < kept; j++) {
            df s = sh.keptE[j];
            if (df_gt(s, bestScore)) { bestScore = s; best = j; }
        }
        sh.pick = sh.keptIdx[best];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pick = sh.pick;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return pick;
}

inline void load_rng(device const ulong* rng, uint slot, thread ulong* s) {
    for (uint i = 0; i < 4; i++) s[i] = rng[slot * 4 + i];
}

inline void store_rng(device ulong* rng, uint slot, thread ulong* s) {
    for (uint i = 0; i < 4; i++) rng[slot * 4 + i] = s[i];
}

// Head `head` of the tower: samples code head + 1 from its 4096 logits and
// writes it where the next residual layer and the backbone read it. After an
// eos main token the CPU never runs the tower, so no draw is taken and the
// code is 0.
[[max_total_threads_per_threadgroup(SAMPLE_TG)]]
kernel void sample_tower(device const float* logits [[buffer(0)]],
                         device ulong* rng [[buffer(1)]],
                         device int* frames [[buffer(2)]],
                         device int* codeOut [[buffer(3)]],
                         device uint* status [[buffer(4)]],
                         constant SamplerParams& p [[buffer(5)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    if (frames[p.frameRow * 11] == p.eos) {
        if (tid == 0) {
            codeOut[0] = 0;
            frames[p.frameRow * 11 + 2 + p.head] = 0;
        }
        return;
    }
    threadgroup SamplerShared sh;
    ulong s[4];
    load_rng(rng, p.rngIn, s);
    bool bad = false;
    uint pick = sample_one(logits, p.n, df_make(p.tHi, p.tLo), df_make(p.topPHi, p.topPLo), p.topK, s, sh, tid, lane, sg, bad);
    if (tid == 0) {
        store_rng(rng, p.rngOut, s);
        int code = bad ? 0 : int(pick);
        codeOut[0] = code;
        frames[p.frameRow * 11 + 2 + p.head] = code;
        if (bad) status[p.rngOut] |= 1u;
    }
}

// The main token: two draws over the 4097 candidates (eos, then the semantic
// band), the second with the retry settings; the retry wins when the first pick
// is semantic and repeats a main token of the last `windowFrames` rows.
[[max_total_threads_per_threadgroup(SAMPLE_TG)]]
kernel void sample_main(device const float* logits [[buffer(0)]],
                        device ulong* rng [[buffer(1)]],
                        device int* frames [[buffer(2)]],
                        device int* code0 [[buffer(3)]],
                        device uint* status [[buffer(4)]],
                        constant SamplerParams& p [[buffer(5)]],
                        device int* semantic [[buffer(6)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup SamplerShared sh;
    ulong s[4];
    load_rng(rng, p.rngIn, s);
    bool bad = false;
    uint first = sample_one(logits, p.n, df_make(p.tHi, p.tLo), df_make(p.topPHi, p.topPLo), p.topK, s, sh, tid, lane, sg, bad);
    uint retry = sample_one(logits, p.n, df_make(p.retryTHi, p.retryTLo), df_make(p.retryTopPHi, p.retryTopPLo), p.retryTopK, s, sh, tid, lane, sg, bad);
    if (tid == 0) {
        store_rng(rng, p.rngOut, s);
        int firstId = first == 0 ? p.eos : p.semanticStart + int(first) - 1;
        int retryId = retry == 0 ? p.eos : p.semanticStart + int(retry) - 1;
        bool repeated = false;
        if (first != 0) {
            uint from = p.position + 1 >= p.windowFrames ? p.position + 1 - p.windowFrames : 0;
            for (uint r = from; r <= p.position; r++) repeated = repeated || frames[r * 11] == firstId;
        }
        int mainId = bad ? p.eos : (repeated ? retryId : firstId);
        int c0 = clamp(mainId - p.semanticStart, 0, int(p.codebookSize) - 1);
        frames[p.frameRow * 11] = mainId;
        frames[p.frameRow * 11 + 1] = c0;
        code0[0] = c0;
        semantic[0] = mainId >= p.semanticStart && mainId < p.semanticStart + int(p.codebookSize) ? 1 : 0;
        status[p.rngOut] = bad ? 1u : 0u;
    }
}
