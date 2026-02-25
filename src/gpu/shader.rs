//! Metal shader source for GPU-accelerated MSM.
//!
//! Contains all MSL kernels as a Rust string constant, compiled at runtime.

pub(crate) const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BN254 Fq field arithmetic — 8×32-bit limbs, Montgomery form
// ============================================================================

// Field modulus p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
constant uint FQ_MOD[8] = {
    0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
};

// -p^{-1} mod 2^32
constant uint FQ_INV = 0xe4866389u;

// R = 2^256 mod p (Montgomery "one")
constant uint FQ_ONE[8] = {
    0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u,
    0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u
};

// Typedef for clarity
typedef uint Fq[8];

// ---- Modular addition: c = a + b mod p ----
inline void fq_add(thread const uint* a, thread const uint* b, thread uint* c) {
    ulong carry = 0;
    uint sum[8];
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(a[i]) + ulong(b[i]) + carry;
        sum[i] = uint(s & 0xFFFFFFFFul);
        carry = s >> 32;
    }
    // Conditional subtraction of modulus
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(sum[i]) - ulong(FQ_MOD[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    // If carry >= borrow, result >= p, use diff; else use sum
    uint use_diff = uint((carry >= borrow) ? 1 : 0);
    for (int i = 0; i < 8; i++) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

// ---- Modular subtraction: c = a - b mod p ----
inline void fq_sub(thread const uint* a, thread const uint* b, thread uint* c) {
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(a[i]) - ulong(b[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    // If borrow, add modulus back
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(diff[i]) + (borrow ? ulong(FQ_MOD[i]) : 0ul) + carry;
        c[i] = uint(s & 0xFFFFFFFFul);
        carry = s >> 32;
    }
}

// ---- Modular negation: c = -a mod p (= p - a if a != 0, else 0) ----
inline void fq_neg(thread const uint* a, thread uint* c) {
    // Check if a == 0
    uint is_zero = 1;
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) { is_zero = 0; break; }
    }
    if (is_zero) {
        for (int i = 0; i < 8; i++) c[i] = 0;
        return;
    }
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(FQ_MOD[i]) - ulong(a[i]) - borrow;
        c[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
}

// ---- Montgomery multiplication: c = a * b * R^{-1} mod p (CIOS, 32-bit) ----
inline void fq_mul(thread const uint* a, thread const uint* b, thread uint* c) {
    // CIOS (Coarsely Integrated Operand Scanning)
    ulong t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        // Step 1: t += a[0..8] * b[i]
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong prod = ulong(a[j]) * ulong(b[i]) + t[j] + carry;
            t[j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[8] += carry;

        // Step 2: Montgomery reduction
        uint m = uint(t[0] & 0xFFFFFFFFul) * FQ_INV;
        carry = 0;
        ulong prod0 = ulong(m) * ulong(FQ_MOD[0]) + t[0];
        carry = prod0 >> 32;
        for (int j = 1; j < 8; j++) {
            ulong prod = ulong(m) * ulong(FQ_MOD[j]) + t[j] + carry;
            t[j - 1] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        ulong sum = t[8] + carry;
        t[7] = sum & 0xFFFFFFFFul;
        t[8] = sum >> 32;
    }

    // Final conditional subtraction
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = t[i] - ulong(FQ_MOD[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    // If no borrow (t >= p), use diff; else use t
    uint use_t = uint(borrow);
    for (int i = 0; i < 8; i++) {
        c[i] = use_t ? uint(t[i] & 0xFFFFFFFFul) : diff[i];
    }
}

// ---- Montgomery squaring: c = a^2 * R^{-1} mod p ----
inline void fq_sqr(thread const uint* a, thread uint* c) {
    fq_mul(a, a, c);
}

// ---- Double: c = a + a mod p ----
inline void fq_dbl(thread const uint* a, thread uint* c) {
    fq_add(a, a, c);
}

// ---- Check if field element is zero ----
inline bool fq_is_zero(thread const uint* a) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

// ---- Copy ----
inline void fq_copy(thread const uint* src, thread uint* dst) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

// ---- Load from device memory ----
inline void fq_load(device const uint* src, thread uint* dst) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

// ---- Store to device memory ----
inline void fq_store(thread const uint* src, device uint* dst) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

// ============================================================================
// Projective point operations — matches halo2curves' RCB2015 formulas
// ============================================================================

// Projective point: (X : Y : Z), identity = (0 : Y : 0) for any Y
// halo2curves identity: x=0, y=1_mont, z=0
// Affine point: (x, y), 16 u32s = 2 × Fq
// Curve: y² = x³ + b, where b = 3 for BN254 G1, so 3b = 9

// mul_by_3b(x) = 9*x = 8*x + x (matching halo2curves BN256 G1 specialization)
inline void fq_mul_by_3b(thread const uint* a, thread uint* c) {
    uint t1[8], t2[8], t3[8];
    fq_dbl(a, t1);      // 2x
    fq_dbl(t1, t2);     // 4x
    fq_dbl(t2, t3);     // 8x
    fq_add(t3, a, c);   // 9x
}

// ---- Mixed addition: R = P + Q, P=(X1:Y1:Z1) projective, Q=(x2,y2) affine ----
// Algorithm 8, https://eprint.iacr.org/2015/1060.pdf (a=0 case)
// This is a COMPLETE addition formula — handles all cases including
// doubling, identity, and P+(-P) without branches.
// IMPORTANT: When Q is identity (affine 0,0), caller must handle separately
// since this formula assumes Q is a valid non-identity affine point.
inline void projective_madd(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,  // Projective input P
    thread const uint* x2, thread const uint* y2,                          // Affine input Q
    thread uint* x3, thread uint* y3, thread uint* z3                      // Projective output
) {
    uint t0[8], t1[8], t2[8], t3[8], t4[8];
    uint tmp[8], tmp2[8];

    // Algorithm 8 (a=0), eprint.iacr.org/2015/1060.pdf:
    // t0 = X1 * x2
    fq_mul(x1, x2, t0);
    // t1 = Y1 * y2
    fq_mul(y1, y2, t1);
    // t3 = x2 + y2
    fq_add(x2, y2, t3);
    // t4 = X1 + Y1
    fq_add(x1, y1, t4);
    // t3 = t3 * t4
    fq_mul(t3, t4, tmp);
    fq_copy(tmp, t3);
    // t4 = t0 + t1
    fq_add(t0, t1, t4);
    // t3 = t3 - t4
    fq_sub(t3, t4, tmp);
    fq_copy(tmp, t3);
    // t4 = y2 * Z1
    fq_mul(y2, z1, t4);
    // t4 = t4 + Y1
    fq_add(t4, y1, tmp);
    fq_copy(tmp, t4);
    // y3 = x2 * Z1
    fq_mul(x2, z1, y3);
    // y3 = y3 + X1
    fq_add(y3, x1, tmp);
    fq_copy(tmp, y3);
    // x3 = t0 + t0
    fq_dbl(t0, x3);
    // t0 = x3 + t0
    fq_add(x3, t0, tmp);
    fq_copy(tmp, t0);
    // t2 = mul_by_3b(Z1)
    fq_mul_by_3b(z1, t2);
    // z3 = t1 + t2
    fq_add(t1, t2, z3);
    // t1 = t1 - t2
    fq_sub(t1, t2, tmp);
    fq_copy(tmp, t1);
    // y3 = mul_by_3b(y3)
    fq_mul_by_3b(y3, tmp);
    fq_copy(tmp, y3);
    // x3 = t4 * y3
    fq_mul(t4, y3, x3);
    // t2 = t3 * t1
    fq_mul(t3, t1, t2);
    // x3 = t2 - x3
    fq_sub(t2, x3, tmp);
    fq_copy(tmp, x3);
    // y3 = y3 * t0
    fq_mul(y3, t0, tmp);
    fq_copy(tmp, y3);
    // t1 = t1 * z3
    fq_mul(t1, z3, tmp);
    fq_copy(tmp, t1);
    // y3 = t1 + y3
    fq_add(t1, y3, tmp);
    fq_copy(tmp, y3);
    // t0 = t0 * t3
    fq_mul(t0, t3, tmp);
    fq_copy(tmp, t0);
    // z3 = z3 * t4
    fq_mul(z3, t4, tmp);
    fq_copy(tmp, z3);
    // z3 = z3 + t0
    fq_add(z3, t0, tmp);
    fq_copy(tmp, z3);
}

// ---- Full projective addition: R = P + Q, both projective ----
// Algorithm 1, https://eprint.iacr.org/2015/1060.pdf (a=0 case)
// Complete addition formula — handles all cases without branches.
inline void projective_add(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,
    thread const uint* x2, thread const uint* y2, thread const uint* z2,
    thread uint* x3, thread uint* y3, thread uint* z3
) {
    uint t0[8], t1[8], t2[8], t3[8], t4[8];
    uint tmp[8];

    // t0 = X1 * X2
    fq_mul(x1, x2, t0);
    // t1 = Y1 * Y2
    fq_mul(y1, y2, t1);
    // t2 = Z1 * Z2
    fq_mul(z1, z2, t2);
    // t3 = X1 + Y1
    fq_add(x1, y1, t3);
    // t4 = X2 + Y2
    fq_add(x2, y2, t4);
    // t3 = t3 * t4
    fq_mul(t3, t4, tmp);
    fq_copy(tmp, t3);
    // t4 = t0 + t1
    fq_add(t0, t1, t4);
    // t3 = t3 - t4  (= X1*Y2 + X2*Y1)
    fq_sub(t3, t4, tmp);
    fq_copy(tmp, t3);
    // t4 = Y1 + Z1
    fq_add(y1, z1, t4);
    // x3 = Y2 + Z2  (temporary)
    fq_add(y2, z2, x3);
    // t4 = t4 * x3
    fq_mul(t4, x3, tmp);
    fq_copy(tmp, t4);
    // x3 = t1 + t2
    fq_add(t1, t2, x3);
    // t4 = t4 - x3  (= Y1*Z2 + Y2*Z1)
    fq_sub(t4, x3, tmp);
    fq_copy(tmp, t4);
    // x3 = X1 + Z1
    fq_add(x1, z1, x3);
    // y3 = X2 + Z2
    fq_add(x2, z2, y3);
    // x3 = x3 * y3
    fq_mul(x3, y3, tmp);
    fq_copy(tmp, x3);
    // y3 = t0 + t2
    fq_add(t0, t2, y3);
    // y3 = x3 - y3  (= X1*Z2 + X2*Z1)
    fq_sub(x3, y3, tmp);
    fq_copy(tmp, y3);
    // x3 = t0 + t0
    fq_dbl(t0, x3);
    // t0 = x3 + t0  (= 3*t0)
    fq_add(x3, t0, tmp);
    fq_copy(tmp, t0);
    // t2 = mul_by_3b(t2)  (= 3*b*t2 = 9*t2)
    fq_mul_by_3b(t2, tmp);
    fq_copy(tmp, t2);
    // z3 = t1 + t2
    fq_add(t1, t2, z3);
    // t1 = t1 - t2
    fq_sub(t1, t2, tmp);
    fq_copy(tmp, t1);
    // y3 = mul_by_3b(y3)
    fq_mul_by_3b(y3, tmp);
    fq_copy(tmp, y3);
    // x3 = t4 * y3
    fq_mul(t4, y3, x3);
    // t2 = t3 * t1
    fq_mul(t3, t1, t2);
    // x3 = t2 - x3
    fq_sub(t2, x3, tmp);
    fq_copy(tmp, x3);
    // y3 = y3 * t0
    fq_mul(y3, t0, tmp);
    fq_copy(tmp, y3);
    // t1 = t1 * z3
    fq_mul(t1, z3, tmp);
    fq_copy(tmp, t1);
    // y3 = t1 + y3
    fq_add(t1, y3, tmp);
    fq_copy(tmp, y3);
    // t0 = t0 * t3
    fq_mul(t0, t3, tmp);
    fq_copy(tmp, t0);
    // z3 = z3 * t4
    fq_mul(z3, t4, tmp);
    fq_copy(tmp, z3);
    // z3 = z3 + t0
    fq_add(z3, t0, tmp);
    fq_copy(tmp, z3);
}

// ---- Point doubling: R = 2*P (projective, a=0) ----
// Algorithm 9, https://eprint.iacr.org/2015/1060.pdf
inline void projective_dbl(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,
    thread uint* x3, thread uint* y3, thread uint* z3
) {
    uint t0[8], t1[8], t2[8];
    uint tmp[8];

    // t0 = Y1^2
    fq_sqr(y1, t0);
    // z3 = t0 + t0
    fq_dbl(t0, z3);
    // z3 = z3 + z3
    fq_dbl(z3, tmp);
    fq_copy(tmp, z3);
    // z3 = z3 + z3  (= 8*Y1^2)
    fq_dbl(z3, tmp);
    fq_copy(tmp, z3);
    // t1 = Y1 * Z1
    fq_mul(y1, z1, t1);
    // t2 = Z1^2
    fq_sqr(z1, t2);
    // t2 = mul_by_3b(t2)  (= 9*Z1^2)
    fq_mul_by_3b(t2, tmp);
    fq_copy(tmp, t2);
    // x3 = t2 * z3  (= 9*Z1^2 * 8*Y1^2)
    fq_mul(t2, z3, x3);
    // y3 = t0 + t2  (= Y1^2 + 9*Z1^2)
    fq_add(t0, t2, y3);
    // z3 = t1 * z3  (= Y1*Z1 * 8*Y1^2)
    fq_mul(t1, z3, tmp);
    fq_copy(tmp, z3);
    // t1 = t2 + t2  (= 2*9*Z1^2)
    fq_dbl(t2, t1);
    // t2 = t1 + t2  (= 3*9*Z1^2 = 27*Z1^2)
    fq_add(t1, t2, tmp);
    fq_copy(tmp, t2);
    // t0 = t0 - t2  (= Y1^2 - 27*Z1^2)
    fq_sub(t0, t2, tmp);
    fq_copy(tmp, t0);
    // y3 = t0 * y3  (= (Y1^2-27Z1^2)*(Y1^2+9Z1^2))
    fq_mul(t0, y3, tmp);
    fq_copy(tmp, y3);
    // y3 = x3 + y3
    fq_add(x3, y3, tmp);
    fq_copy(tmp, y3);
    // t1 = X1 * Y1
    fq_mul(x1, y1, t1);
    // x3 = t0 * t1  (= (Y1^2-27Z1^2)*X1*Y1)
    fq_mul(t0, t1, x3);
    // x3 = x3 + x3  (= 2*(Y1^2-27Z1^2)*X1*Y1)
    fq_dbl(x3, tmp);
    fq_copy(tmp, x3);
}

// ---- Double-and-add: R = scalar * P (for small scalars, used in PBPR) ----
inline void double_and_add(
    thread const uint* px, thread const uint* py, thread const uint* pz,
    uint scalar,
    thread uint* rx, thread uint* ry, thread uint* rz
) {
    // Start with identity
    for (int i = 0; i < 8; i++) rx[i] = 0;
    for (int i = 0; i < 8; i++) ry[i] = FQ_ONE[i];
    for (int i = 0; i < 8; i++) rz[i] = 0;

    if (scalar == 0) return;

    uint tx[8], ty[8], tz[8];  // temp = current power of P
    fq_copy(px, tx);
    fq_copy(py, ty);
    fq_copy(pz, tz);

    uint s = scalar;
    while (s != 0) {
        if (s & 1) {
            uint ox[8], oy[8], oz[8];
            projective_add(rx, ry, rz, tx, ty, tz, ox, oy, oz);
            fq_copy(ox, rx);
            fq_copy(oy, ry);
            fq_copy(oz, rz);
        }
        uint dx[8], dy[8], dz[8];
        projective_dbl(tx, ty, tz, dx, dy, dz);
        fq_copy(dx, tx);
        fq_copy(dy, ty);
        fq_copy(dz, tz);
        s >>= 1;
    }
}

// ============================================================================
// GPU Kernels
// ============================================================================

// ---- Test kernel: field operations ----
// For each thread i: compute c[i] = a[i] * b[i] (Montgomery mul)
kernel void test_field_ops(
    device const uint* a [[buffer(0)]],    // n × 8 u32s
    device const uint* b [[buffer(1)]],    // n × 8 u32s
    device uint* c       [[buffer(2)]],    // n × 8 u32s (output)
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 8;
    uint la[8], lb[8], lc[8];
    fq_load(a + off, la);
    fq_load(b + off, lb);
    fq_mul(la, lb, lc);
    fq_store(lc, c + off);
}

// ---- Test kernel: field add/sub/neg ----
// op: 0=add, 1=sub, 2=neg(a)
kernel void test_field_addsub(
    device const uint* a   [[buffer(0)]],   // n × 8 u32s
    device const uint* b   [[buffer(1)]],   // n × 8 u32s
    device uint* c         [[buffer(2)]],   // n × 8 u32s (output)
    device const uint* op  [[buffer(3)]],   // n × 1 u32
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 8;
    uint la[8], lb[8], lc[8];
    fq_load(a + off, la);
    fq_load(b + off, lb);
    uint operation = op[id];
    if (operation == 0) {
        fq_add(la, lb, lc);
    } else if (operation == 1) {
        fq_sub(la, lb, lc);
    } else {
        fq_neg(la, lc);
    }
    fq_store(lc, c + off);
}

// ---- Test kernel: Jacobian mixed-addition ----
// For each thread i: out[i] = jac[i] + aff[i]
// jac: n × 24 u32s (X,Y,Z), aff: n × 16 u32s (x,y), sign: n × u32
// out: n × 24 u32s (X,Y,Z)
kernel void test_jacobian_madd(
    device const uint* jac  [[buffer(0)]],   // n × 24 u32s
    device const uint* aff  [[buffer(1)]],   // n × 16 u32s
    device const uint* sign [[buffer(2)]],   // n × 1 u32 (1=add, 0=subtract)
    device uint* out        [[buffer(3)]],   // n × 24 u32s
    uint id [[thread_position_in_grid]]
) {
    uint joff = id * 24;
    uint aoff = id * 16;

    uint jx[8], jy[8], jz[8], ax[8], ay[8];
    fq_load(jac + joff, jx);
    fq_load(jac + joff + 8, jy);
    fq_load(jac + joff + 16, jz);
    fq_load(aff + aoff, ax);
    fq_load(aff + aoff + 8, ay);

    // Apply sign: if sign==0, negate y-coordinate of affine point
    if (sign[id] == 0) {
        fq_neg(ay, ay);
    }

    uint rx[8], ry[8], rz[8];
    projective_madd(jx, jy, jz, ax, ay, rx, ry, rz);

    fq_store(rx, out + joff);
    fq_store(ry, out + joff + 8);
    fq_store(rz, out + joff + 16);
}

// ---- Main kernel: bucket accumulation (all windows in one dispatch) ----
// 2D grid: thread.x = bucket index, thread.y = window index.
// Each thread owns one bucket in one window.
//
// Scatter entries are packed: each entry is a single u32 with
//   base_idx = packed & 0x7FFFFFFFu  (low 31 bits)
//   sign     = packed >> 31           (MSB: 1 = positive, 0 = negate y)
//
// window_params layout: per-window [scatter_start, offsets_start, buckets_start] × 3 u32s
//   scatter_start: index into scatter_entries where this window's data begins
//   offsets_start: index into bucket_offsets where this window's offsets begin
//   buckets_start: index into buckets_out where this window's buckets begin (in 24-u32 units)
kernel void bucket_accumulate_all(
    device const uint* bases           [[buffer(0)]],  // n × 16 u32s (affine x,y)
    device const uint* scatter_entries [[buffer(1)]],  // ALL windows concatenated, packed u32s
    device const uint* bucket_offsets  [[buffer(2)]],  // ALL windows concatenated, (num_buckets+1) × num_windows u32s
    device uint* buckets_out           [[buffer(3)]],  // ALL windows, num_windows × num_buckets × 24 u32s
    device const uint* window_params   [[buffer(4)]],  // num_windows × 3 u32s [scatter_start, offsets_start, buckets_start]
    uint2 tid [[thread_position_in_grid]]
) {
    uint bucket_idx = tid.x;
    uint window_idx = tid.y;

    uint scatter_start = window_params[window_idx * 3];
    uint offsets_start = window_params[window_idx * 3 + 1];
    uint buckets_start = window_params[window_idx * 3 + 2];

    uint start = bucket_offsets[offsets_start + bucket_idx];
    uint end   = bucket_offsets[offsets_start + bucket_idx + 1];

    uint boff = (buckets_start + bucket_idx) * 24;

    if (start == end) {
        // Empty bucket → identity (0, 1_mont, 0)
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + i] = 0;
        }
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + 8 + i] = FQ_ONE[i];
        }
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + 16 + i] = 0;
        }
        return;
    }

    // Initialize accumulator to identity
    uint accx[8] = {0,0,0,0,0,0,0,0};
    uint accy[8] = {FQ_ONE[0],FQ_ONE[1],FQ_ONE[2],FQ_ONE[3],
                     FQ_ONE[4],FQ_ONE[5],FQ_ONE[6],FQ_ONE[7]};
    uint accz[8] = {0,0,0,0,0,0,0,0};

    for (uint e = scatter_start + start; e < scatter_start + end; e++) {
        uint packed   = scatter_entries[e];
        uint base_idx = packed & 0x7FFFFFFFu;
        uint s        = packed >> 31;

        uint poff = base_idx * 16;
        uint ax[8], ay[8];
        fq_load(bases + poff, ax);
        fq_load(bases + poff + 8, ay);

        if (s == 0) {
            fq_neg(ay, ay);
        }

        uint rx[8], ry[8], rz[8];
        projective_madd(accx, accy, accz, ax, ay, rx, ry, rz);
        fq_copy(rx, accx);
        fq_copy(ry, accy);
        fq_copy(rz, accz);
    }

    fq_store(accx, buckets_out + boff);
    fq_store(accy, buckets_out + boff + 8);
    fq_store(accz, buckets_out + boff + 16);
}

// ---- Original single-window bucket accumulation (kept for tests) ----
kernel void bucket_accumulate(
    device const uint* bases           [[buffer(0)]],  // n × 16 u32s (affine x,y)
    device const uint* scatter_entries [[buffer(1)]],  // total packed u32s (base_idx|sign<<31)
    device const uint* bucket_offsets  [[buffer(2)]],  // (num_buckets+1) u32s
    device uint* buckets_out           [[buffer(3)]],  // num_buckets × 24 u32s (Jac x,y,z)
    uint tid [[thread_position_in_grid]]
) {
    uint start = bucket_offsets[tid];
    uint end   = bucket_offsets[tid + 1];

    if (start == end) {
        // Empty bucket → identity (0, 1_mont, 0)
        uint boff = tid * 24;
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + i] = 0;           // x = 0
        }
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + 8 + i] = FQ_ONE[i]; // y = 1_mont
        }
        for (int i = 0; i < 8; i++) {
            buckets_out[boff + 16 + i] = 0;      // z = 0
        }
        return;
    }

    // Initialize accumulator to identity: (0, 1_mont, 0)
    // halo2curves identity = (x=0, y=Fq::ONE, z=0)
    uint accx[8] = {0,0,0,0,0,0,0,0};
    uint accy[8] = {FQ_ONE[0],FQ_ONE[1],FQ_ONE[2],FQ_ONE[3],
                     FQ_ONE[4],FQ_ONE[5],FQ_ONE[6],FQ_ONE[7]};
    uint accz[8] = {0,0,0,0,0,0,0,0};

    for (uint e = start; e < end; e++) {
        uint packed   = scatter_entries[e];
        uint base_idx = packed & 0x7FFFFFFFu;
        uint s        = packed >> 31;

        // Load affine point
        uint poff = base_idx * 16;
        uint ax[8], ay[8];
        fq_load(bases + poff, ax);
        fq_load(bases + poff + 8, ay);

        // Apply sign
        if (s == 0) {
            fq_neg(ay, ay);
        }

        // Mixed-add into accumulator
        uint rx[8], ry[8], rz[8];
        projective_madd(accx, accy, accz, ax, ay, rx, ry, rz);
        fq_copy(rx, accx);
        fq_copy(ry, accy);
        fq_copy(rz, accz);
    }

    // Write result
    uint boff = tid * 24;
    fq_store(accx, buckets_out + boff);
    fq_store(accy, buckets_out + boff + 8);
    fq_store(accz, buckets_out + boff + 16);
}

// ============================================================================
// PBPR: Parallel Bucket Point Reduction (two-stage)
// Adapted from cuZK paper (ePrint 2022/1321) and zkmopro/gpu-acceleration.
//
// Standard summation-by-parts for buckets B[0..N-1] where B[b] has weight (b+1):
//   result = Σ_{b=0}^{N-1} (b+1)·B[b]
//          = B[N-1] + (B[N-1]+B[N-2]) + ... + (B[N-1]+...+B[0])
// This is computed by iterating high→low: running_sum += B[i], total += running_sum.
//
// PBPR splits buckets into T segments processed in parallel:
//   Thread t handles segment [seg_start .. seg_start - bpt + 1] (high→low)
//   where seg_start = (N-1) - t*bpt.
//
//   Stage 1: Each thread computes local line_sum (m) and triangle_sum (g).
//   Stage 2: Correct g by adding m * position_offset via double-and-add,
//            where offset = (T-t-1)*bpt accounts for lower segments.
//
// After both stages, the T partial g-points are summed on CPU (tiny: ~256 points).
// ============================================================================

// ---- PBPR Stage 1: Partial bucket reduction ----
// Thread t processes bpt buckets starting from seg_start = (N-1)-t*bpt going DOWN.
// Computes: m = sum of all buckets in segment (line sum)
//           g = triangle sum = Σ_{i=0}^{bpt-1} partial_sum_i
//
// params: [num_buckets, num_threads]
kernel void bucket_reduce_stage1(
    device uint* buckets       [[buffer(0)]],   // num_buckets × 24 u32s (read/write)
    device uint* g_points      [[buffer(1)]],   // num_threads × 24 u32s (output)
    device const uint* params  [[buffer(2)]],   // [num_buckets, num_threads]
    uint tid [[thread_position_in_grid]]
) {
    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (tid >= num_threads) return;

    // Thread t processes the segment starting at (N-1 - t*bpt) going down.
    // Thread 0 → highest buckets, Thread T-1 → lowest buckets.
    uint seg_start = num_buckets - 1 - tid * bpt;

    // Load highest bucket in segment as initial m and g
    uint mx[8], my[8], mz[8];
    uint gx[8], gy[8], gz[8];
    {
        uint boff = seg_start * 24;
        fq_load(buckets + boff,      mx);
        fq_load(buckets + boff + 8,  my);
        fq_load(buckets + boff + 16, mz);
    }
    fq_copy(mx, gx);
    fq_copy(my, gy);
    fq_copy(mz, gz);

    // Iterate over remaining buckets in the segment (descending bucket index)
    for (uint i = 1; i < bpt; i++) {
        uint bi = seg_start - i;
        uint boff = bi * 24;
        uint bx[8], by[8], bz[8];
        fq_load(buckets + boff,      bx);
        fq_load(buckets + boff + 8,  by);
        fq_load(buckets + boff + 16, bz);

        // m += bucket[bi]  (running line sum)
        uint ox[8], oy[8], oz[8];
        projective_add(mx, my, mz, bx, by, bz, ox, oy, oz);
        fq_copy(ox, mx);
        fq_copy(oy, my);
        fq_copy(oz, mz);

        // g += m  (triangle sum accumulation)
        projective_add(gx, gy, gz, mx, my, mz, ox, oy, oz);
        fq_copy(ox, gx);
        fq_copy(oy, gy);
        fq_copy(oz, gz);
    }

    // Store m at seg_start position for stage 2 to read
    {
        uint boff = seg_start * 24;
        fq_store(mx, buckets + boff);
        fq_store(my, buckets + boff + 8);
        fq_store(mz, buckets + boff + 16);
    }

    // Store g to output g_points
    {
        uint goff = tid * 24;
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}

// ---- PBPR Stage 2: Position correction ----
// Thread t's local triangle sum assumed weights bpt, bpt-1, ..., 1.
// But the true weights are (seg_start+1), seg_start, ..., (seg_start-bpt+2).
// Difference = (seg_start+1) - bpt = N - (t+1)*bpt = (T-t-1)*bpt (constant per segment).
// So: g_corrected = g + (T-t-1)*bpt * m
//
// params: [num_buckets, num_threads]
kernel void bucket_reduce_stage2(
    device const uint* buckets [[buffer(0)]],   // num_buckets × 24 u32s (read only: m from stage 1)
    device uint* g_points      [[buffer(1)]],   // num_threads × 24 u32s (read/write)
    device const uint* params  [[buffer(2)]],   // [num_buckets, num_threads]
    uint tid [[thread_position_in_grid]]
) {
    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (tid >= num_threads) return;

    // Load m from the segment's start bucket (written by stage 1)
    uint seg_start = num_buckets - 1 - tid * bpt;
    uint mx[8], my[8], mz[8];
    {
        uint boff = seg_start * 24;
        fq_load(buckets + boff,      mx);
        fq_load(buckets + boff + 8,  my);
        fq_load(buckets + boff + 16, mz);
    }

    // Load g from g_points
    uint gx[8], gy[8], gz[8];
    {
        uint goff = tid * 24;
        fq_load(g_points + goff,      gx);
        fq_load(g_points + goff + 8,  gy);
        fq_load(g_points + goff + 16, gz);
    }

    // Correction scalar = (T - t - 1) * bpt
    // Thread 0 (highest segment) gets the largest correction
    // Thread T-1 (lowest segment) gets correction 0
    uint scalar = bpt * (num_threads - tid - 1);
    if (scalar > 0) {
        // g += double_and_add(m, scalar)
        uint dax[8], day[8], daz[8];
        double_and_add(mx, my, mz, scalar, dax, day, daz);

        uint ox[8], oy[8], oz[8];
        projective_add(gx, gy, gz, dax, day, daz, ox, oy, oz);
        fq_copy(ox, gx);
        fq_copy(oy, gy);
        fq_copy(oz, gz);
    }

    // Write corrected g back
    {
        uint goff = tid * 24;
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}

// ============================================================================
// Multi-window PBPR: All windows reduced in one dispatch
// 2D grid: (num_reduce_threads, num_windows)
// params: [num_buckets_per_window, num_reduce_threads]
// ============================================================================

kernel void bucket_reduce_stage1_all(
    device uint* buckets       [[buffer(0)]],   // ALL windows: num_windows × num_buckets × 24 u32s
    device uint* g_points      [[buffer(1)]],   // ALL windows: num_windows × num_threads × 24 u32s
    device const uint* params  [[buffer(2)]],   // [num_buckets_per_window, num_reduce_threads]
    uint2 tid [[thread_position_in_grid]]
) {
    uint thread_idx = tid.x;
    uint window_idx = tid.y;

    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (thread_idx >= num_threads) return;

    // Offset into this window's buckets
    uint window_bucket_offset = window_idx * num_buckets;
    uint seg_start = window_bucket_offset + num_buckets - 1 - thread_idx * bpt;

    // Load highest bucket in segment
    uint mx[8], my[8], mz[8];
    uint gx[8], gy[8], gz[8];
    {
        uint boff = seg_start * 24;
        fq_load(buckets + boff,      mx);
        fq_load(buckets + boff + 8,  my);
        fq_load(buckets + boff + 16, mz);
    }
    fq_copy(mx, gx);
    fq_copy(my, gy);
    fq_copy(mz, gz);

    for (uint i = 1; i < bpt; i++) {
        uint bi = seg_start - i;
        uint boff = bi * 24;
        uint bx[8], by[8], bz[8];
        fq_load(buckets + boff,      bx);
        fq_load(buckets + boff + 8,  by);
        fq_load(buckets + boff + 16, bz);

        uint ox[8], oy[8], oz[8];
        projective_add(mx, my, mz, bx, by, bz, ox, oy, oz);
        fq_copy(ox, mx);
        fq_copy(oy, my);
        fq_copy(oz, mz);

        projective_add(gx, gy, gz, mx, my, mz, ox, oy, oz);
        fq_copy(ox, gx);
        fq_copy(oy, gy);
        fq_copy(oz, gz);
    }

    // Store m at seg_start for stage 2
    {
        uint boff = seg_start * 24;
        fq_store(mx, buckets + boff);
        fq_store(my, buckets + boff + 8);
        fq_store(mz, buckets + boff + 16);
    }

    // Store g to output g_points (offset by window)
    {
        uint goff = (window_idx * num_threads + thread_idx) * 24;
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}

kernel void bucket_reduce_stage2_all(
    device const uint* buckets [[buffer(0)]],
    device uint* g_points      [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint thread_idx = tid.x;
    uint window_idx = tid.y;

    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (thread_idx >= num_threads) return;

    uint window_bucket_offset = window_idx * num_buckets;
    uint seg_start = window_bucket_offset + num_buckets - 1 - thread_idx * bpt;

    uint mx[8], my[8], mz[8];
    {
        uint boff = seg_start * 24;
        fq_load(buckets + boff,      mx);
        fq_load(buckets + boff + 8,  my);
        fq_load(buckets + boff + 16, mz);
    }

    uint goff = (window_idx * num_threads + thread_idx) * 24;
    uint gx[8], gy[8], gz[8];
    {
        fq_load(g_points + goff,      gx);
        fq_load(g_points + goff + 8,  gy);
        fq_load(g_points + goff + 16, gz);
    }

    uint scalar = bpt * (num_threads - thread_idx - 1);
    if (scalar > 0) {
        uint dax[8], day[8], daz[8];
        double_and_add(mx, my, mz, scalar, dax, day, daz);

        uint ox[8], oy[8], oz[8];
        projective_add(gx, gy, gz, dax, day, daz, ox, oy, oz);
        fq_copy(ox, gx);
        fq_copy(oy, gy);
        fq_copy(oz, gz);
    }

    {
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}
"#;
