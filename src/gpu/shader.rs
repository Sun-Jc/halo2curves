//! Metal shader source for GPU-accelerated MSM.
//!
//! Contains all MSL kernels as a Rust string constant, compiled at runtime.
//!
//! Architecture:
//! - Bucket accumulation uses **Jacobian coordinates** with Bernstein-Lange 2007 formulas
//!   for maximum performance: madd = 7M+4S (vs 11M in projective RCB2015)
//! - PBPR reduction uses Jacobian add (11M+5S) and dbl (2M+5S)
//! - GPU outputs Jacobian points; host converts to halo2curves projective
//!
//! Key optimizations:
//! - Dedicated fq_sqr exploits a[i]*a[j] symmetry (36 muls vs 64 in fq_mul)
//! - madd-2007-bl: 7M+4S vs previous 11M — main bucket kernel ~25% faster
//! - Alias-safe fq_mul/fq_add/fq_sub eliminate fq_copy overhead
//! - First point loaded directly as accumulator (skip identity madd)
//! - Branchless fq_neg

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

// ---- Modular addition: c = a + b mod p (supports c aliasing a or b) ----
inline void fq_add(thread const uint* a, thread const uint* b, thread uint* c) {
    ulong carry = 0;
    uint sum[8];
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(a[i]) + ulong(b[i]) + carry;
        sum[i] = uint(s & 0xFFFFFFFFul);
        carry = s >> 32;
    }
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(sum[i]) - ulong(FQ_MOD[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    uint use_diff = uint((carry >= borrow) ? 1 : 0);
    for (int i = 0; i < 8; i++) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

// ---- Modular subtraction: c = a - b mod p (supports c aliasing a or b) ----
inline void fq_sub(thread const uint* a, thread const uint* b, thread uint* c) {
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(a[i]) - ulong(b[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = ulong(diff[i]) + (borrow ? ulong(FQ_MOD[i]) : 0ul) + carry;
        c[i] = uint(s & 0xFFFFFFFFul);
        carry = s >> 32;
    }
}

// ---- Modular negation: c = -a mod p (branchless) ----
inline void fq_neg(thread const uint* a, thread uint* c) {
    uint is_nonzero = 0;
    for (int i = 0; i < 8; i++) is_nonzero |= a[i];
    uint mask = (is_nonzero != 0) ? 0xFFFFFFFFu : 0u;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong d = ulong(FQ_MOD[i] & mask) - ulong(a[i]) - borrow;
        c[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
}

// ---- Montgomery multiplication: c = a * b * R^{-1} mod p (CIOS, 32-bit) ----
// Supports c aliasing a or b.
inline void fq_mul(thread const uint* a, thread const uint* b, thread uint* c) {
    ulong t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong prod = ulong(a[j]) * ulong(b[i]) + t[j] + carry;
            t[j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[8] += carry;

        uint m = uint(t[0] & 0xFFFFFFFFul) * FQ_INV;
        ulong um = ulong(m);
        carry = (um * ulong(FQ_MOD[0]) + t[0]) >> 32;
        for (int j = 1; j < 8; j++) {
            ulong prod = um * ulong(FQ_MOD[j]) + t[j] + carry;
            t[j - 1] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        ulong sum = t[8] + carry;
        t[7] = sum & 0xFFFFFFFFul;
        t[8] = sum >> 32;
    }

    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = t[i] - ulong(FQ_MOD[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    uint use_t = uint(borrow);
    for (int i = 0; i < 8; i++) {
        c[i] = use_t ? uint(t[i] & 0xFFFFFFFFul) : diff[i];
    }
}

// ---- Montgomery squaring: c = a^2 * R^{-1} mod p (optimized) ----
// Exploits symmetry: off-diagonal products computed once and doubled.
// 28 off-diagonal + 8 diagonal = 36 multiplies (vs 64 in fq_mul).
// Plus 64 for reduction = 100 total (vs 128 in fq_mul = ~22% faster).
// Supports c aliasing a.
inline void fq_sqr(thread const uint* a, thread uint* c) {
    ulong t[17] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Off-diagonal: a[i]*a[j] for i < j (28 multiplies)
    for (int i = 0; i < 7; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 8; j++) {
            ulong prod = ulong(a[i]) * ulong(a[j]) + t[i+j] + carry;
            t[i+j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i+8] += carry;
    }

    // Double off-diagonal terms
    ulong dbl_carry = 0;
    for (int i = 1; i < 16; i++) {
        ulong v = (t[i] << 1) | dbl_carry;
        t[i] = v & 0xFFFFFFFFul;
        dbl_carry = v >> 32;
    }
    t[16] = dbl_carry;

    // Add diagonal: a[i]^2 (8 multiplies)
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong prod = ulong(a[i]) * ulong(a[i]) + t[2*i] + carry;
        t[2*i] = prod & 0xFFFFFFFFul;
        carry = prod >> 32;
        ulong s = t[2*i+1] + carry;
        t[2*i+1] = s & 0xFFFFFFFFul;
        carry = s >> 32;
    }
    t[16] += carry;

    // Montgomery reduction: 8 rounds (64 multiplies, same as fq_mul)
    for (int i = 0; i < 8; i++) {
        uint m = uint(t[i] & 0xFFFFFFFFul) * FQ_INV;
        ulong um = ulong(m);
        ulong rc = (um * ulong(FQ_MOD[0]) + t[i]) >> 32;
        for (int j = 1; j < 8; j++) {
            ulong prod = um * ulong(FQ_MOD[j]) + t[i+j] + rc;
            t[i+j] = prod & 0xFFFFFFFFul;
            rc = prod >> 32;
        }
        ulong s = t[i+8] + rc;
        t[i+8] = s & 0xFFFFFFFFul;
        rc = s >> 32;
        for (int k = i+9; k < 17 && rc != 0; k++) {
            s = t[k] + rc;
            t[k] = s & 0xFFFFFFFFul;
            rc = s >> 32;
        }
    }

    // Result in t[8..15], conditional subtraction
    ulong borrow = 0;
    uint diff[8];
    for (int i = 0; i < 8; i++) {
        ulong d = t[i+8] - ulong(FQ_MOD[i]) - borrow;
        diff[i] = uint(d & 0xFFFFFFFFul);
        borrow = (d >> 63) & 1;
    }
    uint use_t = uint(borrow);
    for (int i = 0; i < 8; i++) {
        c[i] = use_t ? uint(t[i+8] & 0xFFFFFFFFul) : diff[i];
    }
}

// ---- Double: c = 2*a mod p (supports c aliasing a) ----
inline void fq_dbl(thread const uint* a, thread uint* c) {
    fq_add(a, a, c);
}

// ---- Check if field element is zero ----
inline bool fq_is_zero(thread const uint* a) {
    uint v = 0;
    for (int i = 0; i < 8; i++) v |= a[i];
    return v == 0;
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
// Jacobian point operations — BN254 G1 (y² = x³ + 3, a=0)
// ============================================================================
//
// Jacobian coordinates: (X : Y : Z) represents affine (X/Z², Y/Z³)
// Identity: any (X, Y, 0) — we use Z=0 convention
// Affine point (x, y) → Jacobian (x, y, 1)
// Negation: (X, -Y, Z)

// ---- Jacobian mixed addition: R = P + Q ----
// P = (X1, Y1, Z1) Jacobian, Q = (x2, y2) affine (Z2 = 1)
// madd-2007-bl from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// Cost: 7M + 4S + 9add + 3×2 + 1×4
// NOT complete: fails when P = ±Q. Safe for MSM bucket accumulation
// since different base points are essentially random.
// Supports output aliasing input (x3=x1, etc.) via local temporaries.
inline void jacobian_madd(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,
    thread const uint* x2, thread const uint* y2,
    thread uint* x3, thread uint* y3, thread uint* z3
) {
    uint Z1Z1[8], U2[8], S2[8], H[8], HH[8], I[8], J[8], r[8], V[8];
    uint t1[8];

    fq_sqr(z1, Z1Z1);           // Z1Z1 = Z1^2              (1S)
    fq_mul(x2, Z1Z1, U2);       // U2 = x2 * Z1Z1           (1M)
    fq_mul(z1, Z1Z1, t1);       // t1 = Z1 * Z1Z1 = Z1^3    (2M)
    fq_mul(y2, t1, S2);         // S2 = y2 * Z1^3            (3M)
    fq_sub(U2, x1, H);          // H = U2 - X1
    fq_sqr(H, HH);              // HH = H^2                  (2S)
    fq_dbl(HH, I);              // I = 2*HH
    fq_dbl(I, I);               // I = 4*HH
    fq_mul(H, I, J);            // J = H * I                 (4M)
    fq_sub(S2, y1, r);          // r = S2 - Y1
    fq_dbl(r, r);               // r = 2*(S2 - Y1)
    fq_mul(x1, I, V);           // V = X1 * I                (5M)

    // X3 = r^2 - J - 2*V
    fq_sqr(r, x3);              // x3 = r^2                  (3S)
    fq_sub(x3, J, x3);          // x3 = r^2 - J
    fq_dbl(V, t1);              // t1 = 2*V
    fq_sub(x3, t1, x3);         // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*Y1*J
    fq_mul(y1, J, t1);          // t1 = Y1*J                 (6M)  — must read y1 BEFORE writing y3
    fq_dbl(t1, t1);             // t1 = 2*Y1*J
    uint t2[8];
    fq_sub(V, x3, t2);          // t2 = V - X3
    fq_mul(r, t2, y3);          // y3 = r*(V-X3)             (7M)
    fq_sub(y3, t1, y3);         // Y3 = r*(V-X3) - 2*Y1*J

    // Z3 = (Z1+H)^2 - Z1Z1 - HH
    fq_add(z1, H, t1);          // t1 = Z1 + H
    fq_sqr(t1, z3);             // z3 = (Z1+H)^2             (4S)
    fq_sub(z3, Z1Z1, z3);       // z3 -= Z1Z1
    fq_sub(z3, HH, z3);         // Z3 = (Z1+H)^2 - Z1Z1 - HH
}

// ---- Full Jacobian addition: R = P + Q ----
// P = (X1, Y1, Z1), Q = (X2, Y2, Z2), both Jacobian
// add-2007-bl from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// Cost: 11M + 5S + 9add + 4×2
// NOT complete: fails when P = ±Q. Safe for PBPR since bucket points are random.
// Supports output aliasing input via local temporaries.
inline void jacobian_add(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,
    thread const uint* x2, thread const uint* y2, thread const uint* z2,
    thread uint* x3, thread uint* y3, thread uint* z3
) {
    uint Z1Z1[8], Z2Z2[8], U1[8], U2[8], S1[8], S2[8];
    uint H[8], I[8], J[8], r[8], V[8];
    uint t1[8], t2[8];

    fq_sqr(z1, Z1Z1);              // Z1Z1 = Z1^2
    fq_sqr(z2, Z2Z2);              // Z2Z2 = Z2^2
    fq_mul(x1, Z2Z2, U1);          // U1 = X1 * Z2Z2
    fq_mul(x2, Z1Z1, U2);          // U2 = X2 * Z1Z1
    fq_mul(z2, Z2Z2, t1);          // t1 = Z2^3
    fq_mul(y1, t1, S1);            // S1 = Y1 * Z2^3
    fq_mul(z1, Z1Z1, t1);          // t1 = Z1^3
    fq_mul(y2, t1, S2);            // S2 = Y2 * Z1^3
    fq_sub(U2, U1, H);             // H = U2 - U1
    fq_dbl(H, t1);                 // t1 = 2*H
    fq_sqr(t1, I);                 // I = (2*H)^2
    fq_mul(H, I, J);               // J = H * I
    fq_sub(S2, S1, r);             // r = S2 - S1
    fq_dbl(r, r);                  // r = 2*(S2-S1)
    fq_mul(U1, I, V);              // V = U1 * I

    // X3 = r^2 - J - 2*V
    fq_sqr(r, x3);
    fq_sub(x3, J, x3);
    fq_dbl(V, t1);
    fq_sub(x3, t1, x3);

    // Y3 = r*(V-X3) - 2*S1*J
    fq_sub(V, x3, t1);
    fq_mul(r, t1, y3);
    fq_mul(S1, J, t1);
    fq_dbl(t1, t1);
    fq_sub(y3, t1, y3);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
    fq_add(z1, z2, t1);
    fq_sqr(t1, t2);
    fq_sub(t2, Z1Z1, t2);
    fq_sub(t2, Z2Z2, t2);
    fq_mul(t2, H, z3);
}

// ---- Jacobian point doubling: R = 2*P (a=0) ----
// dbl-2009-l from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// Cost: 2M + 5S + 6add + 3×2 + 1×3 + 1×8
// Supports output aliasing input via local temporaries.
inline void jacobian_dbl(
    thread const uint* x1, thread const uint* y1, thread const uint* z1,
    thread uint* x3, thread uint* y3, thread uint* z3
) {
    uint A[8], B[8], C[8], D[8], E[8], F[8];
    uint t1[8];

    fq_sqr(x1, A);               // A = X1^2                  (1S)
    fq_sqr(y1, B);               // B = Y1^2                  (2S)
    fq_sqr(B, C);                // C = B^2 = Y1^4            (3S)

    // D = 2*((X1+B)^2 - A - C)
    fq_add(x1, B, t1);           // t1 = X1 + B
    fq_sqr(t1, D);               // D = (X1+B)^2              (4S)
    fq_sub(D, A, D);             // D -= A
    fq_sub(D, C, D);             // D -= C  (now D = X1*B... hmm, it's X1*Y1^2 effectively)
    fq_dbl(D, D);                // D = 2*(X1*B)

    // E = 3*A  (= 3*X1^2)
    fq_dbl(A, E);                // E = 2*A
    fq_add(E, A, E);             // E = 3*A

    fq_sqr(E, F);                // F = E^2 = (3*X1^2)^2      (5S)

    // X3 = F - 2*D
    fq_dbl(D, t1);               // t1 = 2*D
    fq_sub(F, t1, x3);           // X3 = F - 2*D

    // Y3 = E*(D-X3) - 8*C
    fq_sub(D, x3, t1);           // t1 = D - X3
    fq_mul(E, t1, y3);           // y3 = E*(D-X3)             (1M)
    fq_dbl(C, t1);               // t1 = 2*C
    fq_dbl(t1, t1);              // t1 = 4*C
    fq_dbl(t1, t1);              // t1 = 8*C
    fq_sub(y3, t1, y3);          // Y3 = E*(D-X3) - 8*C

    // Z3 = 2*Y1*Z1
    fq_mul(y1, z1, z3);          // z3 = Y1*Z1                (2M)
    fq_dbl(z3, z3);              // Z3 = 2*Y1*Z1
}

// ---- Double-and-add: R = scalar * P (for small scalars, used in PBPR) ----
inline void double_and_add(
    thread const uint* px, thread const uint* py, thread const uint* pz,
    uint scalar,
    thread uint* rx, thread uint* ry, thread uint* rz
) {
    // Start with identity (Z=0)
    for (int i = 0; i < 8; i++) rx[i] = FQ_ONE[i]; // X = 1 (arbitrary)
    for (int i = 0; i < 8; i++) ry[i] = FQ_ONE[i]; // Y = 1 (arbitrary)
    for (int i = 0; i < 8; i++) rz[i] = 0;         // Z = 0 → identity

    if (scalar == 0) return;

    uint tx[8], ty[8], tz[8];
    fq_copy(px, tx);
    fq_copy(py, ty);
    fq_copy(pz, tz);

    uint s = scalar;
    while (s != 0) {
        if (s & 1) {
            // Check if R is identity (Z=0)
            if (fq_is_zero(rz)) {
                fq_copy(tx, rx);
                fq_copy(ty, ry);
                fq_copy(tz, rz);
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(rx, ry, rz, tx, ty, tz, ox, oy, oz);
                fq_copy(ox, rx);
                fq_copy(oy, ry);
                fq_copy(oz, rz);
            }
        }
        s >>= 1;
        if (s != 0) {
            uint dx[8], dy[8], dz[8];
            jacobian_dbl(tx, ty, tz, dx, dy, dz);
            fq_copy(dx, tx);
            fq_copy(dy, ty);
            fq_copy(dz, tz);
        }
    }
}

// ============================================================================
// GPU Kernels
// ============================================================================

// ---- Test kernel: field operations ----
kernel void test_field_ops(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* c       [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 8;
    uint la[8], lb[8], lc[8];
    fq_load(a + off, la);
    fq_load(b + off, lb);
    fq_mul(la, lb, lc);
    fq_store(lc, c + off);
}

// ---- Test kernel: fq_sqr vs fq_mul(a,a) ----
// Writes two results: sqr result at c[id*16..id*16+8], mul(a,a) at c[id*16+8..id*16+16]
kernel void test_fq_sqr(
    device const uint* a [[buffer(0)]],
    device uint* c       [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 8;
    uint la[8], sqr_result[8], mul_result[8];
    fq_load(a + off, la);
    fq_sqr(la, sqr_result);
    fq_mul(la, la, mul_result);
    uint out_off = id * 16;
    fq_store(sqr_result, c + out_off);
    fq_store(mul_result, c + out_off + 8);
}

// ---- Test kernel: Jacobian add ----
// Takes two Jacobian points, outputs one Jacobian point
kernel void test_jacobian_add(
    device const uint* p1  [[buffer(0)]],
    device const uint* p2  [[buffer(1)]],
    device uint* out       [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 24;
    uint x1[8], y1[8], z1[8], x2[8], y2[8], z2[8];
    uint rx[8], ry[8], rz[8];
    fq_load(p1 + off, x1);
    fq_load(p1 + off + 8, y1);
    fq_load(p1 + off + 16, z1);
    fq_load(p2 + off, x2);
    fq_load(p2 + off + 8, y2);
    fq_load(p2 + off + 16, z2);
    jacobian_add(x1, y1, z1, x2, y2, z2, rx, ry, rz);
    fq_store(rx, out + off);
    fq_store(ry, out + off + 8);
    fq_store(rz, out + off + 16);
}

// ---- Test kernel: Jacobian dbl ----
kernel void test_jacobian_dbl(
    device const uint* p1  [[buffer(0)]],
    device uint* out       [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 24;
    uint x1[8], y1[8], z1[8];
    uint rx[8], ry[8], rz[8];
    fq_load(p1 + off, x1);
    fq_load(p1 + off + 8, y1);
    fq_load(p1 + off + 16, z1);
    jacobian_dbl(x1, y1, z1, rx, ry, rz);
    fq_store(rx, out + off);
    fq_store(ry, out + off + 8);
    fq_store(rz, out + off + 16);
}

// ---- Test kernel: double_and_add ----
// Takes a Jacobian point and a scalar, outputs scalar*P
kernel void test_double_and_add(
    device const uint* p1      [[buffer(0)]],
    device const uint* scalars [[buffer(1)]],
    device uint* out           [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint off = id * 24;
    uint x1[8], y1[8], z1[8];
    uint rx[8], ry[8], rz[8];
    fq_load(p1 + off, x1);
    fq_load(p1 + off + 8, y1);
    fq_load(p1 + off + 16, z1);
    uint s = scalars[id];
    double_and_add(x1, y1, z1, s, rx, ry, rz);
    fq_store(rx, out + off);
    fq_store(ry, out + off + 8);
    fq_store(rz, out + off + 16);
}

// ---- Test kernel: field add/sub/neg ----
kernel void test_field_addsub(
    device const uint* a   [[buffer(0)]],
    device const uint* b   [[buffer(1)]],
    device uint* c         [[buffer(2)]],
    device const uint* op  [[buffer(3)]],
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
kernel void test_jacobian_madd(
    device const uint* jac  [[buffer(0)]],
    device const uint* aff  [[buffer(1)]],
    device const uint* sign [[buffer(2)]],
    device uint* out        [[buffer(3)]],
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

    if (sign[id] == 0) {
        fq_neg(ay, ay);
    }

    uint rx[8], ry[8], rz[8];
    jacobian_madd(jx, jy, jz, ax, ay, rx, ry, rz);

    fq_store(rx, out + joff);
    fq_store(ry, out + joff + 8);
    fq_store(rz, out + joff + 16);
}

// ---- Main kernel: bucket accumulation (all windows in one dispatch) ----
// Uses Jacobian coordinates with madd-2007-bl (7M+4S per point added).
// First scatter entry initializes accumulator directly (skip identity madd).
// Subsequent entries use jacobian_madd with alias-safe in-place updates.
// Output: Jacobian (X,Y,Z) per bucket — converted to projective on host.
kernel void bucket_accumulate_all(
    device const uint* bases           [[buffer(0)]],
    device const uint* scatter_entries [[buffer(1)]],
    device const uint* bucket_offsets  [[buffer(2)]],
    device uint* buckets_out           [[buffer(3)]],
    device const uint* window_params   [[buffer(4)]],
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
        // Empty bucket → identity in Jacobian: (1, 1, 0)
        for (int i = 0; i < 8; i++) buckets_out[boff + i] = FQ_ONE[i];
        for (int i = 0; i < 8; i++) buckets_out[boff + 8 + i] = FQ_ONE[i];
        for (int i = 0; i < 8; i++) buckets_out[boff + 16 + i] = 0;
        return;
    }

    // Load first point directly as Jacobian accumulator: (x, ±y, 1)
    uint e0 = scatter_start + start;
    uint packed0 = scatter_entries[e0];
    uint base_idx0 = packed0 & 0x7FFFFFFFu;
    uint s0 = packed0 >> 31;
    uint poff0 = base_idx0 * 16;

    uint accx[8], accy[8], accz[8];
    fq_load(bases + poff0, accx);
    fq_load(bases + poff0 + 8, accy);
    if (s0 == 0) {
        fq_neg(accy, accy);
    }
    for (int i = 0; i < 8; i++) accz[i] = FQ_ONE[i]; // Z = 1 (affine → Jacobian)

    // Process remaining entries — jacobian_madd is alias-safe for output aliasing input
    for (uint e = e0 + 1; e < scatter_start + end; e++) {
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

        jacobian_madd(accx, accy, accz, ax, ay, accx, accy, accz);
    }

    fq_store(accx, buckets_out + boff);
    fq_store(accy, buckets_out + boff + 8);
    fq_store(accz, buckets_out + boff + 16);
}

// ---- Original single-window bucket accumulation (kept for tests) ----
kernel void bucket_accumulate(
    device const uint* bases           [[buffer(0)]],
    device const uint* scatter_entries [[buffer(1)]],
    device const uint* bucket_offsets  [[buffer(2)]],
    device uint* buckets_out           [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint start = bucket_offsets[tid];
    uint end   = bucket_offsets[tid + 1];

    uint boff_out = tid * 24;

    if (start == end) {
        for (int i = 0; i < 8; i++) buckets_out[boff_out + i] = FQ_ONE[i];
        for (int i = 0; i < 8; i++) buckets_out[boff_out + 8 + i] = FQ_ONE[i];
        for (int i = 0; i < 8; i++) buckets_out[boff_out + 16 + i] = 0;
        return;
    }

    uint packed0 = scatter_entries[start];
    uint base_idx0 = packed0 & 0x7FFFFFFFu;
    uint s0 = packed0 >> 31;
    uint poff0 = base_idx0 * 16;

    uint accx[8], accy[8], accz[8];
    fq_load(bases + poff0, accx);
    fq_load(bases + poff0 + 8, accy);
    if (s0 == 0) { fq_neg(accy, accy); }
    for (int i = 0; i < 8; i++) accz[i] = FQ_ONE[i];

    for (uint e = start + 1; e < end; e++) {
        uint packed   = scatter_entries[e];
        uint base_idx = packed & 0x7FFFFFFFu;
        uint s        = packed >> 31;

        uint poff = base_idx * 16;
        uint ax[8], ay[8];
        fq_load(bases + poff, ax);
        fq_load(bases + poff + 8, ay);

        if (s == 0) { fq_neg(ay, ay); }

        jacobian_madd(accx, accy, accz, ax, ay, accx, accy, accz);
    }

    fq_store(accx, buckets_out + boff_out);
    fq_store(accy, buckets_out + boff_out + 8);
    fq_store(accz, buckets_out + boff_out + 16);
}

// ============================================================================
// PBPR: Parallel Bucket Point Reduction (two-stage)
// Uses Jacobian coordinates throughout.
// ============================================================================

kernel void bucket_reduce_stage1(
    device uint* buckets       [[buffer(0)]],
    device uint* g_points      [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (tid >= num_threads) return;

    uint seg_start = num_buckets - 1 - tid * bpt;

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

    // Track if g and m are the same point (after initial copy).
    // When g==m, jacobian_add(g,m) is P+P which fails with incomplete formula.
    // We must use jacobian_dbl instead.
    bool g_equals_m = true;

    for (uint i = 1; i < bpt; i++) {
        uint bi = seg_start - i;
        uint boff = bi * 24;
        uint bx[8], by[8], bz[8];
        fq_load(buckets + boff,      bx);
        fq_load(buckets + boff + 8,  by);
        fq_load(buckets + boff + 16, bz);

        // Skip identity buckets (Z=0)
        if (!fq_is_zero(bz)) {
            if (fq_is_zero(mz)) {
                // m was identity, replace with bucket
                fq_copy(bx, mx);
                fq_copy(by, my);
                fq_copy(bz, mz);
            } else {
                // m += bucket[bi]
                uint ox[8], oy[8], oz[8];
                jacobian_add(mx, my, mz, bx, by, bz, ox, oy, oz);
                fq_copy(ox, mx);
                fq_copy(oy, my);
                fq_copy(oz, mz);
            }
            g_equals_m = false; // m changed, so g != m now
        }

        // g += m (only if m is not identity)
        if (!fq_is_zero(mz)) {
            if (fq_is_zero(gz)) {
                fq_copy(mx, gx);
                fq_copy(my, gy);
                fq_copy(mz, gz);
                g_equals_m = true; // g was just copied from m
            } else if (g_equals_m) {
                // g == m, use doubling instead of addition
                uint ox[8], oy[8], oz[8];
                jacobian_dbl(gx, gy, gz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
                // g and m are no longer the same (g = 2*m, m = m)
                g_equals_m = false;
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(gx, gy, gz, mx, my, mz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
            }
        }
    }

    {
        uint boff = seg_start * 24;
        fq_store(mx, buckets + boff);
        fq_store(my, buckets + boff + 8);
        fq_store(mz, buckets + boff + 16);
    }
    {
        uint goff = tid * 24;
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}

kernel void bucket_reduce_stage2(
    device const uint* buckets [[buffer(0)]],
    device uint* g_points      [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint num_buckets = params[0];
    uint num_threads = params[1];
    uint bpt = num_buckets / num_threads;

    if (tid >= num_threads) return;

    uint seg_start = num_buckets - 1 - tid * bpt;
    uint mx[8], my[8], mz[8];
    {
        uint boff = seg_start * 24;
        fq_load(buckets + boff,      mx);
        fq_load(buckets + boff + 8,  my);
        fq_load(buckets + boff + 16, mz);
    }

    uint gx[8], gy[8], gz[8];
    {
        uint goff = tid * 24;
        fq_load(g_points + goff,      gx);
        fq_load(g_points + goff + 8,  gy);
        fq_load(g_points + goff + 16, gz);
    }

    uint scalar = bpt * (num_threads - tid - 1);
    if (scalar > 0 && !fq_is_zero(mz)) {
        uint dax[8], day[8], daz[8];
        double_and_add(mx, my, mz, scalar, dax, day, daz);

        if (!fq_is_zero(daz)) {
            if (fq_is_zero(gz)) {
                fq_copy(dax, gx);
                fq_copy(day, gy);
                fq_copy(daz, gz);
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(gx, gy, gz, dax, day, daz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
            }
        }
    }

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
// ============================================================================

kernel void bucket_reduce_stage1_all(
    device uint* buckets       [[buffer(0)]],
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

    bool g_equals_m = true;

    for (uint i = 1; i < bpt; i++) {
        uint bi = seg_start - i;
        uint boff = bi * 24;
        uint bx[8], by[8], bz[8];
        fq_load(buckets + boff,      bx);
        fq_load(buckets + boff + 8,  by);
        fq_load(buckets + boff + 16, bz);

        if (!fq_is_zero(bz)) {
            if (fq_is_zero(mz)) {
                fq_copy(bx, mx);
                fq_copy(by, my);
                fq_copy(bz, mz);
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(mx, my, mz, bx, by, bz, ox, oy, oz);
                fq_copy(ox, mx);
                fq_copy(oy, my);
                fq_copy(oz, mz);
            }
            g_equals_m = false;
        }

        if (!fq_is_zero(mz)) {
            if (fq_is_zero(gz)) {
                fq_copy(mx, gx);
                fq_copy(my, gy);
                fq_copy(mz, gz);
                g_equals_m = true;
            } else if (g_equals_m) {
                uint ox[8], oy[8], oz[8];
                jacobian_dbl(gx, gy, gz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
                g_equals_m = false;
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(gx, gy, gz, mx, my, mz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
            }
        }
    }

    {
        uint boff = seg_start * 24;
        fq_store(mx, buckets + boff);
        fq_store(my, buckets + boff + 8);
        fq_store(mz, buckets + boff + 16);
    }
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
    if (scalar > 0 && !fq_is_zero(mz)) {
        uint dax[8], day[8], daz[8];
        double_and_add(mx, my, mz, scalar, dax, day, daz);

        if (!fq_is_zero(daz)) {
            if (fq_is_zero(gz)) {
                fq_copy(dax, gx);
                fq_copy(day, gy);
                fq_copy(daz, gz);
            } else {
                uint ox[8], oy[8], oz[8];
                jacobian_add(gx, gy, gz, dax, day, daz, ox, oy, oz);
                fq_copy(ox, gx);
                fq_copy(oy, gy);
                fq_copy(oz, gz);
            }
        }
    }

    {
        fq_store(gx, g_points + goff);
        fq_store(gy, g_points + goff + 8);
        fq_store(gz, g_points + goff + 16);
    }
}
"#;
