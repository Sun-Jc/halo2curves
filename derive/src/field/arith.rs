#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use proc_macro2::TokenStream;
use quote::{format_ident as fmtid, quote};

fn select(cond: bool, this: TokenStream, other: TokenStream) -> TokenStream {
    if cond {
        this
    } else {
        other
    }
}

pub(crate) fn impl_arith(field: &syn::Ident, num_limbs: usize, inv: u64, modulus_limbs: &[u64]) -> TokenStream {
    let impl_add = impl_add(field, num_limbs);
    let impl_sub = impl_sub(field, num_limbs);
    let impl_neg = impl_neg(field, num_limbs);
    let impl_mont = impl_mont(field, num_limbs, inv);
    let impl_from_mont = impl_from_mont(field, num_limbs, inv);
    let impl_mul = impl_mul(field, num_limbs, false);
    let impl_square = impl_square(field, num_limbs);
    let impl_mul_by_u64 = impl_mul_by_u64(field, num_limbs, modulus_limbs);
    let impl_mul_by_u32 = impl_mul_by_u32(field);
    let impl_mul_by_i32 = impl_mul_by_i32(field);
    let impl_mul_by_i128 = impl_mul_by_i128(field, num_limbs);
    let wide_num_limbs = num_limbs * 2;
    quote::quote! {
        impl #field {
            #[inline(always)]
            pub const fn add(&self, rhs: &Self) -> Self {
                #impl_add
            }

            #[inline]
            pub const fn double(&self) -> Self {
                self.add(self)
            }

            #[inline(always)]
            pub const fn sub(&self, rhs: &Self) -> Self {
                #impl_sub
            }

            #[inline(always)]
            pub const fn neg(&self) -> Self {
                #impl_neg
            }

            #[inline(always)]
            pub const fn mul(&self, rhs: &Self) -> Self{
                #impl_mul
            }

            #[inline(always)]
            pub const fn square(&self) -> Self{
                #impl_square
            }

            /// Multiply a field element by a u64 scalar using Barrett reduction.
            ///
            /// For self in Montgomery form (stores a*R), computes (a*rhs)*R mod p
            /// via sparse widening multiply (N macs) + Barrett reduction (~2N ops).
            /// Much faster than full field multiply (2N^2 mac ops).
            #[inline(always)]
            pub fn mul_by_u64(&self, rhs: u64) -> Self {
                #impl_mul_by_u64
            }

            /// Low-level sparse Montgomery multiply by u32. Delegates to `mul_by_u64`.
            #[inline(always)]
            pub fn mul_by_u32(&self, rhs: u32) -> Self {
                #impl_mul_by_u32
            }

            /// Low-level sparse Montgomery multiply by i32.
            /// Computes `self.0 * |rhs| * R^{-1} mod p`, then negates if rhs < 0.
            #[inline(always)]
            pub fn mul_by_i32(&self, rhs: i32) -> Self {
                #impl_mul_by_i32
            }

            /// Low-level sparse Montgomery multiply by i128.
            /// Fast path when |rhs| fits in u64 (num_limbs mac ops),
            /// otherwise uses 2-limb sparse multiply (2*num_limbs mac ops).
            #[inline]
            pub fn mul_by_i128(&self, rhs: i128) -> Self {
                #impl_mul_by_i128
            }

            #[inline(always)]
            pub(crate) const fn montgomery_reduce(r: &[u64; #wide_num_limbs]) -> Self {
                #impl_mont
            }

            #[inline(always)]
            pub(crate) const fn from_mont(&self) -> [u64; #num_limbs] {
                #impl_from_mont
            }
        }
    }
}

pub(crate) fn impl_arith_always_const(
    field: &syn::Ident,
    num_limbs: usize,
    inv: u64,
) -> TokenStream {
    let impl_sub = impl_sub(field, num_limbs);
    let impl_mont = impl_mont(field, num_limbs, inv);
    let impl_mul = impl_mul(field, num_limbs, true);
    let wide_num_limbs = num_limbs * 2;
    quote::quote! {
        impl #field {

            #[inline(always)]
            pub(crate) const fn sub_const(&self, rhs: &Self) -> Self {
                #impl_sub
            }


            #[inline(always)]
            pub(crate) const fn mul_const(&self, rhs: &Self) -> Self{
                #impl_mul
            }

            #[inline(always)]
            pub(crate) const fn montgomery_reduce_const(r: &[u64; #wide_num_limbs]) -> Self {
                #impl_mont
            }
        }
    }
}

fn impl_mul(field: &syn::Ident, num_limbs: usize, constant: bool) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac}; };
    for i in 0..num_limbs {
        for j in 0..num_limbs {
            let r_out = fmtid!("r_{}", i + j);
            let r_next = fmtid!("r_{}", i + j + 1);
            let r_in = select(i == 0, quote! {0}, quote! {#r_out});
            let carry_in = select(j == 0, quote! {0}, quote! {carry});
            let carry_out = select(j == num_limbs - 1, quote! {#r_next}, quote! {carry});
            gen.extend(
                quote! { let (#r_out, #carry_out) = mac(#r_in, self.0[#i], rhs.0[#j], #carry_in); },
            );
        }
    }

    let r: Vec<_> = (0..num_limbs * 2).map(|i| fmtid!("r_{}", i)).collect();
    let mont_red = if constant {
        quote! { #field::montgomery_reduce_const(&[#(#r),*]) }
    } else {
        quote! { #field::montgomery_reduce(&[#(#r),*]) }
    };
    quote! {
        #gen
        #mont_red
    }
}

fn impl_square(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac}; };
    for i in 0..num_limbs - 1 {
        let start_index = i * 2 + 1;
        for j in 0..num_limbs - i - 1 {
            let r_out = fmtid!("r_{}", start_index + j);
            let r_in = select(i == 0, quote! {0}, quote! {#r_out});
            let r_next = fmtid!("r_{}", start_index + j + 1);
            let carry_in = select(j == 0, quote! {0}, quote! {carry});
            let carry_out = select(j == num_limbs - i - 2, quote! {#r_next}, quote! {carry});
            let j = i + j + 1;
            gen.extend(quote! { let (#r_out, #carry_out) = mac(#r_in, self.0[#i], self.0[#j], #carry_in); });
        }
    }

    for i in (1..num_limbs * 2).rev() {
        let (r_cur, r_next) = (fmtid!("r_{}", i), fmtid!("r_{}", i - 1));
        if i == num_limbs * 2 - 1 {
            gen.extend(quote! { let #r_cur = #r_next >> 63; });
        } else if i == 1 {
            gen.extend(quote! { let #r_cur = (#r_cur << 1); });
        } else {
            gen.extend(quote! { let #r_cur = (#r_cur << 1) | (#r_next >> 63); });
        }
    }

    for i in 0..num_limbs {
        let index = i * 2;
        let r_cur = fmtid!("r_{}", index);
        let r_next = fmtid!("r_{}", index + 1);
        let r_cur_in = select(i == 0, quote! {0}, quote! {#r_cur});
        let carry_in = select(i == 0, quote! {0}, quote! {carry});
        let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
        gen.extend(quote! {
            let (#r_cur, carry) = mac(#r_cur_in, self.0[#i], self.0[#i], #carry_in);
            let (#r_next, #carry_out) = adc(0, #r_next, carry);
        });
    }

    let r: Vec<_> = (0..num_limbs * 2).map(|i| fmtid!("r_{}", i)).collect();
    quote! {
        #gen
        #field::montgomery_reduce(&[#(#r),*])
    }
}

fn impl_add(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb}; };

    (0..num_limbs).for_each(|i| {
        let carry = select(i == 0, quote! {0}, quote! {carry});
        let d_i = fmtid!("d_{}", i);
        gen.extend(quote! { let ( #d_i, carry) = adc(self.0[#i], rhs.0[#i], #carry); });
    });

    // Attempt to subtract the modulus, to ensure the value
    // is smaller than the modulus.
    (0..num_limbs).for_each(|i| {
        let borrow = select(i == 0, quote! {0}, quote! {borrow});
        let d_i = fmtid!("d_{}", i);
        gen.extend(quote! { let (#d_i, borrow) = sbb(#d_i, Self::MODULUS_LIMBS[#i], #borrow); });
    });
    gen.extend(quote! {let (_, borrow) = sbb(carry, 0, borrow);});

    (0..num_limbs).for_each(|i| {
        let carry_in = select(i == 0, quote! {0}, quote! {carry});
        let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
        let d_i = fmtid!("d_{}", i);
        gen.extend(
            quote! { let (#d_i, #carry_out) = adc(#d_i, Self::MODULUS_LIMBS[#i] & borrow, #carry_in); },
        );
    });

    let ret: Vec<_> = (0..num_limbs).map(|i| fmtid!("d_{}", i)).collect();

    quote! {
        #gen
        #field([#(#ret),*])
    }
}

fn impl_sub(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb}; };

    (0..num_limbs).for_each(|i| {
        let borrow = select(i == 0, quote! {0}, quote! {borrow});
        let d_i = fmtid!("d_{}", i);
        gen.extend(quote! { let (#d_i, borrow) = sbb(self.0[#i], rhs.0[#i], #borrow); });
    });

    (0..num_limbs).for_each(|i| {
        let carry_in = select(i == 0, quote! {0}, quote! {carry});
        let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
        let d_i = fmtid!("d_{}", i);
        gen.extend(
            quote! { let (#d_i, #carry_out) = adc(#d_i, Self::MODULUS_LIMBS[#i] & borrow, #carry_in); },
        );
    });

    let ret: Vec<_> = (0..num_limbs).map(|i| fmtid!("d_{}", i)).collect();

    quote! {
        #gen
        #field([#(#ret),*])
    }
}

fn impl_neg(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb}; };

    (0..num_limbs).for_each(|i| {
        let borrow_in = select(i == 0, quote! {0}, quote! {borrow});
        let borrow_out = select(i == num_limbs - 1, quote! {_}, quote! {borrow});
        let d_i = fmtid!("d_{}", i);
        gen.extend(quote! { let (#d_i, #borrow_out) = sbb(Self::MODULUS_LIMBS[#i], self.0[#i], #borrow_in); })
    });

    let mask_limbs: Vec<_> = (0..num_limbs)
        .map(|i| quote::quote! { self.0[#i] })
        .collect();
    gen.extend(quote! { let mask = (((#(#mask_limbs)|*) == 0) as u64).wrapping_sub(1); });

    let ret: Vec<_> = (0..num_limbs)
        .map(|i| {
            let d_i = fmtid!("d_{}", i);
            quote! { #d_i & mask }
        })
        .collect();

    quote! {
        #gen
        #field([#(#ret),*])
    }
}

fn impl_mont(field: &syn::Ident, num_limbs: usize, inv: u64) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac}; };

    for i in 0..num_limbs {
        if i == 0 {
            gen.extend(quote! { let k = r[0].wrapping_mul(#inv); });

            for j in 0..num_limbs {
                let r_out = fmtid!("r_{}", j);
                let r_out = select(j == 0, quote! {_}, quote! {#r_out});
                let carry_in = select(j == 0, quote! {0}, quote! {carry});
                gen.extend(quote! { let (#r_out, carry) = mac(r[#j], k, Self::MODULUS_LIMBS[#j], #carry_in); });
            }
            let r_out = fmtid!("r_{}", num_limbs);
            gen.extend(quote! { let (#r_out, carry2) = adc(r[#num_limbs], 0, carry); });
        } else {
            let r_i = fmtid!("r_{}", i);
            gen.extend(quote! { let k = #r_i.wrapping_mul(#inv); });

            for j in 0..num_limbs {
                let r_in = fmtid!("r_{}", j + i);
                let r_out = select(j == 0, quote! {_}, quote! {#r_in});
                let carry_in = select(j == 0, quote! {0}, quote! {carry});
                gen.extend(quote! { let (#r_out, carry) = mac(#r_in, k, Self::MODULUS_LIMBS[#j], #carry_in); });
            }
            let idx = num_limbs + i;
            let r_out = fmtid!("r_{}", idx);
            gen.extend(quote! { let (#r_out, carry2) = adc(r[#idx], carry2, carry); });
        }
    }

    (0..num_limbs).for_each(|i| {
        let borrow = select(i == 0, quote! {0}, quote! {borrow});
        let d_i = fmtid!("d_{}", i);
        let r_in = fmtid!("r_{}", num_limbs + i);
        gen.extend(quote! { let (#d_i, borrow) = sbb(#r_in, Self::MODULUS_LIMBS[#i], #borrow); });
    });

    gen.extend(quote! {let (_, borrow) = sbb(carry2, 0, borrow);});

    (0..num_limbs).for_each(|i| {
        let carry_in = select(i == 0, quote! {0}, quote! {carry});
        let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
        let d_i = fmtid!("d_{}", i);
        gen.extend(
            quote! { let (#d_i, #carry_out) = adc(#d_i, Self::MODULUS_LIMBS[#i] & borrow, #carry_in); },
        );
    });
    let ret: Vec<_> = (0..num_limbs).map(|i| fmtid!("d_{}", i)).collect();

    quote! {
        #gen
        #field([#(#ret),*])
    }
}

fn impl_mul_by_u64(field: &syn::Ident, num_limbs: usize, modulus_limbs: &[u64]) -> TokenStream {
    // Optimized field x u64 using sparse widening multiply + Barrett reduction.
    //
    // For self in Montgomery form (stores a*R), product = self.0 x rhs = a*R*rhs (N+1 limbs).
    // Barrett reduce mod p gives a*R*rhs mod p = (a*rhs)*R mod p  --  correct Montgomery form!
    //
    // Optimized Barrett with variable-time branching (like arkworks):
    //   1. Sparse widening multiply (N mac ops)
    //   2. Barrett estimate m (1 u128 mul)
    //   3. Subtract m x 2p (N mac ops + N sbb ops)
    //   4. Single branching conditional subtract (N compare + N sbb)
    //
    // Total: ~2N mac + ~3N sbb ~ 8 mac + 12 sbb for N=4
    // vs previous: 8 mac + 24 sbb (3 constant-time rounds)
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac, macx}; };

    // --- Compute Barrett constants at proc-macro time ---
    use num_bigint::BigUint;
    use num_traits::One;

    let top_limb = modulus_limbs[num_limbs - 1];
    let spare_bits = top_limb.leading_zeros();

    // Reconstruct modulus as BigUint
    let p_big = modulus_limbs.iter().enumerate().fold(
        BigUint::from(0u64),
        |acc, (i, &limb)| acc + BigUint::from(limb) * (BigUint::one() << (64 * i)),
    );

    // BARRETT_MU = floor(2^(63 - spare + 64*N) / p)
    let k = (63 - spare_bits) as usize + 64 * num_limbs;
    let numerator = BigUint::one() << k;
    let mu_big = &numerator / &p_big;
    let mu_bytes = mu_big.to_bytes_le();
    let mut barrett_mu = 0u64;
    for (i, &byte) in mu_bytes.iter().enumerate().take(8) {
        barrett_mu |= (byte as u64) << (8 * i);
    }

    // Helper to extract u64 limbs from BigUint
    let big_to_limbs = |val: &BigUint, n: usize| -> Vec<u64> {
        (0..n).map(|i| {
            let shifted = val >> (64 * i);
            let bytes = shifted.to_bytes_le();
            let mut v = 0u64;
            for (j, &b) in bytes.iter().enumerate().take(8) {
                v |= (b as u64) << (8 * j);
            }
            v
        }).collect()
    };

    // 2p and 3p limbs (precomputed at proc-macro time)
    let two_p = &p_big * 2u32;
    let three_p = &p_big * 3u32;
    let two_p_limbs = big_to_limbs(&two_p, num_limbs);
    let three_p_limbs = big_to_limbs(&three_p, num_limbs);

    let spare_bits_lit = spare_bits;

    // Step 1: Sparse widening multiply self.0[i] x rhs -> w[0..N+1]  (N mac ops)
    for i in 0..num_limbs {
        let w_out = fmtid!("w_{}", i);
        let w_next = fmtid!("w_{}", i + 1);
        if i == 0 {
            gen.extend(quote! { let (#w_out, carry) = macx(0, self.0[#i], rhs); });
        } else if i == num_limbs - 1 {
            gen.extend(quote! { let (#w_out, #w_next) = mac(0, self.0[#i], rhs, carry); });
        } else {
            gen.extend(quote! { let (#w_out, carry) = mac(0, self.0[#i], rhs, carry); });
        }
    }

    // Step 2: Barrett quotient estimate
    let w_top = fmtid!("w_{}", num_limbs);
    let w_second = fmtid!("w_{}", num_limbs - 1);
    if spare_bits > 0 {
        gen.extend(quote! {
            let tilde = (#w_top << #spare_bits_lit) | (#w_second >> (64 - #spare_bits_lit));
            let m = ((tilde as u128 * #barrett_mu as u128) >> 64) as u64;
        });
    } else {
        gen.extend(quote! {
            let m = ((#w_top as u128 * #barrett_mu as u128) >> 64) as u64;
        });
    }

    // Step 3: Compute m x 2p and subtract from w -> r  (N mac + N sbb)
    for j in 0..num_limbs {
        let two_p_j = two_p_limbs[j];
        let mp_lo = fmtid!("mp_lo_{}", j);
        let mp_hi = fmtid!("mp_hi_{}", j);
        if j == 0 {
            gen.extend(quote! {
                let (#mp_lo, #mp_hi) = macx(0, m, #two_p_j);
            });
        } else {
            let mp_hi_prev = fmtid!("mp_hi_{}", j - 1);
            gen.extend(quote! {
                let (#mp_lo, #mp_hi) = mac(#mp_hi_prev, m, #two_p_j, 0);
            });
        }
    }

    for j in 0..num_limbs {
        let w_j = fmtid!("w_{}", j);
        let r_j = fmtid!("r_{}", j);
        let mp_lo_j = fmtid!("mp_lo_{}", j);
        let br_out = fmtid!("br_{}", j);
        if j == 0 {
            gen.extend(quote! { let (#r_j, #br_out) = sbb(#w_j, #mp_lo_j, 0); });
        } else {
            let br_prev = fmtid!("br_{}", j - 1);
            gen.extend(quote! { let (#r_j, #br_out) = sbb(#w_j, #mp_lo_j, #br_prev); });
        }
    }

    // Step 4: Variable-time conditional subtract using precomputed 2p, 3p.
    // After m*2p subtraction, remainder r < 4p. We need r mod p.
    // Since spare_bits >= 2 for BN254, 3p fits in N limbs (no overflow).
    // Use branching comparisons (variable-time, but not security-sensitive for scalars).
    //
    // Strategy: compare r against 3p, 2p, p and subtract the appropriate multiple.
    // Only ONE subtraction path executes.
    let r_limbs: Vec<_> = (0..num_limbs).map(|i| fmtid!("r_{}", i)).collect();

    // Build the r array for comparison
    gen.extend(quote! {
        let r_arr = [#(#r_limbs),*];
    });

    // Precomputed constants as arrays
    gen.extend(quote! {
        const THREE_P: [u64; #num_limbs] = [#(#three_p_limbs),*];
        const TWO_P: [u64; #num_limbs] = [#(#two_p_limbs),*];
    });

    // Comparison helper: returns true if a >= b (lexicographic on [u64; N], little-endian)
    // We compare from MSB down  --  first difference determines result
    gen.extend(quote! {
        // Compare r against threshold, subtract the right multiple of p
        // ge_n compares two N-limb little-endian numbers: returns true if a >= b
        let ge_n = |a: &[u64; #num_limbs], b: &[u64; #num_limbs]| -> bool {
            let mut i = #num_limbs;
            while i > 0 {
                i -= 1;
                if a[i] > b[i] { return true; }
                if a[i] < b[i] { return false; }
            }
            true // equal
        };

        let sub_n = |a: &[u64; #num_limbs], b: &[u64; #num_limbs]| -> [u64; #num_limbs] {
            use crate::arithmetic::sbb;
            let mut result = [0u64; #num_limbs];
            let mut borrow = 0u64;
            let mut j = 0;
            while j < #num_limbs {
                let (val, b_out) = sbb(a[j], b[j], borrow);
                result[j] = val;
                borrow = b_out;
                j += 1;
            }
            result
        };

        let result_arr = if ge_n(&r_arr, &THREE_P) {
            sub_n(&r_arr, &THREE_P)
        } else if ge_n(&r_arr, &TWO_P) {
            sub_n(&r_arr, &TWO_P)
        } else if ge_n(&r_arr, &#field::MODULUS_LIMBS) {
            sub_n(&r_arr, &#field::MODULUS_LIMBS)
        } else {
            r_arr
        };

        #field(result_arr)
    });

    quote! { #gen }
}

fn impl_mul_by_u32(_field: &syn::Ident) -> TokenStream {
    quote! {
        self.mul_by_u64(rhs as u64)
    }
}

fn impl_mul_by_i32(_field: &syn::Ident) -> TokenStream {
    quote! {
        let abs = rhs.unsigned_abs() as u64;
        let result = self.mul_by_u64(abs);
        if rhs < 0 { result.neg() } else { result }
    }
}

fn impl_mul_by_i128(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    // Multiply field element by i128 scalar.
    // Fast path (hi == 0): delegates to mul_by_u64 (Barrett reduction).
    // Slow path: 2-limb sparse widening multiply into N+2 limbs,
    //   then Barrett reduce twice (N+2 -> N+1 -> N).
    //
    // The 2-limb sparse multiply computes self.0 x [lo, hi, 0, 0]:
    //   Pass 1: self.0[i] x lo -> r[0..N+1]   (N mac ops)
    //   Pass 2: self.0[i] x hi -> r[1..N+2]   (N mac ops, shifted by 1 limb)
    // Total: 2N mac ops for the widening multiply.
    //
    // For the Barrett reduction from N+2 limbs, we split into two rounds:
    //   Round 1: reduce high part -> N+1 limbs
    //   Round 2: fold back -> N limbs
    // Each round uses the same Barrett kernel as mul_by_u64.
    //
    // Total cost: 2N mac (sparse) + ~2x Barrett reduce ~ 2N + 2x(N + 3N) = 10N ops
    // vs current: 2x Barrett mul_by_u64 + 1 field mul = 2x(N+2N) + 2N^2 = 6N + 2N^2 ~ 38 ops
    let wide_num_limbs = num_limbs * 2;
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac, macx}; };

    gen.extend(quote! {
        let negative = rhs < 0;
        let abs = rhs.unsigned_abs();
        let lo = abs as u64;
        let hi = (abs >> 64) as u64;
    });

    // Fast path: hi == 0 -> delegate to mul_by_u64
    gen.extend(quote! {
        if hi == 0 {
            let result = self.mul_by_u64(lo);
            return if negative { result.neg() } else { result };
        }
    });

    // Slow path: 2-limb sparse multiply self.0 x [lo, hi]
    // Pass 1: multiply by lo -> r[0..N+1]
    for i in 0..num_limbs {
        let r_out = fmtid!("r_{}", i);
        let r_next = fmtid!("r_{}", i + 1);
        if i == 0 {
            gen.extend(quote! { let (#r_out, carry) = macx(0, self.0[#i], lo); });
        } else if i == num_limbs - 1 {
            gen.extend(quote! { let (#r_out, #r_next) = mac(0, self.0[#i], lo, carry); });
        } else {
            gen.extend(quote! { let (#r_out, carry) = mac(0, self.0[#i], lo, carry); });
        }
    }

    // Pass 2: multiply by hi, shifted by 1 limb -> accumulate into r[1..N+2]
    for i in 0..num_limbs {
        let r_target = fmtid!("r_{}", i + 1);
        let r_next = fmtid!("r_{}", i + 2);
        if i == 0 {
            gen.extend(quote! { let (#r_target, carry) = macx(#r_target, self.0[#i], hi); });
        } else if i == num_limbs - 1 {
            gen.extend(quote! { let (#r_target, #r_next) = mac(#r_target, self.0[#i], hi, carry); });
        } else {
            gen.extend(quote! { let (#r_target, carry) = mac(#r_target, self.0[#i], hi, carry); });
        }
    }

    // Now r[0..N+2] contains the 2-limb sparse product = self.0 x (lo + hi*2^64)
    // This is (a*R) x |rhs| as an integer, and (a*R*|rhs|) mod p = (a*|rhs|)*R mod p.
    // We need to reduce N+2 limbs down to N limbs.
    //
    // Pad remaining limbs to 2N with zeros, then use Montgomery reduce.
    // Cost: N^2 mac ops for reduce. Total = 2N + N^2 = 24 for N=4.
    for i in (num_limbs + 2)..wide_num_limbs {
        let r_i = fmtid!("r_{}", i);
        gen.extend(quote! { let #r_i = 0u64; });
    }

    let r: Vec<_> = (0..wide_num_limbs).map(|i| fmtid!("r_{}", i)).collect();
    gen.extend(quote! {
        // Montgomery reduce gives (a*R*|rhs|)*R^{-1} = a*|rhs| (not Montgomery form).
        // Multiply by R2 to get back to Montgomery form: a*|rhs|*R^2 *R^{-1} = a*|rhs|*R 
        let reduced = #field::montgomery_reduce(&[#(#r),*]);
        let result = reduced.mul(&Self::R2);
        if negative { result.neg() } else { result }
    });

    quote! { #gen }
}

fn impl_from_mont(field: &syn::Ident, num_limbs: usize, inv: u64) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb, mac}; };

    for i in 0..num_limbs {
        let r_i = fmtid!("r_{}", i);
        if i == 0 {
            gen.extend(quote! { let k = self.0[0].wrapping_mul(#inv); });
        } else {
            gen.extend(quote! { let k = #r_i.wrapping_mul(#inv); });
        }

        for j in 0..num_limbs {
            let r_ij = fmtid!("r_{}", (j + i) % num_limbs);
            let r_out = select(j == 0, quote! {_}, quote! {#r_ij});
            let r_ij = select(i == 0, quote! {self.0[#j]}, quote! {#r_ij});
            let carry_in = select(j == 0, quote! {0}, quote! {#r_i});
            gen.extend(
                quote! { let (#r_out, #r_i) = mac(#r_ij, k, Self::MODULUS_LIMBS[#j], #carry_in); },
            );
        }
    }
    let ret: Vec<_> = (0..num_limbs).map(|i| fmtid!("r_{}", i)).collect();
    quote! {
        #gen
        #field([#(#ret),*]).sub(&#field(Self::MODULUS_LIMBS)).0
    }
}
