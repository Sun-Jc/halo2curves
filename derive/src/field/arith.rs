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

pub(crate) fn impl_arith(field: &syn::Ident, num_limbs: usize, inv: u64) -> TokenStream {
    impl_arith_inner(field, num_limbs, inv, /* modulus_fits_in_capacity */ false)
}

/// Same as `impl_arith` but tells the codegen that the modulus uses strictly
/// fewer than `num_limbs * 64` bits. Under that assumption `a + b < 2 * p` is
/// guaranteed to fit in the limbs without overflowing the high limb, so the
/// add path can drop the awkward "combine the original add carry with the
/// try-subtract-modulus borrow" step. That step was the source of the
/// `MRS NZCV` / `MSR NZCV` flag-shuffle that LLVM emits on aarch64 and was
/// making `Fr::add` ~1.5x slower than `Fr::sub` for primes such as BN254 Fr.
pub(crate) fn impl_arith_capacity_lt_limbs(
    field: &syn::Ident,
    num_limbs: usize,
    inv: u64,
) -> TokenStream {
    impl_arith_inner(field, num_limbs, inv, /* modulus_fits_in_capacity */ true)
}

fn impl_arith_inner(
    field: &syn::Ident,
    num_limbs: usize,
    inv: u64,
    modulus_fits_in_capacity: bool,
) -> TokenStream {
    let impl_add = impl_add(field, num_limbs, modulus_fits_in_capacity);
    let impl_sub = impl_sub(field, num_limbs);
    let impl_neg = impl_neg(field, num_limbs);
    let impl_mont = impl_mont(field, num_limbs, inv);
    let impl_from_mont = impl_from_mont(field, num_limbs, inv);
    let impl_mul = impl_mul(field, num_limbs, false);
    let impl_square = impl_square(field, num_limbs);
    let wide_num_limbs = num_limbs * 2;

    // For 4-limb fields (256-bit) where the modulus fits in 255 bits, we
    // can emit a hand-written aarch64 inline asm `add`/`sub` that mirrors
    // the x86 hand-asm shape (in halo2derive's asm/limb4.rs). LLVM's Rust
    // fallback codegen for these routines spends most of its instructions
    // re-materializing the modulus as immediates; the asm path forces a
    // clean adcs/sbcs chain plus a csel/and-based correction, with the
    // modulus loaded once via adrp+ldp.
    let emit_aarch64 = num_limbs == 4 && modulus_fits_in_capacity;
    let m_ident = quote::format_ident!("__MODULUS_LIMBS_S_{}", field);

    let aarch64_add = if emit_aarch64 {
        quote! {
            #[inline(always)]
            #[cfg(target_arch = "aarch64")]
            pub fn add(&self, rhs: &Self) -> Self {
                // Load limbs as values to avoid forcing a stack-spill of
                // `self`/`rhs`. The asm receives the 4-limb operands as
                // four input regs plus four input modulus regs, and
                // returns four output regs.
                let a0 = self.0[0]; let a1 = self.0[1];
                let a2 = self.0[2]; let a3 = self.0[3];
                let b0 = rhs.0[0];  let b1 = rhs.0[1];
                let b2 = rhs.0[2];  let b3 = rhs.0[3];
                let m = #m_ident;
                let m0 = m[0]; let m1 = m[1]; let m2 = m[2]; let m3 = m[3];
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    core::arch::asm!(
                        "adds {a0}, {a0}, {b0}",
                        "adcs {a1}, {a1}, {b1}",
                        "adcs {a2}, {a2}, {b2}",
                        "adc  {a3}, {a3}, {b3}",
                        "subs {b0}, {a0}, {m0}",
                        "sbcs {b1}, {a1}, {m1}",
                        "sbcs {b2}, {a2}, {m2}",
                        "sbcs {b3}, {a3}, {m3}",
                        "csel {a0}, {b0}, {a0}, hs",
                        "csel {a1}, {b1}, {a1}, hs",
                        "csel {a2}, {b2}, {a2}, hs",
                        "csel {a3}, {b3}, {a3}, hs",
                        a0 = inout(reg) a0 => r0,
                        a1 = inout(reg) a1 => r1,
                        a2 = inout(reg) a2 => r2,
                        a3 = inout(reg) a3 => r3,
                        b0 = inout(reg) b0 => _,
                        b1 = inout(reg) b1 => _,
                        b2 = inout(reg) b2 => _,
                        b3 = inout(reg) b3 => _,
                        m0 = in(reg) m0,
                        m1 = in(reg) m1,
                        m2 = in(reg) m2,
                        m3 = in(reg) m3,
                        options(pure, nomem, nostack)
                    );
                }
                Self([r0, r1, r2, r3])
            }
        }
    } else {
        quote! {}
    };

    let aarch64_sub = if emit_aarch64 {
        quote! {
            #[inline(always)]
            #[cfg(target_arch = "aarch64")]
            pub fn sub(&self, rhs: &Self) -> Self {
                let a0 = self.0[0]; let a1 = self.0[1];
                let a2 = self.0[2]; let a3 = self.0[3];
                let b0 = rhs.0[0];  let b1 = rhs.0[1];
                let b2 = rhs.0[2];  let b3 = rhs.0[3];
                let m = #m_ident;
                let m0 = m[0]; let m1 = m[1]; let m2 = m[2]; let m3 = m[3];
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    core::arch::asm!(
                        "subs {a0}, {a0}, {b0}",
                        "sbcs {a1}, {a1}, {b1}",
                        "sbcs {a2}, {a2}, {b2}",
                        "sbcs {a3}, {a3}, {b3}",
                        "csetm {b0}, lo",
                        "and {m0}, {m0}, {b0}",
                        "and {m1}, {m1}, {b0}",
                        "and {m2}, {m2}, {b0}",
                        "and {m3}, {m3}, {b0}",
                        "adds {a0}, {a0}, {m0}",
                        "adcs {a1}, {a1}, {m1}",
                        "adcs {a2}, {a2}, {m2}",
                        "adc  {a3}, {a3}, {m3}",
                        a0 = inout(reg) a0 => r0,
                        a1 = inout(reg) a1 => r1,
                        a2 = inout(reg) a2 => r2,
                        a3 = inout(reg) a3 => r3,
                        b0 = inout(reg) b0 => _,
                        b1 = inout(reg) b1 => _,
                        b2 = inout(reg) b2 => _,
                        b3 = inout(reg) b3 => _,
                        m0 = inout(reg) m0 => _,
                        m1 = inout(reg) m1 => _,
                        m2 = inout(reg) m2 => _,
                        m3 = inout(reg) m3 => _,
                        options(pure, nomem, nostack)
                    );
                }
                Self([r0, r1, r2, r3])
            }
        }
    } else {
        quote! {}
    };

    // Generic fallback `add`/`sub`. When the aarch64 fast path is emitted,
    // gate the fallback off on aarch64 to avoid duplicate definitions.
    let fallback_add_cfg = if emit_aarch64 {
        quote! { #[cfg(not(target_arch = "aarch64"))] }
    } else {
        quote! {}
    };
    let fallback_sub_cfg = fallback_add_cfg.clone();

    quote::quote! {
        impl #field {
            #aarch64_add
            #aarch64_sub

            #fallback_add_cfg
            #[inline(always)]
            pub fn add(&self, rhs: &Self) -> Self {
                #impl_add
            }

            #[inline]
            pub fn double(&self) -> Self {
                self.add(self)
            }

            #fallback_sub_cfg
            #[inline(always)]
            pub fn sub(&self, rhs: &Self) -> Self {
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
    let impl_sub = impl_sub_const(field, num_limbs);
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

/// Generate sub/neg/mul/square/from_mont/montgomery_reduce using generic Rust.
/// Used when asm handles add/double but mul/sub need generic implementation
/// (e.g., for full-width 256-bit moduli where the asm Montgomery mul overflows).
pub(crate) fn impl_arith_mul_only(field: &syn::Ident, num_limbs: usize, inv: u64) -> TokenStream {
    let impl_sub = impl_sub(field, num_limbs);
    let impl_neg = impl_neg(field, num_limbs);
    let impl_mont = impl_mont(field, num_limbs, inv);
    let impl_from_mont = impl_from_mont(field, num_limbs, inv);
    let impl_mul = impl_mul(field, num_limbs, false);
    let impl_square = impl_square(field, num_limbs);
    let wide_num_limbs = num_limbs * 2;
    quote::quote! {
        impl #field {
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

fn impl_add(
    field: &syn::Ident,
    num_limbs: usize,
    modulus_fits_in_capacity: bool,
) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb}; };

    // First phase: compute a + b limb-by-limb. When the modulus fits in
    // num_limbs * 64 - 1 bits we know a, b < p < 2^(num_limbs*64 - 1), so
    // a + b < 2p < 2^(num_limbs*64) -- i.e. the high-limb add carry is
    // always 0 and we can drop the carry-out bookkeeping. This shortens the
    // dependency chain considerably; on aarch64 it removes the MRS/MSR NZCV
    // pair that LLVM otherwise emits to thread the carry across phases.
    if modulus_fits_in_capacity {
        // Phase 1: a + b. Drop the carry-out on the high limb; the modulus
        // capacity bound guarantees no overflow.
        (0..num_limbs).for_each(|i| {
            let carry_in = select(i == 0, quote! {0}, quote! {carry});
            let carry_out =
                select(i == num_limbs - 1, quote! {_}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, #carry_out) = adc(self.0[#i], rhs.0[#i], #carry_in);
            });
        });

        // Hoist the modulus through a `static` array so LLVM emits an
        // `adrp + ldr` data-segment load (2 instructions per limb pair)
        // instead of materializing each limb as `mov + 3 × movk`
        // (4 instructions per limb).
        gen.extend(quote! {
            let _m: &'static [u64; #num_limbs] = Self::modulus_static_ref();
        });

        // Phase 2: try (a+b) - p, mirror impl_sub's shape so LLVM picks
        // the same clean codegen template (subs/ngc on aarch64).
        (0..num_limbs).for_each(|i| {
            let borrow = select(i == 0, quote! {0}, quote! {borrow});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, borrow) = sbb(#d_i, _m[#i], #borrow);
            });
        });

        // Phase 3: conditional add of p, mask-and-adc tail.
        (0..num_limbs).for_each(|i| {
            let carry_in = select(i == 0, quote! {0}, quote! {carry});
            let carry_out =
                select(i == num_limbs - 1, quote! {_}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, #carry_out) =
                    adc(#d_i, _m[#i] & borrow, #carry_in);
            });
        });
    } else {
        // General path: modulus uses every available bit, so a + b might
        // overflow the high limb. We must combine the original add carry
        // with the try-subtract-modulus borrow before deciding whether to
        // add p back. This is the historical implementation.
        (0..num_limbs).for_each(|i| {
            let carry = select(i == 0, quote! {0}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, carry) = adc(self.0[#i], rhs.0[#i], #carry);
            });
        });

        (0..num_limbs).for_each(|i| {
            let borrow = select(i == 0, quote! {0}, quote! {borrow});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, borrow) = sbb(#d_i, Self::MODULUS_LIMBS[#i], #borrow);
            });
        });
        gen.extend(quote! { let (_, borrow) = sbb(carry, 0, borrow); });

        (0..num_limbs).for_each(|i| {
            let carry_in = select(i == 0, quote! {0}, quote! {carry});
            let carry_out =
                select(i == num_limbs - 1, quote! {_}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(quote! {
                let (#d_i, #carry_out) =
                    adc(#d_i, Self::MODULUS_LIMBS[#i] & borrow, #carry_in);
            });
        });
    }

    let ret: Vec<_> = (0..num_limbs).map(|i| fmtid!("d_{}", i)).collect();

    quote! {
        #gen
        #field([#(#ret),*])
    }
}

fn impl_sub(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    impl_sub_inner(field, num_limbs, /* use_static_modulus */ true)
}

/// Const-fn variant: uses `Self::MODULUS_LIMBS` (a `const`) so the body
/// remains usable from `const fn sub_const`. Pays the immediate-
/// materialization cost LLVM-side, but that's acceptable for the
/// const-eval path where the function isn't on a hot path.
fn impl_sub_const(field: &syn::Ident, num_limbs: usize) -> TokenStream {
    impl_sub_inner(field, num_limbs, /* use_static_modulus */ false)
}

fn impl_sub_inner(
    field: &syn::Ident,
    num_limbs: usize,
    use_static_modulus: bool,
) -> TokenStream {
    let mut gen = quote! { use crate::arithmetic::{adc, sbb}; };

    (0..num_limbs).for_each(|i| {
        let borrow = select(i == 0, quote! {0}, quote! {borrow});
        let d_i = fmtid!("d_{}", i);
        gen.extend(quote! { let (#d_i, borrow) = sbb(self.0[#i], rhs.0[#i], #borrow); });
    });

    if use_static_modulus {
        gen.extend(quote! {
            let _m: &'static [u64; #num_limbs] = Self::modulus_static_ref();
        });
        (0..num_limbs).for_each(|i| {
            let carry_in = select(i == 0, quote! {0}, quote! {carry});
            let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(
                quote! { let (#d_i, #carry_out) = adc(#d_i, _m[#i] & borrow, #carry_in); },
            );
        });
    } else {
        (0..num_limbs).for_each(|i| {
            let carry_in = select(i == 0, quote! {0}, quote! {carry});
            let carry_out = select(i == num_limbs - 1, quote! {_}, quote! {carry});
            let d_i = fmtid!("d_{}", i);
            gen.extend(
                quote! { let (#d_i, #carry_out) = adc(#d_i, Self::MODULUS_LIMBS[#i] & borrow, #carry_in); },
            );
        });
    }

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
        #field([#(#ret),*]).sub_const(&#field(Self::MODULUS_LIMBS)).0
    }
}
