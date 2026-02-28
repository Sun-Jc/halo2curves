#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::convert::TryInto;
use halo2derive::impl_field;
use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::ff_ext::ExtField;

impl_field!(
    bn256_base,
    Fq,
    modulus = "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
    mul_gen = "3",
    zeta = "30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48",
    from_uniform = [64, 48],
    endian = "little",
);

crate::extend_field_legendre!(Fq);
crate::impl_binops_calls!(Fq);
crate::impl_binops_additive!(Fq, Fq);
crate::impl_binops_multiplicative!(Fq, Fq);
crate::field_bits!(Fq);
crate::serialize_deserialize_primefield!(Fq);
crate::impl_from_u64!(Fq);
crate::impl_from_bool!(Fq);
crate::impl_integer_mul_ops!(Fq);

use ff::Field;
const NEGATIVE_ONE: Fq = Fq::ZERO.sub_const(&Fq::ONE);
impl ExtField for Fq {
    const NON_RESIDUE: Self = NEGATIVE_ONE;
    fn mul_by_nonresidue(&self) -> Self {
        self.neg()
    }
    fn frobenius_map(&mut self, _: usize) {}
}

#[cfg(test)]
mod test {
    use super::Fq;
    use crate::{
        arith_test, constants_test, from_uniform_bytes_test, legendre_test, serde_test, test,
    };

    constants_test!(Fq);

    arith_test!(Fq);
    legendre_test!(Fq);
    test!(arith, Fq, sqrt_test, 1000);

    serde_test!(Fq PrimeFieldBits);
    from_uniform_bytes_test!(Fq, 1000, L 64, L 48);

    // ── Integer multiplication tests ─────────────────────────────────

    use ff::Field;
    use rand_core::{RngCore, SeedableRng};
    use rand_xorshift::XorShiftRng;

    const INT_TEST_SEED: [u8; 16] = [
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
        0xbc, 0xe5,
    ];

    // Helper: compute a * b (as i128) using field arithmetic as reference
    fn ref_mul_i128(a: Fq, b: i128) -> Fq {
        let abs = b.unsigned_abs();
        let lo = abs as u64;
        let hi = (abs >> 64) as u64;
        // 2^64 in the field = u64::MAX + 1
        let two_64 = Fq::from(u64::MAX) + Fq::ONE;
        let b_field = Fq::from(lo) + Fq::from(hi) * two_64;
        let result = a * b_field;
        if b < 0 { -result } else { result }
    }

    #[test]
    fn test_mul_by_u64_via_from() {
        // mul_by_u64 is a low-level Montgomery helper. It is correct when used
        // via From<u64>: `Self::R.mul_by_u64(val)` produces `Fq::from(val)`.
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);

        for _ in 0..1000 {
            let val: u64 = rng.next_u64();
            // From<u64> uses mul_by_u64 internally
            let from_result = Fq::from(val);
            // Reference: build from repeated addition
            // For small values, we can cross-check directly
            if val < 1000 {
                let mut acc = Fq::ZERO;
                for _ in 0..val {
                    acc = acc + Fq::ONE;
                }
                assert_eq!(from_result, acc, "From<u64>({}) via addition", val);
            }
            // Always check: from(val) * from(1) == from(val)
            assert_eq!(from_result * Fq::ONE, from_result);
        }

        // Edge cases for From<u64>
        assert_eq!(Fq::from(0u64), Fq::ZERO, "From(0) == ZERO");
        assert_eq!(Fq::from(1u64), Fq::ONE, "From(1) == ONE");
        assert_eq!(Fq::from(2u64), Fq::ONE + Fq::ONE, "From(2) == ONE+ONE");
        assert_eq!(
            Fq::from(u64::MAX),
            Fq::from(u64::MAX - 1) + Fq::ONE,
            "From(MAX) == From(MAX-1) + 1"
        );
    }

    #[test]
    fn test_mul_u64_operator() {
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);

        for _ in 0..1000 {
            let a = Fq::random(&mut rng);
            let b: u64 = rng.next_u64();
            let result = a * b;
            let expected = a * Fq::from(b);
            assert_eq!(result, expected, "Fq * u64 failed for b={}", b);
        }

        // Edge cases
        let a = Fq::random(&mut rng);
        assert_eq!(a * 0u64, Fq::ZERO, "a * 0u64 should be zero");
        assert_eq!(a * 1u64, a, "a * 1u64 should be identity");
        assert_eq!(a * 2u64, a.double(), "a * 2u64 should equal double");
        assert_eq!(a * u64::MAX, a * Fq::from(u64::MAX), "a * u64::MAX");
        assert_eq!(Fq::ZERO * 42u64, Fq::ZERO, "ZERO * u64 should be ZERO");
        assert_eq!(Fq::ONE * 42u64, Fq::from(42u64), "ONE * 42u64 = From(42)");

        // MulAssign
        let mut a_copy = a;
        a_copy *= 5u64;
        assert_eq!(a_copy, a * 5u64, "Fq *= u64");

        // &Fq * u64
        assert_eq!(&a * 5u64, a * 5u64, "&Fq * u64");
    }

    #[test]
    fn test_mul_u32_operator() {
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);

        for _ in 0..1000 {
            let a = Fq::random(&mut rng);
            let b: u32 = rng.next_u32();
            let result = a * b;
            let expected = a * Fq::from(b as u64);
            assert_eq!(result, expected, "Fq * u32 failed for b={}", b);
        }

        // Edge cases
        let a = Fq::random(&mut rng);
        assert_eq!(a * 0u32, Fq::ZERO, "a * 0u32 should be zero");
        assert_eq!(a * 1u32, a, "a * 1u32 should be identity");
        assert_eq!(a * u32::MAX, a * Fq::from(u32::MAX as u64), "a * u32::MAX");

        // MulAssign
        let mut a_copy = a;
        a_copy *= 5u32;
        assert_eq!(a_copy, a * 5u32, "Fq *= u32");

        // &Fq * u32
        assert_eq!(&a * 5u32, a * 5u32, "&Fq * u32");
    }

    #[test]
    fn test_mul_i32_operator() {
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);

        for _ in 0..1000 {
            let a = Fq::random(&mut rng);
            let b: i32 = rng.next_u32() as i32;
            let result = a * b;
            let abs = b.unsigned_abs() as u64;
            let expected = if b < 0 {
                -(a * Fq::from(abs))
            } else {
                a * Fq::from(abs)
            };
            assert_eq!(result, expected, "Fq * i32 failed for b={}", b);
        }

        // Edge cases
        let a = Fq::random(&mut rng);
        assert_eq!(a * 0i32, Fq::ZERO, "a * 0i32 should be zero");
        assert_eq!(a * 1i32, a, "a * 1i32 should be identity");
        assert_eq!(a * (-1i32), -a, "a * (-1i32) should negate");
        assert_eq!(
            a * i32::MAX,
            a * Fq::from(i32::MAX as u64),
            "a * i32::MAX"
        );
        assert_eq!(
            a * i32::MIN,
            -(a * Fq::from(i32::MIN.unsigned_abs() as u64)),
            "a * i32::MIN"
        );

        // MulAssign
        let mut a_copy = a;
        a_copy *= -5i32;
        assert_eq!(a_copy, a * (-5i32), "Fq *= i32");

        // &Fq * i32
        assert_eq!(&a * (-5i32), a * (-5i32), "&Fq * i32");
    }

    #[test]
    fn test_mul_i128_operator() {
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);

        for _ in 0..1000 {
            let a = Fq::random(&mut rng);
            // Generate i128 values spanning different sizes
            let lo = rng.next_u64() as i128;
            let hi = (rng.next_u64() as i128) << 64;
            let sign = if rng.next_u32() & 1 == 0 { 1i128 } else { -1i128 };
            let b = sign * (lo | hi).abs();

            let result = a * b;
            let expected = ref_mul_i128(a, b);
            assert_eq!(result, expected, "Fq * i128 failed for b={}", b);
        }

        // Edge cases
        let a = Fq::random(&mut rng);
        assert_eq!(a * 0i128, Fq::ZERO, "a * 0i128 should be zero");
        assert_eq!(a * 1i128, a, "a * 1i128 should be identity");
        assert_eq!(a * (-1i128), -a, "a * (-1i128) should negate");

        // Small values should match u64 path
        assert_eq!(a * 42i128, a * 42u64, "small i128 == u64");
        assert_eq!(a * (-42i128), -(a * 42u64), "small negative i128");

        // Large value > 2^64
        let big: i128 = (1i128 << 100) + 7;
        assert_eq!(a * big, ref_mul_i128(a, big), "large i128 > 2^64");

        // i128::MIN
        assert_eq!(
            a * i128::MIN,
            ref_mul_i128(a, i128::MIN),
            "a * i128::MIN"
        );

        // i128::MAX
        assert_eq!(
            a * i128::MAX,
            ref_mul_i128(a, i128::MAX),
            "a * i128::MAX"
        );

        // MulAssign
        let mut a_copy = a;
        a_copy *= -5i128;
        assert_eq!(a_copy, a * (-5i128), "Fq *= i128");

        // &Fq * i128
        assert_eq!(&a * big, a * big, "&Fq * i128");
    }

    #[test]
    fn test_integer_mul_consistency() {
        // Cross-check: all integer types give the same result for the same value
        let mut rng = XorShiftRng::from_seed(INT_TEST_SEED);
        let a = Fq::random(&mut rng);

        // Positive values representable in all types
        assert_eq!(a * 42u32, a * 42u64, "u32 vs u64");
        assert_eq!(a * 42u64, a * 42i128, "u64 vs i128");
        assert_eq!(a * 42i32, a * 42i128, "i32 vs i128");

        // Negative values (i32 vs i128)
        assert_eq!(a * (-42i32), a * (-42i128), "i32 vs i128 negative");

        // Powers of 2 via u64 operator
        for shift in 0..63 {
            let val: u64 = 1u64 << shift;
            let via_op = a * val;
            let via_field = a * Fq::from(val);
            assert_eq!(
                via_op, via_field,
                "a * (1 << {}) mismatch",
                shift
            );
        }
    }

    #[test]
    fn test_from_u64_still_correct() {
        // Verify From<u64> still works correctly (it now uses mul_by_u64 internally)
        assert_eq!(Fq::from(0u64), Fq::ZERO, "From<u64>(0) should be zero");
        assert_eq!(Fq::from(1u64), Fq::ONE, "From<u64>(1) should be one");

        let two = Fq::ONE + Fq::ONE;
        assert_eq!(Fq::from(2u64), two, "From<u64>(2) should be 1+1");

        let hundred = Fq::from(100u64);
        let also_hundred = Fq::from(10u64) * Fq::from(10u64);
        assert_eq!(hundred, also_hundred, "From<u64>(100) = From(10)*From(10)");

        assert_eq!(
            Fq::from(u64::MAX),
            Fq::from(u64::MAX - 1) + Fq::ONE,
            "From(u64::MAX) = From(u64::MAX-1) + 1"
        );

        // From<bool> also uses mul_by_u64 internally
        assert_eq!(Fq::from(false), Fq::ZERO, "From(false) == ZERO");
        assert_eq!(Fq::from(true), Fq::ONE, "From(true) == ONE");
    }
}
