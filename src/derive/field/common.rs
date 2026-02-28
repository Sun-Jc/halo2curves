#[macro_export]
macro_rules! field_bits {
    ($field:ident) => {
        #[cfg(feature = "bits")]
        #[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
        impl ff::PrimeFieldBits for $field {
            #[cfg(target_pointer_width = "64")]
            type ReprBits = [u64; Self::NUM_LIMBS];
            #[cfg(not(target_pointer_width = "64"))]
            type ReprBits = [u32; Self::NUM_LIMBS * 2];

            fn to_le_bits(&self) -> ff::FieldBits<Self::ReprBits> {
                use ff::PrimeField;
                let bytes: [u8; Self::SIZE] = self.to_repr().into();

                #[cfg(target_pointer_width = "64")]
                const STEP: usize = 8;
                #[cfg(not(target_pointer_width = "64"))]
                const STEP: usize = 4;

                let limbs = (0..Self::NUM_LIMBS * 8 / STEP)
                    .map(|off| {
                        #[cfg(target_pointer_width = "64")]
                        let limb = u64::from_le_bytes(
                            bytes[off * STEP..(off + 1) * STEP].try_into().unwrap(),
                        );
                        #[cfg(not(target_pointer_width = "64"))]
                        let limb = u32::from_le_bytes(
                            bytes[off * STEP..(off + 1) * STEP].try_into().unwrap(),
                        );

                        limb
                    })
                    .collect::<Vec<_>>();

                ff::FieldBits::new(limbs.try_into().unwrap())
            }

            fn char_le_bits() -> ff::FieldBits<Self::ReprBits> {
                #[cfg(target_pointer_width = "64")]
                let bits = ff::FieldBits::new(Self::MODULUS_LIMBS);
                #[cfg(not(target_pointer_width = "64"))]
                let bits = ff::FieldBits::new(Self::MODULUS_LIMBS_32);

                bits
            }
        }
    };
}

#[macro_export]
macro_rules! impl_from_u64 {
    ($field:ident) => {
        impl From<u64> for $field {
            fn from(val: u64) -> $field {
                // mul_by_u64 uses Barrett reduction: self.0 x val mod p.
                // For R (Montgomery one, stores R in limbs): R x val mod p = val*R mod p 
                Self::R.mul_by_u64(val)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_from_bool {
    ($field:ident) => {
        impl From<bool> for $field {
            fn from(val: bool) -> $field {
                Self::R.mul_by_u64(u64::from(val))
            }
        }
    };
}

/// A macro to help define serialization and deserialization for prime field
/// implementations that use `$field::Repr`` representations. This assumes the
/// concerned type implements PrimeField (for from_repr, to_repr).
#[macro_export]
macro_rules! serialize_deserialize_primefield {
    ($field:ident) => {
        #[cfg(feature = "derive_serde")]
        impl<'de> ::serde::Deserialize<'de> for $field {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                use ::serde::de::Error as _;
                let bytes = if deserializer.is_human_readable() {
                    hex::serde::deserialize(deserializer)?
                } else {
                    ::serde_arrays::deserialize::<_, u8, { $field::SIZE }>(deserializer)?
                };
                use ff::PrimeField;
                Option::from(Self::from_repr(bytes.into())).ok_or_else(|| {
                    D::Error::custom("deserialized bytes don't encode a valid field element")
                })
            }
        }
        #[cfg(feature = "derive_serde")]
        impl ::serde::Serialize for $field {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use ff::PrimeField;
                if serializer.is_human_readable() {
                    hex::serde::serialize(self.to_repr().as_ref(), serializer)
                } else {
                    let bytes: [u8; $field::SIZE] = self.to_repr().into();
                    ::serde_arrays::serialize(&bytes, serializer)
                }
            }
        }
    };
}

/// Implements `Mul` and `MulAssign` for integer scalar types (u32, u64, i32, i128).
///
/// `mul_by_u64` uses Barrett reduction: sparse multiply (N macs) + Barrett reduce (~2N ops).
/// This is faster than From(rhs) + field_mul which costs N+N^2 + N^2 = 2N^2+N mac ops.
/// For signed types, multiplies by absolute value and conditionally negates.
#[macro_export]
macro_rules! impl_integer_mul_ops {
    ($field:ident) => {
        // --- Mul<u64> ---
        impl ::core::ops::Mul<u64> for $field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: u64) -> $field {
                self.mul_by_u64(rhs)
            }
        }
        impl ::core::ops::Mul<u64> for &$field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: u64) -> $field {
                self.mul_by_u64(rhs)
            }
        }
        impl ::core::ops::MulAssign<u64> for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: u64) {
                *self = self.mul_by_u64(rhs);
            }
        }

        // --- Mul<u32> ---
        impl ::core::ops::Mul<u32> for $field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: u32) -> $field {
                self.mul_by_u32(rhs)
            }
        }
        impl ::core::ops::Mul<u32> for &$field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: u32) -> $field {
                self.mul_by_u32(rhs)
            }
        }
        impl ::core::ops::MulAssign<u32> for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: u32) {
                *self = self.mul_by_u32(rhs);
            }
        }

        // --- Mul<i32> ---
        impl ::core::ops::Mul<i32> for $field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: i32) -> $field {
                self.mul_by_i32(rhs)
            }
        }
        impl ::core::ops::Mul<i32> for &$field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: i32) -> $field {
                self.mul_by_i32(rhs)
            }
        }
        impl ::core::ops::MulAssign<i32> for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: i32) {
                *self = self.mul_by_i32(rhs);
            }
        }

        // --- Mul<i128> ---
        impl ::core::ops::Mul<i128> for $field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: i128) -> $field {
                self.mul_by_i128(rhs)
            }
        }
        impl ::core::ops::Mul<i128> for &$field {
            type Output = $field;
            #[inline]
            fn mul(self, rhs: i128) -> $field {
                self.mul_by_i128(rhs)
            }
        }
        impl ::core::ops::MulAssign<i128> for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: i128) {
                *self = self.mul_by_i128(rhs);
            }
        }
    };
}
