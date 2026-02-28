//! This benchmarks the basic FF operations.
//! It measures the base field `Fq` and scalar field `Fr` from the BN256 curve.
//!
//! To run this benchmark:
//!
//!     cargo bench --bench field_arith
//!
//! To run only integer multiplication benchmarks:
//!
//!     cargo bench --features std --bench field_arith -- integer

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use halo2curves::{
    bn256::{Fq, Fr},
    ff::Field,
    ff_ext::Legendre,
};
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];
fn bench_field_arithmetic<F: Field + Legendre>(c: &mut Criterion, name: &'static str) {
    let mut rng = XorShiftRng::from_seed(SEED);

    let a = <F as Field>::random(&mut rng);
    let b = <F as Field>::random(&mut rng);
    let exp = rng.next_u64();

    let mut group = c.benchmark_group(format!("{} arithmetic", name));

    group.significance_level(0.1).sample_size(1000);
    group.throughput(Throughput::Elements(1));

    group.bench_function(format!("{}_add", name), |bencher| {
        bencher.iter(|| black_box(&a).add(black_box(&b)))
    });
    group.bench_function(format!("{}_double", name), |bencher| {
        bencher.iter(|| black_box(&a).double())
    });
    group.bench_function(format!("{}_sub", name), |bencher| {
        bencher.iter(|| black_box(&a).sub(black_box(&b)))
    });
    group.bench_function(format!("{}_neg", name), |bencher| {
        bencher.iter(|| black_box(&a).neg())
    });
    group.bench_function(format!("{}_mul", name), |bencher| {
        bencher.iter(|| black_box(&a).mul(black_box(&b)))
    });
    group.bench_function(format!("{}_square", name), |bencher| {
        bencher.iter(|| black_box(&a).square())
    });
    group.bench_function(format!("{}_pow_vartime", name), |bencher| {
        bencher.iter(|| black_box(&a).pow_vartime(black_box(&[exp])))
    });
    group.bench_function(format!("{}_invert", name), |bencher| {
        bencher.iter(|| black_box(&a).invert())
    });
    group.bench_function(format!("{}_legendre", name), |bencher| {
        bencher.iter(|| black_box(&a).legendre())
    });
    group.finish()
}

/// Benchmark integer scalar multiplication for Fq.
///
/// Compares Field×Field (baseline) against Field×integer via operator overloads,
/// and also benchmarks From<u64> conversion (which uses the optimized mul_by_u64
/// internally for Montgomery encoding).
fn bench_integer_mul(c: &mut Criterion) {
    let mut rng = XorShiftRng::from_seed(SEED);
    let a = Fq::random(&mut rng);
    let b = Fq::random(&mut rng);

    // Integer scalars of various sizes
    let small_u32: u32 = 42;
    let medium_u32: u32 = 0xDEAD_BEEF;
    let large_u64: u64 = u64::MAX;
    let medium_u64: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let small_i32: i32 = -42;
    let large_i32: i32 = i32::MIN;
    let small_i128: i128 = 42;
    let large_i128: i128 = (1i128 << 100) + 7;

    let mut group = c.benchmark_group("Fq integer_mul");
    group.significance_level(0.1).sample_size(1000);
    group.throughput(Throughput::Elements(1));

    // Baseline: Field × Field (full Montgomery multiply)
    group.bench_function("field_x_field", |bencher| {
        bencher.iter(|| black_box(&a).mul(black_box(&b)))
    });

    // From<u64> conversion (uses optimized mul_by_u64 internally)
    group.bench_function("from_u64_small", |bencher| {
        bencher.iter(|| Fq::from(black_box(small_u32 as u64)))
    });
    group.bench_function("from_u64_medium", |bencher| {
        bencher.iter(|| Fq::from(black_box(medium_u64)))
    });
    group.bench_function("from_u64_large", |bencher| {
        bencher.iter(|| Fq::from(black_box(large_u64)))
    });

    // Field × u32 (operator overload)
    group.bench_function("field_x_u32_small", |bencher| {
        bencher.iter(|| black_box(a) * black_box(small_u32))
    });
    group.bench_function("field_x_u32_large", |bencher| {
        bencher.iter(|| black_box(a) * black_box(medium_u32))
    });

    // Field × u64 (operator overload)
    group.bench_function("field_x_u64_medium", |bencher| {
        bencher.iter(|| black_box(a) * black_box(medium_u64))
    });
    group.bench_function("field_x_u64_large", |bencher| {
        bencher.iter(|| black_box(a) * black_box(large_u64))
    });

    // Field × i32 (operator overload, includes sign handling)
    group.bench_function("field_x_i32_neg", |bencher| {
        bencher.iter(|| black_box(a) * black_box(small_i32))
    });
    group.bench_function("field_x_i32_min", |bencher| {
        bencher.iter(|| black_box(a) * black_box(large_i32))
    });

    // Field × i128 (operator overload)
    group.bench_function("field_x_i128_small", |bencher| {
        bencher.iter(|| black_box(a) * black_box(small_i128))
    });
    group.bench_function("field_x_i128_large", |bencher| {
        bencher.iter(|| black_box(a) * black_box(large_i128))
    });

    // Explicit baseline comparison: manual From + mul (what the operator does)
    group.bench_function("manual_from_then_mul", |bencher| {
        bencher.iter(|| {
            let rhs = Fq::from(black_box(medium_u64));
            black_box(&a).mul(&rhs)
        })
    });

    group.finish();
}

fn bench_bn256_base_field(c: &mut Criterion) {
    bench_field_arithmetic::<Fq>(c, "Fq")
}
fn bench_bn256_scalar_field(c: &mut Criterion) {
    bench_field_arithmetic::<Fr>(c, "Fr")
}

criterion_group!(
    benches,
    bench_bn256_base_field,
    bench_bn256_scalar_field,
    bench_integer_mul,
);
criterion_main!(benches);
