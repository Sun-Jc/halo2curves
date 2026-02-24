//! Cross-library MSM benchmark: arkworks vs halo2curves (msm_best vs msm_best2)
//!
//! Run: cargo bench --bench msm_compare

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use group::prime::PrimeCurveAffine;
use group::Curve;
use std::time::Duration;

// ── Shared random bytes for deterministic generation ─────────────────────
// We generate random field elements independently in each library from a
// shared seed so that the *scalars* are the same bit-length distribution
// even though the representations differ.

const SAMPLE_SIZE: usize = 10;

/// Generate arkworks BN254 test data (G1Affine bases + Fr scalars as BigInt)
fn ark_generate(
    k: u32,
) -> (
    Vec<ark_bn254::G1Affine>,
    Vec<<ark_bn254::Fr as ark_ff::PrimeField>::BigInt>,
) {
    use ark_ec::CurveGroup;
    use ark_ff::{PrimeField, UniformRand};

    let n = 1usize << k;
    let mut rng = ark_std::test_rng(); // deterministic

    let bases_proj: Vec<ark_bn254::G1Projective> =
        (0..n).map(|_| ark_bn254::G1Projective::rand(&mut rng)).collect();
    let bases = ark_bn254::G1Projective::normalize_batch(&bases_proj);

    let scalars: Vec<_> = (0..n)
        .map(|_| ark_bn254::Fr::rand(&mut rng).into_bigint())
        .collect();

    (bases, scalars)
}

/// Generate halo2curves BN256(=BN254) test data
fn halo2_generate(
    k: u32,
) -> (
    Vec<halo2curves::bn256::G1Affine>,
    Vec<halo2curves::bn256::Fr>,
) {
    use ff::Field;
    use group::Group;
    use rand_core::OsRng;
    use rayon::prelude::*;

    let n = 1usize << k;

    let points_proj: Vec<halo2curves::bn256::G1> = (0..n)
        .into_par_iter()
        .map(|_| halo2curves::bn256::G1::random(OsRng))
        .collect();

    let mut bases = vec![halo2curves::bn256::G1Affine::identity(); n];
    halo2curves::bn256::G1::batch_normalize(&points_proj, &mut bases);

    let scalars: Vec<halo2curves::bn256::Fr> = (0..n)
        .into_par_iter()
        .map(|_| halo2curves::bn256::Fr::random(OsRng))
        .collect();

    (bases, scalars)
}

fn bench_msm(c: &mut Criterion) {

    let mut group = c.benchmark_group("msm_compare");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(3));

    // Test sizes: 2^k
    let sizes: Vec<u32> = vec![14, 16, 18, 20, 24, 26];

    let max_k = *sizes.iter().max().unwrap();

    // Pre-generate data at max size, then slice for smaller sizes
    println!("Generating arkworks test data (2^{max_k})...");
    let (ark_bases, ark_scalars) = ark_generate(max_k);
    println!("Generating halo2curves test data (2^{max_k})...");
    let (h2_bases, h2_scalars) = halo2_generate(max_k);
    println!("Data generation complete.\n");

    for &k in &sizes {
        let n = 1usize << k;

        // ── arkworks MSM ────────────────────────────────────────────
        {
            let bases = &ark_bases[..n];
            let scalars = &ark_scalars[..n];

            group.bench_function(BenchmarkId::new("arkworks", k), |b| {
                use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
                b.iter(|| {
                    <ark_bn254::G1Projective as VariableBaseMSM>::msm_bigint(bases, scalars)
                });
            });
        }

        // ── halo2curves msm_best ────────────────────────────────────
        {
            let bases = &h2_bases[..n];
            let scalars = &h2_scalars[..n];

            group.bench_function(BenchmarkId::new("halo2_msm_best", k), |b| {
                b.iter(|| halo2curves::msm::msm_best(scalars, bases));
            });
        }

        // ── halo2curves msm_best2 (GLV) ─────────────────────────────
        {
            let bases = &h2_bases[..n];
            let scalars = &h2_scalars[..n];

            group.bench_function(BenchmarkId::new("halo2_msm_best2", k), |b| {
                b.iter(|| halo2curves::msm::msm_best2(scalars, bases));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
