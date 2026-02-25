//! A/B test: opt/msm-fast vs upstream main for halo2curves msm_best.
//!
//! Both libraries are pulled from remote git repositories:
//! - Optimized: https://github.com/Sun-Jc/halo2curves (branch: opt/msm-fast)
//! - Baseline:  https://github.com/privacy-scaling-explorations/halo2curves (branch: main)
//!
//! This test:
//! 1. Generates shared raw scalar bytes and point bytes
//! 2. Parses them into both library's types
//! 3. Runs msm_best in both and compares results byte-for-byte
//! 4. Measures and prints timing for each
//!
//! Run: cargo run --release

use std::time::Instant;

use rayon::prelude::*;

fn main() {
    println!("=== MSM A/B Test: opt/msm-fast vs upstream main ===\n");

    let sizes: Vec<usize> = vec![2, 10, 15, 20, 22, 24, 26];

    let max_k = *sizes.iter().max().unwrap();
    let max_n: usize = 1 << max_k;

    // ── Step 1: Generate random scalars and points, serialize to raw bytes
    //    so both versions see identical inputs.
    println!("Generating {} random scalars and points...", max_n);
    let gen_start = Instant::now();

    let scalar_bytes: Vec<[u8; 32]> = {
        use ff::{Field, PrimeField};
        use halo2curves::bn256::Fr;
        use rand_core::OsRng;
        (0..max_n)
            .into_par_iter()
            .map(|_| {
                let s = Fr::random(OsRng);
                let repr = s.to_repr();
                let mut buf = [0u8; 32];
                buf.copy_from_slice(repr.as_ref());
                buf
            })
            .collect()
    };

    let point_bytes: Vec<[u8; 64]> = {
        use group::{Curve, Group};
        use group::prime::PrimeCurveAffine;
        use group::UncompressedEncoding;
        use halo2curves::bn256::{G1, G1Affine};
        use rand_core::OsRng;

        let projs: Vec<G1> = (0..max_n)
            .into_par_iter()
            .map(|_| G1::random(OsRng))
            .collect();
        let mut affs = vec![G1Affine::identity(); max_n];
        G1::batch_normalize(&projs, &mut affs);

        affs.iter()
            .map(|p| {
                let unc = p.to_uncompressed();
                let bytes: &[u8] = unc.as_ref();
                let mut buf = [0u8; 64];
                buf.copy_from_slice(&bytes[..64]);
                buf
            })
            .collect()
    };

    // ── Step 2: Parse bytes into both versions' types
    let opt_scalars: Vec<halo2curves::bn256::Fr> = {
        use ff::PrimeField;
        scalar_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves::bn256::Fr as PrimeField>::Repr::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves::bn256::Fr::from_repr(repr).expect("valid scalar")
            })
            .collect()
    };

    let opt_points: Vec<halo2curves::bn256::G1Affine> = {
        use group::UncompressedEncoding;
        point_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves::bn256::G1Affine as UncompressedEncoding>::Uncompressed::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves::bn256::G1Affine::from_uncompressed(&repr).expect("valid point")
            })
            .collect()
    };

    let bas_scalars: Vec<halo2curves_baseline::bn256::Fr> = {
        use ff::PrimeField;
        scalar_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_baseline::bn256::Fr as PrimeField>::Repr::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_baseline::bn256::Fr::from_repr(repr).expect("valid scalar")
            })
            .collect()
    };

    let bas_points: Vec<halo2curves_baseline::bn256::G1Affine> = {
        use group::UncompressedEncoding;
        point_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_baseline::bn256::G1Affine as UncompressedEncoding>::Uncompressed::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_baseline::bn256::G1Affine::from_uncompressed(&repr)
                    .expect("valid point")
            })
            .collect()
    };

    println!(
        "Data generation + parsing: {:.2}s\n",
        gen_start.elapsed().as_secs_f64()
    );

    // ── Step 3: Run A/B comparison for each k
    println!(
        "{:>4}  {:>14} {:>14}  {:>8}  {:>6}",
        "k", "baseline(s)", "optimized(s)", "speedup", "match?"
    );
    println!("{}", "-".repeat(64));

    for &k in &sizes {
        let n = 1usize << k;

        let opt_s = &opt_scalars[..n];
        let opt_p = &opt_points[..n];
        let bas_s = &bas_scalars[..n];
        let bas_p = &bas_points[..n];

        // Run baseline (upstream main)
        let t = Instant::now();
        let bas_result: halo2curves_baseline::bn256::G1 =
            halo2curves_baseline::msm::msm_best(bas_s, bas_p);
        let bas_time = t.elapsed().as_secs_f64();

        // Run optimized (opt/msm-fast)
        let t = Instant::now();
        let opt_result: halo2curves::bn256::G1 =
            halo2curves::msm::msm_best(opt_s, opt_p);
        let opt_time = t.elapsed().as_secs_f64();

        // Compare results via uncompressed point bytes
        let bas_result_bytes: [u8; 64] = {
            use group::{Curve, UncompressedEncoding};
            let aff = bas_result.to_affine();
            let unc = aff.to_uncompressed();
            let bytes: &[u8] = unc.as_ref();
            let mut buf = [0u8; 64];
            buf.copy_from_slice(&bytes[..64]);
            buf
        };

        let opt_result_bytes: [u8; 64] = {
            use group::{Curve, UncompressedEncoding};
            let aff = opt_result.to_affine();
            let unc = aff.to_uncompressed();
            let bytes: &[u8] = unc.as_ref();
            let mut buf = [0u8; 64];
            buf.copy_from_slice(&bytes[..64]);
            buf
        };

        let results_match = bas_result_bytes == opt_result_bytes;

        assert!(
            results_match,
            "MISMATCH at k={}: baseline and optimized MSM produce different results!",
            k
        );

        let speedup = if opt_time > 0.0 { bas_time / opt_time } else { f64::NAN };
        println!(
            "k={:2}  {:>14.6} {:>14.6}  {:>7.2}x  {}",
            k,
            bas_time,
            opt_time,
            speedup,
            if results_match { "OK" } else { "FAIL" }
        );
    }

    println!("\nAll results match. Optimization is correct.");
}
