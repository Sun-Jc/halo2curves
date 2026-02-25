//! Benchmark GPU MSM with k=25 and all-same base points.
//!
//! This tests the "same generator" scenario: every base point is identical,
//! only the 256-bit scalars differ. Useful for profiling bucket collision
//! patterns and memory access when all points land in the same bucket set.
//!
//! Run with:
//!     rustup run stable cargo run --release --features gpu --example gpu_same_base

use std::time::Instant;

use ff::Field;
use group::{Curve, Group};
use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::gpu::{msm_gpu, msm_gpu_timed};
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use rayon::current_thread_index;

const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d,
    0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

const K: u32 = 24;

fn main() {
    let n = 1usize << K;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GPU MSM — same base point, k={}, n=2^{}={:>12}   ║", K, K, n);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --- Generate a single random base point, replicate it n times ---
    println!("Generating single base point and replicating {} times...", n);
    let t0 = Instant::now();
    let mut rng = XorShiftRng::from_seed(SEED);
    let single_base = G1::random(&mut rng).to_affine();
    let bases = vec![single_base; n];
    println!("  done in {:.2}s ({:.0} MB)", t0.elapsed().as_secs_f64(),
        (n * std::mem::size_of::<G1Affine>()) as f64 / 1_048_576.0);

    // --- Generate random 256-bit scalars ---
    println!("Generating {} random 256-bit scalars...", n);
    let t0 = Instant::now();
    let scalars: Vec<Fr> = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut seed = SEED;
                let idx = current_thread_index().unwrap().to_ne_bytes();
                for i in 0..idx.len() {
                    seed[i] = seed[i].wrapping_add(idx[i]);
                    seed[i + 8] = seed[i + 8].wrapping_add(idx[i]);
                }
                XorShiftRng::from_seed(seed)
            },
            |rng, _| Fr::random(rng),
        )
        .collect();
    println!("  done in {:.2}s\n", t0.elapsed().as_secs_f64());

    // --- Warmup GPU pipeline ---
    println!("Warming up GPU pipeline...");
    let _ = msm_gpu(&scalars[..1024], &bases[..1024]);
    println!("  done\n");

    // --- Run GPU MSM (3 iterations) ---
    let iters = 3;
    println!("Running msm_gpu × {} iterations (k={})...\n", iters, K);

    let mut times_ms = Vec::with_capacity(iters);
    for i in 0..iters {
        let t0 = Instant::now();
        let _result = msm_gpu(&scalars, &bases);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed);
        println!("  run {}: {:.1} ms", i + 1, elapsed);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[times_ms.len() / 2];
    println!("\n  median: {:.1} ms", median);
    println!("  all:    [{}]",
        times_ms.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>().join(", "));

    // --- Phase timing breakdown ---
    println!("\nGPU phase timing breakdown:");
    let (result, timing) = msm_gpu_timed(&scalars, &bases);
    println!("{}", timing);

    // --- Correctness sanity check ---
    // sum_of_scalars * base == MSM result
    println!("\nCorrectness check...");
    let scalar_sum: Fr = scalars.par_iter().copied().reduce(|| Fr::ZERO, |a, b| a + b);
    let expected = (G1::from(single_base) * scalar_sum).to_affine();
    let got = result.to_affine();
    if expected == got {
        println!("  ✅ PASS (scalar_sum * base == msm_gpu result)");
    } else {
        println!("  ❌ FAIL! Results do not match.");
    }
}
