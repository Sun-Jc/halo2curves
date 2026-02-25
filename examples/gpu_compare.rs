//! Benchmark comparing GPU vs GLV+GPU vs CPU-multicore MSM for 256-bit scalars,
//! with detailed GPU phase timing breakdown.
//!
//! Run with:
//!     rustup run stable cargo run --release --features gpu --example gpu_compare
//!
//! Sizes: k = 22, 24, 26 (configurable via SIZES below)

use std::time::Instant;

use ff::Field;
use group::{Curve, Group};
use group::prime::PrimeCurveAffine;
use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::gpu::{msm_gpu, msm_gpu_timed, msm_gpu_glv, GpuMsmTiming};
use halo2curves::msm::msm_best;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use rayon::current_thread_index;

const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d,
    0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

/// How many iterations per (method, k) pair
const ITERS: usize = 3;

/// Default sizes to benchmark (can override via command-line args, e.g. `-- 22 24`)
const DEFAULT_SIZES: &[u8] = &[22, 24, 26];

fn gen_points(n: usize) -> Vec<G1Affine> {
    let projs: Vec<G1> = (0..n)
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
            |rng, _| G1::random(rng),
        )
        .collect();
    let mut affines = vec![G1Affine::identity(); n];
    G1::batch_normalize(&projs, &mut affines);
    affines
}

fn gen_scalars(n: usize) -> Vec<Fr> {
    (0..n)
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
        .collect()
}

/// Run `f` for `iters` times, return (median_ms, all_times_ms)
fn bench_fn<F: FnMut()>(mut f: F, iters: usize) -> (f64, Vec<f64>) {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    (median, times)
}

fn main() {
    // Parse sizes from command line, or use defaults
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sizes: Vec<u8> = if args.is_empty() {
        DEFAULT_SIZES.to_vec()
    } else {
        args.iter()
            .map(|s| s.parse::<u8>().expect("Usage: gpu_compare [k1 k2 ...]"))
            .collect()
    };

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   MSM Benchmark: GPU vs GLV+GPU vs CPU-multicore (256-bit)     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let max_k = *sizes.iter().max().unwrap();
    let max_n = 1usize << max_k;

    println!("Generating {} points (2^{})...", max_n, max_k);
    let t0 = Instant::now();
    let bases = gen_points(max_n);
    println!("  done in {:.1}s", t0.elapsed().as_secs_f64());

    println!("Generating {} 256-bit scalars...", max_n);
    let t0 = Instant::now();
    let scalars = gen_scalars(max_n);
    println!("  done in {:.1}s\n", t0.elapsed().as_secs_f64());

    // Warmup GPU pipeline
    println!("Warming up GPU pipeline...");
    let _ = msm_gpu(&scalars[..1024], &bases[..1024]);
    let _ = msm_gpu_glv(&scalars[..1024], &bases[..1024]);
    println!("  done\n");

    // Header
    println!("{:<6} {:>14} {:>14} {:>14} {:>12} {:>12}",
        "k", "CPU (ms)", "GPU (ms)", "GLV+GPU (ms)", "GPU/CPU", "GLV+GPU/CPU");
    println!("{}", "─".repeat(80));

    for &k in &sizes {
        let n = 1usize << k;

        // CPU multicore
        let (cpu_ms, cpu_times) = bench_fn(|| { msm_best(&scalars[..n], &bases[..n]); }, ITERS);

        // GPU plain (with timing on last run)
        let (gpu_ms, gpu_times) = bench_fn(|| { msm_gpu(&scalars[..n], &bases[..n]); }, ITERS);

        // GPU + GLV
        let (glv_ms, glv_times) = bench_fn(|| { msm_gpu_glv(&scalars[..n], &bases[..n]); }, ITERS);

        let gpu_speedup = cpu_ms / gpu_ms;
        let glv_speedup = cpu_ms / glv_ms;

        println!("{:<6} {:>14.1} {:>14.1} {:>14.1} {:>11.2}x {:>11.2}x",
            k, cpu_ms, gpu_ms, glv_ms, gpu_speedup, glv_speedup);

        // Detail line
        println!("       {:>14} {:>14} {:>14}",
            format!("[{}]", cpu_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>().join(", ")),
            format!("[{}]", gpu_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>().join(", ")),
            format!("[{}]", glv_times.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>().join(", ")),
        );

        // GPU phase timing breakdown
        let (_, timing) = msm_gpu_timed(&scalars[..n], &bases[..n]);
        println!("\n  GPU phase breakdown (k={}):", k);
        println!("{}", timing);
        println!();
    }

    println!("\n{}", "─".repeat(80));
    println!("Notes:");
    println!("  - Times are median of {} runs; [all runs] shown below each row", ITERS);
    println!("  - Speedup > 1.0x means GPU is faster than CPU");
    println!("  - 256-bit scalars (full Fr field elements)");
    println!("  - GPU phases: scalar_encode → base_pack → [scatter → upload → kernel → reduce] × windows");

    // Correctness check on smallest size
    let check_k = *sizes.iter().min().unwrap();
    let check_n = 1usize << check_k;
    let cpu_res = msm_best(&scalars[..check_n], &bases[..check_n]);
    let gpu_res = msm_gpu(&scalars[..check_n], &bases[..check_n]);
    let glv_res = msm_gpu_glv(&scalars[..check_n], &bases[..check_n]);
    let all_match = gpu_res.to_affine() == cpu_res.to_affine()
        && glv_res.to_affine() == cpu_res.to_affine();
    println!("\nCorrectness check (k={}): {}", check_k,
        if all_match { "✅ ALL MATCH" } else { "❌ MISMATCH!" });
}
