//! 3-way MSM benchmark: baseline (v0.9.0) vs cpu-improved (v0.9.1) vs gpu-metal (v0.9.2)
//!
//! Dependencies (all from git, no local paths):
//!   - halo2curves_baseline: upstream PSE main (v0.9.0)
//!   - halo2curves_cpu:      Sun-Jc fork v0.9.1 (CPU optimizations)
//!   - halo2curves_gpu:      Sun-Jc fork v0.9.2 (GPU Metal + CPU optimizations)
//!
//! For each problem size k:
//!   1. Run all 3 methods × 5 iterations
//!   2. Verify results match (first iteration)
//!   3. Report median, mean, and speedups
//!
//! Base points and scalars are cached in /tmp/msm_bench_k{K}.bin to avoid
//! re-generating them on every run. The cache uses portable serialized bytes
//! ([u8;32] for scalars, [u8;64] for points) so it works across library versions.
//!
//! Run: cargo run --release

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

const NUM_RUNS: usize = 5;
const CACHE_MAGIC: [u8; 8] = *b"MSMBN01\0";

// ── Cache for scalars + points ──────────────────────────────────────────────
//
// Format: [magic: 8B][n: 8B (u64 LE)][scalars: n×32B][points: n×64B]
// Uses portable uncompressed encoding, not internal Montgomery form.

fn cache_path(k: usize) -> PathBuf {
    PathBuf::from(format!("/tmp/msm_bench_k{}.bin", k))
}

fn load_cache(k: usize) -> Option<(Vec<[u8; 32]>, Vec<[u8; 64]>)> {
    let path = cache_path(k);
    let mut file = std::fs::File::open(&path).ok()?;

    let mut header = [0u8; 16];
    file.read_exact(&mut header).ok()?;
    if &header[..8] != &CACHE_MAGIC {
        eprintln!("  cache {}: bad magic, regenerating", path.display());
        return None;
    }
    let n = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let expected_n = 1usize << k;
    if n != expected_n {
        eprintln!("  cache {}: n={} expected {}, regenerating", path.display(), n, expected_n);
        return None;
    }

    // Read scalars
    let mut scalar_flat = vec![0u8; n * 32];
    file.read_exact(&mut scalar_flat).ok()?;
    let scalars: Vec<[u8; 32]> = scalar_flat
        .chunks_exact(32)
        .map(|c| { let mut buf = [0u8; 32]; buf.copy_from_slice(c); buf })
        .collect();

    // Read points
    let mut point_flat = vec![0u8; n * 64];
    file.read_exact(&mut point_flat).ok()?;
    let points: Vec<[u8; 64]> = point_flat
        .chunks_exact(64)
        .map(|c| { let mut buf = [0u8; 64]; buf.copy_from_slice(c); buf })
        .collect();

    Some((scalars, points))
}

fn save_cache(k: usize, scalars: &[[u8; 32]], points: &[[u8; 64]]) {
    let path = cache_path(k);
    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  warning: cannot create cache {}: {}", path.display(), e);
            return;
        }
    };

    let n = scalars.len() as u64;
    let mut header = [0u8; 16];
    header[..8].copy_from_slice(&CACHE_MAGIC);
    header[8..16].copy_from_slice(&n.to_le_bytes());
    if file.write_all(&header).is_err() { return; }

    // Write scalars as flat bytes
    let scalar_flat: Vec<u8> = scalars.iter().flat_map(|s| s.iter().copied()).collect();
    if file.write_all(&scalar_flat).is_err() { return; }

    // Write points as flat bytes
    let point_flat: Vec<u8> = points.iter().flat_map(|p| p.iter().copied()).collect();
    let _ = file.write_all(&point_flat);
}

fn generate_data(max_k: usize) -> (Vec<[u8; 32]>, Vec<[u8; 64]>) {
    let max_n = 1usize << max_k;

    // Try cache
    let path = cache_path(max_k);
    if path.exists() {
        if let Some(data) = load_cache(max_k) {
            println!(
                "Loaded cached 2^{} = {} scalars+points from {}",
                max_k, max_n, path.display()
            );
            return data;
        }
    }

    println!("Generating 2^{} = {} random scalars and points...", max_k, max_n);
    let gen_start = Instant::now();

    let scalar_bytes: Vec<[u8; 32]> = {
        use ff::{Field, PrimeField};
        use halo2curves_gpu::bn256::Fr;
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
        use group::prime::PrimeCurveAffine;
        use group::UncompressedEncoding;
        use group::{Curve, Group};
        use halo2curves_gpu::bn256::{G1, G1Affine};
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

    println!("  generated in {:.2}s", gen_start.elapsed().as_secs_f64());
    save_cache(max_k, &scalar_bytes, &point_bytes);
    println!("  saved to {}", path.display());

    (scalar_bytes, point_bytes)
}

// ── Result conversion helpers ───────────────────────────────────────────────

fn result_to_bytes_baseline(r: halo2curves_baseline::bn256::G1) -> [u8; 64] {
    use group::{Curve, UncompressedEncoding};
    let aff = r.to_affine();
    let unc = aff.to_uncompressed();
    let bytes: &[u8] = unc.as_ref();
    let mut buf = [0u8; 64];
    buf.copy_from_slice(&bytes[..64]);
    buf
}

fn result_to_bytes_cpu(r: halo2curves_cpu::bn256::G1) -> [u8; 64] {
    use group::{Curve, UncompressedEncoding};
    let aff = r.to_affine();
    let unc = aff.to_uncompressed();
    let bytes: &[u8] = unc.as_ref();
    let mut buf = [0u8; 64];
    buf.copy_from_slice(&bytes[..64]);
    buf
}

fn result_to_bytes_gpu(r: halo2curves_gpu::bn256::G1) -> [u8; 64] {
    use group::{Curve, UncompressedEncoding};
    let aff = r.to_affine();
    let unc = aff.to_uncompressed();
    let bytes: &[u8] = unc.as_ref();
    let mut buf = [0u8; 64];
    buf.copy_from_slice(&bytes[..64]);
    buf
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MSM Benchmark: baseline(0.9.0) vs cpu(0.9.1) vs gpu(0.9.2)║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sizes: Vec<usize> = vec![10, 15, 18, 20, 22, 23, 24];
    let max_k = *sizes.iter().max().unwrap();

    // ── Step 1: Load or generate scalars and points ──
    let (scalar_bytes, point_bytes) = generate_data(max_k);

    // ── Step 2: Parse bytes into all 3 libraries' types ──
    println!("Parsing into library types...");
    let parse_start = Instant::now();

    // Baseline (v0.9.0)
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

    // CPU-improved (v0.9.1)
    let cpu_scalars: Vec<halo2curves_cpu::bn256::Fr> = {
        use ff::PrimeField;
        scalar_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_cpu::bn256::Fr as PrimeField>::Repr::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_cpu::bn256::Fr::from_repr(repr).expect("valid scalar")
            })
            .collect()
    };
    let cpu_points: Vec<halo2curves_cpu::bn256::G1Affine> = {
        use group::UncompressedEncoding;
        point_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_cpu::bn256::G1Affine as UncompressedEncoding>::Uncompressed::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_cpu::bn256::G1Affine::from_uncompressed(&repr)
                    .expect("valid point")
            })
            .collect()
    };

    // GPU Metal (v0.9.2)
    let gpu_scalars: Vec<halo2curves_gpu::bn256::Fr> = {
        use ff::PrimeField;
        scalar_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_gpu::bn256::Fr as PrimeField>::Repr::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_gpu::bn256::Fr::from_repr(repr).expect("valid scalar")
            })
            .collect()
    };
    let gpu_points: Vec<halo2curves_gpu::bn256::G1Affine> = {
        use group::UncompressedEncoding;
        point_bytes
            .iter()
            .map(|b| {
                let mut repr =
                    <halo2curves_gpu::bn256::G1Affine as UncompressedEncoding>::Uncompressed::default();
                repr.as_mut().copy_from_slice(b);
                halo2curves_gpu::bn256::G1Affine::from_uncompressed(&repr)
                    .expect("valid point")
            })
            .collect()
    };

    println!("  parsed in {:.2}s\n", parse_start.elapsed().as_secs_f64());

    // ── Step 3: Warm up GPU ──
    println!("Warming up GPU...");
    {
        let n_warmup = 1 << 10;
        let _ = halo2curves_gpu::gpu::msm_gpu(&gpu_scalars[..n_warmup], &gpu_points[..n_warmup]);
    }
    println!("  done\n");

    // ── Step 4: Run benchmark ──
    println!(
        "{:>4} {:>10}  {:>22}  {:>22}  {:>22}  {:>8} {:>8}  {:>5}",
        "k", "n",
        "baseline(ms)", "cpu(ms)", "gpu(ms)",
        "cpu-spd", "gpu-spd", "match"
    );
    println!("{}", "─".repeat(112));

    for &k in &sizes {
        let n = 1usize << k;

        let bas_s = &bas_scalars[..n];
        let bas_p = &bas_points[..n];
        let cpu_s = &cpu_scalars[..n];
        let cpu_p = &cpu_points[..n];
        let gpu_s = &gpu_scalars[..n];
        let gpu_p = &gpu_points[..n];

        let mut bas_times = Vec::with_capacity(NUM_RUNS);
        let mut cpu_times = Vec::with_capacity(NUM_RUNS);
        let mut gpu_times = Vec::with_capacity(NUM_RUNS);
        let mut results_match = true;

        for run in 0..NUM_RUNS {
            // Baseline
            let t = Instant::now();
            let bas_result = halo2curves_baseline::msm::msm_best(bas_s, bas_p);
            bas_times.push(t.elapsed().as_secs_f64() * 1000.0);

            // CPU-improved
            let t = Instant::now();
            let cpu_result = halo2curves_cpu::msm::msm_best(cpu_s, cpu_p);
            cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);

            // GPU Metal
            let t = Instant::now();
            let gpu_result = halo2curves_gpu::gpu::msm_gpu(gpu_s, gpu_p);
            gpu_times.push(t.elapsed().as_secs_f64() * 1000.0);

            // Check correctness on first run
            if run == 0 {
                let bas_bytes = result_to_bytes_baseline(bas_result);
                let cpu_bytes = result_to_bytes_cpu(cpu_result);
                let gpu_bytes = result_to_bytes_gpu(gpu_result);

                if bas_bytes != cpu_bytes {
                    eprintln!("MISMATCH at k={}: baseline vs cpu differ!", k);
                    results_match = false;
                }
                if bas_bytes != gpu_bytes {
                    eprintln!("MISMATCH at k={}: baseline vs gpu differ!", k);
                    results_match = false;
                }
                assert!(
                    results_match,
                    "Results mismatch at k={}! Aborting.",
                    k
                );
            }
        }

        // Statistics
        let bas_med = median(&mut bas_times);
        let cpu_med = median(&mut cpu_times);
        let gpu_med = median(&mut gpu_times);
        let bas_avg = mean(&bas_times);
        let cpu_avg = mean(&cpu_times);
        let gpu_avg = mean(&gpu_times);

        let cpu_speedup = bas_med / cpu_med;
        let gpu_speedup = bas_med / gpu_med;

        println!(
            "k={:2} {:>9}  {:>10.1} / {:>8.1}  {:>10.1} / {:>8.1}  {:>10.1} / {:>8.1}  {:>7.2}x {:>7.2}x  {:>5}",
            k,
            n,
            bas_med, bas_avg,
            cpu_med, cpu_avg,
            gpu_med, gpu_avg,
            cpu_speedup,
            gpu_speedup,
            if results_match { "OK" } else { "FAIL" }
        );
        // Print all runs for reference (sorted)
        println!(
            "     {:>9}  {:>44}  {:>22}  {:>22}",
            "",
            format!("[{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                    bas_times[0], bas_times[1], bas_times[2], bas_times[3], bas_times[4]),
            format!("[{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                    cpu_times[0], cpu_times[1], cpu_times[2], cpu_times[3], cpu_times[4]),
            format!("[{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                    gpu_times[0], gpu_times[1], gpu_times[2], gpu_times[3], gpu_times[4]),
        );
    }

    println!("\n{}", "─".repeat(112));
    println!("  median / mean (ms)  │  speedup = baseline_median / method_median");
    println!("  [sorted individual runs shown below each row]");
    println!("\nAll results matched. ✓");
}
