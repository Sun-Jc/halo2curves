/// Batch MSM benchmark: compares serial msm loop vs batch_msm.
///
/// GPU mode (Apple Silicon): cargo run --release --features gpu --example bench_batch_msm
/// CPU mode (any platform):  cargo run --release --features std --example bench_batch_msm
use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::msm::{msm_best, batch_msm_best};
use ff::Field;
use group::{Curve, Group};
use group::prime::PrimeCurveAffine;
use rand_core::OsRng;
use std::time::Instant;

fn main() {
    // Avoid OOM: large k + large batch needs too much memory.
    // k=22, batch=256 → 256 × 4M × 32B = 32GB scalars alone.
    // Use batch_sizes that scale down with k.
    let configs: Vec<(usize, Vec<usize>)> = vec![
        (18, vec![1, 16, 64, 256]),
        (20, vec![1, 16, 64]),
        (22, vec![1, 16]),
    ];

    #[cfg(feature = "gpu")]
    let gpu_available = {
        // Try to initialize GPU; if it fails, fall back to CPU
        match std::panic::catch_unwind(|| {
            halo2curves::gpu::msm_gpu_warmup(1 << 14);
        }) {
            Ok(_) => true,
            Err(_) => {
                eprintln!("GPU not available, using CPU-only mode.");
                false
            }
        }
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    // Generate max-size bases once
    let max_k = configs.iter().map(|(k, _)| *k).max().unwrap();
    let max_n = 1usize << max_k;

    eprintln!("Generating {max_n} random points (k={max_k})...");
    let pts: Vec<G1> = (0..max_n).map(|_| G1::random(OsRng)).collect();
    let mut points = vec![G1Affine::identity(); max_n];
    G1::batch_normalize(&pts, &mut points);
    drop(pts);

    if gpu_available {
        #[cfg(feature = "gpu")]
        {
            eprintln!("Warming up GPU for k={max_k}...");
            halo2curves::gpu::msm_gpu_warmup(max_n);
            // Throwaway to fully wake GPU
            let s: Vec<Fr> = (0..1 << 14).map(|_| Fr::random(OsRng)).collect();
            let _ = halo2curves::gpu::msm_gpu::<G1Affine>(&s, &points[..1 << 14]);
        }
    }

    let mode = if gpu_available { "GPU" } else { "CPU" };
    eprintln!();
    eprintln!("Mode: {mode}");
    eprintln!("{:>4} {:>6} {:>12} {:>12} {:>8}",
        "k", "batch", "serial(ms)", "batch(ms)", "speedup");
    eprintln!("{}", "-".repeat(50));

    for (k, batch_sizes) in &configs {
        let k = *k;
        let n = 1usize << k;
        let bases = &points[..n];

        for &batch_size in batch_sizes {
            let all_scalars: Vec<Vec<Fr>> = (0..batch_size)
                .map(|_| (0..n).map(|_| Fr::random(OsRng)).collect())
                .collect();

            let coeffs_refs: Vec<&[Fr]> = all_scalars.iter().map(|s| s.as_slice()).collect();
            let bases_refs: Vec<&[G1Affine]> = vec![bases; batch_size];

            if gpu_available {
                #[cfg(feature = "gpu")]
                {
                    use halo2curves::gpu::{msm_gpu, batch_msm_gpu};

                    // Warm up
                    let _ = msm_gpu::<G1Affine>(&all_scalars[0], bases);
                    let _ = batch_msm_gpu::<G1Affine>(&coeffs_refs, &bases_refs);

                    // Serial
                    let t0 = Instant::now();
                    let serial_results: Vec<G1> = all_scalars
                        .iter()
                        .map(|s| msm_gpu::<G1Affine>(s, bases))
                        .collect();
                    let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    // Batch
                    let t0 = Instant::now();
                    let batch_results = batch_msm_gpu::<G1Affine>(&coeffs_refs, &bases_refs);
                    let batch_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    assert_eq!(serial_results, batch_results,
                        "Mismatch at k={k}, batch={batch_size}");

                    let speedup = serial_ms / batch_ms;
                    eprintln!("{k:>4} {batch_size:>6} {serial_ms:>12.1} {batch_ms:>12.1} {speedup:>7.2}x");
                }
            } else {
                // CPU mode: msm_best serial vs batch_msm_best

                // Warm up
                let _ = msm_best(&all_scalars[0], bases);
                let _ = batch_msm_best(&coeffs_refs, &bases_refs);

                // Serial
                let t0 = Instant::now();
                let serial_results: Vec<G1> = all_scalars
                    .iter()
                    .map(|s| msm_best(s, bases))
                    .collect();
                let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

                // Batch
                let t0 = Instant::now();
                let batch_results = batch_msm_best(&coeffs_refs, &bases_refs);
                let batch_ms = t0.elapsed().as_secs_f64() * 1000.0;

                assert_eq!(serial_results, batch_results,
                    "Mismatch at k={k}, batch={batch_size}");

                let speedup = serial_ms / batch_ms;
                eprintln!("{k:>4} {batch_size:>6} {serial_ms:>12.1} {batch_ms:>12.1} {speedup:>7.2}x");
            }
        }
        eprintln!();
    }
}
