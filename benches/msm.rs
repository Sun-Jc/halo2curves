//! This benchmarks Multi Scalar Multiplication (MSM).
//! It measures `G1` from the BN256 curve.
//!
//! To run this benchmark:
//!
//!     cargo bench --bench msm
//!
//! Caveat:  The multicore benchmark assumes:
//!     1. a multi-core system
//!     2. that the `multicore` feature is enabled.  It is by default.
//!
//! Base points are cached in /tmp/halo2_msm_bases_k{K}.bin as raw bytes
//! to avoid expensive regeneration on every run.

#[macro_use]
extern crate criterion;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::SystemTime;

use criterion::{BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
#[cfg(feature = "gpu")]
use halo2curves::gpu::msm_gpu;
use halo2curves::{
    bn256::{Fr as Scalar, G1Affine as Point},
    msm::{msm_best, msm_serial},
};
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::{
    current_thread_index,
    prelude::{IntoParallelIterator, ParallelIterator},
};
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const SINGLECORE_RANGE: [u8; 6] = [3, 8, 10, 12, 14, 16];
const MULTICORE_RANGE: [u8; 10] = [3, 8, 10, 12, 14, 16, 18, 20, 22, 24];
const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

// ── Raw binary cache for base points ────────────────────────────────────────
//
// Format:  [magic: 8 bytes][n: 8 bytes (u64 LE)][points: n * size_of::<Point>() bytes]
// Points are stored as raw memory (Montgomery-form [u64;4] × 2 per G1Affine).
// No per-element serialization — single read()/write() syscall.

const CACHE_MAGIC: [u8; 8] = *b"H2CPTS01";

fn cache_path(k: u8) -> PathBuf {
    PathBuf::from(format!("/tmp/halo2_msm_bases_k{}.bin", k))
}

fn load_cached_points(k: u8) -> Option<Vec<Point>> {
    let path = cache_path(k);
    let mut file = std::fs::File::open(&path).ok()?;

    let mut header = [0u8; 16];
    file.read_exact(&mut header).ok()?;
    if &header[..8] != &CACHE_MAGIC {
        return None;
    }
    let n = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    if n != (1usize << k) {
        return None;
    }

    let byte_len = n * std::mem::size_of::<Point>();
    let mut points = vec![Point::identity(); n];
    let dst = unsafe {
        std::slice::from_raw_parts_mut(points.as_mut_ptr() as *mut u8, byte_len)
    };
    file.read_exact(dst).ok()?;
    Some(points)
}

fn save_cached_points(k: u8, points: &[Point]) {
    let path = cache_path(k);
    let Ok(mut file) = std::fs::File::create(&path) else { return };

    let n = points.len() as u64;
    let mut header = [0u8; 16];
    header[..8].copy_from_slice(&CACHE_MAGIC);
    header[8..16].copy_from_slice(&n.to_le_bytes());
    if file.write_all(&header).is_err() { return; }

    let byte_len = points.len() * std::mem::size_of::<Point>();
    let src = unsafe {
        std::slice::from_raw_parts(points.as_ptr() as *const u8, byte_len)
    };
    let _ = file.write_all(src);
}

fn get_or_generate_curvepoints(k: u8) -> Vec<Point> {
    if let Some(pts) = load_cached_points(k) {
        println!("Loaded 2^{k} = {} cached curve points from {}", 1u64 << k, cache_path(k).display());
        return pts;
    }
    let pts = generate_curvepoints(k);
    save_cached_points(k, &pts);
    println!("  (saved to {})", cache_path(k).display());
    pts
}

// ── Point / coefficient generation ──────────────────────────────────────────

fn generate_curvepoints(k: u8) -> Vec<Point> {
    let n: u64 = {
        assert!(k < 64);
        1 << k
    };

    println!("Generating 2^{k} = {n} curve points..",);
    let timer = SystemTime::now();
    let bases = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| Point::random(rng),
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} curve points took: {} sec.\n\n",
        end.as_secs()
    );
    bases
}

fn generate_coefficients(k: u8, bits: usize) -> Vec<Scalar> {
    let n: u64 = {
        assert!(k < 64);
        1 << k
    };
    let max_val: Option<u128> = match bits {
        1 => Some(1),
        8 => Some(0xff),
        16 => Some(0xffff),
        32 => Some(0xffff_ffff),
        64 => Some(0xffff_ffff_ffff_ffff),
        128 => Some(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff),
        256 => None,
        _ => panic!("unexpected bit size {}", bits),
    };

    println!("Generating 2^{k} = {n} coefficients..",);
    let timer = SystemTime::now();
    let coeffs = (0..n)
        .into_par_iter()
        .map_init(
            || {
                let mut thread_seed = SEED;
                let uniq = current_thread_index().unwrap().to_ne_bytes();
                assert!(std::mem::size_of::<usize>() == 8);
                for i in 0..uniq.len() {
                    thread_seed[i] += uniq[i];
                    thread_seed[i + 8] += uniq[i];
                }
                XorShiftRng::from_seed(thread_seed)
            },
            |rng, _| {
                if let Some(max_val) = max_val {
                    let v_lo = rng.next_u64() as u128;
                    let v_hi = rng.next_u64() as u128;
                    let mut v = v_lo + (v_hi << 64);
                    v &= max_val; // Mask the 128bit value to get a lower number of bits
                    Scalar::from_u128(v)
                } else {
                    Scalar::random(rng)
                }
            },
        )
        .collect();
    let end = timer.elapsed().unwrap();
    println!(
        "Generating 2^{k} = {n} coefficients took: {} sec.\n\n",
        end.as_secs()
    );
    coeffs
}

fn msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    let max_k = *SINGLECORE_RANGE
        .iter()
        .chain(MULTICORE_RANGE.iter())
        .max()
        .unwrap_or(&16);
    let bases = get_or_generate_curvepoints(max_k);
    let bits = [1, 8, 16, 32, 64, 128, 256];
    let coeffs: Vec<_> = bits
        .iter()
        .map(|b| generate_coefficients(max_k, *b))
        .collect();

    for (b_index, b) in bits.iter().enumerate() {
        for k in SINGLECORE_RANGE {
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("singlecore", id), |b| {
                    assert!(k < 64);
                    let n: usize = 1 << k;
                    let mut acc = Point::identity().into();
                    b.iter(|| msm_serial(&coeffs[b_index][..n], &bases[..n], &mut acc));
                })
                .sample_size(10);
        }
        for k in MULTICORE_RANGE {
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("multicore", id), |b| {
                    assert!(k < 64);
                    let n: usize = 1 << k;
                    b.iter(|| {
                        msm_best(&coeffs[b_index][..n], &bases[..n]);
                    })
                })
                .sample_size(SAMPLE_SIZE);
        }
        #[cfg(feature = "gpu")]
        for k in MULTICORE_RANGE {
            if k < 14 {
                continue;
            } // GPU only useful for large inputs
            let id = format!("{b}b_{k}");
            group
                .bench_function(BenchmarkId::new("gpu", id), |b| {
                    assert!(k < 64);
                    let n: usize = 1 << k;
                    b.iter(|| {
                        msm_gpu(&coeffs[b_index][..n], &bases[..n]);
                    })
                })
                .sample_size(if k >= 24 { 10 } else { SAMPLE_SIZE })
                .warm_up_time(if k >= 24 {
                    Duration::from_millis(100)
                } else {
                    Duration::from_secs(3)
                })
                .measurement_time(if k >= 24 {
                    Duration::from_secs(30)
                } else {
                    Duration::from_secs(5)
                });
        }
    }
    group.finish();
}

criterion_group!(benches, msm);
criterion_main!(benches);
