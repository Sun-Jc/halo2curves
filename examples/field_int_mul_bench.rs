//! Benchmark: arkworks vs halo2curves field arithmetic
//!
//! Direct comparison of FxF and Fxinteger multiplication on BN254,
//! testing all supported integer types: u32, u64, i32, i128.
//!
//! Run:
//!     cargo run --release --features std --example field_int_mul_bench

use std::time::Instant;

// arkworks
use ark_ff::UniformRand;
use ark_test_curves::bn254::Fr as ArkFr;

// halo2curves
use halo2curves::bn256::Fq as H2cFq;
use halo2curves::ff::Field as H2cField;

use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d,
    0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

fn bench(log_n: u32) {
    let n = 1usize << log_n;
    let mut rng = XorShiftRng::from_seed(SEED);

    // Generate scalar data for all types
    let scalars_u64: Vec<u64> = (0..n).map(|_| rand_core::RngCore::next_u64(&mut rng)).collect();
    let scalars_u32: Vec<u32> = scalars_u64.iter().map(|&s| s as u32).collect();
    let scalars_i32: Vec<i32> = scalars_u64.iter().map(|&s| s as i32).collect(); // wrapping cast
    let scalars_i128: Vec<i128> = scalars_u64.iter().enumerate().map(|(i, &lo)| {
        let hi = scalars_u64[(i + 1) % n] as i128;
        let sign: i128 = if lo & 1 == 0 { 1 } else { -1 };
        sign * ((hi << 64) | (lo as i128))
    }).collect();
    // small i128 (fits in u64)
    let scalars_i128_small: Vec<i128> = scalars_u64.iter().map(|&s| {
        let sign: i128 = if s & 1 == 0 { 1 } else { -1 };
        sign * ((s >> 1) as i128)
    }).collect();

    // arkworks elements
    let mut rng_ark = XorShiftRng::from_seed(SEED);
    let ark_elems: Vec<ArkFr> = (0..n).map(|_| ArkFr::rand(&mut rng_ark)).collect();
    let ark_scalar_fields: Vec<ArkFr> = scalars_u64.iter().map(|&s| ArkFr::from(s)).collect();

    // halo2curves elements
    let mut rng_h2c = XorShiftRng::from_seed(SEED);
    let h2c_elems: Vec<H2cFq> = (0..n).map(|_| H2cFq::random(&mut rng_h2c)).collect();
    let h2c_scalar_fields: Vec<H2cFq> = scalars_u64.iter().map(|&s| H2cFq::from(s)).collect();

    // Helper: time a loop
    macro_rules! time_loop {
        ($init:expr, $body:expr) => {{
            let mut acc = $init;
            let t = Instant::now();
            for i in 0..n { acc += $body(i); }
            let elapsed = t.elapsed();
            std::hint::black_box(&acc);
            elapsed
        }};
    }

    // ===== FxF =====
    let ark_fxf = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]) * std::hint::black_box(ark_scalar_fields[i]));
    let h2c_fxf = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(h2c_scalar_fields[i]));

    // ===== Fxu64 =====
    let ark_u64 = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]).mul_u64::<5>(std::hint::black_box(scalars_u64[i])));
    let h2c_u64 = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(scalars_u64[i]));

    // ===== Fxu32 =====
    let ark_u32 = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]).mul_u64::<5>(std::hint::black_box(scalars_u32[i]) as u64));
    let h2c_u32 = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(scalars_u32[i]));

    // ===== Fxi32 =====
    let ark_i32 = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]).mul_i64::<5>(std::hint::black_box(scalars_i32[i]) as i64));
    let h2c_i32 = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(scalars_i32[i]));

    // ===== Fxi128 (large, > u64) =====
    let ark_i128 = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]).mul_i128::<5, 6>(std::hint::black_box(scalars_i128[i])));
    let h2c_i128 = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(scalars_i128[i]));

    // ===== Fxi128 (small, fits u64) =====
    let ark_i128s = time_loop!(ArkFr::from(0u64), |i: usize|
        std::hint::black_box(ark_elems[i]).mul_i128::<5, 6>(std::hint::black_box(scalars_i128_small[i])));
    let h2c_i128s = time_loop!(H2cFq::ZERO, |i: usize|
        std::hint::black_box(h2c_elems[i]) * std::hint::black_box(scalars_i128_small[i]));

    // ===== From<u64> =====
    let ark_from = time_loop!(ArkFr::from(0u64), |i: usize|
        ArkFr::from(std::hint::black_box(scalars_u64[i])));
    let h2c_from = time_loop!(H2cFq::ZERO, |i: usize|
        H2cFq::from(std::hint::black_box(scalars_u64[i])));

    // Print
    let ns = |d: std::time::Duration| d.as_nanos() as f64 / n as f64;
    let w = |a: f64, h: f64| if a < h {
        format!("ark {:+.0}%", (h/a-1.0)*100.0)
    } else {
        format!("h2c {:+.0}%", (a/h-1.0)*100.0)
    };

    let rows: Vec<(&str, f64, f64)> = vec![
        ("FxF",          ns(ark_fxf),   ns(h2c_fxf)),
        ("Fxu64",        ns(ark_u64),   ns(h2c_u64)),
        ("Fxu32",        ns(ark_u32),   ns(h2c_u32)),
        ("Fxi32",        ns(ark_i32),   ns(h2c_i32)),
        ("Fxi128 large", ns(ark_i128),  ns(h2c_i128)),
        ("Fxi128 small", ns(ark_i128s), ns(h2c_i128s)),
        ("From<u64>",    ns(ark_from),  ns(h2c_from)),
    ];

    println!("log_n={:2}, n={:>9}", log_n, n);
    println!("  {:14} {:>10} {:>10} {:>12}", "operation", "arkworks", "h2c", "faster");
    println!("  {}", "-".repeat(50));
    for (name, a, h) in &rows {
        println!("  {:14} {:>7.2}ns {:>7.2}ns {:>12}", name, a, h, w(*a, *h));
    }
    let af = ns(ark_fxf);
    let hf = ns(h2c_fxf);
    println!("  --- speedup vs FxF ---");
    println!("  {:14} {:>7.2}x  {:>7.2}x", "Fxu64/FxF", af/ns(ark_u64), hf/ns(h2c_u64));
    println!("  {:14} {:>7.2}x  {:>7.2}x", "Fxu32/FxF", af/ns(ark_u32), hf/ns(h2c_u32));
    println!("  {:14} {:>7.2}x  {:>7.2}x", "Fxi32/FxF", af/ns(ark_i32), hf/ns(h2c_i32));
    println!("  {:14} {:>7.2}x  {:>7.2}x", "Fxi128L/FxF", af/ns(ark_i128), hf/ns(h2c_i128));
    println!("  {:14} {:>7.2}x  {:>7.2}x", "Fxi128S/FxF", af/ns(ark_i128s), hf/ns(h2c_i128s));
    println!();
}

fn main() {
    println!("=== arkworks vs halo2curves: BN254 Field x Integer ===");
    println!("    Types: u32, u64, i32, i128 (large & small)\n");
    bench(16); // warmup
    for log_n in [18, 20, 22, 24] { bench(log_n); }
}
