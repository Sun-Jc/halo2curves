/// Reproduce the i16 overflow bug in GPU MSM booth encoding.
///
/// At c=16 (selected for k>=20), get_booth_index() returns values up to
/// ±32768 = ±2^15, which overflows i16 (max 32767). This causes silent
/// sign corruption in the scatter table, leading to wrong MSM results.
///
/// Run: cargo run --release --features gpu --example repro_i16_bug
use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::msm::msm_best;
use halo2curves::gpu::msm_gpu;
use ff::Field;
use group::{Curve, Group};
use rand_core::OsRng;

fn main() {
    eprintln!("=== Reproducing i16 overflow bug in GPU MSM ===\n");
    eprintln!("get_optimal_c_gpu selects c=16 for k>=20.");
    eprintln!("At c=16, booth indices reach ±32768, overflowing i16 (max 32767).\n");

    // First, demonstrate the overflow at the numeric level
    eprintln!("--- Numeric demonstration ---");
    let max_booth_val: i32 = 1 << 15; // 32768
    let as_i16 = max_booth_val as i16;
    eprintln!("  booth index = {max_booth_val} (i32)");
    eprintln!("  cast to i16 = {as_i16}  ← OVERFLOWED! (wraps to -32768)");
    eprintln!("  This flips the sign, putting the point in the wrong bucket.\n");

    // Now run actual GPU vs CPU comparison at k values that trigger c=16
    for k in [14, 16, 18, 20, 21, 22] {
        let n = 1usize << k;
        let pts: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let points: Vec<G1Affine> = pts.iter().map(|p| p.to_affine()).collect();

        let mut pass = 0;
        let mut fail = 0;
        let trials = 3;

        for _ in 0..trials {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
            let cpu = msm_best(&scalars, &points);
            let gpu = msm_gpu::<G1Affine>(&scalars, &points);

            if cpu.to_affine() == gpu.to_affine() {
                pass += 1;
            } else {
                fail += 1;
            }
        }

        let c_gpu = match k {
            0..=13  => 10,
            14..=17 => 13,
            18..=19 => 15,
            20..=24 => 16,
            _       => 18,
        };
        let status = if fail == 0 { "✅ OK" } else { "❌ FAIL" };
        eprintln!("k={k:2}  n={n:>8}  c={c_gpu:2}  {status}  ({pass} pass, {fail} fail of {trials})");
    }

    eprintln!("\nExpected: k<20 passes (c<16, no overflow), k>=20 FAILS (c=16, overflow)");
}
