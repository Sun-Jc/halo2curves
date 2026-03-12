/// Verify GPU correctness fix: msm_gpu vs msm_best across all k values.
use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::msm::msm_best;
use halo2curves::gpu::{msm_gpu, msm_gpu_warmup};
use ff::Field;
use group::Curve;
use group::Group;
use rand_core::OsRng;

fn main() {
    eprintln!("=== GPU correctness test (after i16→i32 fix) ===\n");

    for k in [14, 16, 18, 19, 20, 21, 22] {
        let n = 1usize << k;
        let pts: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let points: Vec<G1Affine> = pts.iter().map(|p| p.to_affine()).collect();

        msm_gpu_warmup(n);

        let mut pass = 0;
        let mut fail = 0;
        let trials = 3;

        for _trial in 0..trials {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
            let cpu_result = msm_best(&scalars, &points);
            let gpu_result = msm_gpu::<G1Affine>(&scalars, &points);

            if cpu_result == gpu_result {
                pass += 1;
            } else {
                fail += 1;
            }
        }

        let status = if fail == 0 { "OK" } else { "FAIL" };
        eprintln!("k={k:2}  n={n:>8}  {status}  ({pass} pass, {fail} fail of {trials})");
    }
}
