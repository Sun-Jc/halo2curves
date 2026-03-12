/// Verify: GPU mixed-add bug when bucket contains duplicate base points.
///
/// When two identical base points land in the same bucket, the GPU kernel
/// does mixed-add(P, P) which gives wrong result (should be doubling).
use ff::Field;
use group::{Curve, Group};
use group::prime::PrimeCurveAffine;
use halo2curves::bn256::{Fr, G1, G1Affine};
use halo2curves::gpu::msm_gpu;
use halo2curves::msm::msm_best;
use rand_core::OsRng;

fn main() {
    eprintln!("=== Duplicate bases bug diagnosis ===\n");

    // Minimum size for GPU path: 2^14
    let n = 1usize << 14;

    // Generate unique bases first
    let unique_bases: Vec<G1Affine> = (0..n)
        .map(|_| G1::random(OsRng).to_affine())
        .collect();

    // Test 1: all unique bases → should pass
    eprintln!("--- Test 1: all unique bases ---");
    {
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let cpu = msm_best(&scalars, &unique_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &unique_bases);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
    }

    // Test 2: all bases are the SAME point
    eprintln!("--- Test 2: all bases = same point ---");
    {
        let same_bases: Vec<G1Affine> = vec![unique_bases[0]; n];
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let cpu = msm_best(&scalars, &same_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &same_bases);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
    }

    // Test 3: bases repeated in groups of 2 (adjacent duplicates)
    eprintln!("--- Test 3: bases repeated in pairs ---");
    {
        let mut pair_bases = Vec::with_capacity(n);
        for i in 0..n/2 {
            pair_bases.push(unique_bases[i]);
            pair_bases.push(unique_bases[i]);
        }
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let cpu = msm_best(&scalars, &pair_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &pair_bases);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
    }

    // Test 4: ~50% duplicates (realistic scenario like par_iter with few seeds)
    eprintln!("--- Test 4: ~50% duplicates ---");
    {
        let mut dup_bases = unique_bases.clone();
        // Copy first half over second half
        for i in 0..n/2 {
            dup_bases[n/2 + i] = dup_bases[i];
        }
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let cpu = msm_best(&scalars, &dup_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &dup_bases);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
    }

    // Test 5: duplicates but with scalar=1 (so duplicates in same bucket for sure)
    eprintln!("--- Test 5: all scalars=1, all same base → pure doubling chain ---");
    {
        let same_bases: Vec<G1Affine> = vec![unique_bases[0]; n];
        let scalars: Vec<Fr> = vec![Fr::ONE; n];
        let cpu = msm_best(&scalars, &same_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &same_bases);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
        if cpu != gpu {
            eprintln!("    cpu: {:?}", cpu.to_affine());
            eprintln!("    gpu: {:?}", gpu.to_affine());
            // expected: n * G where G = unique_bases[0]
        }
    }

    // Test 6: Only 2 duplicate bases matter - put same scalar on 2 identical points
    eprintln!("--- Test 6: 2 duplicate bases at start, rest unique, scalar=1 ---");
    {
        let mut bases_2dup = unique_bases.clone();
        bases_2dup[1] = bases_2dup[0]; // make [0] and [1] identical
        let scalars: Vec<Fr> = vec![Fr::ONE; n];
        let cpu = msm_best(&scalars, &bases_2dup);
        let gpu = msm_gpu::<G1Affine>(&scalars, &bases_2dup);
        eprintln!("  {}", if cpu == gpu { "✅ MATCH" } else { "❌ MISMATCH" });
    }

    // Test 7: Duplicate bases but scalars ensure they DON'T land in same bucket
    // (different scalar values → different booth indices → different buckets)
    eprintln!("--- Test 7: duplicate bases, different random scalars ---");
    for trial in 0..5 {
        let same_bases: Vec<G1Affine> = vec![unique_bases[0]; n];
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let cpu = msm_best(&scalars, &same_bases);
        let gpu = msm_gpu::<G1Affine>(&scalars, &same_bases);
        eprint!("  trial {trial}: {} ", if cpu == gpu { "✅" } else { "❌" });
    }
    eprintln!();
}
