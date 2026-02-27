// Tests for GPU-accelerated MSM.
// This file is included as `mod tests` under `#[cfg(test)]` in gpu/mod.rs.

use crate::bn256::{Fq, Fr, G1Affine, G1};
use crate::gpu::{gpu_ctx, u64x4_to_u32x8, u32x8_to_u64x4};
use crate::CurveAffine;
use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
use group::{Curve, Group};
use metal::*;
use rand_core::OsRng;

/// Test that GPU fq_add, fq_sub, fq_neg match CPU.
#[test]
fn test_gpu_field_addsub() {
    let ctx = gpu_ctx();
    let n = 1024usize;

    let a_vals: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
    let b_vals: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();

    // Test all 3 operations: 0=add, 1=sub, 2=neg
    for (op_id, op_name) in [(0u32, "add"), (1, "sub"), (2, "neg")] {
        let expected: Vec<Fq> = match op_id {
            0 => a_vals.iter().zip(b_vals.iter()).map(|(a, b)| *a + *b).collect(),
            1 => a_vals.iter().zip(b_vals.iter()).map(|(a, b)| *a - *b).collect(),
            2 => a_vals.iter().map(|a| -*a).collect(),
            _ => unreachable!(),
        };

        let mut a_data = vec![0u32; n * 8];
        let mut b_data = vec![0u32; n * 8];
        for i in 0..n {
            a_data[i * 8..(i + 1) * 8].copy_from_slice(&u64x4_to_u32x8(&a_vals[i].0));
            b_data[i * 8..(i + 1) * 8].copy_from_slice(&u64x4_to_u32x8(&b_vals[i].0));
        }
        let op_data = vec![op_id; n];

        let buf_a = ctx.device.new_buffer((a_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_b = ctx.device.new_buffer((b_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_c = ctx.device.new_buffer((n * 8 * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_op = ctx.device.new_buffer((op_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(a_data.as_ptr() as *const u8, buf_a.contents() as *mut u8, a_data.len() * 4);
            std::ptr::copy_nonoverlapping(b_data.as_ptr() as *const u8, buf_b.contents() as *mut u8, b_data.len() * 4);
            std::ptr::copy_nonoverlapping(op_data.as_ptr() as *const u8, buf_op.contents() as *mut u8, op_data.len() * 4);
        }

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.field_addsub_test_pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        enc.set_buffer(3, Some(&buf_op), 0);
        let max_tg = ctx.field_addsub_test_pipeline.max_total_threads_per_threadgroup() as u64;
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let c_ptr = buf_c.contents() as *const u32;
        let mut mismatches = 0;
        for i in 0..n {
            let mut gpu_limbs = [0u32; 8];
            for j in 0..8 { gpu_limbs[j] = unsafe { *c_ptr.add(i * 8 + j) }; }
            let gpu_result = Fq(u32x8_to_u64x4(&gpu_limbs));
            if gpu_result != expected[i] {
                if mismatches < 3 {
                    eprintln!("fq_{} mismatch at i={}: GPU={:?} CPU={:?}", op_name, i, gpu_result.0, expected[i].0);
                    eprintln!("  a={:?}", a_vals[i].0);
                    eprintln!("  b={:?}", b_vals[i].0);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "fq_{}: {} / {} mismatched", op_name, mismatches, n);
        eprintln!("✅ {} GPU fq_{} operations match CPU", n, op_name);
    }
}

/// Test that GPU fq_sqr matches fq_mul(a, a) for random inputs.
/// This isolates the dedicated squaring function from the rest of the pipeline.
#[test]
fn test_gpu_fq_sqr() {
    let ctx = gpu_ctx();
    let n = 4096usize;

    let a_vals: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();

    // Expected: a*a via CPU
    let expected: Vec<Fq> = a_vals.iter().map(|a| a.square()).collect();

    let mut a_data = vec![0u32; n * 8];
    for i in 0..n {
        a_data[i * 8..(i + 1) * 8].copy_from_slice(&u64x4_to_u32x8(&a_vals[i].0));
    }

    let buf_a = ctx.device.new_buffer((a_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    // Output: 16 u32s per element (sqr_result[8] + mul_result[8])
    let buf_c = ctx.device.new_buffer((n * 16 * 4) as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(a_data.as_ptr() as *const u8, buf_a.contents() as *mut u8, a_data.len() * 4);
    }

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.field_sqr_test_pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_c), 0);
    let max_tg = ctx.field_sqr_test_pipeline.max_total_threads_per_threadgroup() as u64;
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let c_ptr = buf_c.contents() as *const u32;
    let mut sqr_mismatches = 0;
    let mut mul_mismatches = 0;
    let mut sqr_vs_mul_mismatches = 0;
    for i in 0..n {
        let out_off = i * 16;
        let mut sqr_limbs = [0u32; 8];
        let mut mul_limbs = [0u32; 8];
        for j in 0..8 {
            sqr_limbs[j] = unsafe { *c_ptr.add(out_off + j) };
            mul_limbs[j] = unsafe { *c_ptr.add(out_off + 8 + j) };
        }
        let sqr_result = Fq(u32x8_to_u64x4(&sqr_limbs));
        let mul_result = Fq(u32x8_to_u64x4(&mul_limbs));

        if sqr_result != mul_result {
            if sqr_vs_mul_mismatches < 3 {
                eprintln!("fq_sqr != fq_mul(a,a) at i={}", i);
                eprintln!("  a    = {:?}", a_vals[i].0);
                eprintln!("  sqr  = {:?}", sqr_result.0);
                eprintln!("  mul  = {:?}", mul_result.0);
            }
            sqr_vs_mul_mismatches += 1;
        }
        if sqr_result != expected[i] {
            if sqr_mismatches < 3 {
                eprintln!("fq_sqr != CPU at i={}", i);
                eprintln!("  a    = {:?}", a_vals[i].0);
                eprintln!("  GPU  = {:?}", sqr_result.0);
                eprintln!("  CPU  = {:?}", expected[i].0);
            }
            sqr_mismatches += 1;
        }
        if mul_result != expected[i] {
            mul_mismatches += 1;
        }
    }
    eprintln!("fq_sqr vs fq_mul(a,a): {} mismatches", sqr_vs_mul_mismatches);
    eprintln!("fq_sqr vs CPU:         {} mismatches", sqr_mismatches);
    eprintln!("fq_mul(a,a) vs CPU:    {} mismatches", mul_mismatches);
    assert_eq!(sqr_vs_mul_mismatches, 0, "fq_sqr != fq_mul(a,a): {}/{} mismatched", sqr_vs_mul_mismatches, n);
    assert_eq!(sqr_mismatches, 0, "fq_sqr != CPU: {}/{} mismatched", sqr_mismatches, n);
    eprintln!("✅ {} GPU fq_sqr operations match CPU and fq_mul(a,a)", n);
}

/// Test that GPU fq_mul matches CPU Fq::mul for random inputs.
#[test]
fn test_gpu_field_mul() {
    let ctx = gpu_ctx();
    let n = 1024usize;

    // Generate random field elements
    let a_vals: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
    let b_vals: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();

    // Expected results (CPU)
    let expected: Vec<Fq> = a_vals.iter().zip(b_vals.iter()).map(|(a, b)| *a * *b).collect();

    // Pack into GPU buffers
    let mut a_data = vec![0u32; n * 8];
    let mut b_data = vec![0u32; n * 8];
    for i in 0..n {
        let a_limbs = u64x4_to_u32x8(&a_vals[i].0);
        let b_limbs = u64x4_to_u32x8(&b_vals[i].0);
        a_data[i * 8..(i + 1) * 8].copy_from_slice(&a_limbs);
        b_data[i * 8..(i + 1) * 8].copy_from_slice(&b_limbs);
    }

    let buf_a = ctx.device.new_buffer(
        (a_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_b = ctx.device.new_buffer(
        (b_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_c = ctx.device.new_buffer(
        (n * 8 * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            a_data.as_ptr() as *const u8,
            buf_a.contents() as *mut u8,
            a_data.len() * 4,
        );
        std::ptr::copy_nonoverlapping(
            b_data.as_ptr() as *const u8,
            buf_b.contents() as *mut u8,
            b_data.len() * 4,
        );
    }

    // Dispatch
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.field_test_pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_c), 0);
    let max_tg = ctx.field_test_pipeline.max_total_threads_per_threadgroup() as u64;
    enc.dispatch_threads(
        MTLSize::new(n as u64, 1, 1),
        MTLSize::new(max_tg.min(n as u64), 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Read back and compare
    let c_ptr = buf_c.contents() as *const u32;
    let mut mismatches = 0;
    for i in 0..n {
        let mut gpu_limbs = [0u32; 8];
        for j in 0..8 {
            gpu_limbs[j] = unsafe { *c_ptr.add(i * 8 + j) };
        }
        let gpu_result = Fq(u32x8_to_u64x4(&gpu_limbs));
        if gpu_result != expected[i] {
            if mismatches < 5 {
                eprintln!(
                    "Mismatch at i={}: GPU={:?} CPU={:?}",
                    i, gpu_result.0, expected[i].0
                );
                eprintln!("  a={:?}", a_vals[i].0);
                eprintln!("  b={:?}", b_vals[i].0);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{} / {} field multiplications mismatched", mismatches, n);
    eprintln!("✅ {} GPU field multiplications match CPU", n);
}

/// Test that GPU Jacobian mixed-add matches CPU point addition.
/// The GPU now uses Jacobian coordinates internally.
/// Input: Jacobian (X,Y,Z) + Affine (x,y)
/// Output: Jacobian (X',Y',Z')
/// We convert Jacobian output to affine for comparison.
#[test]
fn test_gpu_jacobian_madd() {
    use ff::Field;
    let ctx = gpu_ctx();

    // Test: random affine (as Jacobian Z=1) + random affine
    // This mimics the MSM kernel's usage pattern:
    //   first point loaded as (x, y, 1), then madd with subsequent points.
    {
        let n = 16usize;
        // "Jacobian" accumulator points: affine points as (x, y, 1)
        let jac_aff_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
        let aff_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();

        // Expected: jac + aff (both as affine, result is in affine for comparison)
        let expected_aff: Vec<G1Affine> = jac_aff_points.iter().zip(aff_points.iter())
            .map(|(j, a)| (j.to_curve() + a).to_affine())
            .collect();

        // Pack Jacobian (x, y, 1_mont) — affine as Jacobian
        let mut jac_data = vec![0u32; n * 24];
        let one = Fq::ONE;
        for i in 0..n {
            let coords = jac_aff_points[i].coordinates().unwrap();
            jac_data[i * 24..i * 24 + 8].copy_from_slice(&u64x4_to_u32x8(&coords.x().0));
            jac_data[i * 24 + 8..i * 24 + 16].copy_from_slice(&u64x4_to_u32x8(&coords.y().0));
            jac_data[i * 24 + 16..i * 24 + 24].copy_from_slice(&u64x4_to_u32x8(&one.0));
        }

        let mut aff_data = vec![0u32; n * 16];
        for i in 0..n {
            let coords = aff_points[i].coordinates().unwrap();
            aff_data[i * 16..i * 16 + 8].copy_from_slice(&u64x4_to_u32x8(&coords.x().0));
            aff_data[i * 16 + 8..i * 16 + 16].copy_from_slice(&u64x4_to_u32x8(&coords.y().0));
        }

        let sign_data: Vec<u32> = vec![1u32; n];

        let buf_jac = ctx.device.new_buffer((jac_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_aff = ctx.device.new_buffer((aff_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_sign = ctx.device.new_buffer((sign_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_out = ctx.device.new_buffer((n * 24 * 4) as u64, MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(jac_data.as_ptr() as *const u8, buf_jac.contents() as *mut u8, jac_data.len() * 4);
            std::ptr::copy_nonoverlapping(aff_data.as_ptr() as *const u8, buf_aff.contents() as *mut u8, aff_data.len() * 4);
            std::ptr::copy_nonoverlapping(sign_data.as_ptr() as *const u8, buf_sign.contents() as *mut u8, sign_data.len() * 4);
        }

        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.jacobian_madd_test_pipeline);
        enc.set_buffer(0, Some(&buf_jac), 0);
        enc.set_buffer(1, Some(&buf_aff), 0);
        enc.set_buffer(2, Some(&buf_sign), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        let max_tg = ctx.jacobian_madd_test_pipeline.max_total_threads_per_threadgroup() as u64;
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let out_ptr = buf_out.contents() as *const u64;
        let mut mismatches = 0;
        for i in 0..n {
            let off = i * 12; // 12 u64s per Jacobian point (3 × Fq of 4 u64s)
            // Convert Jacobian output to G1 then affine
            let gpu_point = unsafe { super::read_jacobian_point(out_ptr, off) };
            let gpu_aff = gpu_point.map(|p| p.to_affine())
                .unwrap_or(G1Affine::identity());

            if gpu_aff != expected_aff[i] {
                if mismatches < 3 {
                    eprintln!("Affine+Affine(Jacobian) mismatch at i={}:", i);
                    eprintln!("  GPU aff: {:?}", gpu_aff);
                    eprintln!("  CPU aff: {:?}", expected_aff[i]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "Jacobian madd (Z=1): {} / {} mismatched", mismatches, n);
        eprintln!("✅ {} Jacobian madd (Z=1 + affine) tests passed", n);
    }

    // Test chained madd: start with affine (x,y,1), add 4 more affine points
    // This tests the accumulator with Z ≠ 1
    {
        let n = 64usize;
        let num_adds = 4;
        let all_points: Vec<Vec<G1Affine>> = (0..n)
            .map(|_| (0..num_adds + 1).map(|_| G1Affine::random(OsRng)).collect())
            .collect();

        // Expected: chain of additions
        let expected_aff: Vec<G1Affine> = all_points.iter().map(|pts| {
            let mut acc = pts[0].to_curve();
            for p in &pts[1..] {
                acc = acc + p;
            }
            acc.to_affine()
        }).collect();

        // For each test case, run num_adds madd operations on GPU
        let one = Fq::ONE;
        let mut mismatches = 0;
        for i in 0..n {
            // Start with first point as Jacobian (x, y, 1)
            let mut jac_x = u64x4_to_u32x8(&all_points[i][0].coordinates().unwrap().x().0);
            let mut jac_y = u64x4_to_u32x8(&all_points[i][0].coordinates().unwrap().y().0);
            let mut jac_z = u64x4_to_u32x8(&one.0);

            for add_idx in 1..=num_adds {
                let aff = &all_points[i][add_idx];
                let coords = aff.coordinates().unwrap();
                let aff_x = u64x4_to_u32x8(&coords.x().0);
                let aff_y = u64x4_to_u32x8(&coords.y().0);

                let mut jac_data = vec![0u32; 24];
                jac_data[0..8].copy_from_slice(&jac_x);
                jac_data[8..16].copy_from_slice(&jac_y);
                jac_data[16..24].copy_from_slice(&jac_z);

                let mut aff_data = vec![0u32; 16];
                aff_data[0..8].copy_from_slice(&aff_x);
                aff_data[8..16].copy_from_slice(&aff_y);

                let sign_data = vec![1u32; 1];

                let buf_jac = ctx.device.new_buffer((jac_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
                let buf_aff = ctx.device.new_buffer((aff_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
                let buf_sign = ctx.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
                let buf_out = ctx.device.new_buffer((24 * 4) as u64, MTLResourceOptions::StorageModeShared);

                unsafe {
                    std::ptr::copy_nonoverlapping(jac_data.as_ptr() as *const u8, buf_jac.contents() as *mut u8, jac_data.len() * 4);
                    std::ptr::copy_nonoverlapping(aff_data.as_ptr() as *const u8, buf_aff.contents() as *mut u8, aff_data.len() * 4);
                    std::ptr::copy_nonoverlapping(sign_data.as_ptr() as *const u8, buf_sign.contents() as *mut u8, 4);
                }

                let cb = ctx.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.jacobian_madd_test_pipeline);
                enc.set_buffer(0, Some(&buf_jac), 0);
                enc.set_buffer(1, Some(&buf_aff), 0);
                enc.set_buffer(2, Some(&buf_sign), 0);
                enc.set_buffer(3, Some(&buf_out), 0);
                enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                // Read back Jacobian output and feed into next iteration
                let out_ptr = buf_out.contents() as *const u32;
                for j in 0..8 {
                    jac_x[j] = unsafe { *out_ptr.add(j) };
                    jac_y[j] = unsafe { *out_ptr.add(8 + j) };
                    jac_z[j] = unsafe { *out_ptr.add(16 + j) };
                }
            }

            // Convert final Jacobian to affine
            let xj = Fq(u32x8_to_u64x4(&jac_x));
            let yj = Fq(u32x8_to_u64x4(&jac_y));
            let zj = Fq(u32x8_to_u64x4(&jac_z));
            let gpu_g1 = super::jacobian_to_g1(xj, yj, zj);
            let gpu_aff = gpu_g1.to_affine();

            if gpu_aff != expected_aff[i] {
                if mismatches < 3 {
                    eprintln!("Chained madd mismatch at i={}:", i);
                    eprintln!("  GPU aff: {:?}", gpu_aff);
                    eprintln!("  CPU aff: {:?}", expected_aff[i]);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "Chained madd: {} / {} mismatched", mismatches, n);
        eprintln!("✅ {} chained Jacobian madd (5-point chains) tests passed", n);
    }
}

/// Test GPU jacobian_add correctness.
#[test]
fn test_gpu_jacobian_add() {
    let ctx = gpu_ctx();
    let n = 256usize;

    // Generate random Jacobian points: use affine (x, y, 1)
    let pts1: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
    let pts2: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();

    let expected: Vec<G1Affine> = pts1.iter().zip(pts2.iter())
        .map(|(a, b)| (a.to_curve() + b).to_affine())
        .collect();

    let one = Fq::ONE;
    let mut p1_data = vec![0u32; n * 24];
    let mut p2_data = vec![0u32; n * 24];
    for i in 0..n {
        let c1 = pts1[i].coordinates().unwrap();
        p1_data[i*24..i*24+8].copy_from_slice(&u64x4_to_u32x8(&c1.x().0));
        p1_data[i*24+8..i*24+16].copy_from_slice(&u64x4_to_u32x8(&c1.y().0));
        p1_data[i*24+16..i*24+24].copy_from_slice(&u64x4_to_u32x8(&one.0));
        let c2 = pts2[i].coordinates().unwrap();
        p2_data[i*24..i*24+8].copy_from_slice(&u64x4_to_u32x8(&c2.x().0));
        p2_data[i*24+8..i*24+16].copy_from_slice(&u64x4_to_u32x8(&c2.y().0));
        p2_data[i*24+16..i*24+24].copy_from_slice(&u64x4_to_u32x8(&one.0));
    }

    let buf_p1 = ctx.device.new_buffer((p1_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_p2 = ctx.device.new_buffer((p2_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_out = ctx.device.new_buffer((n * 24 * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(p1_data.as_ptr() as *const u8, buf_p1.contents() as *mut u8, p1_data.len() * 4);
        std::ptr::copy_nonoverlapping(p2_data.as_ptr() as *const u8, buf_p2.contents() as *mut u8, p2_data.len() * 4);
    }

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.jacobian_add_test_pipeline);
    enc.set_buffer(0, Some(&buf_p1), 0);
    enc.set_buffer(1, Some(&buf_p2), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    let max_tg = ctx.jacobian_add_test_pipeline.max_total_threads_per_threadgroup() as u64;
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out_ptr = buf_out.contents() as *const u64;
    let mut mismatches = 0;
    for i in 0..n {
        let off = i * 12;
        let gpu_point = unsafe { super::read_jacobian_point(out_ptr, off) };
        let gpu_aff = gpu_point.map(|p| p.to_affine()).unwrap_or(G1Affine::identity());
        if gpu_aff != expected[i] {
            if mismatches < 3 {
                eprintln!("jacobian_add mismatch at i={}", i);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "jacobian_add: {}/{} mismatched", mismatches, n);
    eprintln!("✅ {} GPU jacobian_add operations match CPU", n);
}

/// Test GPU jacobian_dbl correctness.
#[test]
fn test_gpu_jacobian_dbl() {
    let ctx = gpu_ctx();
    let n = 256usize;

    let pts: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
    let expected: Vec<G1Affine> = pts.iter()
        .map(|a| a.to_curve().double().to_affine())
        .collect();

    let one = Fq::ONE;
    let mut p_data = vec![0u32; n * 24];
    for i in 0..n {
        let c = pts[i].coordinates().unwrap();
        p_data[i*24..i*24+8].copy_from_slice(&u64x4_to_u32x8(&c.x().0));
        p_data[i*24+8..i*24+16].copy_from_slice(&u64x4_to_u32x8(&c.y().0));
        p_data[i*24+16..i*24+24].copy_from_slice(&u64x4_to_u32x8(&one.0));
    }

    let buf_p = ctx.device.new_buffer((p_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_out = ctx.device.new_buffer((n * 24 * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(p_data.as_ptr() as *const u8, buf_p.contents() as *mut u8, p_data.len() * 4);
    }

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.jacobian_dbl_test_pipeline);
    enc.set_buffer(0, Some(&buf_p), 0);
    enc.set_buffer(1, Some(&buf_out), 0);
    let max_tg = ctx.jacobian_dbl_test_pipeline.max_total_threads_per_threadgroup() as u64;
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out_ptr = buf_out.contents() as *const u64;
    let mut mismatches = 0;
    for i in 0..n {
        let off = i * 12;
        let gpu_point = unsafe { super::read_jacobian_point(out_ptr, off) };
        let gpu_aff = gpu_point.map(|p| p.to_affine()).unwrap_or(G1Affine::identity());
        if gpu_aff != expected[i] {
            if mismatches < 3 {
                eprintln!("jacobian_dbl mismatch at i={}", i);
                eprintln!("  GPU: {:?}", gpu_aff);
                eprintln!("  CPU: {:?}", expected[i]);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "jacobian_dbl: {}/{} mismatched", mismatches, n);
    eprintln!("✅ {} GPU jacobian_dbl operations match CPU", n);
}

/// Test GPU double_and_add correctness with various scalar sizes.
#[test]
fn test_gpu_double_and_add() {
    let ctx = gpu_ctx();

    // Test with small scalars and also large ones (up to 4096)
    let test_scalars: Vec<u32> = vec![
        1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256,
        511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192,
    ];
    let n = test_scalars.len();

    let pts: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();

    // CPU expected: scalar * P via repeated doubling
    let expected: Vec<G1Affine> = pts.iter().zip(test_scalars.iter())
        .map(|(p, &s)| {
            let mut acc = G1::identity();
            let pg = p.to_curve();
            let mut tmp = pg;
            let mut sc = s;
            while sc != 0 {
                if sc & 1 != 0 { acc = acc + tmp; }
                tmp = tmp.double();
                sc >>= 1;
            }
            acc.to_affine()
        })
        .collect();

    let one = Fq::ONE;
    let mut p_data = vec![0u32; n * 24];
    for i in 0..n {
        let c = pts[i].coordinates().unwrap();
        p_data[i*24..i*24+8].copy_from_slice(&u64x4_to_u32x8(&c.x().0));
        p_data[i*24+8..i*24+16].copy_from_slice(&u64x4_to_u32x8(&c.y().0));
        p_data[i*24+16..i*24+24].copy_from_slice(&u64x4_to_u32x8(&one.0));
    }

    let buf_p = ctx.device.new_buffer((p_data.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_s = ctx.device.new_buffer((test_scalars.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_out = ctx.device.new_buffer((n * 24 * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(p_data.as_ptr() as *const u8, buf_p.contents() as *mut u8, p_data.len() * 4);
        std::ptr::copy_nonoverlapping(test_scalars.as_ptr() as *const u8, buf_s.contents() as *mut u8, test_scalars.len() * 4);
    }

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.double_and_add_test_pipeline);
    enc.set_buffer(0, Some(&buf_p), 0);
    enc.set_buffer(1, Some(&buf_s), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    let max_tg = ctx.double_and_add_test_pipeline.max_total_threads_per_threadgroup() as u64;
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(max_tg.min(n as u64), 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out_ptr = buf_out.contents() as *const u64;
    let mut mismatches = 0;
    for i in 0..n {
        let off = i * 12;
        let gpu_point = unsafe { super::read_jacobian_point(out_ptr, off) };
        let gpu_aff = gpu_point.map(|p| p.to_affine()).unwrap_or(G1Affine::identity());
        if gpu_aff != expected[i] {
            if mismatches < 5 {
                eprintln!("double_and_add mismatch at scalar={}: GPU={:?} CPU={:?}",
                    test_scalars[i], gpu_aff, expected[i]);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "double_and_add: {}/{} mismatched", mismatches, n);
    eprintln!("✅ {} GPU double_and_add operations match CPU", n);
}

/// Test full msm_gpu correctness against msm_best.
#[test]
fn test_msm_gpu_correctness() {
    use crate::msm::msm_best;

    for k in 3..=16 {
        let n = 1usize << k;
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        let points_proj: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let mut points = vec![G1Affine::identity(); n];
        G1::batch_normalize(&points_proj, &mut points);

        let expected: G1 = msm_best(&scalars, &points);
        let gpu_result: G1 = crate::gpu::msm_gpu(&scalars, &points);

        assert_eq!(
            gpu_result.to_affine(),
            expected.to_affine(),
            "msm_gpu != msm_best at k={}",
            k
        );
        eprintln!("✅ msm_gpu matches msm_best at k={} (n={})", k, n);
    }
}

/// Test msm_gpu edge cases.
#[test]
fn test_msm_gpu_edge_cases() {
    use crate::msm::msm_best;

    // All-zero scalars
    {
        let n = 1 << 14;
        let scalars = vec![Fr::ZERO; n];
        let points: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
        let result: G1 = crate::gpu::msm_gpu(&scalars, &points);
        assert_eq!(result, G1::identity(), "All-zero scalars should give identity");
    }

    // All-ones scalars
    {
        let n = 1 << 14;
        let scalars = vec![Fr::ONE; n];
        let points_proj: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let mut points = vec![G1Affine::identity(); n];
        G1::batch_normalize(&points_proj, &mut points);

        let expected: G1 = msm_best(&scalars, &points);
        let result: G1 = crate::gpu::msm_gpu(&scalars, &points);
        assert_eq!(result.to_affine(), expected.to_affine(), "All-ones test failed");
    }
}

/// Test msm_gpu_glv correctness against msm_best.
#[test]
fn test_msm_gpu_glv_correctness() {
    use crate::msm::msm_best;

    for k in 3..=16 {
        let n = 1usize << k;
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();

        let points_proj: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let mut points = vec![G1Affine::identity(); n];
        G1::batch_normalize(&points_proj, &mut points);

        let expected: G1 = msm_best(&scalars, &points);
        let gpu_result: G1 = crate::gpu::msm_gpu_glv(&scalars, &points);

        assert_eq!(
            gpu_result.to_affine(),
            expected.to_affine(),
            "msm_gpu_glv != msm_best at k={}",
            k
        );
        eprintln!("✅ msm_gpu_glv matches msm_best at k={} (n={})", k, n);
    }
}

/// Timing breakdown test — prints per-phase times for GPU MSM.
/// Run with: cargo test --features gpu --release -- test_msm_gpu_timing --nocapture --ignored
#[test]
#[ignore]
fn test_msm_gpu_timing() {
    for k in [14, 18, 20, 22, 24] {
        let n = 1usize << k;
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
        let points_proj: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let mut points = vec![G1Affine::identity(); n];
        G1::batch_normalize(&points_proj, &mut points);

        let (_result, timing) = crate::gpu::msm_gpu_timed(&scalars, &points);
        eprintln!("k={k:2} | n={n:>10} | c={c} | windows={w} | \
                   encode={enc:.1}ms scatter={scat:.1}ms base_pack={bp:.1}ms \
                   gpu_upload={up:.1}ms gpu_kernel={gk:.1}ms gpu_reduce={gr:.1}ms \
                   cpu_reduce={cr:.1}ms | total={tot:.1}ms",
            c = timing.c,
            w = timing.num_windows,
            enc = timing.scalar_encode_ms,
            scat = timing.scatter_build_ms,
            bp = timing.base_pack_ms,
            up = timing.gpu_upload_ms,
            gk = timing.gpu_kernel_ms,
            gr = timing.gpu_reduce_ms,
            cr = timing.cpu_reduce_ms,
            tot = timing.total_ms,
        );
    }
}

/// Buffer pool reuse test — calls MSM 5 times at the same k to measure warm-pool benefit.
/// Run with: cargo test --features gpu --release -- test_msm_gpu_pool_reuse --nocapture --ignored
#[test]
#[ignore]
fn test_msm_gpu_pool_reuse() {
    for k in [18, 20, 22] {
        let n = 1usize << k;
        let points_proj: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let mut points = vec![G1Affine::identity(); n];
        G1::batch_normalize(&points_proj, &mut points);

        eprintln!("\n=== k={k} (n={n}) — 5 consecutive MSM calls ===");
        for iter in 0..5 {
            // Fresh scalars each time to avoid any caching of scalar-dependent work
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
            let (_result, timing) = crate::gpu::msm_gpu_timed(&scalars, &points);
            eprintln!("  iter {iter}: upload={up:.1}ms kernel={gk:.1}ms total={tot:.1}ms",
                up = timing.gpu_upload_ms,
                gk = timing.gpu_kernel_ms,
                tot = timing.total_ms,
            );
        }
    }
}
