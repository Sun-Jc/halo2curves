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
#[test]
fn test_gpu_jacobian_madd() {
    let ctx = gpu_ctx();

    // Start simple: identity + affine_point
    {
        let n = 16usize;
        let aff_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();

        // Jacobian identity: x=0, y=0, z=0 (but halo2curves uses x=0, y=1, z=0)
        // Let's use the actual G1 identity
        let identity = G1::identity();
        eprintln!("G1::identity() = x={:?}, y={:?}, z={:?}", identity.x.0, identity.y.0, identity.z.0);

        // Expected: identity + aff = aff (as projective)
        let expected: Vec<G1> = aff_points.iter().map(|a| identity + a).collect();

        // Pack Jacobian (identity repeated n times): n × 24 u32s
        let mut jac_data = vec![0u32; n * 24];
        for i in 0..n {
            let jx = u64x4_to_u32x8(&identity.x.0);
            let jy = u64x4_to_u32x8(&identity.y.0);
            let jz = u64x4_to_u32x8(&identity.z.0);
            jac_data[i * 24..i * 24 + 8].copy_from_slice(&jx);
            jac_data[i * 24 + 8..i * 24 + 16].copy_from_slice(&jy);
            jac_data[i * 24 + 16..i * 24 + 24].copy_from_slice(&jz);
        }

        let mut aff_data = vec![0u32; n * 16];
        for i in 0..n {
            let coords = aff_points[i].coordinates().unwrap();
            let ax = u64x4_to_u32x8(&coords.x().0);
            let ay = u64x4_to_u32x8(&coords.y().0);
            aff_data[i * 16..i * 16 + 8].copy_from_slice(&ax);
            aff_data[i * 16 + 8..i * 16 + 16].copy_from_slice(&ay);
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

        let out_ptr = buf_out.contents() as *const u32;
        let mut mismatches = 0;
        for i in 0..n {
            let off = i * 24;
            let mut x_l = [0u32; 8]; let mut y_l = [0u32; 8]; let mut z_l = [0u32; 8];
            for j in 0..8 {
                x_l[j] = unsafe { *out_ptr.add(off + j) };
                y_l[j] = unsafe { *out_ptr.add(off + 8 + j) };
                z_l[j] = unsafe { *out_ptr.add(off + 16 + j) };
            }
            let gpu_point = G1 {
                x: Fq(u32x8_to_u64x4(&x_l)),
                y: Fq(u32x8_to_u64x4(&y_l)),
                z: Fq(u32x8_to_u64x4(&z_l)),
            };

            let gpu_aff = gpu_point.to_affine();
            let exp_aff = expected[i].to_affine();

            if gpu_aff != exp_aff {
                if mismatches < 3 {
                    eprintln!("Identity+Affine mismatch at i={}:", i);
                    eprintln!("  GPU raw: x={:?} y={:?} z={:?}", gpu_point.x.0, gpu_point.y.0, gpu_point.z.0);
                    eprintln!("  GPU aff: {:?}", gpu_aff);
                    eprintln!("  CPU aff: {:?}", exp_aff);
                    eprintln!("  CPU raw: x={:?} y={:?} z={:?}", expected[i].x.0, expected[i].y.0, expected[i].z.0);
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            panic!("Identity+Affine: {} / {} mismatched", mismatches, n);
        }
        eprintln!("✅ {} identity + affine tests passed", n);
    }

    // Then test general case: random_jac + random_aff
    {
        let n = 256usize;
        let jac_points: Vec<G1> = (0..n).map(|_| G1::random(OsRng)).collect();
        let aff_points: Vec<G1Affine> = (0..n).map(|_| G1Affine::random(OsRng)).collect();
        let expected: Vec<G1> = jac_points.iter().zip(aff_points.iter()).map(|(j, a)| *j + a).collect();

        let mut jac_data = vec![0u32; n * 24];
        for i in 0..n {
            jac_data[i * 24..i * 24 + 8].copy_from_slice(&u64x4_to_u32x8(&jac_points[i].x.0));
            jac_data[i * 24 + 8..i * 24 + 16].copy_from_slice(&u64x4_to_u32x8(&jac_points[i].y.0));
            jac_data[i * 24 + 16..i * 24 + 24].copy_from_slice(&u64x4_to_u32x8(&jac_points[i].z.0));
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

        let out_ptr = buf_out.contents() as *const u32;
        let mut mismatches = 0;
        for i in 0..n {
            let off = i * 24;
            let mut x_l = [0u32; 8]; let mut y_l = [0u32; 8]; let mut z_l = [0u32; 8];
            for j in 0..8 { x_l[j] = unsafe { *out_ptr.add(off + j) }; y_l[j] = unsafe { *out_ptr.add(off + 8 + j) }; z_l[j] = unsafe { *out_ptr.add(off + 16 + j) }; }
            let gpu_point = G1 { x: Fq(u32x8_to_u64x4(&x_l)), y: Fq(u32x8_to_u64x4(&y_l)), z: Fq(u32x8_to_u64x4(&z_l)) };
            let gpu_aff = gpu_point.to_affine();
            let exp_aff = expected[i].to_affine();
            if gpu_aff != exp_aff {
                if mismatches < 3 {
                    eprintln!("General madd mismatch at i={}:", i);
                    eprintln!("  GPU raw: x={:?} y={:?} z={:?}", gpu_point.x.0, gpu_point.y.0, gpu_point.z.0);
                    eprintln!("  CPU raw: x={:?} y={:?} z={:?}", expected[i].x.0, expected[i].y.0, expected[i].z.0);
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "General madd: {} / {} mismatched", mismatches, n);
        eprintln!("✅ {} general Jacobian mixed-adds match CPU", n);
    }
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
