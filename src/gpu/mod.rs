//! Metal GPU-accelerated MSM for BN254.
//!
//! This module provides `msm_gpu`, a GPU-accelerated multi-scalar multiplication
//! using Apple's Metal compute framework. The architecture is hybrid CPU+GPU:
//!
//! - **CPU**: Booth-encodes scalars, builds a scatter table (CSR format)
//! - **GPU**: Bucket accumulation via Jacobian mixed-addition (the bottleneck)
//! - **CPU**: Summation-by-parts, window shift, final combination
//!
//! Requires the `gpu` feature flag and an Apple Silicon (or other Metal-capable) device.

use metal::*;
use std::sync::OnceLock;

mod shader;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Metal infrastructure (lazy-initialized singletons)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(crate) struct GpuContext {
    device: Device,
    queue: CommandQueue,
    field_test_pipeline: ComputePipelineState,
    field_addsub_test_pipeline: ComputePipelineState,
    jacobian_madd_test_pipeline: ComputePipelineState,
    bucket_accumulate_pipeline: ComputePipelineState,
    bucket_accumulate_all_pipeline: ComputePipelineState,
    bucket_reduce_stage1_pipeline: ComputePipelineState,
    bucket_reduce_stage2_pipeline: ComputePipelineState,
    bucket_reduce_stage1_all_pipeline: ComputePipelineState,
    bucket_reduce_stage2_all_pipeline: ComputePipelineState,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

pub(crate) fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        let device = Device::system_default().expect("No Metal GPU device found");
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader::SHADER_SOURCE, &options)
            .expect("Failed to compile Metal shader");

        let field_test_fn = library
            .get_function("test_field_ops", None)
            .expect("Missing test_field_ops kernel");
        let field_test_pipeline = device
            .new_compute_pipeline_state_with_function(&field_test_fn)
            .expect("Failed to create field_test pipeline");

        let field_addsub_fn = library
            .get_function("test_field_addsub", None)
            .expect("Missing test_field_addsub kernel");
        let field_addsub_test_pipeline = device
            .new_compute_pipeline_state_with_function(&field_addsub_fn)
            .expect("Failed to create field_addsub_test pipeline");

        let jacobian_madd_test_fn = library
            .get_function("test_jacobian_madd", None)
            .expect("Missing test_jacobian_madd kernel");
        let jacobian_madd_test_pipeline = device
            .new_compute_pipeline_state_with_function(&jacobian_madd_test_fn)
            .expect("Failed to create jacobian_madd_test pipeline");

        let bucket_fn = library
            .get_function("bucket_accumulate", None)
            .expect("Missing bucket_accumulate kernel");
        let bucket_accumulate_pipeline = device
            .new_compute_pipeline_state_with_function(&bucket_fn)
            .expect("Failed to create bucket_accumulate pipeline");

        let reduce1_fn = library
            .get_function("bucket_reduce_stage1", None)
            .expect("Missing bucket_reduce_stage1 kernel");
        let bucket_reduce_stage1_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce1_fn)
            .expect("Failed to create bucket_reduce_stage1 pipeline");

        let reduce2_fn = library
            .get_function("bucket_reduce_stage2", None)
            .expect("Missing bucket_reduce_stage2 kernel");
        let bucket_reduce_stage2_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce2_fn)
            .expect("Failed to create bucket_reduce_stage2 pipeline");

        let bucket_all_fn = library
            .get_function("bucket_accumulate_all", None)
            .expect("Missing bucket_accumulate_all kernel");
        let bucket_accumulate_all_pipeline = device
            .new_compute_pipeline_state_with_function(&bucket_all_fn)
            .expect("Failed to create bucket_accumulate_all pipeline");

        let reduce1_all_fn = library
            .get_function("bucket_reduce_stage1_all", None)
            .expect("Missing bucket_reduce_stage1_all kernel");
        let bucket_reduce_stage1_all_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce1_all_fn)
            .expect("Failed to create bucket_reduce_stage1_all pipeline");

        let reduce2_all_fn = library
            .get_function("bucket_reduce_stage2_all", None)
            .expect("Missing bucket_reduce_stage2_all kernel");
        let bucket_reduce_stage2_all_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce2_all_fn)
            .expect("Failed to create bucket_reduce_stage2_all pipeline");

        GpuContext {
            device,
            queue,
            field_test_pipeline,
            field_addsub_test_pipeline,
            jacobian_madd_test_pipeline,
            bucket_accumulate_pipeline,
            bucket_accumulate_all_pipeline,
            bucket_reduce_stage1_pipeline,
            bucket_reduce_stage2_pipeline,
            bucket_reduce_stage1_all_pipeline,
            bucket_reduce_stage2_all_pipeline,
        }
    })
}

// ---------------------------------------------------------------------------
// Data packing: Rust <-> GPU buffer conversion
// ---------------------------------------------------------------------------

/// Repack a Rust `[u64; 4]` (Montgomery form, little-endian) into `[u32; 8]`.
///
/// Rust stores Fq as 4 × u64 limbs. Metal uses 8 × u32 limbs.
/// The mapping is: u64 limb `i` → (u32 lo at index `2*i`, u32 hi at index `2*i+1`).
#[inline]
pub(crate) fn u64x4_to_u32x8(src: &[u64; 4]) -> [u32; 8] {
    [
        src[0] as u32,
        (src[0] >> 32) as u32,
        src[1] as u32,
        (src[1] >> 32) as u32,
        src[2] as u32,
        (src[2] >> 32) as u32,
        src[3] as u32,
        (src[3] >> 32) as u32,
    ]
}

/// Inverse of `u64x4_to_u32x8`: repack 8 × u32 back into 4 × u64.
#[inline]
pub(crate) fn u32x8_to_u64x4(src: &[u32; 8]) -> [u64; 4] {
    [
        src[0] as u64 | ((src[1] as u64) << 32),
        src[2] as u64 | ((src[3] as u64) << 32),
        src[4] as u64 | ((src[5] as u64) << 32),
        src[6] as u64 | ((src[7] as u64) << 32),
    ]
}

// ---------------------------------------------------------------------------
// GPU MSM timing
// ---------------------------------------------------------------------------

use crate::arithmetic::CurveEndo;
use crate::bn256::{Fq, Fr, G1Affine, G1};
use crate::CurveAffine;
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::Group;

/// Per-phase timing breakdown for GPU MSM.
#[derive(Default, Clone)]
pub struct GpuMsmTiming {
    pub scalar_encode_ms: f64,
    pub base_pack_ms: f64,
    pub scatter_build_ms: f64,
    pub gpu_upload_ms: f64,
    pub gpu_kernel_ms: f64,
    pub gpu_reduce_ms: f64,
    pub cpu_reduce_ms: f64,
    pub total_ms: f64,
    /// Window size (c)
    pub c: usize,
    /// Number of buckets per window (2^{c-1})
    pub num_buckets: usize,
    /// Number of Pippenger windows
    pub num_windows: usize,
    /// Number of input points
    pub n: usize,
}

impl std::fmt::Display for GpuMsmTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  n={}, c={}, buckets={}, windows={}",
            self.n, self.c, self.num_buckets, self.num_windows)?;
        writeln!(f, "  scalar_encode:   {:>8.1} ms ({:>5.1}%)",
            self.scalar_encode_ms, 100.0 * self.scalar_encode_ms / self.total_ms)?;
        writeln!(f, "  base_pack:       {:>8.1} ms ({:>5.1}%)",
            self.base_pack_ms, 100.0 * self.base_pack_ms / self.total_ms)?;
        writeln!(f, "  scatter_build:   {:>8.1} ms ({:>5.1}%)",
            self.scatter_build_ms, 100.0 * self.scatter_build_ms / self.total_ms)?;
        writeln!(f, "  gpu_upload:      {:>8.1} ms ({:>5.1}%)",
            self.gpu_upload_ms, 100.0 * self.gpu_upload_ms / self.total_ms)?;
        writeln!(f, "  gpu_kernel:      {:>8.1} ms ({:>5.1}%)",
            self.gpu_kernel_ms, 100.0 * self.gpu_kernel_ms / self.total_ms)?;
        writeln!(f, "  gpu_reduce:      {:>8.1} ms ({:>5.1}%)",
            self.gpu_reduce_ms, 100.0 * self.gpu_reduce_ms / self.total_ms)?;
        writeln!(f, "  cpu_reduce:      {:>8.1} ms ({:>5.1}%)",
            self.cpu_reduce_ms, 100.0 * self.cpu_reduce_ms / self.total_ms)?;
        writeln!(f, "  ────────────────────────────")?;
        write!(f, "  TOTAL:           {:>8.1} ms", self.total_ms)
    }
}

/// GPU-accelerated multi-scalar multiplication for BN254 G1.
///
/// Falls back to CPU `msm_best` for small inputs (< 2^14 points).
pub fn msm_gpu(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
    msm_gpu_inner(coeffs, bases, false).0
}

/// GPU MSM with detailed phase timing breakdown.
pub fn msm_gpu_timed(coeffs: &[Fr], bases: &[G1Affine]) -> (G1, GpuMsmTiming) {
    msm_gpu_inner(coeffs, bases, true)
}

fn msm_gpu_inner(coeffs: &[Fr], bases: &[G1Affine], timed: bool) -> (G1, GpuMsmTiming) {
    use rayon::prelude::*;
    use std::time::Instant;

    assert_eq!(coeffs.len(), bases.len());

    let n = bases.len();
    let mut timing = GpuMsmTiming::default();
    timing.n = n;

    // Fall back for small inputs where GPU overhead dominates
    if n < (1 << 14) {
        let t0 = Instant::now();
        let result = crate::msm::msm_best(coeffs, bases);
        timing.total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return (result, timing);
    }

    let total_start = Instant::now();
    let ctx = gpu_ctx();
    let c = get_optimal_c_gpu(n);
    let num_buckets = 1usize << (c - 1);
    timing.c = c;
    timing.num_buckets = num_buckets;

    // 1. Serialize scalars to bytes (parallelized with rayon)
    let t0 = Instant::now();
    let coeffs_bytes: Vec<_> = coeffs.par_iter().map(|s| s.to_repr()).collect();
    if timed { timing.scalar_encode_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // 2. Zero-copy base points
    let t0 = Instant::now();
    assert_eq!(std::mem::size_of::<G1Affine>(), 64,
        "G1Affine must be 64 bytes for zero-copy GPU transfer");
    let bases_byte_len = n * std::mem::size_of::<G1Affine>();
    let bases_buf = ctx.device.new_buffer(
        bases_byte_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            bases.as_ptr() as *const u8,
            bases_buf.contents() as *mut u8,
            bases_byte_len,
        );
    }
    if timed { timing.base_pack_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // Number of windows
    let num_bits = Fr::NUM_BITS as usize;
    let number_of_windows = num_bits / c + 1;
    timing.num_windows = number_of_windows;

    // 3. Build ALL scatter tables in parallel (rayon)
    let t0 = Instant::now();
    let scatter_tables: Vec<_> = (0..number_of_windows)
        .into_par_iter()
        .map(|w| build_scatter_table(w, c, num_buckets, &coeffs_bytes))
        .collect();
    if timed { timing.scatter_build_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // 4. Build concatenated GPU buffers for all-windows-in-one-dispatch
    let t0 = Instant::now();

    // Count total scatter entries across all windows and identify non-empty windows
    let mut active_windows: Vec<usize> = Vec::new();
    let mut total_scatter_entries = 0usize;
    for w in 0..number_of_windows {
        if !scatter_tables[w].1.is_empty() {
            active_windows.push(w);
            total_scatter_entries += scatter_tables[w].1.len();
        }
    }
    let num_active = active_windows.len();

    // Compute per-window metadata and buffer sizes
    let total_offsets = num_active * (num_buckets + 1);
    let total_bucket_slots = num_active * num_buckets;

    let scatter_buf_bytes = (total_scatter_entries.max(1) * 4) as u64;
    let offsets_buf_bytes = (total_offsets.max(1) * 4) as u64;
    let buckets_buf_bytes = (total_bucket_slots.max(1) * 24 * 4) as u64;
    let params_buf_bytes  = (num_active.max(1) * 3 * 4) as u64;

    // Diagnostic logging only when timed (avoids hot-path overhead)
    if timed {
        let max_buf_len = ctx.device.max_buffer_length();
        let total_gpu_bytes = (bases_byte_len as u64) + scatter_buf_bytes + offsets_buf_bytes + buckets_buf_bytes + params_buf_bytes;
        let k = (n as f64).log2() as usize;
        eprintln!("[msm_gpu] n={n} (k={k}), c={c}, num_buckets={num_buckets}");
        eprintln!("[msm_gpu] Metal max_buffer_length: {:.2} GB", max_buf_len as f64 / (1u64 << 30) as f64);
        eprintln!("[msm_gpu] windows={number_of_windows}, active={num_active}, scatter_entries={total_scatter_entries}");
        eprintln!("[msm_gpu] Buffers: bases={:.3}GB scatter={:.3}GB offsets={:.3}GB buckets={:.3}GB TOTAL={:.3}GB",
            bases_byte_len as f64 / (1u64 << 30) as f64,
            scatter_buf_bytes as f64 / (1u64 << 30) as f64,
            offsets_buf_bytes as f64 / (1u64 << 30) as f64,
            buckets_buf_bytes as f64 / (1u64 << 30) as f64,
            total_gpu_bytes as f64 / (1u64 << 30) as f64);
        for (name, size) in [("bases_buf", bases_byte_len as u64), ("scatter_buf", scatter_buf_bytes), ("buckets_buf", buckets_buf_bytes)] {
            if size > max_buf_len {
                eprintln!("[msm_gpu] *** {name} ({:.2} GB) EXCEEDS Metal limit ({:.2} GB) ***",
                    size as f64 / (1u64 << 30) as f64, max_buf_len as f64 / (1u64 << 30) as f64);
            }
        }
    }

    // Debug-only safety checks (compiled out in release builds)
    debug_assert!(total_scatter_entries <= u32::MAX as usize,
        "msm_gpu: total_scatter_entries ({total_scatter_entries}) exceeds u32::MAX");

    // Allocate Metal buffers directly — write scatter/offsets data straight into
    // shared memory, avoiding intermediate Vec allocations + double-copy.
    let scatter_buf = ctx.device.new_buffer(
        scatter_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer(
        offsets_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let buckets_buf = ctx.device.new_buffer(
        buckets_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let window_params_buf = ctx.device.new_buffer(
        params_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    // Write data directly into Metal shared memory buffers
    let scatter_dst = scatter_buf.contents() as *mut u32;
    let offsets_dst = offsets_buf.contents() as *mut u32;
    let params_dst = window_params_buf.contents() as *mut u32;

    let mut scatter_cursor = 0u32;
    for (i, &w) in active_windows.iter().enumerate() {
        let (ref offsets, ref entries) = scatter_tables[w];
        let offsets_start = (i * (num_buckets + 1)) as u32;
        let buckets_start = (i * num_buckets) as u32;

        // Write window_params directly
        unsafe {
            *params_dst.add(i * 3) = scatter_cursor;
            *params_dst.add(i * 3 + 1) = offsets_start;
            *params_dst.add(i * 3 + 2) = buckets_start;
        }

        // Copy offsets directly into Metal buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                offsets.as_ptr(),
                offsets_dst.add(i * (num_buckets + 1)),
                num_buckets + 1,
            );
        }

        // Copy packed scatter entries directly into Metal buffer
        if !entries.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    entries.as_ptr(),
                    scatter_dst.add(scatter_cursor as usize),
                    entries.len(),
                );
            }
        }
        scatter_cursor += entries.len() as u32;
    }

    // No need to initialize bucket slots — the GPU kernel writes identity for empty
    // buckets and starts from identity for non-empty ones.
    if timed { timing.gpu_upload_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // 5. Dispatch bucket_accumulate_all kernel (single dispatch for ALL windows)
    let t0 = Instant::now();
    if num_active > 0 {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.bucket_accumulate_all_pipeline);
        enc.set_buffer(0, Some(&bases_buf), 0);
        enc.set_buffer(1, Some(&scatter_buf), 0);
        enc.set_buffer(2, Some(&offsets_buf), 0);
        enc.set_buffer(3, Some(&buckets_buf), 0);
        enc.set_buffer(4, Some(&window_params_buf), 0);

        let max_tg = ctx
            .bucket_accumulate_all_pipeline
            .max_total_threads_per_threadgroup() as u64;
        let grid = MTLSize::new(num_buckets as u64, num_active as u64, 1);
        let tg = MTLSize::new(max_tg.min(num_buckets as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    if timed { timing.gpu_kernel_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // 6. PBPR or CPU reduction
    //    Use PBPR whenever we have enough buckets for at least 2 threads with bpt >= 2.
    //    At num_buckets=4096: ideal threads = 4096/64 = 64, bpt = 64 — fine for PBPR.
    let use_pbpr = num_buckets >= 256;
    let num_reduce_threads = if use_pbpr {
        let max_tg = ctx.bucket_reduce_stage1_all_pipeline
            .max_total_threads_per_threadgroup() as usize;
        let ideal = (num_buckets / 64).max(2).min(max_tg);
        let mut t = 1;
        while t * 2 <= ideal && num_buckets % (t * 2) == 0 { t *= 2; }
        t
    } else { 0 };

    // Per-window results: g_points for each active window
    // Stored in window_results[window_index] = Vec<G1> of g_points per PBPR thread
    // or the full summation-by-parts result for CPU mode.
    //
    // We need per-window results to combine with Horner's method.

    let t0 = Instant::now();
    // For each active window, produce a single G1 point via PBPR or CPU reduction
    let mut window_results: Vec<(usize, G1)> = Vec::with_capacity(num_active);

    if use_pbpr && num_active > 0 {
        // GPU PBPR for all windows at once
        let g_points_buf = ctx.device.new_buffer(
            (num_active * num_reduce_threads * 24 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buf = ctx.device.new_buffer(
            (2 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let p = params_buf.contents() as *mut u32;
            *p = num_buckets as u32;
            *p.add(1) = num_reduce_threads as u32;
        }

        // Stage 1: all windows
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage1_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&g_points_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);

            let max_tg = ctx
                .bucket_reduce_stage1_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Stage 2: all windows
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage2_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&g_points_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);

            let max_tg = ctx
                .bucket_reduce_stage2_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        if timed { timing.gpu_reduce_ms = t0.elapsed().as_secs_f64() * 1000.0; }

        // Read back all g_points
        let t0 = Instant::now();
        let g_ptr = g_points_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let mut window_sum = G1::identity();
            for t in 0..num_reduce_threads {
                let off = (i * num_reduce_threads + t) * 12;
                let z_all_zero = unsafe {
                    (*g_ptr.add(off + 8) | *g_ptr.add(off + 9)
                     | *g_ptr.add(off + 10) | *g_ptr.add(off + 11)) == 0
                };
                if !z_all_zero {
                    let g_point = unsafe {
                        let p = g_ptr.add(off);
                        G1 {
                            x: Fq([*p, *p.add(1), *p.add(2), *p.add(3)]),
                            y: Fq([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]),
                            z: Fq([*p.add(8), *p.add(9), *p.add(10), *p.add(11)]),
                        }
                    };
                    window_sum = window_sum + g_point;
                }
            }
            window_results.push((w, window_sum));
        }
        if timed { timing.cpu_reduce_ms = t0.elapsed().as_secs_f64() * 1000.0; }
    } else {
        // CPU fallback: read back buckets and do summation-by-parts on CPU
        let t0 = Instant::now();
        let bucket_ptr = buckets_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let base_off = i * num_buckets * 12; // 12 u64s per Jacobian point
            // Summation-by-parts: running = B[num_buckets-1], sum += running, running += B[num_buckets-2], ...
            let mut running = G1::identity();
            let mut window_sum = G1::identity();
            for b in (0..num_buckets).rev() {
                let off = base_off + b * 12;
                let z_all_zero = unsafe {
                    (*bucket_ptr.add(off + 8) | *bucket_ptr.add(off + 9)
                     | *bucket_ptr.add(off + 10) | *bucket_ptr.add(off + 11)) == 0
                };
                if !z_all_zero {
                    let bucket_point = unsafe {
                        let p = bucket_ptr.add(off);
                        G1 {
                            x: Fq([*p, *p.add(1), *p.add(2), *p.add(3)]),
                            y: Fq([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]),
                            z: Fq([*p.add(8), *p.add(9), *p.add(10), *p.add(11)]),
                        }
                    };
                    running = running + bucket_point;
                }
                window_sum = window_sum + running;
            }
            window_results.push((w, window_sum));
        }
        if timed { timing.cpu_reduce_ms = t0.elapsed().as_secs_f64() * 1000.0; }
    }

    // 7. Combine window results using Horner's method
    //    total = Σ window_result[w] * 2^{w*c}
    //    Done via iterating from highest window down:
    //    acc = ((...(window[W-1] * 2^c + window[W-2]) * 2^c + ...) * 2^c + window[0])
    let t0 = Instant::now();
    // Build a lookup: window_index -> G1 point
    let mut window_map: Vec<Option<G1>> = vec![None; number_of_windows];
    for (w, pt) in window_results {
        window_map[w] = Some(pt);
    }

    let mut total_acc = G1::identity();
    for w in (0..number_of_windows).rev() {
        for _ in 0..c {
            total_acc = total_acc.double();
        }
        if let Some(pt) = window_map[w].take() {
            total_acc = total_acc + pt;
        }
    }
    if timed { timing.cpu_reduce_ms += t0.elapsed().as_secs_f64() * 1000.0; }

    timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    (total_acc, timing)
}

/// GPU-accelerated MSM with GLV endomorphism for BN254 G1.
///
/// Decomposes each 254-bit scalar `k` into two ~128-bit half-scalars via
/// `k·P = k1·P + k2·phi(P)` where `phi(P) = (zeta·x, y)`.
///
/// This halves the number of Pippenger windows at the cost of doubling
/// the point count per window. Combined with all-windows-in-one-dispatch,
/// the GPU does a single bucket accumulation + single PBPR reduction across
/// all windows in just two kernel launches (instead of 2×W per-window launches).
///
/// Falls back to `msm_gpu` for small inputs (< 2^14 points).
pub fn msm_gpu_glv(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
    use rayon::prelude::*;

    assert_eq!(coeffs.len(), bases.len());

    let n = bases.len();

    if n < (1 << 14) {
        return crate::msm::msm_best(coeffs, bases);
    }

    let ctx = gpu_ctx();

    // 1. GLV scalar decomposition: k = k1 + lambda * k2
    //    decompose_scalar returns (|k1|, k1_neg, |k2|, k2_neg) with k1,k2 ~128 bits
    let decomposed: Vec<(u128, bool, u128, bool)> = coeffs
        .par_iter()
        .map(|s| G1::decompose_scalar(s))
        .collect();

    // Convert k1, k2 to byte representations for Booth encoding
    let k1_bytes: Vec<[u8; 16]> = decomposed
        .iter()
        .map(|(k1, _, _, _): &(u128, bool, u128, bool)| k1.to_le_bytes())
        .collect();

    let k2_bytes: Vec<[u8; 16]> = decomposed
        .iter()
        .map(|(_, _, k2, _): &(u128, bool, u128, bool)| k2.to_le_bytes())
        .collect();

    // Sign flags: true = positive contribution to MSM
    let k1_signs: Vec<bool> = decomposed.iter().map(|(_, neg, _, _)| !neg).collect();
    let k2_signs: Vec<bool> = decomposed.iter().map(|(_, _, _, neg)| *neg).collect();

    // 2. Pack 2n affine bases into GPU buffer (parallelized)
    //    Layout: [P_0, P_1, ..., P_{n-1}, phi(P_0), phi(P_1), ..., phi(P_{n-1})]
    //    Endomorphism: phi(P) = (zeta * x, y) where zeta is cube root of unity in Fq.
    //    Both memcpy of original points and zeta-mul are parallelized with rayon.
    let zeta = <Fq as WithSmallOrderMulGroup<3>>::ZETA;
    let total_points = 2 * n;
    let bases_byte_len = total_points * 16 * 4; // 16 u32s per affine point
    let bases_buf = ctx.device.new_buffer(
        bases_byte_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    {
        let dst_ptr = bases_buf.contents() as usize; // usize for Send
        // Parallel over chunks for cache-friendliness
        bases.par_chunks(1024).enumerate().for_each(|(chunk_idx, chunk)| {
            let base_i = chunk_idx * 1024;
            let dst = dst_ptr as *mut u32;
            for (j, base) in chunk.iter().enumerate() {
                let i = base_i + j;
                let coords = base.coordinates().unwrap();
                let x_limbs = u64x4_to_u32x8(&coords.x().0);
                let y_limbs = u64x4_to_u32x8(&coords.y().0);
                let off = i * 16;
                unsafe {
                    std::ptr::copy_nonoverlapping(x_limbs.as_ptr(), dst.add(off), 8);
                    std::ptr::copy_nonoverlapping(y_limbs.as_ptr(), dst.add(off + 8), 8);
                }

                // Endomorphism base: phi(P) = (zeta * x, y)
                let endo_x = *coords.x() * zeta;
                let endo_x_limbs = u64x4_to_u32x8(&endo_x.0);
                let off2 = (n + i) * 16;
                unsafe {
                    std::ptr::copy_nonoverlapping(endo_x_limbs.as_ptr(), dst.add(off2), 8);
                    std::ptr::copy_nonoverlapping(y_limbs.as_ptr(), dst.add(off2 + 8), 8);
                }
            }
        });
    }

    // 3. Determine window parameters for ~128-bit scalars
    //    Use n (not 2n) for c selection: the scatter table has 2n entries per window
    //    but we want c tuned to the actual MSM density, not the inflated point count.
    let half_bits = 128usize;
    let c = get_optimal_c_gpu(n);
    let num_buckets = 1usize << (c - 1);
    let number_of_windows = half_bits / c + 1;

    // 4. Build ALL scatter tables in parallel
    let scatter_tables: Vec<_> = (0..number_of_windows)
        .into_par_iter()
        .map(|w| build_scatter_table_glv(
            w, c, num_buckets, n,
            &k1_bytes, &k1_signs,
            &k2_bytes, &k2_signs,
        ))
        .collect();

    // 5. Build concatenated GPU buffers for all-windows-in-one-dispatch
    //    Same architecture as msm_gpu_inner: flat buffers + window_params metadata
    let mut active_windows: Vec<usize> = Vec::new();
    let mut total_scatter_entries = 0usize;
    for w in 0..number_of_windows {
        if !scatter_tables[w].1.is_empty() {
            active_windows.push(w);
            total_scatter_entries += scatter_tables[w].1.len();
        }
    }
    let num_active = active_windows.len();

    let total_offsets = num_active * (num_buckets + 1);
    let total_bucket_slots = num_active * num_buckets;

    let scatter_buf_bytes = (total_scatter_entries.max(1) * 4) as u64;
    let offsets_buf_bytes = (total_offsets.max(1) * 4) as u64;
    let buckets_buf_bytes = (total_bucket_slots.max(1) * 24 * 4) as u64;
    let params_buf_bytes  = (num_active.max(1) * 3 * 4) as u64;

    // Debug-only safety checks (compiled out in release builds)
    debug_assert!(total_scatter_entries <= u32::MAX as usize,
        "msm_gpu_glv: total_scatter_entries exceeds u32::MAX");
    debug_assert!((2 * n) <= (1usize << 31),
        "msm_gpu_glv: 2*n exceeds u31 range for base_idx packing");

    // Allocate Metal buffers directly — write scatter/offsets data straight into
    // shared memory, avoiding intermediate Vec allocations + double-copy.
    let scatter_buf = ctx.device.new_buffer(
        scatter_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = ctx.device.new_buffer(
        offsets_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let buckets_buf = ctx.device.new_buffer(
        buckets_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let window_params_buf = ctx.device.new_buffer(
        params_buf_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let scatter_dst = scatter_buf.contents() as *mut u32;
    let offsets_dst = offsets_buf.contents() as *mut u32;
    let params_dst = window_params_buf.contents() as *mut u32;

    let mut scatter_cursor = 0u32;
    for (i, &w) in active_windows.iter().enumerate() {
        let (ref offsets, ref entries) = scatter_tables[w];
        let offsets_start = (i * (num_buckets + 1)) as u32;
        let buckets_start = (i * num_buckets) as u32;

        unsafe {
            *params_dst.add(i * 3) = scatter_cursor;
            *params_dst.add(i * 3 + 1) = offsets_start;
            *params_dst.add(i * 3 + 2) = buckets_start;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                offsets.as_ptr(),
                offsets_dst.add(i * (num_buckets + 1)),
                num_buckets + 1,
            );
        }

        if !entries.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    entries.as_ptr(),
                    scatter_dst.add(scatter_cursor as usize),
                    entries.len(),
                );
            }
        }
        scatter_cursor += entries.len() as u32;
    }

    // 6. Dispatch bucket_accumulate_all kernel (single dispatch for ALL windows)
    if num_active > 0 {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.bucket_accumulate_all_pipeline);
        enc.set_buffer(0, Some(&bases_buf), 0);
        enc.set_buffer(1, Some(&scatter_buf), 0);
        enc.set_buffer(2, Some(&offsets_buf), 0);
        enc.set_buffer(3, Some(&buckets_buf), 0);
        enc.set_buffer(4, Some(&window_params_buf), 0);

        let max_tg = ctx
            .bucket_accumulate_all_pipeline
            .max_total_threads_per_threadgroup() as u64;
        let grid = MTLSize::new(num_buckets as u64, num_active as u64, 1);
        let tg = MTLSize::new(max_tg.min(num_buckets as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // 7. PBPR or CPU reduction (same logic as msm_gpu_inner)
    let use_pbpr = num_buckets >= 256;
    let num_reduce_threads = if use_pbpr {
        let max_tg = ctx.bucket_reduce_stage1_all_pipeline
            .max_total_threads_per_threadgroup() as usize;
        let ideal = (num_buckets / 64).max(2).min(max_tg);
        let mut t = 1;
        while t * 2 <= ideal && num_buckets % (t * 2) == 0 { t *= 2; }
        t
    } else { 0 };

    let mut window_results: Vec<(usize, G1)> = Vec::with_capacity(num_active);

    if use_pbpr && num_active > 0 {
        // GPU PBPR for all windows at once
        let g_points_buf = ctx.device.new_buffer(
            (num_active * num_reduce_threads * 24 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buf = ctx.device.new_buffer(
            (2 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let p = params_buf.contents() as *mut u32;
            *p = num_buckets as u32;
            *p.add(1) = num_reduce_threads as u32;
        }

        // Stage 1: all windows
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage1_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&g_points_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);

            let max_tg = ctx
                .bucket_reduce_stage1_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Stage 2: all windows
        {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage2_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&g_points_buf), 0);
            enc.set_buffer(2, Some(&params_buf), 0);

            let max_tg = ctx
                .bucket_reduce_stage2_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Read back all g_points
        let g_ptr = g_points_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let mut window_sum = G1::identity();
            for t in 0..num_reduce_threads {
                let off = (i * num_reduce_threads + t) * 12;
                let z_all_zero = unsafe {
                    (*g_ptr.add(off + 8) | *g_ptr.add(off + 9)
                     | *g_ptr.add(off + 10) | *g_ptr.add(off + 11)) == 0
                };
                if !z_all_zero {
                    let g_point = unsafe {
                        let p = g_ptr.add(off);
                        G1 {
                            x: Fq([*p, *p.add(1), *p.add(2), *p.add(3)]),
                            y: Fq([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]),
                            z: Fq([*p.add(8), *p.add(9), *p.add(10), *p.add(11)]),
                        }
                    };
                    window_sum = window_sum + g_point;
                }
            }
            window_results.push((w, window_sum));
        }
    } else {
        // CPU fallback: read back buckets and do summation-by-parts on CPU
        let bucket_ptr = buckets_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let base_off = i * num_buckets * 12;
            let mut running = G1::identity();
            let mut window_sum = G1::identity();
            for b in (0..num_buckets).rev() {
                let off = base_off + b * 12;
                let z_all_zero = unsafe {
                    (*bucket_ptr.add(off + 8) | *bucket_ptr.add(off + 9)
                     | *bucket_ptr.add(off + 10) | *bucket_ptr.add(off + 11)) == 0
                };
                if !z_all_zero {
                    let bucket_point = unsafe {
                        let p = bucket_ptr.add(off);
                        G1 {
                            x: Fq([*p, *p.add(1), *p.add(2), *p.add(3)]),
                            y: Fq([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]),
                            z: Fq([*p.add(8), *p.add(9), *p.add(10), *p.add(11)]),
                        }
                    };
                    running = running + bucket_point;
                }
                window_sum = window_sum + running;
            }
            window_results.push((w, window_sum));
        }
    }

    // 8. Combine window results using Horner's method
    let mut window_map: Vec<Option<G1>> = vec![None; number_of_windows];
    for (w, pt) in window_results {
        window_map[w] = Some(pt);
    }

    let mut total_acc = G1::identity();
    for w in (0..number_of_windows).rev() {
        for _ in 0..c {
            total_acc = total_acc.double();
        }
        if let Some(pt) = window_map[w].take() {
            total_acc = total_acc + pt;
        }
    }

    total_acc
}

// ---------------------------------------------------------------------------
// Scatter table builders
// ---------------------------------------------------------------------------

use crate::msm::get_booth_index;

/// Build CSR scatter table for one Pippenger window (GLV variant).
///
/// Handles 2n points: indices 0..n for k1 bases, n..2n for k2 (endomorphism) bases.
fn build_scatter_table_glv(
    window: usize,
    c: usize,
    num_buckets: usize,
    n: usize,
    k1_bytes: &[[u8; 16]],
    k1_signs: &[bool],
    k2_bytes: &[[u8; 16]],
    k2_signs: &[bool],
) -> (Vec<u32>, Vec<u32>) {
    // First pass: count entries per bucket
    let mut counts = vec![0u32; num_buckets];

    for i in 0..n {
        let idx1 = get_booth_index(window, c, &k1_bytes[i]);
        if idx1 != 0 {
            let buck = idx1.unsigned_abs() as usize - 1;
            counts[buck] += 1;
        }
        let idx2 = get_booth_index(window, c, &k2_bytes[i]);
        if idx2 != 0 {
            let buck = idx2.unsigned_abs() as usize - 1;
            counts[buck] += 1;
        }
    }

    // Build prefix sums
    let mut bucket_offsets = vec![0u32; num_buckets + 1];
    for i in 0..num_buckets {
        bucket_offsets[i + 1] = bucket_offsets[i] + counts[i];
    }
    let total = bucket_offsets[num_buckets] as usize;

    // Second pass: fill scatter entries (packed: base_idx | sign<<31)
    let mut scatter_entries = vec![0u32; total];
    let mut write_pos = bucket_offsets[..num_buckets].to_vec();

    for i in 0..n {
        // k1 contribution: base point index = i
        let idx1 = get_booth_index(window, c, &k1_bytes[i]);
        if idx1 != 0 {
            // Effective sign: Booth sign XOR GLV k1 negation
            let booth_positive = idx1.is_positive();
            let effective_positive = booth_positive ^ (!k1_signs[i]);
            let buck = idx1.unsigned_abs() as usize - 1;
            let pos = write_pos[buck] as usize;
            scatter_entries[pos] = (i as u32) | ((effective_positive as u32) << 31);
            write_pos[buck] += 1;
        }

        // k2 contribution: base point index = n + i (endomorphism base)
        let idx2 = get_booth_index(window, c, &k2_bytes[i]);
        if idx2 != 0 {
            let booth_positive = idx2.is_positive();
            let effective_positive = booth_positive ^ (!k2_signs[i]);
            let buck = idx2.unsigned_abs() as usize - 1;
            let pos = write_pos[buck] as usize;
            scatter_entries[pos] = ((n + i) as u32) | ((effective_positive as u32) << 31);
            write_pos[buck] += 1;
        }
    }

    (bucket_offsets, scatter_entries)
}

/// Build CSR scatter table for one Pippenger window.
///
/// Returns `(bucket_offsets, scatter_entries)` where:
/// - `bucket_offsets[i]` = start index in `scatter_entries` for bucket `i`
/// - `scatter_entries[j]` = packed u32: `base_idx | (sign << 31)`
fn build_scatter_table(
    window: usize,
    c: usize,
    num_buckets: usize,
    coeffs_bytes: &[impl AsRef<[u8]>],
) -> (Vec<u32>, Vec<u32>) {
    // First pass: count entries per bucket
    let mut counts = vec![0u32; num_buckets];
    for (base_idx, coeff) in coeffs_bytes.iter().enumerate() {
        let idx = get_booth_index(window, c, coeff.as_ref());
        if idx != 0 {
            let buck = idx.unsigned_abs() as usize - 1;
            counts[buck] += 1;
            let _ = base_idx; // used in second pass
        }
    }

    // Build prefix sums (offsets)
    let mut bucket_offsets = vec![0u32; num_buckets + 1];
    for i in 0..num_buckets {
        bucket_offsets[i + 1] = bucket_offsets[i] + counts[i];
    }
    let total = bucket_offsets[num_buckets] as usize;

    // Second pass: fill scatter entries (packed: base_idx | sign<<31)
    let mut scatter_entries = vec![0u32; total];
    let mut write_pos = bucket_offsets[..num_buckets].to_vec(); // current write position per bucket

    for (base_idx, coeff) in coeffs_bytes.iter().enumerate() {
        let idx = get_booth_index(window, c, coeff.as_ref());
        if idx != 0 {
            let sign = idx.is_positive() as u32;
            let buck = idx.unsigned_abs() as usize - 1;
            let pos = write_pos[buck] as usize;
            scatter_entries[pos] = (base_idx as u32) | (sign << 31);
            write_pos[buck] += 1;
        }
    }

    (bucket_offsets, scatter_entries)
}

/// Optimal Pippenger window size for GPU.
///
/// GPU prefers larger windows (fewer dispatches) because:
/// 1. Bucket accumulation is GPU-parallel (more buckets = more parallelism)
/// 2. Fewer windows = fewer kernel launches (each has fixed overhead)
/// 3. PBPR handles the larger bucket count efficiently on GPU
///
/// Tuned based on zkmopro's empirical values and icicle's ARM decision tree.
fn get_optimal_c_gpu(n: usize) -> usize {
    let k = (n as f64).log2() as usize;
    match k {
        0..=13  => 10,
        14..=17 => 13,
        18..=21 => 15,
        22..=24 => 16,
        _       => 18,
    }
}
