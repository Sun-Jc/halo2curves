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
use std::sync::{Mutex, OnceLock};

mod shader;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Metal infrastructure (lazy-initialized singletons)
// ---------------------------------------------------------------------------

/// Pool of reusable Metal buffers to avoid repeated kernel VM allocation.
///
/// Metal's `new_buffer()` with `StorageModeShared` involves kernel page-fault
/// allocation. For large buffers (48-64MB), this takes 50-200ms. By reusing
/// buffers across MSM calls, we amortize this cost to the first call only.
///
/// Strategy: best-fit with capacity rounding. Buffers are indexed by capacity
/// and reused when a buffer of sufficient size is available.
struct BufferPool {
    /// Available buffers sorted by capacity (ascending).
    available: Vec<(u64, Buffer)>,
}

impl BufferPool {
    fn new() -> Self {
        Self { available: Vec::new() }
    }

    /// Acquire a buffer with at least `min_bytes` capacity.
    /// Returns a pooled buffer if available, otherwise allocates a new one.
    /// The returned buffer's contents are NOT zeroed — caller must initialize.
    fn acquire(&mut self, device: &Device, min_bytes: u64) -> Buffer {
        // Find the smallest buffer that fits (best-fit to minimize waste)
        if let Some(pos) = self.available.iter().position(|(cap, _)| *cap >= min_bytes) {
            return self.available.remove(pos).1;
        }
        // No suitable buffer — allocate exactly what's needed.
        // Don't over-allocate: Metal shared buffers consume physical memory immediately
        // on Apple Silicon unified memory, so wasting capacity = wasting RAM.
        device.new_buffer(min_bytes.max(64), MTLResourceOptions::StorageModeShared)
    }

    /// Return a buffer to the pool for future reuse.
    fn release(&mut self, buf: Buffer) {
        let cap = buf.length();
        // Insert sorted by capacity for efficient best-fit search
        let pos = self.available.partition_point(|(c, _)| *c < cap);
        self.available.insert(pos, (cap, buf));

        // Cap pool size to avoid unbounded memory growth (keep at most 16 buffers)
        while self.available.len() > 16 {
            self.available.pop(); // drop largest
        }
    }
}

#[allow(dead_code)]
pub(crate) struct GpuContext {
    device: Device,
    queue: CommandQueue,
    buffer_pool: Mutex<BufferPool>,
    field_test_pipeline: ComputePipelineState,
    field_addsub_test_pipeline: ComputePipelineState,
    field_sqr_test_pipeline: ComputePipelineState,
    jacobian_madd_test_pipeline: ComputePipelineState,
    jacobian_add_test_pipeline: ComputePipelineState,
    jacobian_dbl_test_pipeline: ComputePipelineState,
    double_and_add_test_pipeline: ComputePipelineState,
    bucket_accumulate_pipeline: ComputePipelineState,
    bucket_accumulate_all_pipeline: ComputePipelineState,
    bucket_reduce_stage1_pipeline: ComputePipelineState,
    bucket_reduce_stage2_pipeline: ComputePipelineState,
    bucket_reduce_stage1_all_pipeline: ComputePipelineState,
    bucket_reduce_stage2_all_pipeline: ComputePipelineState,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

/// Cached CPU-side booth index buffer to avoid repeated large allocations.
/// On k=24 this is 512MB — re-allocating each MSM call causes significant
/// page-fault overhead on first touch. By caching, we pay this cost only once.
static BOOTH_CACHE: Mutex<Vec<i32>> = Mutex::new(Vec::new());

pub(crate) fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        // Retry GPU device acquisition — paravirtual devices may be transiently unavailable
        let device = {
            let mut dev = None;
            for attempt in 0..5 {
                dev = Device::system_default();
                if dev.is_some() { break; }
                if attempt < 4 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
            dev.expect("No Metal GPU device found after 5 attempts")
        };
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = match device.new_library_with_source(shader::SHADER_SOURCE, &options) {
            Ok(lib) => lib,
            Err(e) => panic!("Failed to compile Metal shader: {}", e),
        };

        let field_test_fn = library
            .get_function("test_field_ops", None)
            .expect("Missing test_field_ops kernel");
        let field_test_pipeline = match device
            .new_compute_pipeline_state_with_function(&field_test_fn) {
                Ok(p) => p,
                Err(e) => panic!("Failed to create field_test pipeline: {:?}", e),
            };

        let field_addsub_fn = library
            .get_function("test_field_addsub", None)
            .expect("Missing test_field_addsub kernel");
        let field_addsub_test_pipeline = device
            .new_compute_pipeline_state_with_function(&field_addsub_fn)
            .expect("Failed to create field_addsub_test pipeline");

        let field_sqr_fn = library
            .get_function("test_fq_sqr", None)
            .expect("Missing test_fq_sqr kernel");
        let field_sqr_test_pipeline = device
            .new_compute_pipeline_state_with_function(&field_sqr_fn)
            .expect("Failed to create field_sqr_test pipeline");

        let jacobian_madd_test_fn = library
            .get_function("test_jacobian_madd", None)
            .expect("Missing test_jacobian_madd kernel");
        let jacobian_madd_test_pipeline = device
            .new_compute_pipeline_state_with_function(&jacobian_madd_test_fn)
            .expect("Failed to create jacobian_madd_test pipeline");

        let jacobian_add_test_fn = library
            .get_function("test_jacobian_add", None)
            .expect("Missing test_jacobian_add kernel");
        let jacobian_add_test_pipeline = device
            .new_compute_pipeline_state_with_function(&jacobian_add_test_fn)
            .expect("Failed to create jacobian_add_test pipeline");

        let jacobian_dbl_test_fn = library
            .get_function("test_jacobian_dbl", None)
            .expect("Missing test_jacobian_dbl kernel");
        let jacobian_dbl_test_pipeline = device
            .new_compute_pipeline_state_with_function(&jacobian_dbl_test_fn)
            .expect("Failed to create jacobian_dbl_test pipeline");

        let double_and_add_test_fn = library
            .get_function("test_double_and_add", None)
            .expect("Missing test_double_and_add kernel");
        let double_and_add_test_pipeline = device
            .new_compute_pipeline_state_with_function(&double_and_add_test_fn)
            .expect("Failed to create double_and_add_test pipeline");

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
            buffer_pool: Mutex::new(BufferPool::new()),
            field_test_pipeline,
            field_addsub_test_pipeline,
            field_sqr_test_pipeline,
            jacobian_madd_test_pipeline,
            jacobian_add_test_pipeline,
            jacobian_dbl_test_pipeline,
            double_and_add_test_pipeline,
            bucket_accumulate_pipeline,
            bucket_accumulate_all_pipeline,
            bucket_reduce_stage1_pipeline,
            bucket_reduce_stage2_pipeline,
            bucket_reduce_stage1_all_pipeline,
            bucket_reduce_stage2_all_pipeline,
        }
    })
}

impl GpuContext {
    /// Acquire a buffer from the pool (or allocate if none available).
    fn acquire_buffer(&self, min_bytes: u64) -> Buffer {
        self.buffer_pool.lock().unwrap().acquire(&self.device, min_bytes)
    }

    /// Return a buffer to the pool for future reuse.
    fn release_buffer(&self, buf: Buffer) {
        self.buffer_pool.lock().unwrap().release(buf);
    }

    /// Release multiple buffers back to the pool.
    fn release_buffers(&self, bufs: Vec<Buffer>) {
        let mut pool = self.buffer_pool.lock().unwrap();
        for buf in bufs {
            pool.release(buf);
        }
    }
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
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use group::Group;

/// Convert a point from Jacobian (Xj, Yj, Zj) to halo2curves' internal
/// homogeneous projective representation G1 { x, y, z }.
///
/// Jacobian: affine = (Xj/Zj², Yj/Zj³)
/// Projective: affine = (x/z, y/z)
///
/// So: x_proj = Xj*Zj, y_proj = Yj, z_proj = Zj³
/// No field inversions needed — just 2 muls + 1 squaring.
#[inline]
fn jacobian_to_g1(xj: Fq, yj: Fq, zj: Fq) -> G1 {
    if zj.is_zero().into() {
        return G1::identity();
    }
    let zj_sq = zj.square();
    let zj_cu = zj * zj_sq;
    G1 {
        x: xj * zj,
        y: yj,
        z: zj_cu,
    }
}

/// Read a Jacobian point from GPU buffer (u64 pointer + offset in u64s)
/// and convert to halo2curves G1 (projective).
#[inline]
unsafe fn read_jacobian_point(ptr: *const u64, off: usize) -> Option<G1> {
    let z_all_zero = (*ptr.add(off + 8) | *ptr.add(off + 9)
        | *ptr.add(off + 10) | *ptr.add(off + 11)) == 0;
    if z_all_zero {
        return None;
    }
    let p = ptr.add(off);
    let xj = Fq([*p, *p.add(1), *p.add(2), *p.add(3)]);
    let yj = Fq([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]);
    let zj = Fq([*p.add(8), *p.add(9), *p.add(10), *p.add(11)]);
    Some(jacobian_to_g1(xj, yj, zj))
}

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

/// GPU-accelerated multi-scalar multiplication.
///
/// Generic over any `CurveAffine + GpuMsm` type. When `C` is `bn256::G1Affine`,
/// dispatches to the Metal GPU implementation **at compile time** via the `GpuMsm`
/// trait. For all other curve types, falls back to the CPU-only `msm_best`.
///
/// This allows `msm_gpu` to be used as a drop-in replacement for `msm_best`:
/// ```ignore
/// // Works for BN254 G1 (uses GPU):
/// let result = msm_gpu(&bn254_scalars, &bn254_points);
/// // Works for any other curve (falls back to CPU):
/// let result = msm_gpu(&other_scalars, &other_points);
/// ```
pub fn msm_gpu<C: GpuMsm>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    C::msm_gpu_dispatch(coeffs, bases)
}

/// Compute multiple MSMs in one call with GPU acceleration.
///
/// For BN254 G1: reuses the Metal GPU context, bases buffer, booth cache, and
/// working buffers across all tasks. Other curves fall back to parallel
/// `msm_gpu_dispatch` per task.
///
/// # Panics
/// Panics if `coeffs.len() != bases.len()` or any pair has mismatched lengths.
pub fn batch_msm_gpu<C: GpuMsm>(
    coeffs: &[&[C::Scalar]],
    bases: &[&[C]],
) -> Vec<C::Curve> {
    assert_eq!(coeffs.len(), bases.len());
    C::batch_msm_gpu_dispatch(coeffs, bases)
}

/// Trait for compile-time GPU MSM dispatch.
///
/// `bn256::G1Affine` uses the Metal GPU backend; all other curve types fall
/// back to `msm_best`. This is resolved at compile time — no runtime overhead.
pub trait GpuMsm: CurveAffine {
    #[doc(hidden)]
    fn msm_gpu_dispatch(coeffs: &[Self::Scalar], bases: &[Self]) -> Self::Curve;

    /// Batch dispatch — default runs each task through msm_gpu_dispatch via rayon.
    fn batch_msm_gpu_dispatch(
        coeffs: &[&[Self::Scalar]],
        bases: &[&[Self]],
    ) -> Vec<Self::Curve> {
        use rayon::prelude::*;
        coeffs
            .par_iter()
            .zip(bases.par_iter())
            .map(|(c, b)| Self::msm_gpu_dispatch(c, b))
            .collect()
    }
}

// ── GPU-accelerated: BN254 G1 ──────────────────────────────────────────────

impl GpuMsm for G1Affine {
    fn msm_gpu_dispatch(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
        msm_gpu_bn254(coeffs, bases)
    }

    fn batch_msm_gpu_dispatch(
        coeffs: &[&[Fr]],
        bases: &[&[G1Affine]],
    ) -> Vec<G1> {
        batch_msm_gpu_bn254(coeffs, bases)
    }
}

// ── CPU fallback for all other curves ──────────────────────────────────────
//
// Macro to reduce boilerplate: generates `impl GpuMsm` with `msm_best` fallback.
macro_rules! impl_gpu_msm_fallback {
    ($affine:ty) => {
        impl GpuMsm for $affine {
            fn msm_gpu_dispatch(
                coeffs: &[Self::Scalar],
                bases: &[Self],
            ) -> Self::Curve {
                crate::msm::msm_best(coeffs, bases)
            }
        }
    };
}

impl_gpu_msm_fallback!(crate::bn256::G2Affine);
impl_gpu_msm_fallback!(crate::grumpkin::G1Affine);
impl_gpu_msm_fallback!(crate::pasta::PallasAffine);
impl_gpu_msm_fallback!(crate::pasta::VestaAffine);
impl_gpu_msm_fallback!(crate::secp256k1::Secp256k1Affine);
impl_gpu_msm_fallback!(crate::secp256r1::Secp256r1Affine);
impl_gpu_msm_fallback!(crate::secq256k1::Secq256k1Affine);
impl_gpu_msm_fallback!(crate::bls12381::G1Affine);
impl_gpu_msm_fallback!(crate::bls12381::G2Affine);
impl_gpu_msm_fallback!(crate::pluto_eris::G1Affine);
impl_gpu_msm_fallback!(crate::pluto_eris::ErisAffine);
impl_gpu_msm_fallback!(crate::pluto_eris::G2Affine);
impl_gpu_msm_fallback!(crate::t256::T256Affine);

/// GPU-accelerated MSM for BN254 G1 (concrete implementation).
fn msm_gpu_bn254(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
    msm_gpu_inner(coeffs, bases, false).0
}

/// Batch GPU MSM for BN254 G1.
///
/// Optimized for the case where multiple MSMs share the same bases (typical in
/// Nova IVC batch commits). Reuses:
/// - Metal GPU context and compiled pipelines (always shared via `gpu_ctx()`)
/// - Bases GPU buffer (single memcpy when bases are shared)
/// - Booth index cache (single allocation)
/// - Working buffers (acquire once, reuse per task)
fn batch_msm_gpu_bn254(all_coeffs: &[&[Fr]], all_bases: &[&[G1Affine]]) -> Vec<G1> {
    use ff::PrimeField;
    use rayon::prelude::*;

    assert_eq!(all_coeffs.len(), all_bases.len());
    if all_coeffs.is_empty() {
        return Vec::new();
    }

    let batch_size = all_coeffs.len();

    // Detect shared bases (pointer equality)
    let shared = batch_size > 1
        && all_bases.iter().all(|b| {
            core::ptr::eq(b.as_ptr(), all_bases[0].as_ptr()) && b.len() == all_bases[0].len()
        });

    let n = all_bases[0].len();

    // Fall back to CPU for small inputs
    if n < (1 << 14) {
        return crate::msm::batch_msm_best(all_coeffs, all_bases);
    }

    let ctx = gpu_ctx();
    let c = get_optimal_c_gpu(n);
    let num_buckets = 1usize << (c - 1);
    let num_bits = Fr::NUM_BITS as usize;
    let number_of_windows = num_bits / c + 1;

    assert_eq!(
        std::mem::size_of::<G1Affine>(),
        64,
        "G1Affine must be 64 bytes for zero-copy GPU transfer"
    );
    let bases_byte_len = n * std::mem::size_of::<G1Affine>();

    // ── Shared resources: acquire once ──────────────────────────────────────

    // Bases buffer: acquire once, copy once if shared
    let bases_buf = ctx.acquire_buffer(bases_byte_len as u64);
    if shared {
        unsafe {
            std::ptr::copy_nonoverlapping(
                all_bases[0].as_ptr() as *const u8,
                bases_buf.contents() as *mut u8,
                bases_byte_len,
            );
        }
    }

    // Booth index cache: take once, reuse for each task
    let booth_len = number_of_windows * n;
    let mut booth = {
        let mut cached = BOOTH_CACHE.lock().unwrap();
        if cached.capacity() >= booth_len {
            let mut b = std::mem::take(&mut *cached);
            unsafe { b.set_len(booth_len); }
            b
        } else {
            drop(cached);
            vec![0i32; booth_len]
        }
    };

    // Working buffers: estimate max sizes and acquire once
    // (scatter size varies per task, but we allocate for worst case)
    let max_scatter_entries = number_of_windows * n; // worst case: all scalars non-zero
    let scatter_buf_bytes = (max_scatter_entries.max(1) * 4) as u64;
    let offsets_buf_bytes = (number_of_windows * (num_buckets + 1) * 4) as u64;
    let buckets_buf_bytes = (number_of_windows * num_buckets * 24 * 4) as u64;
    let params_buf_bytes = (number_of_windows * 3 * 4) as u64;
    let use_pbpr = num_buckets >= 256;
    let num_reduce_threads = if use_pbpr {
        let max_tg = ctx
            .bucket_reduce_stage1_all_pipeline
            .max_total_threads_per_threadgroup() as usize;
        let ideal = (num_buckets / 64).max(2).min(max_tg);
        let mut t = 1;
        while t * 2 <= ideal && num_buckets % (t * 2) == 0 {
            t *= 2;
        }
        t
    } else {
        0
    };
    let g_points_bytes = if use_pbpr {
        (number_of_windows * num_reduce_threads * 24 * 4) as u64
    } else {
        0
    };

    let scatter_buf = ctx.acquire_buffer(scatter_buf_bytes);
    let offsets_buf = ctx.acquire_buffer(offsets_buf_bytes);
    let buckets_buf = ctx.acquire_buffer(buckets_buf_bytes);
    let window_params_buf = ctx.acquire_buffer(params_buf_bytes);
    let g_points_buf = if use_pbpr {
        Some(ctx.acquire_buffer(g_points_bytes))
    } else {
        None
    };
    let reduce_params_buf = if use_pbpr {
        Some(ctx.acquire_buffer(8))
    } else {
        None
    };

    // ── Per-task loop ───────────────────────────────────────────────────────

    let mut results = Vec::with_capacity(batch_size);

    for task_idx in 0..batch_size {
        let coeffs = all_coeffs[task_idx];
        let task_bases = all_bases[task_idx];
        assert_eq!(coeffs.len(), task_bases.len());

        // If bases differ per task, re-copy
        if !shared {
            assert_eq!(task_bases.len(), n, "batch tasks must have same base length");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    task_bases.as_ptr() as *const u8,
                    bases_buf.contents() as *mut u8,
                    bases_byte_len,
                );
            }
        }

        // 1. Fused scalar encode + scatter build (per-task, different coeffs)
        let booth_addr = booth.as_mut_ptr() as usize;
        coeffs.par_iter().enumerate().for_each(|(i, s)| {
            let bytes = s.to_repr();
            let ptr = booth_addr as *mut i32;
            for w in 0..number_of_windows {
                let idx = crate::msm::get_booth_index(w, c, bytes.as_ref());
                unsafe { *ptr.add(w * n + i) = idx as i32; }
            }
        });

        let scatter_tables: Vec<_> = (0..number_of_windows)
            .into_par_iter()
            .map(|w| {
                let booth_slice = &booth[w * n..(w + 1) * n];
                let mut counts = vec![0u32; num_buckets];
                for &idx in booth_slice {
                    if idx != 0 {
                        counts[idx.unsigned_abs() as usize - 1] += 1;
                    }
                }
                let mut bucket_offsets = vec![0u32; num_buckets + 1];
                for b in 0..num_buckets {
                    bucket_offsets[b + 1] = bucket_offsets[b] + counts[b];
                }
                let total = bucket_offsets[num_buckets] as usize;
                let mut scatter_entries = vec![0u32; total];
                let mut write_pos = bucket_offsets[..num_buckets].to_vec();
                for (i, &idx) in booth_slice.iter().enumerate() {
                    if idx != 0 {
                        let sign = (idx > 0) as u32;
                        let buck = idx.unsigned_abs() as usize - 1;
                        let pos = write_pos[buck] as usize;
                        scatter_entries[pos] = (i as u32) | (sign << 31);
                        write_pos[buck] += 1;
                    }
                }
                (bucket_offsets, scatter_entries)
            })
            .collect();

        // 2. Build concatenated GPU buffers
        let mut active_windows: Vec<usize> = Vec::new();
        let mut total_scatter_entries = 0usize;
        for w in 0..number_of_windows {
            if !scatter_tables[w].1.is_empty() {
                active_windows.push(w);
                total_scatter_entries += scatter_tables[w].1.len();
            }
        }
        let num_active = active_windows.len();

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

                std::ptr::copy_nonoverlapping(
                    offsets.as_ptr(),
                    offsets_dst.add(i * (num_buckets + 1)),
                    num_buckets + 1,
                );
                std::ptr::copy_nonoverlapping(
                    entries.as_ptr(),
                    scatter_dst.add(scatter_cursor as usize),
                    entries.len(),
                );
            }
            scatter_cursor += entries.len() as u32;
        }

        // 3. GPU dispatch (reuses all buffers)
        let mut window_results: Vec<(usize, G1)> = Vec::with_capacity(num_active);

        if num_active > 0 {
            let cb = ctx.queue.new_command_buffer();

            // Bucket accumulation
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

            if use_pbpr {
                let gpb = g_points_buf.as_ref().unwrap();
                let rpb = reduce_params_buf.as_ref().unwrap();
                unsafe {
                    let p = rpb.contents() as *mut u32;
                    *p = num_buckets as u32;
                    *p.add(1) = num_reduce_threads as u32;
                }

                // PBPR Stage 1
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage1_all_pipeline);
                enc.set_buffer(0, Some(&buckets_buf), 0);
                enc.set_buffer(1, Some(gpb), 0);
                enc.set_buffer(2, Some(rpb), 0);
                let max_tg = ctx
                    .bucket_reduce_stage1_all_pipeline
                    .max_total_threads_per_threadgroup() as u64;
                let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
                let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();

                // PBPR Stage 2
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage2_all_pipeline);
                enc.set_buffer(0, Some(&buckets_buf), 0);
                enc.set_buffer(1, Some(gpb), 0);
                enc.set_buffer(2, Some(rpb), 0);
                let max_tg = ctx
                    .bucket_reduce_stage2_all_pipeline
                    .max_total_threads_per_threadgroup() as u64;
                let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
                let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
            }

            cb.commit();
            cb.wait_until_completed();
        }

        // 4. Read back results
        if use_pbpr && num_active > 0 {
            let gpb = g_points_buf.as_ref().unwrap();
            let g_ptr = gpb.contents() as *const u64;
            for (i, &w) in active_windows.iter().enumerate() {
                let mut window_sum = G1::identity();
                for t in 0..num_reduce_threads {
                    let off = (i * num_reduce_threads + t) * 12;
                    if let Some(g_point) = unsafe { read_jacobian_point(g_ptr, off) } {
                        window_sum = window_sum + g_point;
                    }
                }
                window_results.push((w, window_sum));
            }
        } else {
            let bucket_ptr = buckets_buf.contents() as *const u64;
            for (i, &w) in active_windows.iter().enumerate() {
                let base_off = i * num_buckets * 12;
                let mut running = G1::identity();
                let mut window_sum = G1::identity();
                for b in (0..num_buckets).rev() {
                    let off = base_off + b * 12;
                    if let Some(bucket_point) = unsafe { read_jacobian_point(bucket_ptr, off) } {
                        running = running + bucket_point;
                    }
                    window_sum = window_sum + running;
                }
                window_results.push((w, window_sum));
            }
        }

        // 5. Horner's method to combine windows
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

        results.push(total_acc);
    }

    // ── Cleanup: return all shared resources ────────────────────────────────

    {
        let mut cached = BOOTH_CACHE.lock().unwrap();
        *cached = booth;
    }

    let mut bufs = vec![bases_buf, scatter_buf, offsets_buf, buckets_buf, window_params_buf];
    if let Some(b) = g_points_buf {
        bufs.push(b);
    }
    if let Some(b) = reduce_params_buf {
        bufs.push(b);
    }
    ctx.release_buffers(bufs);

    results
}

/// GPU MSM with detailed phase timing breakdown.
pub fn msm_gpu_timed(coeffs: &[Fr], bases: &[G1Affine]) -> (G1, GpuMsmTiming) {
    msm_gpu_inner(coeffs, bases, true)
}

/// Pre-warm all GPU MSM resources for a given problem size.
///
/// Call once at application startup (e.g., before proving rounds) to eliminate
/// cold-start overhead from the first `msm_gpu` call. This pre-allocates:
///
/// 1. **GPU context** — Metal shader compilation + 14 compute pipelines (~50-100ms)
/// 2. **Metal buffer pool** — 7 GPU buffers sized for `n` points (avoids page faults)
/// 3. **Booth index cache** — CPU-side `n * windows * 2` byte array (avoids 512MB alloc at k=24)
/// 4. **GPU wake** — Dummy dispatch to wake GPU from idle power-down (~10ms on Apple Silicon)
///
/// # Example
/// ```ignore
/// // Before proving:
/// msm_gpu_warmup(1 << 22); // pre-warm for k=22 (4M points)
/// // Now first msm_gpu call is as fast as subsequent calls.
/// ```
pub fn msm_gpu_warmup(n: usize) {
    use ff::PrimeField;

    if n < (1 << 14) { return; } // too small for GPU

    let c = get_optimal_c_gpu(n);
    let num_buckets = 1usize << (c - 1);
    let num_bits = Fr::NUM_BITS as usize;
    let number_of_windows = num_bits / c + 1;

    // 1. Initialize GpuContext (compiles shader, creates pipelines)
    let ctx = gpu_ctx();

    // 2. Pre-allocate Metal buffers at target sizes and return them to the pool.
    //    This forces the OS to back the virtual pages with physical memory now,
    //    avoiding page faults during the first real MSM call.
    let bases_bytes = (n * 64) as u64; // G1Affine = 64 bytes
    let max_scatter_per_window = n; // worst case: all scalars non-zero
    let scatter_bytes = (number_of_windows * max_scatter_per_window * 4) as u64;
    let offsets_bytes = (number_of_windows * (num_buckets + 1) * 4) as u64;
    let buckets_bytes = (number_of_windows * num_buckets * 24 * 4) as u64; // 24 u32 per Jacobian
    let window_params_bytes = (number_of_windows * 3 * 4) as u64;
    let g_points_bytes = (number_of_windows * 24 * 4) as u64; // PBPR g_points
    let reduce_params_bytes = 8u64;

    let bufs = vec![
        ctx.acquire_buffer(bases_bytes),
        ctx.acquire_buffer(scatter_bytes),
        ctx.acquire_buffer(offsets_bytes),
        ctx.acquire_buffer(buckets_bytes),
        ctx.acquire_buffer(window_params_bytes),
        ctx.acquire_buffer(g_points_bytes),
        ctx.acquire_buffer(reduce_params_bytes),
    ];

    // Touch every page to force physical backing (avoid lazy allocation page faults)
    for buf in &bufs {
        let ptr = buf.contents() as *mut u8;
        let len = buf.length() as usize;
        // Write one byte per 4KB page
        for offset in (0..len).step_by(4096) {
            unsafe { ptr.add(offset).write_volatile(0); }
        }
    }

    ctx.release_buffers(bufs);

    // 3. Pre-allocate booth index cache
    {
        let booth_len = number_of_windows * n;
        let mut booth = vec![0i32; booth_len];
        // Touch all pages
        for offset in (0..booth_len).step_by(1024) { // 1024 i32 = 4KB
            booth[offset] = 0;
        }
        let mut cached = BOOTH_CACHE.lock().unwrap();
        *cached = booth;
    }

    // 4. Wake GPU with a trivial dispatch (forces GPU out of idle power-down)
    {
        let dummy = ctx.acquire_buffer(64);
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.bucket_accumulate_all_pipeline);
        enc.set_buffer(0, Some(&dummy), 0);
        enc.set_buffer(1, Some(&dummy), 0);
        enc.set_buffer(2, Some(&dummy), 0);
        enc.set_buffer(3, Some(&dummy), 0);
        enc.set_buffer(4, Some(&dummy), 0);
        let size = metal::MTLSize::new(1, 1, 1);
        enc.dispatch_threads(size, size);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        ctx.release_buffers(vec![dummy]);
    }
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

    // 1+2+3. Pipelined CPU work: scalar encode, base pack, scatter build
    //
    // Dependencies:
    //   scatter_build depends on coeffs_bytes (from scalar encode)
    //   base_pack is independent of scalar encode AND scatter build
    //
    // Strategy: Run base_pack on a background thread while scalar encode + scatter
    // build run on the rayon pool. This overlaps the large memcpy (36-230ms) with
    // all CPU computation, effectively hiding base_pack latency entirely.
    //
    // For k=24: saves ~230ms base_pack (runs concurrently with 38ms encode + 575ms scatter)
    // For k=22: saves ~36ms base_pack (runs concurrently with 13ms encode + 97ms scatter)

    assert_eq!(std::mem::size_of::<G1Affine>(), 64,
        "G1Affine must be 64 bytes for zero-copy GPU transfer");
    let bases_byte_len = n * std::mem::size_of::<G1Affine>();

    // Number of windows
    let num_bits = Fr::NUM_BITS as usize;
    let number_of_windows = num_bits / c + 1;
    timing.num_windows = number_of_windows;

    // Allocate GPU buffer for bases (from pool to avoid repeated VM allocation)
    let bases_buf = ctx.acquire_buffer(bases_byte_len as u64);

    // Start base packing on background thread (pure memcpy, doesn't need rayon)
    let bases_dst = bases_buf.contents() as usize; // usize is Send
    let bases_src = bases.as_ptr() as usize;
    let base_pack_handle = std::thread::spawn(move || {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bases_src as *const u8,
                bases_dst as *mut u8,
                bases_byte_len,
            );
        }
    });

    // Meanwhile: fused scalar encode + scatter build on rayon pool
    //
    // Instead of separate encode and scatter phases, we fuse them:
    // 1. Compute to_repr() once per scalar (in parallel)
    // 2. Extract ALL windows' Booth indices from the repr
    // 3. Store as column-major i32 booth[window][scalar] for sequential scatter access
    //
    // This eliminates the 512MB intermediate coeffs_bytes allocation.
    let t0 = Instant::now();

    let booth_len = number_of_windows * n;
    let mut booth = {
        let mut cached = BOOTH_CACHE.lock().unwrap();
        if cached.capacity() >= booth_len {
            let mut b = std::mem::take(&mut *cached);
            // Resize without zeroing — encode will overwrite every position
            unsafe { b.set_len(booth_len); }
            b
        } else {
            drop(cached);
            // First call or size grew: allocate fresh (this pays the page-fault cost once)
            vec![0i32; booth_len]
        }
    };
    let booth_addr = booth.as_mut_ptr() as usize;

    coeffs.par_iter().enumerate().for_each(|(i, s)| {
        let bytes = s.to_repr();
        let ptr = booth_addr as *mut i32;
        for w in 0..number_of_windows {
            let idx = get_booth_index(w, c, bytes.as_ref());
            // SAFETY: each scalar i writes to distinct positions [w*n + i].
            unsafe { *ptr.add(w * n + i) = idx as i32; }
        }
    });
    if timed { timing.scalar_encode_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    let t0 = Instant::now();
    let scatter_tables: Vec<_> = (0..number_of_windows)
        .into_par_iter()
        .map(|w| {
            let booth_slice = &booth[w * n..(w + 1) * n];

            // Pass 1: count
            let mut counts = vec![0u32; num_buckets];
            for &idx in booth_slice {
                if idx != 0 {
                    counts[idx.unsigned_abs() as usize - 1] += 1;
                }
            }

            // Prefix-sum
            let mut bucket_offsets = vec![0u32; num_buckets + 1];
            for b in 0..num_buckets {
                bucket_offsets[b + 1] = bucket_offsets[b] + counts[b];
            }
            let total = bucket_offsets[num_buckets] as usize;

            // Pass 2: fill
            let mut scatter_entries = vec![0u32; total];
            let mut write_pos = bucket_offsets[..num_buckets].to_vec();
            for (i, &idx) in booth_slice.iter().enumerate() {
                if idx != 0 {
                    let sign = (idx > 0) as u32;
                    let buck = idx.unsigned_abs() as usize - 1;
                    let pos = write_pos[buck] as usize;
                    scatter_entries[pos] = (i as u32) | (sign << 31);
                    write_pos[buck] += 1;
                }
            }

            (bucket_offsets, scatter_entries)
        })
        .collect();
    if timed { timing.scatter_build_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // Return booth buffer to cache for reuse in next MSM call
    {
        let mut cached = BOOTH_CACHE.lock().unwrap();
        *cached = booth;
    }

    // Wait for base packing to finish (should already be done since scatter is slower)
    let t0 = Instant::now();
    base_pack_handle.join().unwrap();
    if timed { timing.base_pack_ms = t0.elapsed().as_secs_f64() * 1000.0; } // time waiting only

    // 4. Build concatenated GPU buffers for all-windows-in-one-dispatch
    let t0 = Instant::now();

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
    }

    debug_assert!(total_scatter_entries <= u32::MAX as usize,
        "msm_gpu: total_scatter_entries ({total_scatter_entries}) exceeds u32::MAX");

    let scatter_buf = ctx.acquire_buffer(scatter_buf_bytes);
    let offsets_buf = ctx.acquire_buffer(offsets_buf_bytes);
    let buckets_buf = ctx.acquire_buffer(buckets_buf_bytes);
    let window_params_buf = ctx.acquire_buffer(params_buf_bytes);

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

    if timed { timing.gpu_upload_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    // 5+6. Dispatch bucket_accumulate_all + PBPR in ONE command buffer
    //
    // Merging all GPU work into a single commit()+wait() eliminates CPU-GPU
    // synchronization overhead between kernel and reduce phases (~5ms).
    // Metal guarantees sequential execution of encoders within one command buffer.
    let use_pbpr = num_buckets >= 256;
    let num_reduce_threads = if use_pbpr {
        let max_tg = ctx.bucket_reduce_stage1_all_pipeline
            .max_total_threads_per_threadgroup() as usize;
        let ideal = (num_buckets / 64).max(2).min(max_tg);
        let mut t = 1;
        while t * 2 <= ideal && num_buckets % (t * 2) == 0 { t *= 2; }
        t
    } else { 0 };

    let g_points_buf: Option<Buffer>;
    let reduce_params_buf: Option<Buffer>;

    let t0 = Instant::now();
    let mut window_results: Vec<(usize, G1)> = Vec::with_capacity(num_active);

    if num_active > 0 {
        let cb = ctx.queue.new_command_buffer();

        // Bucket accumulation kernel
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.bucket_accumulate_all_pipeline);
        enc.set_buffer(0, Some(&bases_buf), 0);
        enc.set_buffer(1, Some(&scatter_buf), 0);
        enc.set_buffer(2, Some(&offsets_buf), 0);
        enc.set_buffer(3, Some(&buckets_buf), 0);
        enc.set_buffer(4, Some(&window_params_buf), 0);
        let max_tg = ctx.bucket_accumulate_all_pipeline
            .max_total_threads_per_threadgroup() as u64;
        let grid = MTLSize::new(num_buckets as u64, num_active as u64, 1);
        let tg = MTLSize::new(max_tg.min(num_buckets as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        if use_pbpr {
            // PBPR Stage 1 + Stage 2 (same command buffer)
            let gpb = ctx.acquire_buffer(
                (num_active * num_reduce_threads * 24 * 4) as u64,
            );
            let rpb = ctx.acquire_buffer(8);
            unsafe {
                let p = rpb.contents() as *mut u32;
                *p = num_buckets as u32;
                *p.add(1) = num_reduce_threads as u32;
            }

            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage1_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&gpb), 0);
            enc.set_buffer(2, Some(&rpb), 0);
            let max_tg = ctx.bucket_reduce_stage1_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage2_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&gpb), 0);
            enc.set_buffer(2, Some(&rpb), 0);
            let max_tg = ctx.bucket_reduce_stage2_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            g_points_buf = Some(gpb);
            reduce_params_buf = Some(rpb);
        } else {
            g_points_buf = None;
            reduce_params_buf = None;
        }

        cb.commit();
        cb.wait_until_completed();
    } else {
        g_points_buf = None;
        reduce_params_buf = None;
    }
    if timed { timing.gpu_kernel_ms = t0.elapsed().as_secs_f64() * 1000.0; }

    if use_pbpr && num_active > 0 {
        // Read back all g_points (Jacobian → projective conversion)
        let t0 = Instant::now();
        let gpb = g_points_buf.as_ref().unwrap();
        let g_ptr = gpb.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let mut window_sum = G1::identity();
            for t in 0..num_reduce_threads {
                let off = (i * num_reduce_threads + t) * 12;
                if let Some(g_point) = unsafe { read_jacobian_point(g_ptr, off) } {
                    window_sum = window_sum + g_point;
                }
            }
            window_results.push((w, window_sum));
        }
        if timed { timing.cpu_reduce_ms = t0.elapsed().as_secs_f64() * 1000.0; }
    } else {
        // CPU fallback: read back buckets (Jacobian) and do summation-by-parts
        let t0 = Instant::now();
        let bucket_ptr = buckets_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let base_off = i * num_buckets * 12;
            let mut running = G1::identity();
            let mut window_sum = G1::identity();
            for b in (0..num_buckets).rev() {
                let off = base_off + b * 12;
                if let Some(bucket_point) = unsafe { read_jacobian_point(bucket_ptr, off) } {
                    running = running + bucket_point;
                }
                window_sum = window_sum + running;
            }
            window_results.push((w, window_sum));
        }
        if timed { timing.cpu_reduce_ms = t0.elapsed().as_secs_f64() * 1000.0; }
    }

    // 7. Combine window results using Horner's method
    let t0 = Instant::now();
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

    // Return all buffers to pool for reuse in subsequent MSM calls
    let mut bufs = vec![bases_buf, scatter_buf, offsets_buf, buckets_buf, window_params_buf];
    if let Some(b) = g_points_buf { bufs.push(b); }
    if let Some(b) = reduce_params_buf { bufs.push(b); }
    ctx.release_buffers(bufs);

    timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    (total_acc, timing)
}

/// GPU-accelerated MSM with GLV endomorphism for BN254 G1.
///
/// Decomposes each 254-bit scalar `k` into two ~128-bit half-scalars via
/// `k·P = k1·P + k2·phi(P)` where `phi(P) = (zeta·x, y)`.
///
/// This halves the number of Pippenger windows at the cost of doubling
/// the point count per window.
pub fn msm_gpu_glv(coeffs: &[Fr], bases: &[G1Affine]) -> G1 {
    use rayon::prelude::*;

    assert_eq!(coeffs.len(), bases.len());

    let n = bases.len();

    if n < (1 << 14) {
        return crate::msm::msm_best(coeffs, bases);
    }

    let ctx = gpu_ctx();

    // 1. GLV scalar decomposition: k = k1 + lambda * k2
    let decomposed: Vec<(u128, bool, u128, bool)> = coeffs
        .par_iter()
        .map(|s| G1::decompose_scalar(s))
        .collect();

    let k1_bytes: Vec<[u8; 16]> = decomposed
        .iter()
        .map(|(k1, _, _, _): &(u128, bool, u128, bool)| k1.to_le_bytes())
        .collect();

    let k2_bytes: Vec<[u8; 16]> = decomposed
        .iter()
        .map(|(_, _, k2, _): &(u128, bool, u128, bool)| k2.to_le_bytes())
        .collect();

    let k1_signs: Vec<bool> = decomposed.iter().map(|(_, neg, _, _)| !neg).collect();
    let k2_signs: Vec<bool> = decomposed.iter().map(|(_, _, _, neg)| *neg).collect();

    // 2+3+4. Pipeline: base packing overlaps with scatter build
    //
    // Base packing: computes endomorphism points + copies to GPU buffer (reads `bases`)
    // Scatter build: Booth-encodes half-scalars, builds CSR tables (reads k1_bytes, k2_bytes)
    // These are independent — overlap them with rayon::join.

    let zeta = <Fq as WithSmallOrderMulGroup<3>>::ZETA;
    let total_points = 2 * n;
    let bases_byte_len = total_points * 16 * 4;
    let bases_buf = ctx.acquire_buffer(bases_byte_len as u64);

    // Window parameters for ~128-bit scalars
    let half_bits = 128usize;
    let c = get_optimal_c_gpu(n);
    let num_buckets = 1usize << (c - 1);
    let number_of_windows = half_bits / c + 1;

    // Run base packing and (fused encode + scatter build) concurrently.
    //
    // Fused approach: precompute ALL booth indices for both k1 and k2 in
    // column-major layout [window][scalar], then build scatter tables with
    // sequential reads. This eliminates the cache-pathological column-stride
    // access pattern in build_scatter_table_glv.
    let (_, scatter_tables) = rayon::join(
        || {
            // Pack 2n affine bases into GPU buffer (parallelized)
            let dst_ptr = bases_buf.contents() as usize;
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

                    let endo_x = *coords.x() * zeta;
                    let endo_x_limbs = u64x4_to_u32x8(&endo_x.0);
                    let off2 = (n + i) * 16;
                    unsafe {
                        std::ptr::copy_nonoverlapping(endo_x_limbs.as_ptr(), dst.add(off2), 8);
                        std::ptr::copy_nonoverlapping(y_limbs.as_ptr(), dst.add(off2 + 8), 8);
                    }
                }
            });
        },
        || {
            // Precompute booth indices in column-major layout [W][n] for both k1 and k2.
            // Each entry stores (booth_index, effective_sign, base_index) packed as:
            //   booth1[w * n + i] = signed booth index for k1 scalar i at window w
            //   booth2[w * n + i] = signed booth index for k2 scalar i at window w
            let mut booth1 = vec![0i32; number_of_windows * n];
            let mut booth2 = vec![0i32; number_of_windows * n];
            let booth1_addr = booth1.as_mut_ptr() as usize;
            let booth2_addr = booth2.as_mut_ptr() as usize;

            // Parallel precompute: each scalar computes all its window indices
            (0..n).into_par_iter().for_each(|i| {
                let p1 = booth1_addr as *mut i32;
                let p2 = booth2_addr as *mut i32;
                for w in 0..number_of_windows {
                    // SAFETY: each scalar i writes to distinct positions [w*n + i]
                    unsafe {
                        *p1.add(w * n + i) = get_booth_index(w, c, &k1_bytes[i]) as i32;
                        *p2.add(w * n + i) = get_booth_index(w, c, &k2_bytes[i]) as i32;
                    }
                }
            });

            // Build scatter tables from precomputed booth indices (sequential reads)
            (0..number_of_windows)
                .into_par_iter()
                .map(|w| {
                    let b1_slice = &booth1[w * n..(w + 1) * n];
                    let b2_slice = &booth2[w * n..(w + 1) * n];

                    // Pass 1: count entries per bucket
                    let mut counts = vec![0u32; num_buckets];
                    for &idx in b1_slice {
                        if idx != 0 {
                            counts[idx.unsigned_abs() as usize - 1] += 1;
                        }
                    }
                    for &idx in b2_slice {
                        if idx != 0 {
                            counts[idx.unsigned_abs() as usize - 1] += 1;
                        }
                    }

                    // Prefix-sum
                    let mut bucket_offsets = vec![0u32; num_buckets + 1];
                    for b in 0..num_buckets {
                        bucket_offsets[b + 1] = bucket_offsets[b] + counts[b];
                    }
                    let total = bucket_offsets[num_buckets] as usize;

                    // Pass 2: fill scatter entries
                    let mut scatter_entries = vec![0u32; total];
                    let mut write_pos = bucket_offsets[..num_buckets].to_vec();

                    for (i, &idx) in b1_slice.iter().enumerate() {
                        if idx != 0 {
                            let booth_positive = idx > 0;
                            let effective_positive = booth_positive ^ (!k1_signs[i]);
                            let buck = idx.unsigned_abs() as usize - 1;
                            let pos = write_pos[buck] as usize;
                            scatter_entries[pos] = (i as u32) | ((effective_positive as u32) << 31);
                            write_pos[buck] += 1;
                        }
                    }

                    for (i, &idx) in b2_slice.iter().enumerate() {
                        if idx != 0 {
                            let booth_positive = idx > 0;
                            let effective_positive = booth_positive ^ (!k2_signs[i]);
                            let buck = idx.unsigned_abs() as usize - 1;
                            let pos = write_pos[buck] as usize;
                            scatter_entries[pos] = ((n + i) as u32) | ((effective_positive as u32) << 31);
                            write_pos[buck] += 1;
                        }
                    }

                    (bucket_offsets, scatter_entries)
                })
                .collect::<Vec<_>>()
        },
    );

    // 5. Build concatenated GPU buffers
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

    debug_assert!(total_scatter_entries <= u32::MAX as usize);
    debug_assert!((2 * n) <= (1usize << 31));

    let scatter_buf = ctx.acquire_buffer(scatter_buf_bytes);
    let offsets_buf = ctx.acquire_buffer(offsets_buf_bytes);
    let buckets_buf = ctx.acquire_buffer(buckets_buf_bytes);
    let window_params_buf = ctx.acquire_buffer(params_buf_bytes);

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

    // 6+7. Dispatch accumulate + PBPR in ONE command buffer.
    //
    // Merging all GPU work into a single commit()+wait() eliminates CPU-GPU
    // synchronization overhead between kernel and reduce phases (~5ms).
    // Metal guarantees sequential execution of encoders within one command buffer.
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

    let g_points_buf: Option<Buffer>;
    let reduce_params_buf: Option<Buffer>;

    if num_active > 0 {
        let cb = ctx.queue.new_command_buffer();

        // Bucket accumulation kernel
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

        if use_pbpr {
            // PBPR Stage 1 + Stage 2 (same command buffer as accumulation)
            let gpb = ctx.acquire_buffer(
                (num_active * num_reduce_threads * 24 * 4) as u64,
            );
            let rpb = ctx.acquire_buffer(8);
            unsafe {
                let p = rpb.contents() as *mut u32;
                *p = num_buckets as u32;
                *p.add(1) = num_reduce_threads as u32;
            }

            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage1_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&gpb), 0);
            enc.set_buffer(2, Some(&rpb), 0);
            let max_tg = ctx
                .bucket_reduce_stage1_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.bucket_reduce_stage2_all_pipeline);
            enc.set_buffer(0, Some(&buckets_buf), 0);
            enc.set_buffer(1, Some(&gpb), 0);
            enc.set_buffer(2, Some(&rpb), 0);
            let max_tg = ctx
                .bucket_reduce_stage2_all_pipeline
                .max_total_threads_per_threadgroup() as u64;
            let grid = MTLSize::new(num_reduce_threads as u64, num_active as u64, 1);
            let tg = MTLSize::new(max_tg.min(num_reduce_threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            g_points_buf = Some(gpb);
            reduce_params_buf = Some(rpb);
        } else {
            g_points_buf = None;
            reduce_params_buf = None;
        }

        cb.commit();
        cb.wait_until_completed();
    } else {
        g_points_buf = None;
        reduce_params_buf = None;
    }

    if use_pbpr && num_active > 0 {
        // Read back all g_points (Jacobian → projective conversion)
        let gpb = g_points_buf.as_ref().unwrap();
        let g_ptr = gpb.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let mut window_sum = G1::identity();
            for t in 0..num_reduce_threads {
                let off = (i * num_reduce_threads + t) * 12;
                if let Some(g_point) = unsafe { read_jacobian_point(g_ptr, off) } {
                    window_sum = window_sum + g_point;
                }
            }
            window_results.push((w, window_sum));
        }
    } else {
        let bucket_ptr = buckets_buf.contents() as *const u64;
        for (i, &w) in active_windows.iter().enumerate() {
            let base_off = i * num_buckets * 12;
            let mut running = G1::identity();
            let mut window_sum = G1::identity();
            for b in (0..num_buckets).rev() {
                let off = base_off + b * 12;
                if let Some(bucket_point) = unsafe { read_jacobian_point(bucket_ptr, off) } {
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

    // Return all buffers to pool for reuse
    let mut bufs = vec![bases_buf, scatter_buf, offsets_buf, buckets_buf, window_params_buf];
    if let Some(b) = g_points_buf { bufs.push(b); }
    if let Some(b) = reduce_params_buf { bufs.push(b); }
    ctx.release_buffers(bufs);

    total_acc
}

// ---------------------------------------------------------------------------
// Scatter table builders
// ---------------------------------------------------------------------------

use crate::msm::get_booth_index;

/// Optimal Pippenger window size for GPU.
///
/// GPU prefers larger windows (fewer dispatches) because:
/// 1. Bucket accumulation is GPU-parallel (more buckets = more parallelism)
/// 2. Fewer windows = fewer kernel launches (each has fixed overhead)
/// 3. PBPR handles the larger bucket count efficiently on GPU
///
/// Tuned for Apple Silicon (M4 Pro) with unified memory.
/// Override with MSM_GPU_C=<value> for tuning experiments.
fn get_optimal_c_gpu(n: usize) -> usize {
    // Allow override via environment variable for tuning
    if let Ok(val) = std::env::var("MSM_GPU_C") {
        if let Ok(c) = val.parse::<usize>() {
            if (8..=20).contains(&c) {
                return c;
            }
        }
    }
    let k = (n as f64).log2() as usize;
    match k {
        0..=13  => 10,
        14..=17 => 13,
        18..=19 => 15,
        20..=24 => 16,
        _       => 18,
    }
}
