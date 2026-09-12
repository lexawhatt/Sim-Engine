//! Event-loop-thread allocation requests and independent frame timing samples.

use sim_engine::{GpuTimingBatch, GpuTimingId, GpuTimingSource};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

pub struct Allocator;

fn record(bytes: usize) {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            let _ = CALLS.try_with(|count| count.set(count.get().saturating_add(1)));
            let _ = BYTES.try_with(|count| count.set(count.get().saturating_add(bytes)));
        }
    });
}

// SAFETY: Requests are forwarded unchanged to System. Thread-local counters
// have constant initialization and do not allocate or access caller memory.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: Same layout as the original request.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: Same layout as the original request.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: Same allocation/layout pair as the original request.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        // SAFETY: Same allocation/layout/new-size tuple as the original request.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

pub fn count<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.set(false);
        }
    }
    CALLS.set(0);
    BYTES.set(0);
    ACTIVE.set(true);
    let guard = Guard;
    let result = operation();
    drop(guard);
    (result, CALLS.get(), BYTES.get())
}

pub fn percentile(samples: &[f64], percent: usize) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
}

pub fn gpu_distribution(samples: &[f64]) -> String {
    if samples.is_empty() {
        return "unavailable".into();
    }
    format!(
        "[p50={:.3},p95={:.3},p99={:.3}]",
        percentile(samples, 50),
        percentile(samples, 95),
        percentile(samples, 99)
    )
}

#[derive(Default)]
pub struct Samples {
    pub work_ms: Vec<f64>,
    pub update_ms: Vec<f64>,
    pub acquire_ms: Vec<f64>,
    pub render_3d_ms: Vec<f64>,
    pub allocations: usize,
    pub allocated_bytes: usize,
    pub uploaded_mesh_bytes: usize,
    pub mesh_upload_calls: usize,
    pub gpu_buffer_allocations: usize,
    pub reused_updates: usize,
    pub alias_detachments: usize,
    pub scratch_reallocations: usize,
    pub preflight_ms: Vec<f64>,
    pub staging_upload_ms: Vec<f64>,
    pub encode_submit_ms: Vec<f64>,
    pub generated_upload_bytes: usize,
    pub submitted_triangles: usize,
    pub generated_triangles: usize,
    pub discarded_triangles: usize,
    pub clipped_triangles: usize,
    pub draw_calls: usize,
    pub texture_uploaded_bytes: usize,
    pub texture_upload_calls: usize,
    pub texture_copy_bytes: usize,
    pub texture_gpu_allocations: usize,
    pub texture_staging_peak: usize,
    pub prepared_text_uploaded_bytes: usize,
    pub frame_upload_bytes: usize,
    pub frame_upload_calls: usize,
    pub frame_buffer_allocations: usize,
    pub composition_upload_bytes: usize,
    pub composition_draw_calls: usize,
    pub frame_staging_peak: usize,
    pub frame_cpu_peak: usize,
    pub frame_gpu_peak: usize,
    pub mesh_staging_peak: usize,
    pub mesh_update_gpu_peak: usize,
    pub mesh_update_recovery_peak: usize,
    pub texture_update_gpu_peak: usize,
    pub texture_update_recovery_peak: usize,
    pub gpu: TimingSamples,
}

impl Samples {
    pub fn with_capacity(frames: usize) -> Self {
        Self {
            work_ms: Vec::with_capacity(frames),
            update_ms: Vec::with_capacity(frames),
            acquire_ms: Vec::with_capacity(frames),
            render_3d_ms: Vec::with_capacity(frames),
            preflight_ms: Vec::with_capacity(frames),
            staging_upload_ms: Vec::with_capacity(frames),
            encode_submit_ms: Vec::with_capacity(frames),
            gpu: TimingSamples::new(frames),
            ..Self::default()
        }
    }
}

/// Only IDs belonging to measured Drawn frames are included. Warmup/other
/// passes can complete in any order and must never shift measured samples.
#[derive(Default)]
pub struct TimingSamples {
    requested: Vec<(GpuTimingId, GpuTimingSource, bool)>,
    received: usize,
    pub scene_ms: Vec<f64>,
    pub composition_ms: Vec<f64>,
}

impl TimingSamples {
    fn new(frames: usize) -> Self {
        Self {
            requested: Vec::with_capacity(frames * 2),
            received: 0,
            scene_ms: Vec::with_capacity(frames),
            composition_ms: Vec::with_capacity(frames),
        }
    }
    pub fn expect(&mut self, scene: Option<GpuTimingId>, composition: Option<GpuTimingId>) {
        for pair in [
            (scene, GpuTimingSource::Scene3d),
            (composition, GpuTimingSource::FrameComposer),
        ] {
            if let Some(id) = pair.0 {
                self.requested.push((id, pair.1, false));
            }
        }
    }
    pub fn collect(&mut self, batch: GpuTimingBatch) -> Result<(), &'static str> {
        for sample in batch.samples() {
            let Ok(index) = self
                .requested
                .binary_search_by_key(&sample.id(), |(id, _, _)| *id)
            else {
                continue;
            };
            let (_, source, received) = &mut self.requested[index];
            if *source != sample.source() || *received {
                return Err("GPU timing source mismatch or duplicate sample");
            }
            *received = true;
            self.received += 1;
            let samples = match source {
                GpuTimingSource::Scene3d => &mut self.scene_ms,
                GpuTimingSource::FrameComposer => &mut self.composition_ms,
            };
            samples.push(sample.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(())
    }
    pub fn missing(&self) -> usize {
        self.requested.len() - self.received
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_keeps_slow_tail() {
        assert_eq!(percentile(&[1.0, 1.0, 100.0], 95), 100.0);
        assert_eq!(percentile(&[3.0, 2.0, 1.0], 50), 2.0);
    }

    #[test]
    fn absent_gpu_samples_are_explicitly_unavailable_not_zero_or_nan() {
        assert_eq!(gpu_distribution(&[]), "unavailable");
        assert_eq!(gpu_distribution(&[0.0]), "[p50=0.000,p95=0.000,p99=0.000]");
    }
}
