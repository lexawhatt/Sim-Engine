//! Event-loop-thread allocation requests and independent frame timing samples.

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
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
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
}

impl Samples {
    pub fn with_capacity(frames: usize) -> Self {
        Self {
            work_ms: Vec::with_capacity(frames),
            update_ms: Vec::with_capacity(frames),
            acquire_ms: Vec::with_capacity(frames),
            render_3d_ms: Vec::with_capacity(frames),
            ..Self::default()
        }
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
}
