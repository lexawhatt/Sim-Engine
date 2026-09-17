//! CPU-only cache-key/retirement measurements; no GPU or presentation is timed.

use super::*;
use std::{hint::black_box, time::Instant};

fn key_lookups(keys: &[BindingKey], clone_description: bool) -> usize {
    black_box(keys)
        .iter()
        .filter(|key| {
            if clone_description {
                // The old description path cloned one provenance Arc before
                // comparing an otherwise unchanged ordered item.
                let description = black_box(key.as_ref().into_owned());
                key.matches(black_box(description.as_ref()))
            } else {
                key.matches(black_box(key.as_ref()))
            }
        })
        .count()
}

fn keys(count: usize) -> Vec<BindingKey> {
    let first = Arc::new(());
    let second = Arc::new(());
    (0..count)
        .map(|index| match index % 3 {
            0 => BindingKey::Camera,
            1 => BindingKey::Image {
                identity: Arc::clone(&first),
                sampling: ImageSampling::Nearest,
                texture_bytes: 4,
            },
            _ => BindingKey::Image {
                identity: Arc::clone(&second),
                sampling: ImageSampling::Linear,
                texture_bytes: 4,
            },
        })
        .collect()
}

fn scratch(count: usize, budget: FrameCacheBudget) -> FrameCache {
    let mut cache = FrameCache {
        budget,
        ..FrameCache::default()
    };
    cache.items.reserve(count * 3);
    cache.ready.reserve(count * 3);
    cache.bindings.reserve(count * 3);
    cache.slots.resize_with(count * 3, || None);
    for _ in 0..count {
        cache.batches.push(Vec::with_capacity(1));
    }
    cache
}

fn old_finish(cache: &mut FrameCache) {
    cache.statistics.peak_cpu_bytes = cache
        .initial_cpu_bytes
        .saturating_add(cache.cpu_bytes().saturating_mul(2));
    if cache.cpu_bytes() > cache.budget.max_cpu_bytes {
        cache.items = Vec::new();
        cache.retained_resources = Vec::new();
        cache.scalar_luts = Vec::new();
        cache.ready = Vec::new();
        cache.bindings = Vec::new();
        cache.batches = Vec::new();
    }
    if cache.cpu_bytes() > cache.budget.max_cpu_bytes {
        cache.slots = Vec::new();
        cache.statistics.uniform_bytes = 0;
        cache.statistics.texture_bytes = 0;
    }
    cache.statistics.cpu_bytes = cache.cpu_bytes();
    cache.statistics.binding_count = cache.slots.iter().flatten().count();
}

#[test]
fn borrowed_key_comparison_preserves_provenance_without_arc_traffic() {
    let identity = Arc::new(());
    let owned = BindingKey::Image {
        identity: Arc::clone(&identity),
        sampling: ImageSampling::Nearest,
        texture_bytes: 4,
    };
    let description = owned.as_ref();
    assert_eq!(Arc::strong_count(&identity), 2);
    assert!(owned.matches(description));
    assert_eq!(Arc::strong_count(&identity), 2);
    let retained = description.into_owned();
    assert!(owned.matches(retained.as_ref()));
    assert_eq!(Arc::strong_count(&identity), 3);
    drop(retained);
    assert_eq!(Arc::strong_count(&identity), 2);
    for count in [1, 100, 1000] {
        let keys = keys(count);
        let (found, allocations) = crate::test_allocations::count(|| key_lookups(&keys, false));
        assert_eq!(found, count);
        assert_eq!(allocations, 0);
        assert_eq!(key_lookups(&keys, true), found);
    }
}

#[test]
fn single_capacity_scan_preserves_eviction_and_peak_accounting() {
    let probe = scratch(100, FrameCacheBudget::default());
    let exact = probe.cpu_bytes();
    let slots = vec_bytes(&probe.slots);
    for limit in [0, slots - 1, slots, exact - 1, exact, usize::MAX] {
        let budget = FrameCacheBudget::new(limit, 4096, 4096, 300);
        let mut old = scratch(100, budget);
        let mut new = scratch(100, budget);
        old.begin();
        new.begin();
        old_finish(&mut old);
        new.finish();
        assert_eq!(new.statistics, old.statistics);
        assert_eq!(new.cpu_bytes(), old.cpu_bytes());
        assert!(new.cpu_bytes() <= limit);
        assert_eq!(new.batches.len(), old.batches.len());
        assert_eq!(new.slots.len(), old.slots.len());
    }
}

#[test]
#[ignore = "opt-in paired release CPU cache diagnostic, excludes GPU work"]
fn frame_cache_cpu_diagnostic() {
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    for count in [1, 100, 1000] {
        let keys = keys(count * 3);
        let budget = FrameCacheBudget::new(usize::MAX, usize::MAX, usize::MAX, count * 3);
        let mut old = scratch(count, budget);
        let mut new = scratch(count, budget);
        for mode in [
            "owned_key",
            "borrowed_key",
            "repeated_finish_scan",
            "single_finish_scan",
        ] {
            let mut work = || match mode {
                "owned_key" => {
                    assert_eq!(black_box(key_lookups(&keys, true)), count * 3);
                }
                "borrowed_key" => {
                    assert_eq!(black_box(key_lookups(&keys, false)), count * 3);
                }
                "repeated_finish_scan" => {
                    old_finish(black_box(&mut old));
                }
                _ => {
                    black_box(&mut new).finish();
                }
            };
            for _ in 0..8 {
                work();
            }
            let (_, allocations) = crate::test_allocations::count(&mut work);
            assert_eq!(allocations, 0);
            let repeats = 512;
            let mut samples = Vec::with_capacity(15);
            for _ in 0..15 {
                let started = Instant::now();
                for _ in 0..repeats {
                    work();
                }
                samples.push(started.elapsed().as_secs_f64() * 1e6 / repeats as f64);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "frame_cache_cpu items={} mode={mode} median_us={:.3} p10_p90_us={:.3}/{:.3} allocations={allocations}",
                count * 3,
                samples[7],
                samples[1],
                samples[13]
            );
        }
    }
}
