//! Test-only CPU accounting study; no GPU resources or presentation are timed.

use super::*;
use std::{collections::HashSet, hint::black_box, time::Instant};

type Key = (usize, u8);

fn key(value: RetainedResourceKey) -> Key {
    (value.address, value.class)
}

fn added_work(
    resources: &[RetainedResourceAccounting],
    contains: impl Fn(Key) -> bool,
) -> FrameStatistics {
    let mut work = FrameStatistics::default();
    for (index, resource) in resources.iter().enumerate() {
        if !contains(key(resource.key))
            && !resources[..index]
                .iter()
                .any(|previous| previous.key == resource.key)
        {
            work.retained_cpu_bytes += resource.cpu_bytes;
            work.retained_buffer_bytes += resource.buffer_bytes;
            work.texture_bytes += resource.texture_bytes;
        }
    }
    work
}

fn accounting(
    mode: &str,
    groups: &[[RetainedResourceAccounting; 3]],
    linear: &mut Vec<RetainedResourceKey>,
    sorted: &mut Vec<Key>,
    hashed: &mut HashSet<Key>,
) -> FrameStatistics {
    linear.clear();
    sorted.clear();
    hashed.clear();
    let mut total = FrameStatistics::default();
    for resources in black_box(groups) {
        let work = match mode {
            "linear" => {
                let (work, missing) =
                    account_new_retained_resources(linear, resources, FrameStatistics::default());
                linear.try_reserve(missing).unwrap();
                // Exactly the existing accounting-plus-commit algorithm.
                for resource in resources {
                    if !linear.contains(&resource.key) {
                        linear.push(resource.key);
                    }
                }
                work
            }
            "sorted" => {
                let work = added_work(resources, |key| sorted.binary_search(&key).is_ok());
                // Keep separate preflight and commit searches: a production
                // caller must validate aggregate budgets between these stages.
                for resource in resources {
                    if let Err(index) = sorted.binary_search(&key(resource.key)) {
                        sorted.insert(index, key(resource.key));
                    }
                }
                work
            }
            _ => {
                let work = added_work(resources, |key| hashed.contains(&key));
                for resource in resources {
                    hashed.insert(key(resource.key));
                }
                work
            }
        };
        total = total.adding(work);
    }
    black_box(total)
}

fn groups(count: usize, order: &str) -> Vec<[RetainedResourceAccounting; 3]> {
    let mut ordering: Vec<_> = (0..count).collect();
    if order == "descending" {
        ordering.reverse();
    } else if order == "shuffled" {
        let mut state = 0x8729_aa21_u64;
        for index in (1..count).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ordering.swap(index, (state as usize) % (index + 1));
        }
    }
    ordering
        .into_iter()
        .map(|index| {
            [
                // Same key classes/sharing as draw_glyph_run_placed: image, atlas,
                // unique run. Synthetic addresses isolate collection costs from GPU.
                RetainedResourceAccounting::new(
                    RetainedResourceKey {
                        address: 1,
                        class: 7,
                    },
                    4,
                    0,
                    4,
                ),
                RetainedResourceAccounting::new(
                    RetainedResourceKey {
                        address: 2,
                        class: 9,
                    },
                    32,
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    RetainedResourceKey {
                        address: index + 3,
                        class: 10,
                    },
                    3072,
                    1024,
                    0,
                ),
            ]
        })
        .collect()
}

#[test]
fn dedup_diagnostic_candidates_preserve_exact_shared_accounting() {
    for order in ["ascending", "descending", "shuffled"] {
        let groups = groups(100, order);
        let mut linear = Vec::with_capacity(102);
        let mut sorted = Vec::with_capacity(102);
        let mut hashed = HashSet::with_capacity(102);
        let baseline = accounting("linear", &groups, &mut linear, &mut sorted, &mut hashed);
        assert_eq!(baseline.retained_cpu_bytes(), 36 + 100 * 3072);
        assert_eq!(baseline.retained_buffer_bytes(), 100 * 1024);
        assert_eq!(baseline.texture_bytes(), 4);
        for mode in ["sorted", "hashed"] {
            assert_eq!(
                accounting(mode, &groups, &mut linear, &mut sorted, &mut hashed),
                baseline
            );
        }
    }
}

#[test]
#[ignore = "opt-in warmed release CPU accounting diagnostic"]
fn retained_resource_dedup_cpu_diagnostic() {
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    for count in [1, 100, 1000] {
        for order in ["ascending", "descending", "shuffled"] {
            let groups = groups(count, order);
            let mut linear = Vec::with_capacity(count + 2);
            let mut sorted = Vec::with_capacity(count + 2);
            let mut hashed = HashSet::with_capacity(count + 2);
            let expected = accounting("linear", &groups, &mut linear, &mut sorted, &mut hashed);
            for mode in ["linear", "sorted", "hashed"] {
                let mut work = || {
                    assert_eq!(
                        accounting(mode, &groups, &mut linear, &mut sorted, &mut hashed),
                        expected
                    )
                };
                for _ in 0..4 {
                    work();
                }
                let (_, allocations) = crate::test_allocations::count(&mut work);
                assert_eq!(allocations, 0);
                let repeats = if count == 1 { 128 } else { 8 };
                let mut samples = Vec::with_capacity(15);
                for _ in 0..15 {
                    let start = Instant::now();
                    for _ in 0..repeats {
                        work();
                    }
                    samples.push(start.elapsed().as_secs_f64() / repeats as f64 * 1e6);
                }
                samples.sort_by(f64::total_cmp);
                println!(
                    "resource_dedup glyph_runs={count} order={order} mode={mode} median_us={:.3} p10_p90_us={:.3}/{:.3} allocations={allocations}",
                    samples[7], samples[1], samples[13]
                );
            }
        }
    }
}

#[test]
#[ignore = "opt-in release CPU glyph projection diagnostic"]
fn glyph_projection_cpu_diagnostic() {
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    let source = ImageTexelRect::new(0, 0, 1, 1).unwrap();
    let glyphs: Vec<_> = (0..32)
        .map(|index| {
            ImageSprite2d::new(
                source,
                LogicalViewportRegion::new(
                    LogicalScreenPosition::new(index as f32 * 3.0, 0.0),
                    LogicalViewport::new(2.0, 6.0).unwrap(),
                )
                .unwrap(),
                Color::WHITE,
            )
            .unwrap()
        })
        .collect();
    for labels in [1, 100, 1000] {
        let runs = vec![glyphs.clone(); labels];
        for passes in [1, 2] {
            let operation = || {
                for _ in 0..passes {
                    for (index, run) in black_box(&runs).iter().enumerate() {
                        let origin =
                            Vec2::new((index % 10) as f32 * 112.0, (index / 10) as f32 * 7.0);
                        assert!(black_box(image_sprites_are_safe_for_target(
                            run,
                            origin,
                            [2.0 / 1280.0, -2.0 / 720.0, -1.0, 1.0],
                        )));
                    }
                }
            };
            for _ in 0..4 {
                operation();
            }
            let (_, allocations) = crate::test_allocations::count(operation);
            let mut samples = Vec::with_capacity(15);
            for _ in 0..15 {
                let start = Instant::now();
                operation();
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "glyph_projection labels={labels} glyphs_per_label=32 passes={passes} median_ms={:.6} p10_p90_ms={:.6}/{:.6} allocations={allocations}",
                samples[7], samples[1], samples[13]
            );
        }
    }
}
