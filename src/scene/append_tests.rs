use super::*;
use std::{hint::black_box, time::Instant};

fn circle(index: usize) -> DrawCommand {
    DrawCommand::circle(
        Vec2::new(index as f32, 0.0),
        1.0,
        ShapeStyle::filled(Color::WHITE),
    )
    .unwrap()
}

fn layers(count: usize, distribution: usize) -> Vec<Layer> {
    let mut values: Vec<_> = (0..count)
        .map(|index| match distribution {
            0 => 0,
            1 => (index * 8 / count.max(1)) as i32,
            2 => ((index.wrapping_mul(2_654_435_761) ^ (index >> 3)) % 8) as i32,
            _ => ((count - index) * 8 / count.max(1)) as i32,
        })
        .map(Layer::new)
        .collect();
    if distribution == 2 {
        // A deterministic shuffled control keeps the paired inputs identical.
        let mut state = 0x1a2b_3c4d_u64;
        for index in (1..count).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.swap(index, state as usize % (index + 1));
        }
    }
    values
}

fn populate<const FAST: bool>(scene: &mut Scene, layers: &[Layer], commands: &[DrawCommand]) {
    for (&layer, command) in layers.iter().zip(commands) {
        scene
            .try_push_to_layer_ordered::<FAST>(layer, command.clone())
            .unwrap();
    }
}

fn assert_equivalent(fast: &Scene, reference: &Scene) {
    assert_eq!(fast.statistics(), reference.statistics());
    assert_eq!(fast.owned_payload_bytes, reference.owned_payload_bytes);
    assert_eq!(fast.next_order, reference.next_order);
    assert_eq!(fast.allocation_bytes(), reference.allocation_bytes());
    assert_eq!(fast.commands.len(), reference.commands.len());
    for (actual, expected) in fast.commands.iter().zip(&reference.commands) {
        assert_eq!(actual.layer, expected.layer);
        assert_eq!(actual.order, expected.order);
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.screen_clip, expected.screen_clip);
        let DrawCommand::Circle(actual) = &actual.command else {
            panic!("circle fixture changed primitive");
        };
        let DrawCommand::Circle(expected) = &expected.command else {
            panic!("circle fixture changed primitive");
        };
        assert_eq!(actual.center(), expected.center());
    }
}

#[test]
fn monotonic_append_matches_partition_order_statistics_and_warmed_storage() {
    for count in [0, 100, 1_000, 10_000] {
        let commands: Vec<_> = (0..count).map(circle).collect();
        for distribution in 0..4 {
            let layers = layers(count, distribution);
            let mut fast = Scene::new(Color::BLACK).unwrap();
            let mut reference = Scene::new(Color::BLACK).unwrap();
            let (_, fast_allocations) = crate::test_allocations::count(|| {
                populate::<true>(&mut fast, &layers, &commands);
            });
            let (_, reference_allocations) = crate::test_allocations::count(|| {
                populate::<false>(&mut reference, &layers, &commands);
            });
            assert_eq!(fast_allocations, reference_allocations);
            assert_equivalent(&fast, &reference);
            let fast_pointer = fast.commands.as_ptr();
            let reference_pointer = reference.commands.as_ptr();
            fast.clear();
            reference.clear();
            let (_, fast_allocations) = crate::test_allocations::count(|| {
                populate::<true>(&mut fast, &layers, &commands);
            });
            let (_, reference_allocations) = crate::test_allocations::count(|| {
                populate::<false>(&mut reference, &layers, &commands);
            });
            assert_eq!((fast_allocations, reference_allocations), (0, 0));
            assert_eq!(fast.commands.as_ptr(), fast_pointer);
            assert_eq!(reference.commands.as_ptr(), reference_pointer);
            assert_equivalent(&fast, &reference);
        }
    }
}

#[test]
fn monotonic_append_preserves_saturated_order_and_budget_rejection() {
    let budget = SceneBudget::new(
        2,
        0,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    );
    let mut fast = Scene::with_budget(Color::BLACK, budget).unwrap();
    let mut reference = Scene::with_budget(Color::BLACK, budget).unwrap();
    fast.next_order = u64::MAX;
    reference.next_order = u64::MAX;
    for (index, layer) in [Layer::DEFAULT, Layer::DEFAULT, Layer::BACKGROUND]
        .into_iter()
        .enumerate()
    {
        let actual = fast.try_push_to_layer_ordered::<true>(layer, circle(index));
        let expected = reference.try_push_to_layer_ordered::<false>(layer, circle(index));
        assert_eq!(actual, expected);
        assert_equivalent(&fast, &reference);
    }
    assert_eq!(fast.statistics().rejected_commands(), 1);
}

#[test]
fn exact_marker_anchor_validates_without_a_renderer() {
    let outward = StrokeMarker2d::arrow(
        LogicalPixels::new(30.0).unwrap(),
        LogicalPixels::new(12.0).unwrap(),
    );
    assert_eq!(outward.anchor(), StrokeMarkerAnchor2d::BaseAtEndpoint);
    let inward = outward.with_anchor(StrokeMarkerAnchor2d::TipAtEndpoint);
    for world_width in [false, true] {
        let style = if world_width {
            StrokeStyle2d::world(WorldLength::new(2.0).unwrap(), Color::WHITE)
        } else {
            StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
        }
        .with_start_marker(inward)
        .with_end_marker(inward);
        for endpoint in [
            Vec2::X,
            -Vec2::X,
            Vec2::new(3.0, 4.0),
            Vec2::new(0.00001, 0.0),
        ] {
            assert!(DrawCommand::styled_line(Vec2::ZERO, endpoint, style).is_ok());
            assert!(DrawCommand::styled_polyline(vec![Vec2::ZERO, endpoint], style).is_ok());
        }
        let dashed =
            style.with_dash_pattern(StrokeDashPattern2d::new(&[2.0, 2.0], 0.0, 10).unwrap());
        assert!(matches!(
            DrawCommand::styled_line(Vec2::ZERO, Vec2::X, dashed),
            Err(SceneError::UnsupportedTipMarkerPath(ScenePrimitive::Line))
        ));
        assert!(matches!(
            DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 1.0)], style),
            Err(SceneError::UnsupportedTipMarkerPath(
                ScenePrimitive::Polyline
            ))
        ));
    }
}

/// Run with `cargo test --release --no-default-features paired_scene_append_benchmark
/// -- --ignored --nocapture --test-threads=1`. Timing is observational, not a gate.
#[test]
#[ignore = "paired release-mode CPU benchmark; explicitly opt in"]
fn paired_scene_append_benchmark() {
    if cfg!(debug_assertions) {
        panic!("use --release for timing");
    }
    for count in [0, 100, 1_000, 10_000] {
        let commands: Vec<_> = (0..count).map(circle).collect();
        for (distribution, name) in ["one_layer", "eight_layers", "shuffled", "descending"]
            .into_iter()
            .enumerate()
        {
            let layers = layers(count, distribution);
            let mut fast = Scene::new(Color::BLACK).unwrap();
            let mut reference = Scene::new(Color::BLACK).unwrap();
            populate::<true>(&mut fast, &layers, &commands);
            populate::<false>(&mut reference, &layers, &commands);
            let capacity = fast.commands.capacity();
            fast.clear();
            reference.clear();
            let (_, fast_allocations) = crate::test_allocations::count(|| {
                populate::<true>(&mut fast, &layers, &commands);
            });
            let (_, reference_allocations) = crate::test_allocations::count(|| {
                populate::<false>(&mut reference, &layers, &commands);
            });
            assert_eq!((fast_allocations, reference_allocations), (0, 0));
            let repetitions = (20_000 / count.max(1)).clamp(1, 200);
            let mut fast_samples = Vec::with_capacity(21);
            let mut reference_samples = Vec::with_capacity(21);
            for round in 0..25 {
                let mut measure = |use_fast: bool| {
                    let scene = if use_fast { &mut fast } else { &mut reference };
                    let started = Instant::now();
                    for _ in 0..repetitions {
                        scene.clear();
                        if use_fast {
                            populate::<true>(scene, black_box(&layers), black_box(&commands));
                        } else {
                            populate::<false>(scene, black_box(&layers), black_box(&commands));
                        }
                        black_box(scene.statistics());
                    }
                    let elapsed = started.elapsed().as_secs_f64() / repetitions as f64;
                    if round >= 4 {
                        if use_fast {
                            fast_samples.push(elapsed);
                        } else {
                            reference_samples.push(elapsed);
                        }
                    }
                };
                measure(round % 2 == 0);
                measure(round % 2 != 0);
                assert_equivalent(&fast, &reference);
                assert_eq!(fast.commands.capacity(), capacity);
            }
            fast_samples.sort_by(f64::total_cmp);
            reference_samples.sort_by(f64::total_cmp);
            let fast_median = fast_samples[fast_samples.len() / 2];
            let reference_median = reference_samples[reference_samples.len() / 2];
            println!(
                "commands={count} order={name} append_us={:.3} partition_us={:.3} ratio={:.3} append_p10_p90_us={:.3}/{:.3} partition_p10_p90_us={:.3}/{:.3} warmed_allocations={fast_allocations}/{reference_allocations} retained_capacity={capacity}",
                fast_median * 1e6,
                reference_median * 1e6,
                fast_median / reference_median,
                fast_samples[2] * 1e6,
                fast_samples[18] * 1e6,
                reference_samples[2] * 1e6,
                reference_samples[18] * 1e6,
            );
        }
    }
}
