use super::*;
use std::{hint::black_box, time::Instant};

fn sprite(x: f32, y: f32, width: f32, height: f32) -> ImageSprite2d {
    ImageSprite2d::new(
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        LogicalViewportRegion::new(
            LogicalScreenPosition::new(x, y),
            LogicalViewport::new(width, height).unwrap(),
        )
        .unwrap(),
        Color::WHITE,
    )
    .unwrap()
}

fn glyphs() -> Vec<ImageSprite2d> {
    (0..32)
        .map(|index| sprite(index as f32 * 3.0, 0.0, 2.0, 6.0))
        .collect()
}

#[test]
fn geometry_comparison_uses_all_four_destination_bits_not_uv_or_tint() {
    let original = sprite(0.0, 2.0, 4.0, 8.0);
    let changed_presentation = ImageSprite2d::new(
        ImageTexelRect::new(1, 0, 1, 1).unwrap(),
        original.destination(),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    assert_eq!(
        destination_bits(original),
        destination_bits(changed_presentation)
    );
    for changed in [
        sprite(-0.0, 2.0, 4.0, 8.0),
        sprite(0.0, f32::from_bits(2.0f32.to_bits() + 1), 4.0, 8.0),
        sprite(0.0, 2.0, f32::from_bits(4.0f32.to_bits() + 1), 8.0),
        sprite(0.0, 2.0, 4.0, f32::from_bits(8.0f32.to_bits() + 1)),
    ] {
        assert_ne!(destination_bits(original), destination_bits(changed));
    }
    assert_eq!(
        crate::test_allocations::count(
            || destination_bits(original) == destination_bits(changed_presentation)
        )
        .1,
        0
    );
}

#[test]
fn exact_cache_matches_uncached_proof_at_normal_extreme_and_ftz_boundaries() {
    let batches = [
        Vec::new(),
        glyphs(),
        vec![sprite(-2.0e38, 0.0, 3.0e38, 1.0)],
        vec![sprite(f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE, 1.0)],
        vec![sprite(f32::from_bits(1), 0.0, 1.0, 1.0)],
        vec![sprite(-1.0e20, 1.0e20, 1.0, 1.0)],
    ];
    let values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::EPSILON,
        1.0,
        -1.0,
        1280.0,
        1.0e20,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    for sprites in &batches {
        let cache = BatchProofCache::default();
        for component in 0..6 {
            for value in values {
                let mut transform = [0.0, 0.0, 2.0 / 1280.0, -2.0 / 720.0, -1.0, 1.0];
                transform[component] = value;
                let origin = Vec2::new(transform[0], transform[1]);
                let clip = transform[2..].try_into().unwrap();
                let expected = sprites_are_safe_for_target(sprites, origin, clip);
                assert_eq!(cache.validate(sprites, origin, clip), expected);
                assert_eq!(cache.validate(sprites, origin, clip), expected);
                let entry = cache.entry.lock().unwrap().unwrap();
                assert_eq!(entry.key, transform.map(f32::to_bits));
                assert_eq!(entry.accepted, expected);
            }
        }
    }
}

#[test]
fn exact_cache_invalidation_does_not_reuse_old_accepted_or_rejected_revision() {
    let mut cache = BatchProofCache::default();
    let origin = Vec2::ZERO;
    let clip = [2.0, -2.0, -1.0, 1.0];
    let safe = [sprite(0.0, 0.0, 1.0, 1.0)];
    let unsafe_source = [sprite(-2.0e38, 0.0, 3.0e38, 1.0)];
    assert!(cache.validate(&safe, origin, clip));
    cache.invalidate();
    assert!(!cache.validate(&unsafe_source, origin, clip));
    cache.invalidate();
    assert!(cache.validate(&safe, origin, clip));
    let (_, allocations) = crate::test_allocations::count(|| {
        for _ in 0..100 {
            assert!(cache.validate(&safe, origin, clip));
        }
    });
    assert_eq!(allocations, 0);
}

#[test]
fn exact_cache_remains_send_sync_with_concurrent_distinct_transform_readers() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ImageBatch2d>();
    let cache = BatchProofCache::default();
    let sprites = glyphs();
    std::thread::scope(|scope| {
        for thread in 0..8 {
            let cache = &cache;
            let sprites = &sprites;
            scope.spawn(move || {
                for pass in 0..100 {
                    let origin = Vec2::new(thread as f32 * 100.0 + pass as f32, 30.0);
                    let clip = if thread % 2 == 0 {
                        [2.0 / 1280.0, -2.0 / 720.0, -1.0, 1.0]
                    } else {
                        [f32::MAX, -2.0, -1.0, 1.0]
                    };
                    assert_eq!(
                        cache.validate(sprites, origin, clip),
                        sprites_are_safe_for_target(sprites, origin, clip)
                    );
                }
            });
        }
    });
}

#[test]
#[ignore = "opt-in paired release CPU image/glyph proof benchmark"]
fn paired_exact_image_proof_cache_benchmark() {
    if cfg!(debug_assertions) {
        panic!("benchmark requires --release");
    }
    for labels in [1, 100, 1000] {
        let runs: Vec<_> = (0..labels)
            .map(|_| (glyphs(), BatchProofCache::default()))
            .collect();
        for moving in [false, true] {
            let operation = |cached: bool, frame: usize| {
                for (index, (sprites, cache)) in black_box(&runs).iter().enumerate() {
                    let offset = if moving {
                        (frame as f32).sin() * 8.0
                    } else {
                        0.0
                    };
                    let origin = Vec2::new(
                        (index % 10) as f32 * 112.0 + offset,
                        (index / 10) as f32 * 7.0,
                    );
                    let clip = [2.0 / 1280.0, -2.0 / 720.0, -1.0, 1.0];
                    assert!(black_box(if cached {
                        cache.validate(sprites, origin, clip)
                    } else {
                        sprites_are_safe_for_target(sprites, origin, clip)
                    }));
                }
            };
            for frame in 0..4 {
                operation(false, frame);
                operation(true, frame);
            }
            let uncached_allocations = crate::test_allocations::count(|| operation(false, 4)).1;
            let cached_allocations = crate::test_allocations::count(|| operation(true, 4)).1;
            let mut samples = [Vec::with_capacity(21), Vec::with_capacity(21)];
            for frame in 0..21 {
                let modes = if frame % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for cached in modes {
                    let start = Instant::now();
                    operation(cached, frame + 5);
                    samples[usize::from(cached)].push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            for values in &mut samples {
                values.sort_by(f64::total_cmp);
            }
            println!(
                "image_proof labels={labels} glyphs=32 moving={moving} uncached_median_ms={:.6} cached_median_ms={:.6} uncached_p10_p90_ms={:.6}/{:.6} cached_p10_p90_ms={:.6}/{:.6} allocations={uncached_allocations}/{cached_allocations}",
                samples[0][10],
                samples[1][10],
                samples[0][2],
                samples[0][18],
                samples[1][2],
                samples[1][18]
            );
        }
    }
}

pub(in crate::renderer::image) fn verify_revision_invalidation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    verify_geometry_only_preservation(device, queue);
    let identity = Arc::new(());
    let image = create_image_resources(
        device,
        queue,
        Arc::clone(&identity),
        1,
        1,
        vec![255; 4],
        ImageBudget::default(),
    )
    .unwrap();
    let sprites = [sprite(0.0, 0.0, 1.0, 1.0)];
    let budget = ImageBatchBudget::new(4, batch_retained_bytes(4)).unwrap();
    let mut batch = create_image_batch_resources(
        device,
        queue,
        Arc::clone(&identity),
        &image,
        sprites.to_vec(),
        budget,
    )
    .unwrap();
    let origin = Vec2::new(MAX_PORTABLE_SHADER_VALUE * 0.75, 0.0);
    let clip = [1.0, -1.0, 0.0, 0.0];
    assert!(batch.sprites_are_safe_for_target(origin, clip));
    let retained = batch.recovery_memory_bytes();
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        sprites.into_iter(),
        1,
        budget,
    )
    .unwrap();
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_some(),
        "equal update discarded valid proof"
    );
    let moved = [sprite(MAX_PORTABLE_SHADER_VALUE * 0.5, 0.0, 1.0, 1.0)];
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        moved.into_iter(),
        1,
        budget,
    )
    .unwrap();
    assert!(batch.geometry_proof.entry.lock().unwrap().is_none());
    assert!(
        !batch.sprites_are_safe_for_target(origin, clip),
        "changed destination reused old accepted proof"
    );
    assert_eq!(batch.recovery_memory_bytes(), retained);
    let error = update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        sprites.into_iter().cycle().take(5),
        5,
        budget,
    );
    assert!(error.is_err());
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_some(),
        "rejection discarded unchanged proof"
    );
    assert!(!batch.sprites_are_safe_for_target(origin, clip));
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        sprites.into_iter(),
        1,
        budget,
    )
    .unwrap();
    assert!(
        batch.sprites_are_safe_for_target(origin, clip),
        "changed revision reused old rejection"
    );
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        std::iter::empty(),
        0,
        budget,
    )
    .unwrap();
    assert!(batch.geometry_proof.entry.lock().unwrap().is_none());
    assert!(batch.sprites_are_safe_for_target(origin, clip));
    let recovery_identity = Arc::new(());
    let recovered_image = create_image_resources(
        recovery_device,
        recovery_queue,
        Arc::clone(&recovery_identity),
        1,
        1,
        image.pixels().to_vec(),
        ImageBudget::default(),
    )
    .unwrap();
    let restored = create_image_batch_resources(
        recovery_device,
        recovery_queue,
        recovery_identity,
        &recovered_image,
        batch.sprites().to_vec(),
        budget,
    )
    .unwrap();
    assert!(
        restored.geometry_proof.entry.lock().unwrap().is_none(),
        "restoration inherited another batch's cached proof"
    );
    assert!(restored.sprites_are_safe_for_target(origin, clip));
}

fn verify_geometry_only_preservation(device: &wgpu::Device, queue: &wgpu::Queue) {
    let identity = Arc::new(());
    let image = create_image_resources(
        device,
        queue,
        Arc::clone(&identity),
        2,
        1,
        vec![255; 8],
        ImageBudget::default(),
    )
    .unwrap();
    let original = sprite(0.0, 0.0, 1.0, 1.0);
    let budget = ImageBatchBudget::new(4, batch_retained_bytes(4)).unwrap();
    let mut batch =
        create_image_batch_resources(device, queue, identity, &image, vec![original; 2], budget)
            .unwrap();
    let clip = [2.0 / 1280.0, -2.0 / 720.0, -1.0, 1.0];
    assert!(batch.sprites_are_safe_for_target(Vec2::ZERO, clip));
    let initial_proof = *batch.geometry_proof.entry.lock().unwrap();
    let cpu_pointer = batch.sprites.as_ptr();
    let retained = batch.recovery_memory_bytes();

    let uv = ImageSprite2d::new(
        ImageTexelRect::new(1, 0, 1, 1).unwrap(),
        original.destination(),
        Color::WHITE,
    )
    .unwrap();
    let tinted = ImageSprite2d::new(
        uv.source(),
        uv.destination(),
        Color::rgba(0.25, 0.5, 1.0, 0.5),
    )
    .unwrap();
    for presentation in [uv, tinted] {
        let report = update_image_batch_resources(
            device,
            queue,
            &image,
            &mut batch,
            [presentation; 2].into_iter(),
            2,
            budget,
        )
        .unwrap();
        assert!(
            report.uploaded_instance_bytes() > 0,
            "fixture must publish a real UV/tint update"
        );
        assert_eq!(
            *batch.geometry_proof.entry.lock().unwrap(),
            initial_proof,
            "UV/tint-only update discarded exact proof"
        );
        assert_eq!(batch.sprites(), &[presentation; 2]);
        assert_eq!(batch.sprites.as_ptr(), cpu_pointer);
        assert_eq!(batch.recovery_memory_bytes(), retained);
    }
    // Candidate validation must finish before even cache metadata changes.
    let invalid = ImageSprite2d::new(
        ImageTexelRect::new(2, 0, 1, 1).unwrap(),
        original.destination(),
        Color::WHITE,
    )
    .unwrap();
    assert!(
        update_image_batch_resources(
            device,
            queue,
            &image,
            &mut batch,
            [original, invalid].into_iter(),
            2,
            budget
        )
        .is_err()
    );
    assert_eq!(*batch.geometry_proof.entry.lock().unwrap(), initial_proof);
    assert_eq!(batch.sprites(), &[tinted; 2]);

    let wider = sprite(0.0, 0.0, 2.0, 1.0);
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [tinted, wider].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_none(),
        "second destination size was not compared"
    );
    assert!(batch.sprites_are_safe_for_target(Vec2::ZERO, clip));
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [tinted].into_iter(),
        1,
        budget,
    )
    .unwrap();
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_none(),
        "count-only shrink preserved old proof"
    );
    assert!(batch.sprites_are_safe_for_target(Vec2::ZERO, clip));
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [tinted; 2].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_none(),
        "count-only growth preserved old proof"
    );
    assert!(batch.sprites_are_safe_for_target(Vec2::ZERO, clip));

    // Force a publication with different zero bits: PartialEq considers +0/-0
    // equal, but the simultaneous color change makes this a real update.
    let negative_zero = sprite(-0.0, 0.0, 1.0, 1.0);
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [negative_zero; 2].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert!(
        batch.geometry_proof.entry.lock().unwrap().is_none(),
        "published signed-zero position reused a non-identical key"
    );
    assert_eq!(
        batch.sprites()[0]
            .destination()
            .origin()
            .to_vec2()
            .x
            .to_bits(),
        (-0.0f32).to_bits()
    );
    let overflowing_origin = Vec2::new(MAX_PORTABLE_SHADER_VALUE, 0.0);
    assert!(!batch.sprites_are_safe_for_target(overflowing_origin, [1.0, -1.0, 0.0, 0.0]));
    let rejected_proof = *batch.geometry_proof.entry.lock().unwrap();
    let tint_only = ImageSprite2d::new(
        negative_zero.source(),
        negative_zero.destination(),
        Color::BLACK,
    )
    .unwrap();
    update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [tint_only; 2].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert_eq!(
        *batch.geometry_proof.entry.lock().unwrap(),
        rejected_proof,
        "presentation-only update discarded exact rejection"
    );
}
