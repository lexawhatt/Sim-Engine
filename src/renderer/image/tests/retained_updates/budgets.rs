use super::*;

pub(super) async fn verify_exact_update_budgets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    sample_count: u32,
) {
    let atlas = glyph::create_glyph_atlas_resources(
        device,
        queue,
        Arc::new(()),
        1,
        1,
        vec![255; 4],
        vec![GlyphAtlasEntry::new(
            GlyphId::new(7),
            ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        )],
        GlyphAtlasBudget::default(),
    )
    .unwrap();
    let positioned = PositionedGlyph2d::new(
        GlyphId::new(7),
        destination(-2.0, -2.0),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    let per_glyph = std::mem::size_of::<PositionedGlyph2d>() + batch_retained_bytes(1);
    let budget = GlyphRunBudget::new(3, per_glyph * 3).unwrap();
    let mut run = glyph::create_glyph_run_resources(
        device,
        queue,
        Arc::clone(&atlas.image.renderer_identity),
        &atlas,
        vec![positioned],
        budget,
    )
    .unwrap();
    let (unchanged, calls) = crate::test_allocations::count(|| {
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[positioned])
    });
    assert_eq!(calls, 0, "unchanged glyph input allocated");
    assert_eq!(unchanged.unwrap().instances().uploaded_instance_bytes(), 0);
    let old_cpu = run.recovery_memory_bytes();
    let old_gpu = run.gpu_allocation_bytes();
    let growth =
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[positioned; 3])
            .unwrap();
    assert_eq!(growth.glyph_count(), 3);
    assert_eq!(growth.capacity(), 3);
    assert_eq!(growth.retained_bytes(), per_glyph * 3);
    assert_eq!(growth.peak_retained_bytes(), old_cpu + per_glyph * 3);
    assert_eq!(
        growth.instances().peak_gpu_bytes(),
        old_gpu + 3 * std::mem::size_of::<ImageInstance>()
    );
    assert_eq!(
        growth.instances().uploaded_instance_bytes(),
        3 * std::mem::size_of::<ImageInstance>()
    );
    glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[positioned]).unwrap();
    let retained_capacity = run.capacity();
    let instance_buffer = run.batch.instance_buffer.clone();
    let cpu_before = run.recovery_memory_bytes();
    let bounds_before = run.bounds();
    let (before, _) = render_placements(
        device,
        queue,
        &atlas.image,
        &run.batch,
        format,
        sample_count,
        64,
    );
    let before_pixels = read_buffer(device, &before).await;
    let (failure, calls) = crate::test_allocations::count(|| {
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[positioned; 4])
    });
    assert_eq!(calls, 0, "count rejection allocated staging");
    assert!(matches!(
        failure,
        Err(GlyphError::EntryBudgetExceeded {
            limit: 3,
            actual: 4
        })
    ));
    let invalid =
        PositionedGlyph2d::new(GlyphId::new(99), destination(30.0, 30.0), Color::WHITE).unwrap();
    let (failure, calls) = crate::test_allocations::count(|| {
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[positioned, invalid])
    });
    assert_eq!(calls, 0, "late missing glyph allocated staging");
    assert!(matches!(
        failure,
        Err(GlyphError::MissingGlyph { index: 1, .. })
    ));
    assert_eq!(run.glyphs(), &[positioned]);
    assert_eq!(run.capacity(), retained_capacity);
    assert_eq!(run.batch.instance_buffer, instance_buffer);
    assert_eq!(run.bounds(), bounds_before);
    assert_eq!(run.recovery_memory_bytes(), cpu_before);
    let (after, _) = render_placements(
        device,
        queue,
        &atlas.image,
        &run.batch,
        format,
        sample_count,
        64,
    );
    assert_eq!(
        read_buffer(device, &after).await,
        before_pixels,
        "rejected glyph update changed drawable pixels"
    );
    let cleared = glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[]).unwrap();
    assert_eq!(cleared.glyph_count(), 0);
    assert_eq!(cleared.retained_bytes(), cpu_before);
    assert_eq!(cleared.instances().uploaded_instance_bytes(), 0);
    assert_eq!(run.batch.instance_buffer, instance_buffer);
    assert!(run.bounds().is_empty());

    let exact_one_budget = GlyphRunBudget::new(2, per_glyph).unwrap();
    let mut exact_one = glyph::create_glyph_run_resources(
        device,
        queue,
        Arc::clone(&atlas.image.renderer_identity),
        &atlas,
        vec![positioned],
        exact_one_budget,
    )
    .unwrap();
    let (failure, calls) = crate::test_allocations::count(|| {
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut exact_one, &[positioned; 2])
    });
    assert_eq!(calls, 0, "byte budget rejection allocated staging");
    assert!(
        matches!(failure, Err(GlyphError::MetadataBudgetExceeded { limit, actual }) if limit == per_glyph && actual == per_glyph * 2)
    );
    assert_eq!(exact_one.glyphs(), &[positioned]);

    let batch_budget = ImageBatchBudget::new(usize::MAX, usize::MAX).unwrap();
    let first_invalid =
        device.limits().max_buffer_size as usize / std::mem::size_of::<ImageInstance>() + 1;
    let (_, calls) = crate::test_allocations::count(|| {
        assert_eq!(
            preflight_image_batch_capacity(device, first_invalid, batch_budget),
            Err(ImageError::DimensionsTooLarge)
        );
    });
    assert_eq!(calls, 0, "GPU capacity preflight allocated");
}
