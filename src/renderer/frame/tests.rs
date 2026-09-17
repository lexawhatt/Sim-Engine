use super::*;

#[test]
fn scalar_lut_accounting_matches_one_entry_cache_runs() {
    const BYTES: usize = COLOR_MAP_LUT_SIZE as usize * 4;
    let a = [1; BYTES];
    let b = [2; BYTES];
    let adjacent = vec![
        ScalarLutPlan {
            sort_key: (0, 0),
            lut: a,
        },
        ScalarLutPlan {
            sort_key: (0, 2),
            lut: a,
        },
    ];
    assert_eq!(
        scalar_lut_counts_with_inserted(&adjacent, 1, &a, Some(&a)),
        (0, 1)
    );
    assert_eq!(
        scalar_lut_counts_with_inserted(&adjacent, 1, &b, Some(&a)),
        (2, 3),
        "A, B, A performs two uploads and keeps three LUT allocations alive"
    );
    assert_eq!(
        scalar_lut_counts_with_inserted(&adjacent[..1], 1, &b, None),
        (2, 2)
    );
}

#[test]
fn image_and_glyph_sprite_bounds_include_final_clip_arithmetic() {
    let source = ImageTexelRect::new(0, 0, 1, 1).unwrap();
    let safe = ImageSprite2d::new(
        source,
        LogicalViewportRegion::new(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalViewport::new(1.0, 1.0).unwrap(),
        )
        .unwrap(),
        Color::WHITE,
    )
    .unwrap();
    let overflowing = ImageSprite2d::new(
        source,
        LogicalViewportRegion::new(
            LogicalScreenPosition::new(-2.0e38, 0.0),
            LogicalViewport::new(3.0e38, 1.0).unwrap(),
        )
        .unwrap(),
        Color::WHITE,
    )
    .unwrap();
    let one_pixel_clip = [2.0, -2.0, -1.0, 1.0];

    assert!(image_sprites_are_safe_for_target(
        &[safe],
        Vec2::ZERO,
        one_pixel_clip,
    ));
    assert!(!image_sprites_are_safe_for_target(
        &[overflowing],
        Vec2::ZERO,
        one_pixel_clip,
    ));
}

#[test]
fn frame_budget_accepts_exact_limit_and_rejects_one_over() {
    let budget = FrameBudget::new(1, 2, 3, 4, 5, 6);
    let exact = FrameStatistics {
        pass_count: 1,
        command_count: 2,
        vertex_count: 3,
        streaming_vertex_count: 0,
        reused_vertex_count: 0,
        upload_bytes: 4,
        streaming_upload_bytes: 0,
        texture_bytes: 5,
        retained_cpu_bytes: 7,
        retained_buffer_bytes: 8,
        draw_calls: 6,
        source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
    };
    assert_eq!(validate_frame_budget(budget, exact), Ok(()));
    assert_eq!(
        validate_frame_budget(
            budget,
            FrameStatistics {
                draw_calls: 7,
                ..exact
            }
        ),
        Err(FrameComposerError::BudgetExceeded {
            resource: FrameBudgetResource::DrawCalls,
            limit: 6,
            actual: 7,
        })
    );
}

#[test]
fn frame_statistics_group_sources_and_retained_memory_without_changing_budgets() {
    let streaming = FrameStatistics {
        pass_count: 1,
        retained_cpu_bytes: 128,
        source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        ..FrameStatistics::default()
    };
    let glyph = FrameStatistics {
        pass_count: 1,
        retained_cpu_bytes: 256,
        retained_buffer_bytes: 512,
        source_counts: FrameSourceStatistics::single(FrameSourceKind::Glyph),
        ..FrameStatistics::default()
    };

    let combined = streaming.adding(glyph);

    assert_eq!(combined.pass_count(), 2);
    assert_eq!(combined.retained_cpu_bytes(), 384);
    assert_eq!(combined.retained_buffer_bytes(), 512);
    assert_eq!(combined.source_counts().streaming_scenes(), 1);
    assert_eq!(combined.source_counts().glyph_runs(), 1);
    assert_eq!(combined.source_counts().total(), 2);
    assert_eq!(
        validate_frame_budget(FrameBudget::new(2, 0, 0, 0, 0, 0), combined),
        Ok(())
    );
}

#[test]
fn repeated_retained_references_count_each_allocation_once() {
    let first = RetainedResourceKey {
        address: 0x1000,
        class: 3,
    };
    let second = RetainedResourceKey {
        address: 0x2000,
        class: 7,
    };
    let resources = [
        RetainedResourceAccounting::new(first, 128, 256, 0),
        RetainedResourceAccounting::new(first, 128, 256, 0),
        RetainedResourceAccounting::new(second, 64, 0, 512),
    ];

    let (first_frame, missing) =
        account_new_retained_resources(&[], &resources, FrameStatistics::default());
    assert_eq!(missing, 2);
    assert_eq!(first_frame.retained_cpu_bytes(), 192);
    assert_eq!(first_frame.retained_buffer_bytes(), 256);
    assert_eq!(first_frame.texture_bytes(), 512);

    let (repeated, missing) =
        account_new_retained_resources(&[first, second], &resources, FrameStatistics::default());
    assert_eq!(missing, 0);
    assert_eq!(repeated.retained_cpu_bytes(), 0);
    assert_eq!(repeated.retained_buffer_bytes(), 0);
    assert_eq!(repeated.texture_bytes(), 0);
}

#[test]
fn frame_report_exposes_single_present_contract_and_skipped_zeroes() {
    let drawn = FrameReport {
        status: RenderStatus::Drawn,
        statistics: FrameStatistics::default(),
        metrics: RendererFrameMetrics::default(),
        gpu_timing_id: None,
    };
    assert_eq!(drawn.command_encoder_count(), 1);
    assert_eq!(drawn.render_pass_count(), 1);
    assert_eq!(drawn.queue_submission_count(), 1);
    assert_eq!(drawn.surface_present_count(), 1);

    let skipped = FrameReport {
        status: RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
        ..drawn
    };
    assert_eq!(skipped.command_encoder_count(), 0);
    assert_eq!(skipped.render_pass_count(), 0);
    assert_eq!(skipped.queue_submission_count(), 0);
    assert_eq!(skipped.surface_present_count(), 0);
    assert_eq!(skipped.gpu_timing_id(), None);

    let planned = FrameStatistics {
        upload_bytes: 128,
        streaming_upload_bytes: 96,
        vertex_count: 12,
        ..FrameStatistics::default()
    };
    let skipped_statistics = planned.without_uploads();
    assert_eq!(skipped_statistics.upload_bytes(), 0);
    assert_eq!(skipped_statistics.streaming_upload_bytes(), 0);
    assert_eq!(skipped_statistics.vertex_count(), 12);
}

#[test]
fn frame_items_sort_stably_by_order_then_insertion() {
    let scene = Scene::new(Color::BLACK).unwrap();
    let camera = Camera2d::default();
    let mut items = [
        FrameItem::Scene {
            scene: &scene,
            camera,
            options: FramePassOptions::new(4),
            insertion: 0,
        },
        FrameItem::Scene {
            scene: &scene,
            camera,
            options: FramePassOptions::new(-1),
            insertion: 1,
        },
        FrameItem::Scene {
            scene: &scene,
            camera,
            options: FramePassOptions::new(4),
            insertion: 2,
        },
    ];
    items.sort_unstable_by_key(FrameItem::sort_key);
    assert_eq!(
        items.iter().map(FrameItem::sort_key).collect::<Vec<_>>(),
        vec![(-1, 1), (4, 0), (4, 2)]
    );
}

#[test]
fn late_streaming_transform_rejection_rolls_back_appended_vertices() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_line(
            Vec2::new(1.0e30, 0.0),
            Vec2::new(1.1e30, 0.0),
            1.0,
            Color::WHITE,
        )
        .unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let camera = Camera2d::new(Vec2::ZERO, 1.0e10).unwrap();
    let uniform = CameraUniform::new_in_region(camera, viewport, Vec2::ZERO, viewport).unwrap();
    let resolved = ResolvedViewport {
        viewport,
        origin: Vec2::ZERO,
        scissor: ScissorRect {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        },
        item_clip: None,
        item_clipped_out: false,
    };
    let mut vertices = Vec::new();
    let mut ready = Vec::new();
    let mut statistics = FrameStatistics::default();
    let mut aggregate = TessellationStats::default();

    assert_eq!(
        prepare_streaming_scene_resolved(
            &scene,
            uniform,
            resolved,
            &mut vertices,
            &mut ready,
            &mut statistics,
            &mut aggregate,
            &mut Vec::new(),
        ),
        Err(FrameComposerError::Frame(
            RendererFrameError::InvalidGeometryTransform
        ))
    );
    assert!(vertices.is_empty());
    assert!(ready.is_empty());
    assert_eq!(statistics, FrameStatistics::default());
    assert_eq!(aggregate, TessellationStats::default());
}

#[test]
fn streaming_screen_scene_rejects_generated_subnormal_geometry() {
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let mut scene = ScreenScene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            LogicalScreenPosition::new(32.0, 32.0),
            crate::LogicalPixels::new(f32::MIN_POSITIVE).unwrap(),
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    let camera = screen_camera(viewport).unwrap();
    let uniform = CameraUniform::new_in_region(camera, viewport, Vec2::ZERO, viewport).unwrap();
    let resolved = ResolvedViewport {
        viewport,
        origin: Vec2::ZERO,
        scissor: ScissorRect {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        },
        item_clip: None,
        item_clipped_out: false,
    };
    let mut vertices = Vec::new();
    let mut ready = Vec::new();
    let mut statistics = FrameStatistics::default();
    let mut aggregate = TessellationStats::default();

    assert_eq!(
        prepare_streaming_scene_resolved(
            scene.as_scene(),
            uniform,
            resolved,
            &mut vertices,
            &mut ready,
            &mut statistics,
            &mut aggregate,
            &mut Vec::new(),
        ),
        Err(FrameComposerError::Frame(
            RendererFrameError::InvalidGeometryTransform
        ))
    );
    assert!(vertices.is_empty());
    assert!(ready.is_empty());
}

#[test]
fn scissor_intersection_rejects_disjoint_rectangles() {
    let first = ScissorRect {
        x: 2,
        y: 3,
        width: 8,
        height: 9,
    };
    let overlapping = ScissorRect {
        x: 8,
        y: 10,
        width: 8,
        height: 8,
    };
    assert_eq!(
        intersect_scissors(first, overlapping),
        Some(ScissorRect {
            x: 8,
            y: 10,
            width: 2,
            height: 2,
        })
    );
    assert_eq!(
        intersect_scissors(
            first,
            ScissorRect {
                x: 10,
                y: 3,
                width: 2,
                height: 2,
            }
        ),
        None
    );
}

#[test]
fn target_destination_maps_logical_region_to_clip_space() {
    let target = LogicalViewport::new(800.0, 600.0).unwrap();
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(200.0, 150.0),
        LogicalViewport::new(400.0, 300.0).unwrap(),
    )
    .unwrap();
    let uniform = CompositeUniform::in_region(0.5, region, target).unwrap();
    assert_eq!(uniform.opacity[0], 0.5);
    assert_eq!(uniform.destination, [0.5, 0.5, 0.0, 0.0]);
}
