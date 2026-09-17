use super::*;

#[test]
fn line_tessellates_with_round_caps() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.line(
        Vec2::new(-10.0, 0.0),
        Vec2::new(10.0, 0.0),
        2.0,
        Color::WHITE,
    );

    let (vertices, _) = tessellate_for_test(&scene);

    assert_eq!(vertices.len(), 6 + ROUND_CAP_SEGMENTS * 6);
}

#[test]
fn polyline_uses_joined_strip_and_only_two_round_caps() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.polyline(
        vec![
            Vec2::new(-10.0, 0.0),
            Vec2::new(-2.0, 5.0),
            Vec2::new(4.0, -3.0),
            Vec2::new(10.0, 0.0),
        ],
        3.0,
        Color::WHITE,
    );

    let (vertices, _) = tessellate_for_test(&scene);

    // Miter fallback candidates are retained for each join because the final
    // limit decision is made after camera projection in WGSL.
    assert_eq!(vertices.len(), 3 * 6 + 2 * 6 + ROUND_CAP_SEGMENTS * 6);
    assert!(vertices.iter().copied().all(Vertex::is_finite));
}

#[test]
fn richer_strokes_have_deterministic_bounded_topology() {
    let logical_width = crate::LogicalPixels::new(2.0).unwrap();
    let marker = crate::StrokeMarker2d::arrow(
        crate::LogicalPixels::new(5.0).unwrap(),
        crate::LogicalPixels::new(4.0).unwrap(),
    );
    let points = vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 1.0)];

    let vertex_count = |style| {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.try_styled_polyline(points.clone(), style).unwrap();
        tessellate_for_test(&scene).0.len()
    };

    let base = crate::StrokeStyle2d::logical(logical_width, Color::WHITE)
        .with_cap(crate::StrokeCap2d::Butt);
    assert_eq!(vertex_count(base.with_join(crate::StrokeJoin2d::Miter)), 18);
    assert_eq!(vertex_count(base.with_join(crate::StrokeJoin2d::Bevel)), 18);
    assert_eq!(
        vertex_count(base.with_join(crate::StrokeJoin2d::Round)),
        12 + ROUND_CAP_SEGMENTS * 6
    );

    let marked = base.with_start_marker(marker).with_end_marker(marker);
    assert_eq!(vertex_count(marked), 24);

    let dash = crate::StrokeDashPattern2d::new(&[2.0, 2.0], 0.0, 4).unwrap();
    let mut dashed = Scene::new(Color::BLACK).unwrap();
    dashed
        .try_styled_line(
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            base.with_dash_pattern(dash),
        )
        .unwrap();
    assert_eq!(tessellate_for_test(&dashed).0.len(), 3 * 6);
}

#[test]
fn short_endpoint_markers_extend_outward_from_a_butt_boundary() {
    let marker = crate::StrokeMarker2d::arrow(
        crate::LogicalPixels::new(3.0).unwrap(),
        crate::LogicalPixels::new(4.0).unwrap(),
    );
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_styled_line(
            Vec2::ZERO,
            Vec2::new(4.0, 0.0),
            crate::StrokeStyle2d::logical(crate::LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Round)
                .with_start_marker(marker)
                .with_end_marker(marker),
        )
        .unwrap();

    let vertices = tessellate_for_test(&scene).0;
    assert_eq!(vertices.len(), 12);
    assert!(
        vertices[..6]
            .iter()
            .all(|vertex| vertex.tangent_distance == 0.0)
    );
    assert_eq!(vertices[6].tangent_distance, -3.0);
    assert_eq!(vertices[7].tangent_distance, 0.0);
    assert_eq!(vertices[8].tangent_distance, 0.0);
    assert_eq!(vertices[9].tangent_distance, 3.0);
    assert_eq!(vertices[10].tangent_distance, 0.0);
    assert_eq!(vertices[11].tangent_distance, 0.0);
}

#[test]
fn dash_run_crossing_a_polyline_vertex_uses_one_join_without_internal_caps() {
    let dash = crate::StrokeDashPattern2d::new(&[6.0, 2.0], 0.0, 8).unwrap();
    let style =
        crate::StrokeStyle2d::logical(crate::LogicalPixels::new(2.0).unwrap(), Color::WHITE)
            .with_cap(crate::StrokeCap2d::Round)
            .with_join(crate::StrokeJoin2d::Round)
            .with_dash_pattern(dash);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_styled_polyline(
            vec![Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(4.0, 4.0)],
            style,
        )
        .unwrap();

    let vertices = tessellate_for_test(&scene).0;

    // Two quads form one visible dash, the bend carries two projected-turn
    // candidates (only one survives in WGSL), and only the two actual dash
    // endpoints receive semicircular caps.
    assert_eq!(
        vertices.len(),
        2 * 6 + ROUND_CAP_SEGMENTS * 6 + 2 * ROUND_CAP_SEGMENTS * 3
    );
    assert!(vertices.iter().copied().all(Vertex::is_finite));
    assert!(scene.statistics().estimated_tessellated_vertices() >= vertices.len());
}

#[test]
fn dash_phase_and_every_cap_join_combination_are_deterministic() {
    let width = crate::LogicalPixels::new(2.0).unwrap();
    let points = vec![Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(4.0, 4.0)];
    for cap in [
        crate::StrokeCap2d::Butt,
        crate::StrokeCap2d::Square,
        crate::StrokeCap2d::Round,
    ] {
        for join in [
            crate::StrokeJoin2d::Bevel,
            crate::StrokeJoin2d::Miter,
            crate::StrokeJoin2d::Round,
        ] {
            let mut scene = Scene::new(Color::BLACK).unwrap();
            scene
                .try_styled_polyline(
                    points.clone(),
                    crate::StrokeStyle2d::logical(width, Color::WHITE)
                        .with_cap(cap)
                        .with_join(join),
                )
                .unwrap();
            let vertices = tessellate_for_test(&scene).0;
            assert!(vertices.iter().copied().all(Vertex::is_finite));
            assert!(scene.statistics().estimated_tessellated_vertices() >= vertices.len());
        }
    }

    let dash_count = |phase| {
        let dash = crate::StrokeDashPattern2d::new(&[2.0, 2.0], phase, 8).unwrap();
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .try_styled_line(
                Vec2::ZERO,
                Vec2::new(10.0, 0.0),
                crate::StrokeStyle2d::logical(width, Color::WHITE)
                    .with_cap(crate::StrokeCap2d::Butt)
                    .with_dash_pattern(dash),
            )
            .unwrap();
        tessellate_for_test(&scene).0.len()
    };
    assert_eq!(dash_count(0.0), 18);
    assert_eq!(dash_count(2.0), 12);
}

#[test]
fn logical_join_shader_branches_accept_validated_right_angles() {
    let viewport = LogicalViewport::new(128.0, 128.0).unwrap();
    let uniform = CameraUniform::new(Camera2d::default(), viewport).unwrap();
    let style =
        crate::StrokeStyle2d::logical(crate::LogicalPixels::new(4.0).unwrap(), Color::WHITE)
            .with_cap(crate::StrokeCap2d::Butt)
            .with_join(crate::StrokeJoin2d::Round);

    let mut right_angle = Scene::new(Color::BLACK).unwrap();
    right_angle
        .try_styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)], style)
        .unwrap();
    let (vertices, _) = tessellate_for_test(&right_angle);
    assert!(geometry_is_safe_for(
        GeometryExtents::from_vertices(&vertices),
        GeometryValidationSource::Tessellated(&vertices),
        uniform,
    ));
}

#[test]
fn styled_strokes_preserve_clip_and_bound_miter_extrusion() {
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(10.0, 12.0),
        LogicalScreenVector::new(40.0, 30.0),
    )
    .unwrap();
    let style =
        crate::StrokeStyle2d::logical(crate::LogicalPixels::new(4.0).unwrap(), Color::WHITE)
            .with_cap(crate::StrokeCap2d::Butt)
            .with_join(crate::StrokeJoin2d::Miter)
            .with_miter_limit(1.0)
            .unwrap();
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .with_screen_clip(clip, |scene| {
            scene
                .try_styled_polyline(
                    vec![Vec2::new(-10.0, 0.0), Vec2::ZERO, Vec2::new(-9.0, 1.0)],
                    style,
                )
                .unwrap();
        })
        .unwrap();

    let (vertices, batches) = tessellate_for_test(&scene);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].screen_clip, Some(clip));
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(200.0, 120.0).unwrap();
    let center = camera
        .world_to_screen(Vec2::ZERO, viewport)
        .unwrap()
        .to_vec2();
    for vertex in vertices
        .iter()
        .copied()
        .filter(|vertex| vertex.world_position == [0.0, 0.0])
    {
        assert!((vertex_screen_position(vertex, camera, viewport) - center).length() <= 2.01);
    }
}

#[test]
fn stroke_caps_and_width_spaces_follow_their_contract() {
    let logical_width = crate::LogicalPixels::new(2.0).unwrap();
    let line_vertices = |style| {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .try_styled_line(Vec2::new(-5.0, 0.0), Vec2::new(5.0, 0.0), style)
            .unwrap();
        tessellate_for_test(&scene).0
    };
    let base = crate::StrokeStyle2d::logical(logical_width, Color::WHITE);
    assert_eq!(
        line_vertices(base.with_cap(crate::StrokeCap2d::Butt)).len(),
        6
    );
    assert_eq!(
        line_vertices(base.with_cap(crate::StrokeCap2d::Square)).len(),
        6
    );
    assert_eq!(
        line_vertices(base.with_cap(crate::StrokeCap2d::Round)).len(),
        6 + ROUND_CAP_SEGMENTS * 6
    );

    let viewport = LogicalViewport::new(200.0, 100.0).unwrap();
    let logical = line_vertices(base.with_cap(crate::StrokeCap2d::Butt));
    let world = line_vertices(
        crate::StrokeStyle2d::world(crate::WorldLength::new(2.0).unwrap(), Color::WHITE)
            .with_cap(crate::StrokeCap2d::Butt),
    );
    let height = |vertices: &[Vertex], camera| {
        let projected: Vec<_> = vertices
            .iter()
            .copied()
            .map(|vertex| vertex_screen_position(vertex, camera, viewport))
            .collect();
        projected
            .iter()
            .map(|point| point.y)
            .fold(f32::NEG_INFINITY, f32::max)
            - projected
                .iter()
                .map(|point| point.y)
                .fold(f32::INFINITY, f32::min)
    };
    let zoom_one = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let zoom_four = Camera2d::new(Vec2::ZERO, 4.0).unwrap();
    assert!((height(&logical, zoom_one) - 2.0).abs() < 0.01);
    assert!((height(&logical, zoom_four) - 2.0).abs() < 0.01);
    assert!((height(&world, zoom_one) - 2.0).abs() < 0.01);
    assert!((height(&world, zoom_four) - 8.0).abs() < 0.01);
}

#[test]
fn world_width_stroke_preserves_width_below_large_base_ulp() {
    let center = Vec2::new(1.0e20, 0.0);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_styled_line(
            Vec2::new(center.x, -1.0),
            Vec2::new(center.x, 1.0),
            crate::StrokeStyle2d::world(crate::WorldLength::new(1.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();

    let (vertices, batches) = tessellate_for_test(&scene);
    assert_eq!(vertices.len(), 6);
    assert_eq!(batches.len(), 1);
    assert!(
        vertices
            .iter()
            .all(|vertex| vertex.world_offset[0].abs() == 0.5)
    );

    let camera = Camera2d::new(center, 100.0).unwrap();
    let viewport = LogicalViewport::new(320.0, 240.0).unwrap();
    let projected: Vec<_> = vertices
        .iter()
        .copied()
        .map(|vertex| vertex_screen_position(vertex, camera, viewport))
        .collect();
    let minimum = projected
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min);
    let maximum = projected
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!((maximum - minimum - 100.0).abs() < 0.01);

    let mut overflow_scene = Scene::new(Color::BLACK).unwrap();
    overflow_scene
        .try_styled_line(
            Vec2::new(center.x, -1.0),
            Vec2::new(center.x, 1.0),
            crate::StrokeStyle2d::world(crate::WorldLength::new(4.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();
    let (overflow_vertices, _) = tessellate_for_test(&overflow_scene);
    let tiny_viewport = LogicalViewport::new(1.0e-38, 1.0).unwrap();
    assert!(CameraUniform::new(Camera2d::new(center, 1.0).unwrap(), tiny_viewport).is_none());
    assert!(!overflow_vertices.is_empty());
}

#[test]
fn near_reversal_is_rejected_before_join_tessellation() {
    let points = vec![Vec2::ZERO, Vec2::X, Vec2::new(0.0, 0.000_000_1)];

    for join in [
        crate::StrokeJoin2d::Bevel,
        crate::StrokeJoin2d::Round,
        crate::StrokeJoin2d::Miter,
    ] {
        for style in [
            crate::StrokeStyle2d::logical(crate::LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(join),
            crate::StrokeStyle2d::world(crate::WorldLength::new(2.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(join),
        ] {
            let mut scene = Scene::new(Color::BLACK).unwrap();
            assert!(matches!(
                scene.try_styled_polyline(points.clone(), style),
                Err(crate::SceneError::DegenerateStrokeTurn {
                    primitive: crate::ScenePrimitive::Polyline,
                    vertex_index: 1,
                })
            ));
            assert_eq!(
                scene.command_count(),
                0,
                "{join:?} {:?}",
                style.width_mode()
            );
        }
    }
}

#[test]
fn short_accepted_line_emits_vertices() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert!(scene.line(Vec2::ZERO, Vec2::new(0.005, 0.0), 2.0, Color::WHITE));

    let (vertices, _) = tessellate_for_test(&scene);

    assert!(!vertices.is_empty());
}

#[test]
fn maximum_radius_rounded_rect_stroke_has_no_collapsed_directions() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::splat(20.0)),
        100.0,
        ShapeStyle::stroked(3.0, Color::WHITE),
    );

    let (vertices, _) = tessellate_for_test(&scene);

    assert!(!vertices.is_empty());
    assert!(vertices.iter().all(|vertex| {
        Vec2::new(vertex.previous_direction[0], vertex.previous_direction[1]).length_squared()
            > f32::EPSILON
            && Vec2::new(vertex.next_direction[0], vertex.next_direction[1]).length_squared()
                > f32::EPSILON
    }));
}

#[test]
fn sub_epsilon_closed_strokes_remain_drawable_at_large_zoom() {
    let mut circle_scene = Scene::new(Color::BLACK).unwrap();
    circle_scene
        .try_circle(Vec2::ZERO, 0.000_1, ShapeStyle::stroked(2.0, Color::WHITE))
        .unwrap();
    let (circle_vertices, circle_batches) = tessellate_for_test(&circle_scene);
    assert_eq!(circle_vertices.len(), CIRCLE_SEGMENTS * 6);
    assert_eq!(circle_batches.len(), 1);

    let mut rounded_rect_scene = Scene::new(Color::BLACK).unwrap();
    rounded_rect_scene
        .try_rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::splat(0.000_1)),
            0.000_025,
            ShapeStyle::stroked(2.0, Color::WHITE),
        )
        .unwrap();
    let (rect_vertices, rect_batches) = tessellate_for_test(&rounded_rect_scene);
    assert_eq!(rect_vertices.len(), 4 * (CORNER_SEGMENTS + 1) * 6);
    assert_eq!(rect_batches.len(), 1);

    let camera = Camera2d::new(Vec2::ZERO, 100_000.0).unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    assert!(GeometryExtents::from_vertices(&circle_vertices).is_safe_for(uniform));
    assert!(GeometryExtents::from_vertices(&rect_vertices).is_safe_for(uniform));
}

#[test]
fn logical_stroke_rejects_backend_dependent_clip_classification() {
    let depth = f32::from_bits(0x627f_fa61);
    let from = Vec2::new(f32::from_bits(0xe1b5_00f9), f32::from_bits(0xe27f_fa60));
    let to = Vec2::new(f32::from_bits(0xe1b5_00e9), f32::from_bits(0xe27f_fa60));
    let viewport_extent = 2.0_f32.powi(45);
    let viewport = LogicalViewport::new(viewport_extent, viewport_extent).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(std::f32::consts::FRAC_PI_4, 1.0).unwrap());
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    let fixed = [from, to].map(|point| uniform.world_to_screen(point, depth));
    assert_eq!(fixed[0], Vec2::splat(viewport_extent * 0.5));
    assert!(fixed[1].x > viewport_extent && fixed[1].y == fixed[0].y);
    let fused_y = uniform.world_to_screen_y[2].mul_add(
        depth,
        uniform.world_to_screen_y[0] * from.x + uniform.world_to_screen_y[1] * from.y,
    ) + uniform.world_to_screen_y[3];
    assert!(fused_y + 10.0 < 0.0);

    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.try_set_depth(depth).unwrap();
    scene
        .try_styled_line(
            from,
            to,
            crate::StrokeStyle2d::logical(crate::LogicalPixels::new(20.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    assert!(!geometry_is_safe_for(
        GeometryExtents::from_vertices(&vertices),
        GeometryValidationSource::Tessellated(&vertices),
        uniform,
    ));
}

#[test]
fn large_center_circle_stroke_and_shadow_preserve_camera_relative_radius() {
    let center = Vec2::splat(1.0e20);
    let camera = Camera2d::new(center, 10.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();

    for style in [
        ShapeStyle::stroked(3.0, Color::WHITE),
        ShapeStyle::new(
            None,
            None,
            Some(crate::Shadow::new(
                LogicalScreenVector::new(4.0, -3.0),
                2.0,
                Color::WHITE,
            )),
        ),
    ] {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.try_circle(center, 1.0, style).unwrap();
        let (vertices, batches) = tessellate_for_test(&scene);
        assert!(!vertices.is_empty());
        assert_eq!(batches.len(), 1);
        assert!(vertices.iter().copied().all(Vertex::is_finite));
        assert!(
            vertices
                .iter()
                .all(|vertex| vertex.world_position == [center.x, center.y])
        );
        assert!(
            vertices
                .iter()
                .any(|vertex| vertex.world_offset != [0.0, 0.0])
        );

        let positions: Vec<_> = vertices
            .iter()
            .copied()
            .map(|vertex| vertex_screen_position(vertex, camera, viewport))
            .collect();
        let minimum = positions
            .iter()
            .copied()
            .fold(Vec2::splat(f32::INFINITY), |minimum, point| {
                Vec2::new(minimum.x.min(point.x), minimum.y.min(point.y))
            });
        let maximum = positions
            .iter()
            .copied()
            .fold(Vec2::splat(f32::NEG_INFINITY), |maximum, point| {
                Vec2::new(maximum.x.max(point.x), maximum.y.max(point.y))
            });
        assert!((maximum.x - minimum.x) > 19.0);
        assert!((maximum.y - minimum.y) > 19.0);
    }
}

#[test]
fn gpu_extrusion_contract_keeps_line_width_in_screen_pixels() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.line(
        Vec2::new(-20.0, -13.0),
        Vec2::new(40.0, 27.0),
        7.0,
        Color::WHITE,
    );
    let (vertices, _) = tessellate_for_test(&scene);
    let Ok(mut camera) = Camera2d::new(Vec2::new(5.0, 9.0), 6.0) else {
        panic!("test camera should be valid");
    };
    let Ok(projection) = crate::Projection2d::new(0.72, 1.0) else {
        panic!("test projection should be valid");
    };
    camera.set_projection(projection);
    if camera.set_rotation(-0.41).is_err() {
        panic!("test rotation should be valid");
    }
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };

    let positive = vertex_screen_position(vertices[0], camera, viewport);
    let negative = vertex_screen_position(vertices[5], camera, viewport);

    assert!(((positive - negative).length() - 7.0).abs() < 0.001);
}

#[test]
fn logical_stroke_direction_remains_normalized_for_extreme_finite_segments() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_styled_line(
            Vec2::ZERO,
            Vec2::new(f32::MAX, 0.0),
            crate::StrokeStyle2d::logical(crate::LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();

    let direction = Vec2::new(vertices[0].next_direction[0], vertices[0].next_direction[1]);
    let positive = vertex_screen_position(vertices[0], camera, viewport);
    let negative = vertex_screen_position(vertices[5], camera, viewport);

    assert!((direction.length() - 1.0).abs() < 0.000_001);
    assert!(((positive - negative).length() - 2.0).abs() < 0.001);
}

#[test]
fn logical_stroke_rejects_screen_overflow_hidden_by_tiny_clip_scale() {
    let maximum = MAX_PORTABLE_SHADER_VALUE;
    let world_screen = (f64::from(maximum * 0.75), f64::from(maximum * 0.75));

    // A sufficiently small clip multiplier could make the final value look
    // harmless, but WGSL must first add the projected center and extrusion in
    // f32 screen space. That intermediate is outside the shared envelope.
    assert!(shader_clip_interval_is_safe(
        world_screen.0,
        world_screen.1,
        2.0_f32.powi(-119),
        0.0
    ));
    assert_eq!(
        shader_stroke_screen_bounds(world_screen, 0.0, maximum * 0.5, 0.0, 1.0),
        None
    );
}
