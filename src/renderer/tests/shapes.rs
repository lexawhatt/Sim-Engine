use super::*;

#[test]
fn filled_circle_tessellates_to_triangle_fan() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 12.0, ShapeStyle::filled(Color::WHITE));

    let (vertices, draw_batches) = tessellate_for_test(&scene);

    assert_eq!(vertices.len(), CIRCLE_SEGMENTS * 3);
    assert_eq!(draw_batches.len(), 1);
}

#[test]
fn overflowing_finite_geometry_is_rejected_by_scene() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert!(!scene.circle(
        Vec2::splat(f32::MAX),
        f32::MAX,
        ShapeStyle::filled(Color::WHITE),
    ));
    assert_eq!(scene.command_count(), 0);
}

#[test]
fn invalid_primitives_do_not_emit_vertices() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 0.0, ShapeStyle::filled(Color::WHITE));
    scene.line(Vec2::ZERO, Vec2::ZERO, 2.0, Color::WHITE);
    scene.line(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), 0.0, Color::WHITE);

    let (vertices, draw_batches) = tessellate_for_test(&scene);

    assert!(vertices.is_empty());
    assert!(draw_batches.is_empty());
}

#[test]
fn gradient_fill_reaches_vertex_colors() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::new(2.0, 2.0)),
        0.0,
        ShapeStyle::filled_with(Fill::LinearGradient(crate::LinearGradient::new(
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Color::BLACK,
            Color::WHITE,
        ))),
    );

    let (vertices, _) = tessellate_for_test(&scene);

    assert!(vertices.iter().any(|vertex| vertex.color[0] < 0.01));
    assert!(vertices.iter().any(|vertex| vertex.color[0] > 0.99));
}

#[test]
fn flat_rectangle_emits_every_fan_sector() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::new(20.0, 10.0)),
        0.0,
        ShapeStyle::filled(Color::WHITE),
    );

    let (vertices, _) = tessellate_for_test(&scene);

    assert_eq!(vertices.len(), 12);
}

#[test]
fn projected_circle_follows_camera_tilt() {
    let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 2.0) else {
        panic!("test camera should be valid");
    };
    let Ok(projection) = crate::Projection2d::new(0.8, 1.0) else {
        panic!("test projection should be valid");
    };
    camera.set_projection(projection);
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();

    tessellate_scene(&scene, &mut vertices, &mut draw_batches)
        .expect("validated circle should tessellate");

    let positions: Vec<_> = vertices
        .iter()
        .copied()
        .map(|vertex| vertex_screen_position(vertex, camera, viewport))
        .collect();
    let width = positions
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        - positions
            .iter()
            .map(|point| point.x)
            .fold(f32::INFINITY, f32::min);
    let height = positions
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - positions
            .iter()
            .map(|point| point.y)
            .fold(f32::INFINITY, f32::min);

    assert!((width - 40.0).abs() < 0.01);
    assert!((height - 40.0 * 0.8_f32.cos()).abs() < 0.05);
}

#[test]
fn scene_budget_estimate_bounds_actual_tessellation_and_upload() {
    let budget = crate::SceneBudget::new(4, 4, 2_000, 16_000, 16_000, 120_000, 4);
    let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
    scene
        .try_circle(
            Vec2::ZERO,
            8.0,
            ShapeStyle::fill_stroke(Color::WHITE, 2.0, Color::BLACK),
        )
        .unwrap();
    scene
        .try_rect(
            Rect::from_center_size(Vec2::new(20.0, 0.0), Vec2::new(8.0, 12.0)),
            2.0,
            ShapeStyle::fill_stroke(Color::WHITE, 1.0, Color::BLACK),
        )
        .unwrap();
    scene
        .try_polyline(
            vec![
                Vec2::ZERO,
                Vec2::X,
                Vec2::new(1.5, 0.25),
                Vec2::new(2.0, 1.0),
            ],
            1.0,
            Color::WHITE,
        )
        .unwrap();

    let mut vertices = Vec::new();
    let mut batches = Vec::new();
    let stats = tessellate_scene(&scene, &mut vertices, &mut batches).unwrap();
    let estimate = scene.statistics();

    assert!(stats.vertex_count() <= estimate.estimated_tessellated_vertices());
    assert!(stats.upload_bytes() <= estimate.estimated_upload_bytes());
    assert!(stats.draw_batch_count() <= estimate.estimated_draw_batches());
    assert_eq!(stats.vertex_count(), vertices.len());
    assert_eq!(stats.draw_batch_count(), batches.len());
    assert_eq!(
        stats.upload_bytes(),
        vertices.len() * std::mem::size_of::<Vertex>()
    );
    assert_eq!(stats.command_counts().circles(), 1);
    assert_eq!(stats.command_counts().rectangles(), 1);
    assert_eq!(stats.command_counts().polylines(), 1);
    assert_eq!(stats.rendered_counts(), stats.command_counts());
    assert_eq!(stats.dropped_counts().total(), 0);
}

#[test]
fn appended_streaming_tessellation_reports_local_work_and_global_ranges() {
    let mut first = Scene::new(Color::BLACK).unwrap();
    first
        .try_rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::splat(8.0)),
            1.5,
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    let mut second = Scene::new(Color::BLACK).unwrap();
    second
        .try_rect(
            Rect::from_center_size(Vec2::new(20.0, 0.0), Vec2::splat(8.0)),
            1.5,
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();

    let mut vertices = Vec::new();
    let mut first_batches = Vec::new();
    let first_stats = tessellate_scene(&first, &mut vertices, &mut first_batches).unwrap();
    let first_vertex_count = vertices.len();
    let mut second_batches = Vec::new();
    let second_stats = tessellate_scene(&second, &mut vertices, &mut second_batches).unwrap();

    assert_eq!(first_stats.vertex_count(), first_vertex_count);
    assert_eq!(
        second_stats.vertex_count(),
        vertices.len() - first_vertex_count
    );
    assert_eq!(second_stats.vertex_count(), first_stats.vertex_count());
    assert_eq!(
        second_stats.upload_bytes(),
        second_stats.vertex_count() * std::mem::size_of::<Vertex>()
    );
    assert_eq!(second_stats.draw_batch_count(), 1);
    assert_eq!(second_batches.len(), 1);
    assert_eq!(
        second_batches[0].vertex_range,
        first_vertex_count as u32..vertices.len() as u32
    );
}

#[test]
fn scene_depth_reaches_tessellated_vertices() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert!(
        scene
            .with_depth(12.5, |scene| {
                scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE));
            })
            .is_ok()
    );

    let (vertices, _) = tessellate_for_test(&scene);

    assert!(!vertices.is_empty());
    assert!(vertices.iter().all(|vertex| vertex.depth == 12.5));
}

#[test]
fn cached_circle_samples_close_exactly_at_large_world_scale() {
    let samples = super::tessellation::unit_circle_points();
    assert_eq!(samples[0], Vec2::X);
    assert_eq!(samples[CIRCLE_SEGMENTS / 4], Vec2::Y);
    assert_eq!(samples[CIRCLE_SEGMENTS / 2], Vec2::new(-1.0, 0.0));
    assert_eq!(samples[CIRCLE_SEGMENTS * 3 / 4], Vec2::new(0.0, -1.0));
    assert_eq!(samples[CIRCLE_SEGMENTS], Vec2::X);

    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 20_000.0, ShapeStyle::stroked(3.0, Color::WHITE));

    let (vertices, _) = tessellate_for_test(&scene);

    assert_eq!(vertices.len(), CIRCLE_SEGMENTS * 6);
    assert!(vertices.iter().copied().all(Vertex::is_finite));
}

#[test]
fn tessellated_fill_preserves_local_offsets_at_large_world_anchor() {
    let anchor = 2.0_f32.powi(30);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(Vec2::splat(anchor), 1.0, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let triangle = [vertices[0], vertices[1], vertices[2]];
    let camera = Camera2d::new(Vec2::splat(anchor), 100.0).unwrap();
    let viewport = LogicalViewport::new(1_000.0, 1_000.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    let fixed = triangle.map(|vertex| {
        let relative = Vec2::new(
            (vertex.world_position[0] - uniform.camera_center[0]) + vertex.world_offset[0],
            (vertex.world_position[1] - uniform.camera_center[1]) + vertex.world_offset[1],
        );
        Vec2::new(
            uniform.world_to_screen_x[0] * relative.x
                + uniform.world_to_screen_x[1] * relative.y
                + uniform.world_to_screen_x[2] * vertex.depth
                + uniform.world_to_screen_x[3]
                + vertex.screen_offset[0],
            uniform.world_to_screen_y[0] * relative.x
                + uniform.world_to_screen_y[1] * relative.y
                + uniform.world_to_screen_y[2] * vertex.depth
                + uniform.world_to_screen_y[3]
                + vertex.screen_offset[1],
        )
    });
    let fixed_area = f64::from(fixed[1].x - fixed[0].x) * f64::from(fixed[2].y - fixed[0].y)
        - f64::from(fixed[1].y - fixed[0].y) * f64::from(fixed[2].x - fixed[0].x);
    assert!(fixed_area.abs() > 500.0);

    let reassociated = triangle.map(|vertex| {
        let relative = Vec2::new(
            vertex.world_position[0] + (vertex.world_offset[0] - uniform.camera_center[0]),
            vertex.world_position[1] + (vertex.world_offset[1] - uniform.camera_center[1]),
        );
        Vec2::new(
            uniform.world_to_screen_x[0] * relative.x
                + uniform.world_to_screen_x[1] * relative.y
                + uniform.world_to_screen_x[2] * vertex.depth
                + uniform.world_to_screen_x[3]
                + vertex.screen_offset[0],
            uniform.world_to_screen_y[0] * relative.x
                + uniform.world_to_screen_y[1] * relative.y
                + uniform.world_to_screen_y[2] * vertex.depth
                + uniform.world_to_screen_y[3]
                + vertex.screen_offset[1],
        )
    });
    assert_eq!(reassociated, [Vec2::splat(500.0); 3]);
    let extents = GeometryExtents::from_vertices(&vertices);
    assert!(geometry_sources_are_portable(
        GeometryValidationSource::Tessellated(&vertices)
    ));
    assert!(geometry_vertex_centers_are_portable(
        GeometryValidationSource::Tessellated(&vertices),
        uniform,
    ));
    for (index, triangle) in vertices.chunks_exact(3).enumerate() {
        let clip = [
            tessellated_vertex_clip_ranges(triangle[0], uniform).unwrap(),
            tessellated_vertex_clip_ranges(triangle[1], uniform).unwrap(),
            tessellated_vertex_clip_ranges(triangle[2], uniform).unwrap(),
        ];
        assert!(
            clip_triangle_is_wholly_outside(clip)
                || clip_triangle_has_stable_signed_area(clip, f64::from(f32::MIN_POSITIVE)),
            "triangle {index}: {clip:?}"
        );
    }
    assert!(tessellated_triangle_topology_is_portable(
        &vertices, uniform
    ));
    assert!(extents.is_safe_for(uniform));
    assert!(geometry_is_safe_for(
        extents,
        GeometryValidationSource::Tessellated(&vertices),
        uniform,
    ));
}

#[test]
fn sub_ulp_rounded_corner_keeps_its_local_arc_offsets() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::splat(1.0)),
            0.000_000_01,
            ShapeStyle::stroked(2.0, Color::WHITE),
        )
        .unwrap();

    let (vertices, batches) = tessellate_for_test(&scene);
    assert_eq!(vertices.len(), 4 * (CORNER_SEGMENTS + 1) * 6);
    assert_eq!(batches.len(), 1);
    assert!(
        vertices
            .iter()
            .any(|vertex| vertex.world_offset != [0.0; 2])
    );
    let camera = Camera2d::new(Vec2::ZERO, 10_000_000_000.0).unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    assert!(
        GeometryExtents::from_vertices(&vertices)
            .is_safe_for(CameraUniform::new(camera, viewport).unwrap())
    );
}

#[test]
fn large_center_circle_fill_preserves_camera_relative_radius() {
    let center = Vec2::new(1.0e20, 0.0);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(center, 1.0, ShapeStyle::filled(Color::WHITE))
        .unwrap();

    let (vertices, batches) = tessellate_for_test(&scene);
    assert_eq!(vertices.len(), CIRCLE_SEGMENTS * 3);
    assert_eq!(batches.len(), 1);
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

    let camera = Camera2d::new(center, 10.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    assert!(GeometryExtents::from_vertices(&vertices).is_safe_for(uniform));
    let triangle: Vec<_> = vertices[..3]
        .iter()
        .copied()
        .map(|vertex| vertex_screen_position(vertex, camera, viewport))
        .collect();
    let area_twice = (triangle[1] - triangle[0]).x.mul_add(
        (triangle[2] - triangle[0]).y,
        -(triangle[1] - triangle[0]).y * (triangle[2] - triangle[0]).x,
    );
    assert!(area_twice.is_finite() && area_twice.abs() > 0.1);
}

#[test]
fn large_center_circle_preserves_radial_gradient_offsets() {
    let center = Vec2::splat(1.0e20);
    let gradient = crate::RadialGradient::new(center, 0.0, 1.0, Color::BLACK, Color::WHITE);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            center,
            1.0,
            ShapeStyle::filled_with(Fill::RadialGradient(gradient)),
        )
        .unwrap();

    let (vertices, _) = tessellate_for_test(&scene);
    assert_eq!(vertices[0].color, Color::BLACK.to_array());
    assert_eq!(vertices[1].color, Color::WHITE.to_array());
    assert_eq!(vertices[2].color, Color::WHITE.to_array());
}

#[test]
fn cached_quarter_circle_samples_keep_large_rounded_rect_tangents_forward() {
    let samples = super::tessellation::unit_quarter_circle_points();
    assert_eq!(samples[0], Vec2::X);
    assert_eq!(samples[CORNER_SEGMENTS], Vec2::Y);

    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::splat(20_000.0)),
        10_000.0,
        ShapeStyle::stroked(3.0, Color::WHITE),
    );

    let (vertices, _) = tessellate_for_test(&scene);
    let directions: Vec<_> = vertices
        .chunks_exact(6)
        .take(CORNER_SEGMENTS * 4)
        .map(|quad| Vec2::new(quad[0].next_direction[0], quad[0].next_direction[1]))
        .collect();

    assert_eq!(directions.len(), CORNER_SEGMENTS * 4);
    assert!(directions.iter().all(|direction| direction.is_finite()));
    assert!(directions.windows(2).all(|pair| pair[0].dot(pair[1]) > 0.0));
    assert!(directions[directions.len() - 1].dot(directions[0]) > 0.0);
}
