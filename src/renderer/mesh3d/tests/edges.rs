//! Edges contract regressions.

use super::*;

#[test]
fn edge_projection_rejects_post_width_shader_overflow() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, 0.0, 0.0).unwrap(),
            Vec3::new(0.5, 0.0, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1_048_576.0)).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            style,
            [1.0, 1.0, 5.0e32, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_projection_rejects_logical_distance_overflow() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, 0.0, 0.0).unwrap(),
            Vec3::new(0.5, 0.0, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1_048_576.0)).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            style,
            [1_024.0, 1_024.0, f32::MIN_POSITIVE, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_projection_rejects_hidden_dash_phase_overflow() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, 0.0, 0.0).unwrap(),
            Vec3::new(0.5, 0.0, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let smallest = logical(f32::MIN_POSITIVE);
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0))
        .unwrap()
        .with_hidden(Color::WHITE, logical(1.0), smallest, smallest)
        .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            style,
            [1_024.0, 1_024.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_projection_bounds_dash_across_legal_shader_associations() {
    let point = Vec3::new(-6.593_279e17, 2.460_645_7e18, 9.006_587e17).unwrap();
    let mesh = Mesh3d::with_display_edges(
        vec![point, point.checked_scale(0.5).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let axis = Vec3::new(1.0, 1.0, 1.0).unwrap().normalized().unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::from_axis_angle(axis, std::f32::consts::FRAC_PI_2).unwrap(),
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Vec3::Y,
        Projection3d::orthographic(world(5.0e10), 1.0, world(1.0), world(3.0e18)).unwrap(),
    )
    .unwrap();
    let tiny = logical(1.5e-38);
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0))
        .unwrap()
        .with_hidden(Color::WHITE, logical(1.0), tiny, tiny)
        .unwrap();

    let model_rows = transform.model_rows().unwrap();
    let model_point = [point.x(), point.y(), point.z(), 1.0];
    assert_eq!(shader_dot(model_rows[0], model_point), Ok(0.0));
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model_rows,
            camera.world_to_clip_rows().unwrap(),
            style,
            [100.0, 100.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_projection_rejects_association_dependent_collapse() {
    let point = Vec3::new(-6.593_279e17, 2.460_645_7e18, 9.006_587e17).unwrap();
    let mesh = Mesh3d::with_display_edges(
        vec![point, point.checked_scale(0.5).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let axis = Vec3::new(1.0, 1.0, 1.0).unwrap().normalized().unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::from_axis_angle(axis, std::f32::consts::FRAC_PI_2).unwrap(),
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Vec3::Y,
        Projection3d::orthographic(world(1.0e13), 1.0, world(1.0), world(3.0e18)).unwrap(),
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, logical(5.0)).unwrap();
    let model_rows = transform.model_rows().unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();

    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Ok(())
    );
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model_rows,
            camera_rows,
            style,
            [1_000.0, 1_000.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_projection_rejects_extreme_homogeneous_sources_before_clipping() {
    let outside_x = f32::from_bits(1.0_f32.to_bits() + 1);
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-outside_x, f32::MAX, 1.0).unwrap(),
            Vec3::new(-outside_x, -f32::MAX, 1.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(1.0), world(10.0)).unwrap(),
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap();
    let model_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();

    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model_rows,
            camera_rows,
            style,
            [128.0, 128.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn edge_projection_rejects_subnormal_component_scaled_clip_boundary() {
    let near = f32::from_bits(0x327e_4a5c) * 0.5;
    let clip_w = f32::from_bits(0x327e_4a5c);
    let clip_x = f32::from_bits(clip_w.to_bits() + 1);
    let clip_y = f32::from_bits(0x41db_c900);
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(near), world(1.0))
            .unwrap(),
    )
    .unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();
    let start = Vec3::new(
        clip_x / camera_rows[0][0],
        clip_y / camera_rows[1][1],
        clip_w,
    )
    .unwrap();
    let end = Vec3::new(
        clip_x / camera_rows[0][0],
        -clip_y / camera_rows[1][1],
        clip_w,
    )
    .unwrap();
    let mesh = Mesh3d::with_display_edges(
        vec![start, end],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let model_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let start_clip = shader_clip_point_ranges(start, model_rows, camera_rows).unwrap();
    assert_eq!(start_clip[0].fixed, clip_x);
    assert_eq!(start_clip[1].fixed, clip_y);
    assert_eq!(start_clip[3].fixed, clip_w);

    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model_rows,
            camera_rows,
            WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap(),
            [128.0, 128.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn edge_projection_rejects_ftz_sensitive_pair_scale_inputs() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(0.0, largest_subnormal, 1.0).unwrap(),
            Vec3::new(0.5, largest_subnormal, 1.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::new(0.0, f32::from_bits(1.0_f32.to_bits() + 1), 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, f32::MAX, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(1.0), world(10.0)).unwrap(),
    )
    .unwrap();
    let model_rows = transform.model_rows().unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();

    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model_rows,
            camera_rows,
            WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap(),
            [128.0, 128.0, 1.0, 0.0],
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn edge_projection_rejects_values_outside_portable_width_envelope() {
    let maximum_width = logical(1_048_576.0);
    let requested_scale = f32::MAX / maximum_width.get();
    let logical_viewport =
        LogicalViewport::new(1.0 / requested_scale, 8_192.0 / requested_scale).unwrap();
    let pixels_per_logical = target_pixels_per_logical(1, 8_192, logical_viewport).unwrap();
    let scaled_width = maximum_width.get() * pixels_per_logical.get();
    assert!(scaled_width.is_finite());

    let view_depth = f32::from_bits(0x7f4d_1ce4);
    let aspect = 1.0 / 8_192.0;
    let vertical_fov = 1.0_f32;
    let horizontal_scale = (vertical_fov * 0.5).tan() * aspect;
    let half_x = view_depth * 0.5 * horizontal_scale;
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-half_x, 0.0, -view_depth).unwrap(),
            Vec3::new(half_x, 0.0, -view_depth).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Vec3::Y,
        Projection3d::perspective(vertical_fov, aspect, world(1.0), world(f32::MAX)).unwrap(),
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, maximum_width).unwrap();

    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            style,
            [1.0, 8_192.0, pixels_per_logical.get(), 0.0],
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection)
    );
}

#[test]
fn edge_crossing_near_plane_is_clipped_before_wgsl_division() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(0.0, 0.0, -0.05).unwrap(),
            Vec3::new(0.5, 0.0, -1.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Vec3::Y,
        Projection3d::perspective(1.0, 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap();
    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            style,
            [100.0, 100.0, 1.0, 0.0],
        ),
        Ok(())
    );
}

#[test]
fn fully_clipped_edges_do_not_reject_the_frame() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(0.0, 0.0, -0.01).unwrap(),
            Vec3::new(0.01, 0.0, -0.05).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Vec3::Y,
        Projection3d::perspective(1.0, 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        validate_edge_projection(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
            WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap(),
            [100.0, 100.0, 1.0, 0.0],
        ),
        Ok(())
    );
}

#[test]
fn homogeneous_edge_clipping_covers_near_far_and_camera_half_spaces() {
    let near = clip_edge_to_frustum([0.0, 0.0, -0.5, 1.0], [0.0, 0.0, 0.5, 1.0])
        .unwrap()
        .unwrap();
    assert!(near[0][2].abs() <= f32::EPSILON);
    assert_eq!(near[1], [0.0, 0.0, 0.5, 1.0]);

    let far = clip_edge_to_frustum([0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 2.0, 1.0])
        .unwrap()
        .unwrap();
    assert!((far[1][2] - far[1][3]).abs() <= f32::EPSILON);

    assert_eq!(
        clip_edge_to_frustum([0.0, 0.0, -2.0, -1.0], [0.0, 0.0, -1.0, -0.5]),
        Ok(None)
    );
    assert_eq!(
        clip_edge_to_frustum([-3.0, 0.0, 0.5, 1.0], [-2.0, 0.0, 0.5, 1.0]),
        Ok(None)
    );
}
