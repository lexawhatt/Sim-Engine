//! Transforms contract regressions.

use super::*;

#[test]
fn shader_transform_validation_rejects_finite_overflow() {
    let maximum = f32::MAX;
    let mesh = Mesh3d::new(
        vec![
            Vec3::new(maximum, 0.0, 0.0).unwrap(),
            Vec3::new(maximum, 1.0, 0.0).unwrap(),
            Vec3::new(maximum, 0.0, 1.0).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        Vec3::new(maximum, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(4.0), 1.0, world(0.1), world(20.0)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        validate_shader_transform(
            &mesh,
            transform.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_ftz_sensitive_vertex_operands() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let mesh = Mesh3d::with_display_edges(
        vec![Vec3::ZERO, Vec3::new(largest_subnormal, 0.0, 0.0).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_shader_transform(
            &mesh,
            Transform3d::IDENTITY.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_subnormal_row_operands_uniformly() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let mesh = Mesh3d::with_display_edges(
        vec![Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        Vec3::new(largest_subnormal, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_shader_transform(
            &mesh,
            transform.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_subnormal_source_absorbed_by_translation() {
    let point = Vec3::new(f32::from_bits(1), 0.0, 0.0).unwrap();
    let mesh = Mesh3d::with_display_edges(
        vec![point, Vec3::new(0.5, 0.0, 0.0).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::new(1.0, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, -4.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(4.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let model_rows = transform.model_rows().unwrap();
    let model_x = shader_dot_range(
        model_rows[0],
        [point.x(), point.y(), point.z(), 1.0].map(ShaderValueRange::exact),
    );

    assert!(matches!(
        model_x,
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    ));
    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera.world_to_clip_rows().unwrap()),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn surface_triangle_ftz_high_scale_is_rejected() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let mesh = Mesh3d::new(
        vec![
            Vec3::ZERO,
            Vec3::new(largest_subnormal, 0.0, 0.0).unwrap(),
            Vec3::new(0.0, largest_subnormal, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        Vec3::new(f32::MAX, f32::MAX, 1.0).unwrap(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 4.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(8.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        validate_shader_transform(
            &mesh,
            transform.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_extreme_component_selection_after_interval_dot() {
    let rotation = Rotation3d::from_euler_xyz(
        0.0,
        std::f32::consts::FRAC_PI_4,
        -(1.0_f32 / 3.0_f32.sqrt()).asin(),
    )
    .unwrap();
    let unit_rows = Transform3d::new(Vec3::ZERO, rotation, Vec3::new(1.0, 1.0, 1.0).unwrap())
        .unwrap()
        .model_rows()
        .unwrap();
    let first_scale = 1.0 / unit_rows[0][0];
    let scale = Vec3::new(
        f32::from_bits(first_scale.to_bits() + 1),
        1.0 / unit_rows[0][1],
        1.0 / unit_rows[0][2],
    )
    .unwrap();
    let transform = Transform3d::new(Vec3::ZERO, rotation, scale).unwrap();
    let model_rows = transform.model_rows().unwrap();
    assert_eq!(model_rows[0], [1.0, 1.0, 1.0, 0.0]);

    let point = Vec3::new(
        f32::from_bits(0x7ea1_07f9),
        f32::from_bits(0x7ecc_fe95),
        f32::from_bits(0x7e91_f970),
    )
    .unwrap();
    let mesh = Mesh3d::with_display_edges(
        vec![point, point.checked_scale(0.5).unwrap()],
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

    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera.world_to_clip_rows().unwrap()),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_pairwise_dot_overflow_after_finite_cpu_transform() {
    let maximum = f32::MAX;
    let angle = 0.5_f32.acos();
    let model_x = -0.2 * maximum;
    let model_z = (0.51 * maximum) / angle.sin();
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(model_x, -1.0e30, model_z).unwrap(),
            Vec3::new(model_x, 1.0e30, model_z).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::new(0.51 * maximum, 0.0, 0.0).unwrap(),
        Rotation3d::from_axis_angle(Vec3::Y, angle).unwrap(),
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let model_center = Vec3::new(model_x, 0.0, model_z).unwrap();
    let world_center = transform.transform_point(model_center).unwrap();
    assert!(
        world_center.x().is_finite()
            && world_center.y().is_finite()
            && world_center.z().is_finite()
    );

    let camera = Camera3d::look_at(
        Vec3::new(
            world_center.x(),
            world_center.y(),
            world_center.z() - 1.0e32,
        )
        .unwrap(),
        world_center,
        Vec3::Y,
        Projection3d::perspective(2.8, 1.0, world(1.0e30), world(1.0e34)).unwrap(),
    )
    .unwrap();
    assert!(
        camera
            .project_world(world_center, LogicalViewport::new(64.0, 64.0).unwrap())
            .unwrap()
            .inside_view()
    );
    assert_eq!(
        validate_shader_transform(
            &mesh,
            transform.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_propagates_model_association_range_into_camera_dot() {
    let point = Vec3::new(-6.593_279_5e37, 2.460_645_7e38, 9.006_588e37).unwrap();
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
        Vec3::new(0.0, 0.0, -1.0e37).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0 / 7.0e7), 1.0, world(1.0), world(3.0e38)).unwrap(),
    )
    .unwrap();
    let model_rows = transform.model_rows().unwrap();
    let model_point = [point.x(), point.y(), point.z(), 1.0];

    assert_eq!(
        shader_dot(model_rows[0], model_point),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera.world_to_clip_rows().unwrap()),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn shader_transform_rejects_association_dependent_clip_classification() {
    let point = Vec3::new(1.732_050_8e20, -1.732_050_8e20, 3.808_820_2e12).unwrap();
    let mesh = Mesh3d::new(
        vec![
            point,
            Vec3::new(1.269_606_6e12, 1.269_606_6e12, 1.269_606_6e12).unwrap(),
            Vec3::new(6.590_675_4e19, -6.590_675_4e19, 3.202_013e12).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let field_of_view = f32::from_bits(std::f32::consts::PI.to_bits() - 1);
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
        Vec3::new(-1.0e20, -9.999_999e19, 2.0e20).unwrap(),
        Projection3d::perspective(field_of_view, 16.0, world(f32::MIN_POSITIVE), world(1.0e30))
            .unwrap(),
    )
    .unwrap();
    let model_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();
    let model_point = [point.x(), point.y(), point.z(), 1.0].map(ShaderValueRange::exact);
    let world = [
        shader_dot_range(model_rows[0], model_point).unwrap(),
        shader_dot_range(model_rows[1], model_point).unwrap(),
        shader_dot_range(model_rows[2], model_point).unwrap(),
        ShaderValueRange::exact(1.0),
    ];
    let clip_w = shader_dot_range(camera_rows[3], world).unwrap();

    assert!(clip_w.minimum <= 0.0 && clip_w.maximum > 1.0e12);
    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}

#[test]
fn surface_triangle_rejects_association_dependent_projected_collapse() {
    let scale = f32::from_bits(0x5d52_e40a);
    let coordinate = f32::from_bits(0x5d72_6836);
    let adjacent = f32::from_bits(0x5d72_6837);
    let product = f32::from_bits(0x7b47_b16b);
    let mesh = Mesh3d::new(
        vec![
            Vec3::new(coordinate, coordinate, 0.0).unwrap(),
            Vec3::new(adjacent, coordinate, 0.0).unwrap(),
            Vec3::new(coordinate, adjacent, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::new(-product, -product, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(scale, scale, 1.0).unwrap(),
    )
    .unwrap();
    let model_rows = transform.model_rows().unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0_f32.powi(103)), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();

    // The shader's legal separate multiply/add association collapses all
    // three vertices, while a fused association retains a visible
    // triangle. Point/plane classification alone cannot detect that.
    for vertex in mesh.vertices() {
        assert_eq!(
            shader_dot(model_rows[0], [vertex.x(), vertex.y(), vertex.z(), 1.0]),
            Ok(0.0)
        );
        assert_eq!(
            shader_dot(model_rows[1], [vertex.x(), vertex.y(), vertex.z(), 1.0]),
            Ok(0.0)
        );
    }
    assert_eq!(
        validate_shader_points(&mesh, model_rows, camera_rows),
        Ok(())
    );
    assert_eq!(
        validate_surface_triangle_topology(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::UnportableSurfaceTopology)
    );
    assert_eq!(
        validate_shader_transform(&mesh, model_rows, camera_rows),
        Err(Mesh3dRenderError::UnportableSurfaceTopology)
    );
}

#[test]
fn surface_triangle_frustum_branches_are_explicit() {
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let model_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let camera_rows = camera.world_to_clip_rows().unwrap();
    let mesh = |vertices| Mesh3d::new(vertices, vec![0, 1, 2]).unwrap();

    let inside = mesh(vec![
        Vec3::new(-0.25, -0.25, 0.0).unwrap(),
        Vec3::new(0.25, -0.25, 0.0).unwrap(),
        Vec3::new(0.0, 0.25, 0.0).unwrap(),
    ]);
    assert_eq!(
        validate_surface_triangle_topology(&inside, model_rows, camera_rows),
        Ok(())
    );

    let crossing = mesh(vec![
        Vec3::new(-0.25, -0.25, 0.0).unwrap(),
        Vec3::new(1.5, -0.25, 0.0).unwrap(),
        Vec3::new(0.0, 0.25, 0.0).unwrap(),
    ]);
    assert_eq!(
        validate_shader_points(&crossing, model_rows, camera_rows),
        Ok(())
    );
    assert_eq!(
        validate_surface_triangle_topology(&crossing, model_rows, camera_rows),
        Ok(())
    );

    let outside = mesh(vec![
        Vec3::new(1.5, -0.25, 0.0).unwrap(),
        Vec3::new(2.0, -0.25, 0.0).unwrap(),
        Vec3::new(1.75, 0.25, 0.0).unwrap(),
    ]);
    assert_eq!(outside.triangle_count(), 1);
    assert_eq!(
        validate_surface_triangle_topology(&outside, model_rows, camera_rows),
        Ok(())
    );
}

#[test]
fn shader_transform_rejects_extreme_correlated_sparse_vertices() {
    let extent = 0.9 * f32::MAX;
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(extent, 0.0, 0.0).unwrap(),
            Vec3::new(0.0, extent, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::from_axis_angle(Vec3::Z, -std::f32::consts::FRAC_PI_4).unwrap(),
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let first = transform.transform_point(mesh.vertices()[0]).unwrap();
    let second = transform.transform_point(mesh.vertices()[1]).unwrap();
    for vertex in [first, second] {
        assert!(vertex.x().is_finite() && vertex.y().is_finite() && vertex.z().is_finite());
    }
    let target = Vec3::new(first.x(), 0.0, 0.0).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(target.x(), 0.0, 2.0e38).unwrap(),
        target,
        Vec3::Y,
        Projection3d::perspective(2.8, 1.0, world(1.0), world(3.0e38)).unwrap(),
    )
    .unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    assert!(camera.project_world(first, viewport).unwrap().inside_view());
    assert!(
        camera
            .project_world(second, viewport)
            .unwrap()
            .inside_view()
    );
    assert_eq!(
        validate_shader_transform(
            &mesh,
            transform.model_rows().unwrap(),
            camera.world_to_clip_rows().unwrap(),
        ),
        Err(Mesh3dRenderError::InvalidGeometryTransform)
    );
}
