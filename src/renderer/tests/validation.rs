use super::*;

#[test]
fn dynamic_mesh_vertices_require_complete_finite_triangles() {
    let vertex = DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::WHITE).unwrap();
    assert!(matches!(
        dynamic_vertices_to_gpu(&[vertex, vertex]),
        Err(DynamicMeshError::InvalidVertexCount)
    ));
    assert_eq!(
        DynamicVertex2d::new(Vec2::new(f32::NAN, 0.0), 0.0, Color::WHITE),
        Err(DynamicMeshError::InvalidVertex)
    );
    assert_eq!(
        DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::rgb(1.01, 0.0, 0.0)),
        Err(DynamicMeshError::InvalidVertex)
    );
    let subnormal =
        DynamicVertex2d::new(Vec2::new(f32::from_bits(1), 0.0), 0.0, Color::WHITE).unwrap();
    assert_eq!(
        dynamic_vertices_to_gpu(&[subnormal; 3]),
        Err(DynamicMeshError::InvalidVertex)
    );

    let vertices = dynamic_vertices_to_gpu(&[vertex; 3]).unwrap();
    assert_eq!(vertices.len(), 3);
    assert_eq!(dynamic_vertex_capacity(3), Some(4));
    assert_eq!(dynamic_vertex_capacity(usize::MAX), None);
    assert_eq!(buffer_allocation_bytes::<Vertex>(usize::MAX), None);

    let triangle_bytes = 3 * std::mem::size_of::<DynamicGpu>();
    assert_eq!(
        DynamicMeshBudget::new(2, triangle_bytes, triangle_bytes),
        Err(DynamicMeshError::InvalidBudget)
    );
    let budget = DynamicMeshBudget::new(3, triangle_bytes, triangle_bytes).unwrap();
    assert_eq!(validate_dynamic_mesh_budget(budget, 3), Ok(()));
    assert_eq!(
        validate_dynamic_mesh_budget(budget, 6),
        Err(DynamicMeshError::BudgetExceeded {
            resource: DynamicMeshBudgetResource::Vertices,
            limit: 3,
            actual: 6,
        })
    );
}

#[test]
fn retained_geometry_validation_cache_is_exact_bounded_and_clearable() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::splat(8.0)),
            0.0,
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let extents = GeometryExtents::from_vertices(&vertices);
    let source = GeometryValidationSource::Tessellated(&vertices);
    let mut cache = GeometryValidationCache::default();

    for width in 128..128 + GEOMETRY_VALIDATION_CACHE_CAPACITY + 1 {
        let uniform = CameraUniform::new(
            Camera2d::default(),
            LogicalViewport::new(width as f32, 128.0).unwrap(),
        )
        .unwrap();
        assert!(cache.validate(extents, source, uniform));
        assert!(cache.validate(extents, source, uniform));
    }
    let state = cache.state.lock().unwrap();
    assert_eq!(
        state.entries.iter().flatten().count(),
        GEOMETRY_VALIDATION_CACHE_CAPACITY
    );
    drop(state);

    cache.clear();
    assert!(
        cache
            .state
            .lock()
            .unwrap()
            .entries
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn camera_uniform_matches_public_camera_projection() {
    let Ok(mut camera) = Camera2d::new(Vec2::new(17.0, -23.0), 2.75) else {
        panic!("test camera should be valid");
    };
    let Ok(projection) = crate::Projection2d::new(0.63, 4.0) else {
        panic!("test projection should be valid");
    };
    camera.set_projection(projection);
    if camera.set_rotation(0.31).is_err() {
        panic!("test rotation should be valid");
    }
    let Ok(viewport) = LogicalViewport::new(1_137.0, 683.0) else {
        panic!("test viewport should be valid");
    };
    let Some(uniform) = CameraUniform::new(camera, viewport) else {
        panic!("test camera uniform should be finite");
    };

    for (world, depth) in [
        (Vec2::ZERO, 0.0),
        (camera.center(), 0.0),
        (Vec2::new(-81.5, 44.25), 7.5),
        (Vec2::new(319.0, -127.0), -13.25),
    ] {
        let expected = camera
            .projected_world_to_screen(world, depth, viewport)
            .unwrap()
            .to_vec2();
        let actual = uniform.world_to_screen(world, depth);
        assert!((actual.x - expected.x).abs() < 0.001);
        assert!((actual.y - expected.y).abs() < 0.001);
    }
}

#[test]
fn ftz_sensitive_world_geometry_is_rejected_before_gpu_submission() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    let mut circle_scene = Scene::new(Color::BLACK).unwrap();
    circle_scene
        .try_circle(
            Vec2::ZERO,
            largest_subnormal,
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    let (circle_vertices, _) = tessellate_for_test(&circle_scene);
    let circle_extents = GeometryExtents::from_vertices(&circle_vertices);
    assert!(!geometry_is_safe_for(
        circle_extents,
        GeometryValidationSource::Tessellated(&circle_vertices),
        uniform,
    ));

    let mut line_scene = Scene::new(Color::BLACK).unwrap();
    line_scene
        .try_styled_line(
            Vec2::ZERO,
            Vec2::new(largest_subnormal, 0.0),
            crate::StrokeStyle2d::logical(crate::LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();
    let (line_vertices, _) = tessellate_for_test(&line_scene);
    let line_extents = GeometryExtents::from_vertices(&line_vertices);
    assert!(!geometry_is_safe_for(
        line_extents,
        GeometryValidationSource::Tessellated(&line_vertices),
        uniform,
    ));

    let particle = ParticleGpu {
        world_position: [largest_subnormal, 0.0],
        depth: 0.0,
        radius: 1.0,
        color: Color::WHITE.to_array(),
    };
    assert!(!particle.is_safe_for(uniform, viewport));
    assert!(
        particle
            .validated_viewport_intersection(uniform, viewport)
            .is_none()
    );
}

#[test]
fn portability_envelope_rejects_subnormal_camera_operands() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let empty_camera = Camera2d::new(Vec2::splat(largest_subnormal), 1.0).unwrap();
    assert!(CameraUniform::new(empty_camera, viewport).is_none());

    let mut tiny_projection = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    tiny_projection.set_projection(crate::Projection2d::new(0.5, largest_subnormal).unwrap());
    assert!(CameraUniform::new(tiny_projection, viewport).is_none());
}

#[test]
fn dynamic_geometry_rejects_subnormal_position_even_when_translation_absorbs_it() {
    let vertices = [
        DynamicGpu {
            world_position: [f32::from_bits(1), 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [0.5, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [0.0, 0.5],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
    ];
    let uniform = CameraUniform::new(
        Camera2d::new(Vec2::ZERO, 1.0).unwrap(),
        LogicalViewport::new(100.0, 100.0).unwrap(),
    )
    .unwrap();
    let extents = GeometryExtents::from_dynamic_vertices(&vertices);

    // Extents alone intentionally describe arithmetic range, while the exact
    // retained-source scan below rejects the backend-dependent subnormal.
    assert!(extents.is_safe_for(uniform));
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&vertices),
        uniform,
    ));
}

#[test]
fn dynamic_triangle_rejects_backend_dependent_ftz_collapse_at_high_zoom() {
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    let vertices = [
        DynamicGpu {
            world_position: [0.0, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [largest_subnormal, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [0.0, largest_subnormal],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
    ];
    let camera = Camera2d::new(Vec2::ZERO, 2.0_f32.powi(119)).unwrap();
    let uniform = CameraUniform::new(camera, LogicalViewport::new(64.0, 64.0).unwrap()).unwrap();
    let extents = GeometryExtents::from_dynamic_vertices(&vertices);

    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&vertices),
        uniform,
    ));
}

#[test]
fn dynamic_triangle_rejects_association_dependent_projected_collapse() {
    let vertices = [
        DynamicGpu {
            world_position: [f32::from_bits(0xe1b5_00f9), f32::from_bits(0xe27f_fa60)],
            depth: f32::from_bits(0x627f_fa61),
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [f32::from_bits(0xe1b5_01e8), f32::from_bits(0xe27f_fbb3)],
            depth: f32::from_bits(0x627f_fbb3),
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [f32::from_bits(0xe1b5_0e07), f32::from_bits(0xe280_066b)],
            depth: f32::from_bits(0x6280_066b),
            color: Color::WHITE.to_array(),
        },
    ];
    let viewport_extent = f32::from_bits(0x5680_0000);
    let viewport = LogicalViewport::new(viewport_extent, viewport_extent).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(std::f32::consts::FRAC_PI_4, 1.0).unwrap());
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let half_viewport = viewport_extent * 0.5;

    let fixed = vertices.map(|vertex| {
        uniform.world_to_screen(
            Vec2::new(vertex.world_position[0], vertex.world_position[1]),
            vertex.depth,
        )
    });
    assert_eq!(fixed, [Vec2::splat(half_viewport); 3]);

    let fused = vertices.map(|vertex| {
        Vec2::new(
            uniform.world_to_screen_x[2].mul_add(vertex.depth, vertex.world_position[0])
                + uniform.world_to_screen_x[3],
            uniform.world_to_screen_y[2].mul_add(
                vertex.depth,
                uniform.world_to_screen_y[1] * vertex.world_position[1],
            ) + uniform.world_to_screen_y[3],
        )
    });
    let fused_area = f64::from(fused[1].x - fused[0].x) * f64::from(fused[2].y - fused[0].y)
        - f64::from(fused[1].y - fused[0].y) * f64::from(fused[2].x - fused[0].x);
    assert!(fused_area.is_finite() && fused_area != 0.0);

    let extents = GeometryExtents::from_dynamic_vertices(&vertices);
    assert!(geometry_sources_are_portable(
        GeometryValidationSource::Dynamic(&vertices)
    ));
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&vertices),
        uniform,
    ));
}

#[test]
fn dynamic_triangle_frustum_branches_are_explicit() {
    let uniform = CameraUniform::new(
        Camera2d::default(),
        LogicalViewport::new(100.0, 100.0).unwrap(),
    )
    .unwrap();
    let vertex = |x, y| DynamicGpu {
        world_position: [x, y],
        depth: 0.0,
        color: Color::WHITE.to_array(),
    };
    let inside = [vertex(-10.0, -10.0), vertex(10.0, -10.0), vertex(0.0, 10.0)];
    assert!(dynamic_triangle_topology_is_portable(&inside, uniform));

    let crossing = [vertex(-10.0, -10.0), vertex(60.0, -10.0), vertex(0.0, 10.0)];
    assert!(!dynamic_triangle_topology_is_portable(&crossing, uniform));

    let outside = [vertex(60.0, -10.0), vertex(80.0, -10.0), vertex(70.0, 10.0)];
    assert!(dynamic_triangle_topology_is_portable(&outside, uniform));
}

#[test]
fn prepared_scene_preflight_rejects_nonportable_sources_without_tessellation() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for index in 0..256 {
        scene
            .try_circle(
                Vec2::new(2.0_f32.powi(121), index as f32),
                1.0,
                ShapeStyle::filled(Color::WHITE),
            )
            .unwrap();
    }

    assert_eq!(scene.command_count(), 256);
    assert!(!scene_command_sources_are_portable(&scene));
}

#[test]
fn circle_world_offsets_remain_inside_shader_arithmetic_validation() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(Vec2::ZERO, f32::MAX, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let camera = Camera2d::new(Vec2::ZERO, 2.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    assert!(!GeometryExtents::from_vertices(&vertices).is_safe_for(uniform));
}

#[test]
fn geometry_extents_reject_shader_arithmetic_overflow() {
    let vertices = [world_vertex(
        Vec2::new(f32::MAX * 0.75, 0.0),
        Vec2::ZERO,
        Color::WHITE,
    )];
    let extents = GeometryExtents::from_vertices(&vertices);
    let Ok(camera) = Camera2d::new(Vec2::ZERO, 2.0) else {
        panic!("test camera should be valid");
    };
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };
    let Some(uniform) = CameraUniform::new(camera, viewport) else {
        panic!("test camera uniform should be finite");
    };

    assert!(!extents.is_safe_for(uniform));
}

#[test]
fn geometry_extents_reject_dot_product_overflow_hidden_by_cancellation() {
    let depth = 3.0e37;
    let tilt = std::f32::consts::FRAC_PI_4;
    let lifted_sine = depth * tilt.sin();
    let center = Vec2::new(-lifted_sine * 0.5, -lifted_sine / tilt.cos());
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 100.0).unwrap();
    camera.set_projection(crate::Projection2d::new(tilt, 1.0).unwrap());

    assert!(
        camera
            .projected_world_to_screen(center, depth, viewport)
            .is_ok(),
        "the public camera formula remains mathematically finite"
    );

    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .with_depth(depth, |scene| {
            scene.try_circle(center, 1.0, ShapeStyle::filled(Color::WHITE))
        })
        .unwrap()
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    // WGSL evaluates each dot-product term in f32. Both camera rows contain
    // products above f32::MAX even though their mathematical sums cancel.
    assert!(!GeometryExtents::from_vertices(&vertices).is_safe_for(uniform));
}

#[test]
fn geometry_validation_rejects_extreme_world_depth_correlation() {
    let depth = 0.9 * f32::MAX;
    let tilt = std::f32::consts::FRAC_PI_4;
    let horizontal = depth * tilt.sin() * 0.5;
    let vertical = depth * tilt.sin() / tilt.cos();
    let vertices = [
        DynamicGpu {
            world_position: [-horizontal, -vertical],
            depth,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [horizontal, vertical],
            depth: -depth,
            color: Color::WHITE.to_array(),
        },
    ];
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(tilt, 1.0).unwrap());
    let uniform = CameraUniform::new(camera, LogicalViewport::new(1.0, 1.0).unwrap()).unwrap();
    let extents = GeometryExtents::from_dynamic_vertices(&vertices);

    assert!(
        vertices.iter().all(
            |vertex| Vec2::new(vertex.world_position[0], vertex.world_position[1]).is_finite()
        )
    );
    assert!(!extents.is_safe_for(uniform));
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&vertices),
        uniform,
    ));
}

#[test]
fn camera_uniform_rejects_extreme_direction_projection_scale() {
    let mut horizontal = world_vertex(Vec2::ZERO, Vec2::ZERO, Color::WHITE);
    horizontal.previous_direction = [1.0, 0.0];
    horizontal.next_direction = [1.0, 0.0];
    horizontal.normal_distance = 1.0;
    let mut vertical = world_vertex(Vec2::ZERO, Vec2::ZERO, Color::WHITE);
    vertical.previous_direction = [0.0, -1.0];
    vertical.next_direction = [0.0, -1.0];
    vertical.normal_distance = 1.0;
    let vertices = [horizontal, vertical];
    let mut camera = Camera2d::new(Vec2::ZERO, f32::MAX).unwrap();
    camera.set_rotation(std::f32::consts::FRAC_PI_4).unwrap();
    assert!(CameraUniform::new(camera, LogicalViewport::new(1.0, 1.0).unwrap()).is_none());
    assert_eq!(vertices.len(), 2);
}

#[test]
fn geometry_validation_propagates_dot_rounding_into_final_clip() {
    let depth = 0.9 * f32::MAX;
    let tilt = std::f32::consts::FRAC_PI_4;
    let vertex = DynamicGpu {
        world_position: [-depth * tilt.sin() * 0.5, -depth * tilt.sin() / tilt.cos()],
        depth,
        color: Color::WHITE.to_array(),
    };
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(tilt, 1.0).unwrap());
    let tiny_extent = 8.0 / f32::MAX;
    let uniform = CameraUniform::new(
        camera,
        LogicalViewport::new(tiny_extent, tiny_extent).unwrap(),
    );

    assert!(uniform.is_none());
    assert!(vertex.world_position.into_iter().all(f32::is_finite));
}

#[test]
fn shader_dot_envelope_rejects_terms_outside_portability_envelope() {
    assert!(!shader_interval_sum_is_safe([
        (-1.47e38, -1.47e38),
        (0.83e38, 0.83e38),
        (1.70e38, 1.70e38),
    ]));
    assert!(!shader_interval_sum_is_safe([
        (2.0e38, 2.0e38),
        (-2.0e38, -2.0e38),
        (2.0e38, 2.0e38),
    ]));
}

#[test]
fn shader_dot_envelope_rejects_identity_outside_portability_envelope() {
    assert_eq!(
        shader_interval_sum_range([
            (f64::from(f32::MAX), f64::from(f32::MAX)),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ]),
        None
    );
}

#[test]
fn shader_dot_envelope_rejects_extreme_sources_before_two_product_dot() {
    let mut camera = Camera2d::new(Vec2::ZERO, f32::from_bits(0x3f35_04f4)).unwrap();
    camera.set_rotation(-std::f32::consts::FRAC_PI_4).unwrap();
    let uniform = CameraUniform::new(camera, LogicalViewport::new(2.0, 2.0).unwrap()).unwrap();
    let vertex = DynamicGpu {
        world_position: [f32::MAX, f32::MAX],
        depth: 0.0,
        color: Color::WHITE.to_array(),
    };

    assert_eq!(uniform.world_to_screen_x, [0.5, 0.5, 0.0, 1.0]);
    let extents = GeometryExtents::from_dynamic_vertices(&[vertex]);
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&[vertex]),
        uniform,
    ));
}

#[test]
fn shader_dot_envelope_rejects_single_extreme_interval_term() {
    let previous = f32::from_bits(f32::MAX.to_bits() - 1);

    assert_eq!(
        shader_interval_sum_range([
            (-f64::from(f32::MAX), -f64::from(previous)),
            (0.0, 0.0),
            (0.0, 0.0),
            (-0.0, -0.0),
        ]),
        None
    );
}

#[test]
fn interval_shader_dot_bounds_cancellation_across_zero() {
    let minimum_normal = f32::MIN_POSITIVE;
    let adjacent_normal = f32::from_bits(minimum_normal.to_bits() + 1);

    let range = shader_interval_sum_range([
        (f64::from(adjacent_normal), f64::from(adjacent_normal)),
        (-f64::from(minimum_normal), -f64::from(minimum_normal)),
    ])
    .unwrap();
    assert!(range.0 < 0.0 && range.1 > 0.0);
}

#[test]
fn actual_vertex_scan_rejects_interior_ftz_product_hidden_by_extrema() {
    let vertices = [
        DynamicGpu {
            world_position: [-2.0, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [0.5, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
        DynamicGpu {
            world_position: [2.0, 0.0],
            depth: 0.0,
            color: Color::WHITE.to_array(),
        },
    ];
    let uniform = CameraUniform {
        camera_center: [0.0; 4],
        world_to_screen_x: [f32::MIN_POSITIVE, 0.0, 0.0, 1.0],
        world_to_screen_y: [0.0, 1.0, 0.0, 1.0],
        screen_to_clip: [1.0, 1.0, 0.0, 0.0],
    };
    let extents = GeometryExtents::from_dynamic_vertices(&vertices);

    assert!(extents.is_safe_for(uniform));
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Dynamic(&vertices),
        uniform,
    ));
}

#[test]
fn maximum_target_scale_is_outside_gpu_portability_envelope() {
    let scale = 0.5 / f32::MIN_POSITIVE;
    let extent = 1.0 / scale;
    let viewport = LogicalViewport::new(extent, extent).unwrap();
    assert!(CameraUniform::new(Camera2d::default(), viewport).is_none());
}

#[test]
fn geometry_extents_reject_world_offset_overflow_before_camera_scale() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(Vec2::ZERO, 2.0e38, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let camera = Camera2d::new(Vec2::splat(-2.0e38), 0.000_1).unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport);

    // The projected f64 envelope is small after zoom, but WGSL first adds
    // 2e38 + 2e38 in f32. Validation must reject that intermediate overflow.
    assert!(uniform.is_none());
    assert!(!vertices.is_empty());
}

#[test]
fn geometry_extents_reject_final_screen_to_clip_overflow() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(Vec2::ZERO, 2.0e38, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(1.0, 1.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    assert!(!GeometryExtents::from_vertices(&vertices).is_safe_for(uniform));

    let dynamic = [DynamicGpu {
        world_position: [2.0e38, 0.0],
        depth: 0.0,
        color: Color::WHITE.to_array(),
    }];
    assert!(!GeometryExtents::from_dynamic_vertices(&dynamic).is_safe_for(uniform));
}

#[test]
fn geometry_validation_rejects_extreme_base_offset_correlation_between_commands() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            Vec2::new(2.0e38, 0.0),
            1.0,
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    scene
        .try_circle(Vec2::ZERO, 2.0e38, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    let (vertices, _) = tessellate_for_test(&scene);
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();

    let extents = GeometryExtents::from_vertices(&vertices);
    assert!(!extents.is_safe_for(uniform));
    assert!(!geometry_is_safe_for(
        extents,
        GeometryValidationSource::Tessellated(&vertices),
        uniform,
    ));
}

#[test]
fn relative_bounds_cover_rounding_reordered_base_offset_pairs() {
    let center = 1.661_066_2e32;
    let pairs = [
        (-6.524_132_6e24, -3.625_333_7e24),
        (2.993_184_3e34, 8.678_791e24),
        (-9.846_69e24, -2.212_276_4e-36),
    ];
    let minimum = pairs
        .iter()
        .map(|pair| pair.0)
        .fold(f32::INFINITY, f32::min);
    let maximum = pairs
        .iter()
        .map(|pair| pair.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let offset_minimum = pairs
        .iter()
        .map(|pair| pair.1)
        .fold(f32::INFINITY, f32::min);
    let offset_maximum = pairs
        .iter()
        .map(|pair| pair.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let bounds =
        shader_relative_component_bounds(minimum, maximum, center, offset_minimum, offset_maximum)
            .unwrap();

    for (base, offset) in pairs {
        let actual = f64::from((base - center) + offset);
        assert!(actual >= bounds.0 && actual <= bounds.1);
    }
}

#[test]
fn renderer_uniform_rejects_camera_center_outside_portability_envelope() {
    let center = Vec2::new(2.0e38, 0.0);
    let world = Vec2::new(center.x + 1.0e33, 0.0);
    let vertices = [world_vertex(world, Vec2::ZERO, Color::WHITE)];
    let extents = GeometryExtents::from_vertices(&vertices);
    let Ok(camera) = Camera2d::new(center, 2.0) else {
        panic!("large finite camera should be valid");
    };
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };
    assert!(CameraUniform::new(camera, viewport).is_none());
    assert!(!extents.empty);
}

#[test]
fn non_finite_and_overflowing_geometry_never_reaches_batches() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert!(!scene.circle(
        Vec2::new(f32::NAN, 0.0),
        10.0,
        ShapeStyle::filled(Color::WHITE)
    ));
    assert!(!scene.circle(
        Vec2::new(f32::MAX, f32::MAX),
        f32::MAX,
        ShapeStyle::filled(Color::WHITE)
    ));

    let (vertices, draw_batches) = tessellate_for_test(&scene);

    assert!(vertices.is_empty());
    assert!(draw_batches.is_empty());
}
