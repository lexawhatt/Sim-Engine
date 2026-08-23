use std::sync::mpsc;

use super::*;
use crate::LogicalScreenVector;

fn tessellate_for_test(scene: &Scene) -> (Vec<Vertex>, Vec<PreparedDrawBatch>) {
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    let _ = tessellate_scene(scene, &mut vertices, &mut draw_batches);
    (vertices, draw_batches)
}

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

    let vertices = dynamic_vertices_to_gpu(&[vertex; 3]).unwrap();
    assert_eq!(vertices.len(), 3);
    assert_eq!(dynamic_vertex_capacity(3), Some(4));
    assert_eq!(dynamic_vertex_capacity(usize::MAX), None);
    assert_eq!(buffer_allocation_bytes::<Vertex>(usize::MAX), None);
}

#[test]
fn particle_instances_validate_visual_contract() {
    let particle = ParticleInstance2d::new(Vec2::new(3.0, -2.0), 4.5, Color::WHITE, 1.0).unwrap();
    assert_eq!(particle.world_position(), Vec2::new(3.0, -2.0));
    assert_eq!(particle.radius(), 4.5);
    assert_eq!(particle.depth(), 1.0);
    assert_eq!(particle.color(), Color::WHITE);
    assert_eq!(
        ParticleInstance2d::new(Vec2::ZERO, 0.0, Color::WHITE, 0.0),
        Err(ParticleInstanceError::InvalidRadius)
    );
    assert_eq!(
        ParticleInstance2d::new(Vec2::ZERO, 1.0, Color::WHITE, f32::NAN),
        Err(ParticleInstanceError::NonFinite)
    );
}

#[test]
fn particle_gpu_instances_preserve_counts_and_capacity_contract() {
    let particle = ParticleInstance2d::new(Vec2::new(2.0, -3.0), 2.0, Color::WHITE, 4.0)
        .expect("finite particle should be valid");
    let gpu_instances = particle_instances_to_gpu(&[particle]).unwrap();
    assert_eq!(gpu_instances.len(), 1);
    assert_eq!(particle_instance_capacity(0), Some(1));
    assert_eq!(particle_instance_capacity(3), Some(4));
    let statistics = particle_statistics(gpu_instances.len(), gpu_instances.len(), 1);
    assert_eq!(statistics.submitted(), 1);
    assert_eq!(statistics.visible(), 1);
    assert_eq!(statistics.culled(), 0);
    assert_eq!(statistics.dropped(), 0);
    assert_eq!(statistics.rendered(), 1);
}

#[test]
fn particle_budget_caps_memory_upload_and_samples_evenly() {
    let instance_bytes = std::mem::size_of::<ParticleGpu>();
    assert_eq!(
        ParticleRenderBudget::new(0, instance_bytes, instance_bytes),
        Err(ParticleBudgetError::InvalidLimit)
    );
    let budget = ParticleRenderBudget::new(30, instance_bytes * 20, instance_bytes * 12)
        .expect("budget fits at least one instance");
    assert_eq!(budget.instance_limit(), 12);
    assert_eq!(particle_budgeted_capacity(100, budget), Some(12));
    assert_eq!(
        budget.with_max_visibility_checks(11),
        Err(ParticleBudgetError::InvalidLimit)
    );
    let bounded_checks = budget.with_max_visibility_checks(12).unwrap();
    assert_eq!(bounded_checks.max_visibility_checks_per_frame(), 12);

    let selected: Vec<_> = (0..100)
        .filter(|index| particle_visible_index_is_selected(*index, 100, 12))
        .collect();
    assert_eq!(selected.len(), 12);
    assert!(selected.windows(2).all(|pair| pair[1] > pair[0]));

    let candidates: Vec<_> = (0..12)
        .map(|index| uniformly_sampled_index(index, 100, 12))
        .collect();
    assert_eq!(candidates.last(), Some(&99));
    assert!(candidates.windows(2).all(|pair| pair[1] > pair[0]));

    let statistics = particle_statistics_with_budget(140, 140, 100, 12, 12);
    assert_eq!(statistics.culled(), 40);
    assert_eq!(statistics.budget_limited(), 88);
    assert_eq!(statistics.rendered(), 12);

    let bounded_statistics = particle_statistics_with_budget(140, 12, 12, 12, 12);
    assert_eq!(bounded_statistics.visibility_checked(), 12);
    assert_eq!(bounded_statistics.budget_limited(), 128);
}

#[test]
fn layered_visualization_options_reject_invalid_composition_contracts() {
    assert!(matches!(
        LayeredVisualizationOptions::new((-f32::MAX, f32::MAX), Color::BLACK, Color::BLACK),
        Err(LayeredVisualizationError::InvalidValueRange { .. })
    ));
    assert_eq!(
        LayeredVisualizationOptions::new(
            (0.0, 1.0),
            Color::rgba(f32::NAN, 0.0, 0.0, 1.0),
            Color::BLACK
        ),
        Err(LayeredVisualizationError::InvalidBackground)
    );
    let options = LayeredVisualizationOptions::new((0.0, 1.0), Color::BLACK, Color::BLACK).unwrap();
    assert_eq!(options.value_range(), (0.0, 1.0));
    assert_eq!(options.sampling(), ScalarFieldSampling::Linear);
    assert_eq!(
        options.with_composition(BlendMode::Alpha, f32::NAN),
        Err(LayeredVisualizationError::InvalidOpacity)
    );
}

#[test]
fn particle_culling_keeps_circles_that_intersect_the_logical_viewport() {
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let visible = ParticleGpu {
        world_position: [55.0, 0.0],
        depth: 0.0,
        radius: 6.0,
        color: Color::WHITE.to_array(),
    };
    let culled = ParticleGpu {
        world_position: [57.0, 0.0],
        depth: 0.0,
        radius: 6.0,
        color: Color::WHITE.to_array(),
    };
    assert!(visible.is_safe_for(uniform));
    assert!(visible.intersects_viewport(uniform, viewport));
    assert!(!culled.intersects_viewport(uniform, viewport));
}

#[test]
fn particle_partial_updates_require_an_existing_contiguous_range() {
    assert_eq!(particle_update_range(2, 3, 5), Ok(2..5));
    assert_eq!(particle_update_range(5, 0, 5), Ok(5..5));
    assert_eq!(
        particle_update_range(4, 2, 5),
        Err(ParticleFieldError::UpdateRangeOutOfBounds)
    );
    assert_eq!(
        particle_update_range(usize::MAX, 1, 5),
        Err(ParticleFieldError::UpdateRangeOutOfBounds)
    );
}

fn vertex_screen_position(vertex: Vertex, camera: Camera2d, viewport: LogicalViewport) -> Vec2 {
    let Some(uniform) = CameraUniform::new(camera, viewport) else {
        panic!("test camera uniform should be finite");
    };
    let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
    let mut screen = uniform.world_to_screen(world, vertex.depth)
        + Vec2::new(vertex.screen_offset[0], vertex.screen_offset[1]);
    if vertex.normal_distance.abs() > 0.0 {
        let previous = uniform.direction_to_screen(Vec2::new(
            vertex.previous_direction[0],
            vertex.previous_direction[1],
        ));
        let next = uniform.direction_to_screen(Vec2::new(
            vertex.next_direction[0],
            vertex.next_direction[1],
        ));
        let previous_normal = previous.normalized().perp();
        let next_normal = next.normalized().perp();
        let combined_normal = previous_normal + next_normal;
        let mut extrusion = next_normal * vertex.normal_distance;
        if combined_normal.length_squared() > 0.000001 {
            let miter = combined_normal.normalized();
            let denominator = miter.dot(next_normal);
            if denominator.abs() > 0.001 {
                extrusion = miter * (vertex.normal_distance / denominator);
            }
        }
        screen += extrusion;
    }
    screen
}

#[test]
fn filled_circle_tessellates_to_triangle_fan() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 12.0, ShapeStyle::filled(Color::WHITE));

    let (vertices, draw_batches) = tessellate_for_test(&scene);

    assert_eq!(vertices.len(), CIRCLE_SEGMENTS * 3);
    assert_eq!(draw_batches.len(), 1);
}

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

    assert_eq!(vertices.len(), 3 * 6 + ROUND_CAP_SEGMENTS * 6);
    assert!(vertices.iter().copied().all(Vertex::is_finite));
}

#[test]
fn short_accepted_line_emits_vertices() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert!(scene.line(Vec2::ZERO, Vec2::new(0.005, 0.0), 2.0, Color::WHITE));

    let (vertices, _) = tessellate_for_test(&scene);

    assert!(!vertices.is_empty());
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

    tessellate_scene(&scene, &mut vertices, &mut draw_batches);

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
fn geometry_extents_accept_valid_geometry_relative_to_large_camera_center() {
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
    let Some(uniform) = CameraUniform::new(camera, viewport) else {
        panic!("relative camera uniform should remain finite");
    };

    assert!(uniform.world_to_screen(world, 0.0).is_finite());
    assert!(extents.is_safe_for(uniform));
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

#[test]
fn clipped_commands_create_scissor_batches() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(10.25, 20.75),
        LogicalScreenVector::new(100.0, 80.0),
    )
    .unwrap();
    scene
        .with_screen_clip(clip, |scene| {
            scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE));
            scene.circle(Vec2::X, 8.0, ShapeStyle::filled(Color::WHITE));
        })
        .unwrap();
    scene.circle(Vec2::Y, 8.0, ShapeStyle::filled(Color::WHITE));

    let (_, draw_batches) = tessellate_for_test(&scene);

    assert_eq!(draw_batches.len(), 2);
    assert_eq!(draw_batches[0].screen_clip, Some(clip));
    assert_eq!(draw_batches[1].screen_clip, None);
}

#[test]
fn offscreen_clip_keeps_prepared_geometry_but_resolves_to_no_scissor() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let screen_clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(900.0, 700.0),
        LogicalScreenVector::new(20.0, 20.0),
    )
    .unwrap();
    scene
        .with_screen_clip(screen_clip, |scene| {
            scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE))
        })
        .unwrap();

    let (vertices, draw_batches) = tessellate_for_test(&scene);
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };

    assert!(!vertices.is_empty());
    assert_eq!(draw_batches[0].screen_clip, Some(screen_clip));
    assert_eq!(screen_clip_to_scissor(screen_clip, viewport, 1.0), None);
}

#[test]
fn non_finite_screen_clip_is_rejected_immediately() {
    let scene = Scene::new(Color::BLACK).unwrap();
    assert_eq!(
        ScreenClipRect::new(
            LogicalScreenPosition::new(f32::NAN, 0.0),
            LogicalScreenPosition::new(100.0, 100.0),
        ),
        Err(crate::SceneError::InvalidScreenClip)
    );
    assert_eq!(scene.command_count(), 0);
}

#[test]
fn logical_clip_converts_to_hidpi_physical_scissor() {
    let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
        panic!("test viewport should be valid");
    };
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(10.25, 20.75),
        LogicalScreenVector::new(100.0, 80.0),
    )
    .unwrap();

    let scissor = screen_clip_to_scissor(clip, viewport, 2.0);

    assert_eq!(
        scissor,
        Some(ScissorRect {
            x: 20,
            y: 41,
            width: 201,
            height: 161,
        })
    );
}

#[test]
fn renderer_screen_position_conversion_is_explicit_at_hidpi() {
    let physical = PhysicalScreenPosition::new(800.0, 600.0);

    let logical = physical_to_logical_screen(physical, 2.0).unwrap();
    let roundtrip = logical_to_physical_screen(logical, 2.0).unwrap();

    assert_eq!(logical, LogicalScreenPosition::new(400.0, 300.0));
    assert_eq!(roundtrip, physical);
    assert_eq!(
        physical_to_logical_screen(PhysicalScreenPosition::new(f32::MAX, 0.0), 0.5),
        Err(RendererCoordinateError::NonFiniteConversion)
    );
}

#[test]
fn prepared_scene_identity_guard_rejects_another_renderer() {
    let first_renderer = Arc::new(());
    let same_renderer = Arc::clone(&first_renderer);
    let second_renderer = Arc::new(());

    assert!(prepared_scene_belongs_to(&first_renderer, &same_renderer));
    assert!(!prepared_scene_belongs_to(
        &first_renderer,
        &second_renderer
    ));
}

#[test]
fn scalar_value_range_rejects_finite_subtraction_overflow() {
    assert_eq!(scalar_value_range_extent(0.0, 1.0), Some(1.0));
    assert_eq!(scalar_value_range_extent(-f32::MAX, f32::MAX), None);
}

#[test]
fn offscreen_gpu_readback_verifies_camera_depth_and_clip_contract() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
        else {
            assert_ne!(
                std::env::var("SIM_ENGINE_REQUIRE_GPU_TESTS").as_deref(),
                Ok("1"),
                "a GPU adapter is required by SIM_ENGINE_REQUIRE_GPU_TESTS=1"
            );
            return;
        };
        let Ok((device, queue)) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine offscreen test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
        else {
            panic!("adapter should create a test device");
        };
        let Ok((recovery_device, recovery_queue)) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine offscreen recovery test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
        else {
            panic!("adapter should create a second recovery device");
        };

        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let recovery_validation_scope =
            recovery_device.push_error_scope(wgpu::ErrorFilter::Validation);
        let vertex_limit = device.limits().max_buffer_size / std::mem::size_of::<Vertex>() as u64;
        if let Ok(first_invalid_capacity) = usize::try_from(vertex_limit.saturating_add(1)) {
            assert!(!buffer_capacity_fits::<Vertex>(
                &device,
                first_invalid_capacity
            ));
        }
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (
            pipeline,
            _dynamic_pipeline,
            _particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            _composition_pipelines,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        ) = create_pipeline(&device, format, 1);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen test target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(30.0, 20.0),
            LogicalScreenVector::new(8.0, 12.0),
        )
        .unwrap();
        assert!(
            scene
                .with_depth(3.0, |scene| {
                    scene
                        .with_screen_clip(clip, |scene| {
                            scene.rect(
                                Rect::from_center_size(Vec2::ZERO, Vec2::splat(4.0)),
                                0.0,
                                ShapeStyle::filled(Color::WHITE),
                            );
                        })
                        .unwrap();
                })
                .is_ok()
        );
        let source_identity = Arc::new(());
        let prepared =
            prepare_scene_resources(&device, &queue, Arc::clone(&source_identity), &scene)
                .expect("small prepared scene should fit the test device");
        let replacement_identity = Arc::new(());
        let restored = restore_prepared_scene_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            &prepared,
        )
        .expect("small prepared scene should restore on the test device");
        assert!(prepared_scene_belongs_to(
            &source_identity,
            &prepared.renderer_identity
        ));
        assert!(!prepared_scene_belongs_to(
            &source_identity,
            &restored.renderer_identity
        ));
        assert!(prepared_scene_belongs_to(
            &replacement_identity,
            &restored.renderer_identity
        ));
        assert_eq!(restored.vertex_count(), prepared.vertex_count());
        assert_eq!(
            restored.recovery_memory_bytes(),
            restored.vertex_count() * std::mem::size_of::<Vertex>()
        );
        assert!(Arc::ptr_eq(&restored.vertices, &prepared.vertices));
        let recovery_identity = Arc::new(());
        let recovered_on_another_device = restore_prepared_scene_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &prepared,
        )
        .expect("prepared scene should migrate to a second device");
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_on_another_device.renderer_identity
        ));
        assert_eq!(
            recovered_on_another_device.vertex_count(),
            prepared.vertex_count()
        );
        assert!(Arc::ptr_eq(
            &recovered_on_another_device.vertices,
            &prepared.vertices
        ));

        let dynamic_vertex = DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::WHITE).unwrap();
        let dynamic_vertices = dynamic_vertices_to_gpu(&[dynamic_vertex; 3]).unwrap();
        let dynamic_source = DynamicMesh2d {
            renderer_identity: Arc::clone(&source_identity),
            vertex_buffer: Arc::new(create_dynamic_vertex_buffer(&device, 8)),
            geometry_extents: GeometryExtents::from_dynamic_vertices(&dynamic_vertices),
            vertices: dynamic_vertices,
            vertex_capacity: 8,
        };
        queue.write_buffer(
            &dynamic_source.vertex_buffer,
            0,
            bytemuck::cast_slice(dynamic_source.vertices.as_slice()),
        );
        let restored_dynamic = restore_dynamic_mesh_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            &dynamic_source,
        )
        .expect("small dynamic mesh should restore on the test device");
        assert_eq!(restored_dynamic.vertex_count(), 3);
        assert_eq!(restored_dynamic.vertex_capacity(), 8);
        assert_eq!(
            restored_dynamic.recovery_memory_bytes(),
            dynamic_source.recovery_memory_bytes()
        );
        assert!(prepared_scene_belongs_to(
            &replacement_identity,
            &restored_dynamic.renderer_identity
        ));
        let recovered_dynamic_on_another_device = restore_dynamic_mesh_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &dynamic_source,
        )
        .expect("dynamic mesh should migrate to a second device");
        assert_eq!(recovered_dynamic_on_another_device.vertex_count(), 3);
        assert_eq!(recovered_dynamic_on_another_device.vertex_capacity(), 8);
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_dynamic_on_another_device.renderer_identity
        ));

        let scalar_source = ScalarField::new(2, 2, vec![0.0, 0.25, 0.5, 1.0]).unwrap();
        let maximum_texture_dimension = device.limits().max_texture_dimension_2d;
        if maximum_texture_dimension < 1_000_000 {
            let oversized = ScalarField::filled(maximum_texture_dimension as usize + 1, 1, 0.0)
                .expect("GPU-limit probe should be a valid CPU scalar field");
            assert!(matches!(
                create_scalar_field_texture(&device, &oversized),
                Err(ScalarFieldTextureError::DimensionsTooLarge)
            ));
        }
        let scalar_texture = create_scalar_field_texture_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            scalar_source,
        )
        .expect("finite scalar texture should upload");
        assert_eq!((scalar_texture.width(), scalar_texture.height()), (2, 2));
        assert_eq!(scalar_texture.recovery_memory_bytes(), 16);
        let restored_scalar_texture = create_scalar_field_texture_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            scalar_texture.field.clone(),
        )
        .expect("scalar texture should restore");
        assert_eq!(
            restored_scalar_texture.field().values(),
            scalar_texture.field().values()
        );
        let recovered_scalar_texture_on_another_device = create_scalar_field_texture_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            scalar_texture.field.clone(),
        )
        .expect("scalar texture should migrate to a second device");
        assert_eq!(
            recovered_scalar_texture_on_another_device.field().values(),
            scalar_texture.field().values()
        );
        let heatmap_color_map = ColorMap::linear(Color::BLACK, Color::WHITE).unwrap();
        let heatmap_lut = create_cached_color_map(&device, &queue, &heatmap_color_map);
        let heatmap_scalar_view = scalar_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_lut_view = &heatmap_lut.view;
        let heatmap_uniform = HeatmapUniform {
            value_range: [0.0, 1.0, 0.0, 0.0],
            dimensions: [2, 2, 0, 0],
        };
        queue.write_buffer(
            &heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&heatmap_uniform),
        );
        let heatmap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen heatmap bind group"),
            layout: &heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heatmap_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: heatmap_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let heatmap_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen heatmap target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let heatmap_view = heatmap_target.create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen heatmap readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 1.0) else {
            panic!("test camera should be valid");
        };
        let Ok(projection) = crate::Projection2d::new(0.5, 2.0) else {
            panic!("test projection should be valid");
        };
        camera.set_projection(projection);
        let Ok(viewport) = LogicalViewport::new(64.0, 64.0) else {
            panic!("test viewport should be valid");
        };
        let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
            panic!("test camera uniform should be finite");
        };
        queue.write_buffer(
            &camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
        let particle_instances = [
            ParticleGpu {
                world_position: [18.0, -18.0],
                depth: 0.0,
                radius: 6.0,
                color: Color::rgba(1.0, 0.0, 0.0, 1.0).to_array(),
            },
            ParticleGpu {
                world_position: [1_000.0, 0.0],
                depth: 0.0,
                radius: 6.0,
                color: Color::WHITE.to_array(),
            },
        ];
        let particle_source = ParticleField2d {
            renderer_identity: Arc::clone(&source_identity),
            instance_buffer: Arc::new(create_particle_instance_buffer(&device, 2)),
            instances: particle_instances.to_vec(),
            visible_instances: Vec::new(),
            instance_capacity: 2,
            budget: ParticleRenderBudget::UNBOUNDED,
            statistics: particle_statistics(2, 1, 1),
        };
        queue.write_buffer(
            &particle_source.instance_buffer,
            0,
            bytemuck::cast_slice(&particle_source.instances),
        );
        let recovered_particles_on_another_device = restore_particle_field_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &particle_source,
        )
        .expect("particle field should migrate to a second device");
        assert_eq!(recovered_particles_on_another_device.instance_count(), 2);
        assert_eq!(recovered_particles_on_another_device.instance_capacity(), 2);
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_particles_on_another_device.renderer_identity
        ));
        assert_eq!(
            recovered_particles_on_another_device.statistics(),
            particle_statistics(2, 2, 0),
            "recovery must not preserve stale culling or draw statistics"
        );
        let recovery_vertex_bytes = bytemuck::cast_slice(prepared.vertices.as_ref());
        let recovery_readback = recovery_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine recovery vertex readback"),
            size: recovery_vertex_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut recovery_encoder =
            recovery_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine recovery readback encoder"),
            });
        recovery_encoder.copy_buffer_to_buffer(
            &recovered_on_another_device.vertex_buffer,
            0,
            &recovery_readback,
            0,
            recovery_vertex_bytes.len() as wgpu::BufferAddress,
        );
        let _recovery_submission = recovery_queue.submit([recovery_encoder.finish()]);
        recovery_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("recovery upload should complete on the second device");
        let recovery_slice = recovery_readback.slice(..);
        let (recovery_sender, recovery_receiver) = mpsc::channel();
        recovery_slice.map_async(wgpu::MapMode::Read, move |result| {
            recovery_sender.send(result).unwrap()
        });
        recovery_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("recovery readback should complete on the second device");
        recovery_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("recovery readback callback")
            .expect("recovery readback should map");
        let recovered_bytes = recovery_slice
            .get_mapped_range()
            .expect("recovery vertex bytes");
        assert_eq!(recovered_bytes.as_ref(), recovery_vertex_bytes);
        drop(recovered_bytes);
        recovery_readback.unmap();
        if let Some(error) = recovery_validation_scope.pop().await {
            panic!("second-device recovery validation failed: {error}");
        }
        let visible_particles =
            visible_particle_instances(&particle_instances, camera_uniform, viewport).unwrap();
        assert_eq!(
            visible_particles.len(),
            1,
            "offscreen particle should be culled"
        );
        let particle_instance_buffer = create_particle_instance_buffer(&device, 1);
        queue.write_buffer(
            &particle_instance_buffer,
            0,
            bytemuck::cast_slice(&visible_particles),
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen readback"),
            size: (256 * 64) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen test encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &heatmap_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_heatmap_pipeline);
            pass.set_bind_group(0, &heatmap_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_vertex_buffer(0, restored.vertex_buffer.slice(..));
            for batch in &restored.draw_batches {
                if let Some(clip) = batch.screen_clip {
                    let scissor = screen_clip_to_scissor(clip, viewport, 1.0).unwrap();
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                }
                pass.draw(batch.vertex_range.clone(), 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_particle_pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_vertex_buffer(0, particle_unit_buffer.slice(..));
            pass.set_vertex_buffer(1, particle_instance_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &heatmap_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &heatmap_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );

        let _submission = queue.submit([encoder.finish()]);
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        }) {
            panic!("offscreen GPU submission did not complete: {error:?}");
        }
        if let Some(error) = validation_scope.pop().await {
            panic!("offscreen GPU validation failed: {error}");
        }
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        }) {
            panic!("offscreen GPU readback did not complete: {error:?}");
        }
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen readback callback")
            .expect("offscreen readback should map");
        let bytes = slice.get_mapped_range().expect("offscreen mapped bytes");
        let pixel = |x: usize, y: usize| &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert!(pixel(33, 29)[0] > 200, "camera/depth pixel was not drawn");
        assert!(pixel(33, 34)[0] < 10, "clip failed to remove outside pixel");
        assert!(
            pixel(50, 48)[0] > 200,
            "instanced particle center was not drawn"
        );
        assert!(
            pixel(50, 48)[1] < 10,
            "instanced particle color was not applied"
        );
        assert!(
            pixel(56, 54)[0] < 10,
            "particle circle mask did not discard its corner"
        );
        drop(bytes);
        readback.unmap();
        let heatmap_slice = heatmap_readback.slice(..);
        let (heatmap_sender, heatmap_receiver) = mpsc::channel();
        heatmap_slice.map_async(wgpu::MapMode::Read, move |result| {
            heatmap_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen heatmap readback should complete");
        heatmap_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen heatmap callback")
            .expect("offscreen heatmap should map");
        let heatmap_bytes = heatmap_slice
            .get_mapped_range()
            .expect("offscreen heatmap bytes");
        let heatmap_pixel =
            |x: usize, y: usize| &heatmap_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert!(
            heatmap_pixel(0, 0)[0] < 8,
            "minimum scalar should map to black"
        );
        assert!(
            heatmap_pixel(1, 1)[0] > 247,
            "maximum scalar should map to white"
        );
        assert!(
            heatmap_pixel(1, 0)[0] > 130 && heatmap_pixel(1, 0)[0] < 145,
            "quarter scalar should map through LUT then sRGB encode"
        );
        assert!(
            heatmap_pixel(0, 1)[0] > 180 && heatmap_pixel(0, 1)[0] < 195,
            "half scalar should map through LUT then sRGB encode"
        );
        drop(heatmap_bytes);
        heatmap_readback.unmap();

        let composition_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen composition target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let composition_target_view =
            composition_target.create_view(&wgpu::TextureViewDescriptor::default());
        let premultiplied_source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen premultiplied composition source"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &premultiplied_source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[188, 0, 0, 128],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let premultiplied_source_view =
            premultiplied_source.create_view(&wgpu::TextureViewDescriptor::default());
        let composition_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen composition uniform"),
            size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &composition_uniform,
            0,
            bytemuck::bytes_of(&CompositeUniform {
                opacity: [1.0, 0.0, 0.0, 0.0],
            }),
        );
        let composition_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen composition bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&premultiplied_source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: composition_uniform.as_entire_binding(),
                },
            ],
        });
        let composition_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen composition readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut composition_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine offscreen composition encoder"),
            });
        {
            let mut pass = composition_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &composition_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        composition_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composition_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &composition_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([composition_encoder.finish()]);
        let composition_slice = composition_readback.slice(..);
        let (composition_sender, composition_receiver) = mpsc::channel();
        composition_slice.map_async(wgpu::MapMode::Read, move |result| {
            composition_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen composition readback should complete");
        composition_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen composition callback")
            .expect("offscreen composition should map");
        let composition_bytes = composition_slice
            .get_mapped_range()
            .expect("offscreen composition bytes");
        assert!(
            composition_bytes[0] > 180 && composition_bytes[0] < 195,
            "premultiplied render-target RGB must be unpremultiplied before alpha composition"
        );
        drop(composition_bytes);
        composition_readback.unmap();

        queue.write_buffer(
            &target_composition_pipelines.secondary_uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform {
                opacity: [0.5, 0.0, 0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &target_composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform {
                opacity: [0.5, 0.0, 0.0, 0.0],
            }),
        );
        let trail_history_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen trail history bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_composition_pipelines
                        .secondary_uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let trail_source_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen trail source bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let mut trail_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen trail encoder"),
        });
        {
            let mut pass = trail_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen trail history pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &trail_history_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = trail_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen trail source pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &trail_source_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        trail_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composition_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &composition_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([trail_encoder.finish()]);
        let trail_slice = composition_readback.slice(..);
        let (trail_sender, trail_receiver) = mpsc::channel();
        trail_slice.map_async(wgpu::MapMode::Read, move |result| {
            trail_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen trail readback should complete");
        trail_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen trail callback")
            .expect("offscreen trail should map");
        let trail_bytes = trail_slice
            .get_mapped_range()
            .expect("offscreen trail bytes");
        assert!(
            trail_bytes[4] > 112 && trail_bytes[4] < 128,
            "half-retained history plus half-opacity source should accumulate predictably"
        );
        drop(trail_bytes);
        composition_readback.unmap();

        let linear_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen linear heatmap uniform"),
            size: std::mem::size_of::<HeatmapUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &linear_uniform_buffer,
            0,
            bytemuck::bytes_of(&HeatmapUniform {
                value_range: [0.0, 1.0, 0.0, 0.0],
                dimensions: [2, 2, ScalarFieldSampling::Linear.shader_value(), 0],
            }),
        );
        let linear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen linear heatmap bind group"),
            layout: &heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heatmap_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: linear_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let linear_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen linear heatmap target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let linear_view = linear_target.create_view(&wgpu::TextureViewDescriptor::default());
        let linear_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen linear heatmap readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut linear_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen linear heatmap encoder"),
        });
        {
            let mut pass = linear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen linear heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &linear_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&heatmap_pipeline);
            pass.set_bind_group(0, &linear_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        linear_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &linear_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &linear_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([linear_encoder.finish()]);
        let linear_slice = linear_readback.slice(..);
        let (linear_sender, linear_receiver) = mpsc::channel();
        linear_slice.map_async(wgpu::MapMode::Read, move |result| {
            linear_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen linear heatmap readback should complete");
        linear_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen linear heatmap callback")
            .expect("offscreen linear heatmap should map");
        let linear_bytes = linear_slice
            .get_mapped_range()
            .expect("offscreen linear heatmap bytes");
        let linear_pixel = |x: usize, y: usize| &linear_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert!(linear_pixel(0, 0)[0] < 8, "top-left texel moved vertically");
        assert!(
            linear_pixel(1, 0)[0] > 130 && linear_pixel(1, 0)[0] < 145,
            "top-right texel moved vertically"
        );
        assert!(
            linear_pixel(0, 1)[0] > 180 && linear_pixel(0, 1)[0] < 195,
            "bottom-left texel moved vertically"
        );
        assert!(
            linear_pixel(1, 1)[0] > 247,
            "bottom-right texel moved vertically"
        );
        drop(linear_bytes);
        linear_readback.unmap();
    });
}

#[test]
fn renderer_options_reject_invalid_scale_factor() {
    assert!(matches!(
        WgpuRendererOptions::new(RendererPresentMode::Vsync, f64::NAN),
        Err(RendererConfigurationError::InvalidScaleFactor { .. })
    ));
    assert!(matches!(
        WgpuRendererOptions::new(RendererPresentMode::Vsync, 0.0),
        Err(RendererConfigurationError::InvalidScaleFactor { .. })
    ));
    assert!(matches!(
        WgpuRendererOptions::new(RendererPresentMode::Vsync, f64::MIN_POSITIVE),
        Err(RendererConfigurationError::InvalidScaleFactor { .. })
    ));
    assert!(matches!(
        WgpuRendererOptions::new(RendererPresentMode::Vsync, f32::MIN_POSITIVE as f64),
        Err(RendererConfigurationError::InvalidScaleFactor { .. })
    ));
}

#[test]
fn render_target_memory_accounting_is_checked_and_format_aware() {
    assert_eq!(
        render_target_allocation_bytes(wgpu::TextureFormat::Rgba8UnormSrgb, 3, 5),
        Some(60)
    );
    assert_eq!(
        render_target_allocation_bytes(wgpu::TextureFormat::Rgba16Float, 3, 5),
        Some(120)
    );
}

#[test]
fn renderer_present_modes_have_explicit_fallback_contracts() {
    assert_eq!(
        RendererPresentMode::Vsync.to_wgpu(),
        wgpu::PresentMode::Fifo
    );
    assert_eq!(
        RendererPresentMode::NoVsync.to_wgpu(),
        wgpu::PresentMode::AutoNoVsync
    );
}
