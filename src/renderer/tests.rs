use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

use super::config::recovery_quarantine_has_capacity;
use super::*;
use crate::LogicalScreenVector;

fn tessellate_for_test(scene: &Scene) -> (Vec<Vertex>, Vec<PreparedDrawBatch>) {
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    tessellate_scene(scene, &mut vertices, &mut draw_batches)
        .expect("validated test scene should tessellate within its budget");
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
    assert_eq!(
        DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::rgb(1.01, 0.0, 0.0)),
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
    assert_eq!(
        LayeredVisualizationOptions::new(
            (0.0, 1.0),
            Color::BLACK,
            Color::rgba(0.0, 0.0, 0.0, 1.01),
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
        + uniform.direction_to_screen(Vec2::new(vertex.world_offset[0], vertex.world_offset[1]))
        + Vec2::new(vertex.screen_offset[0], vertex.screen_offset[1]);
    if vertex.normal_distance.abs() > 0.0 || vertex.tangent_distance.abs() > 0.0 {
        let previous = uniform.direction_to_screen(Vec2::new(
            vertex.previous_direction[0],
            vertex.previous_direction[1],
        ));
        let next = uniform.direction_to_screen(Vec2::new(
            vertex.next_direction[0],
            vertex.next_direction[1],
        ));
        let previous_tangent = previous.normalized();
        let next_tangent = next.normalized();
        let previous_normal = previous_tangent.perp();
        let next_normal = next_tangent.perp();
        let combined_normal = previous_normal + next_normal;
        let turn = previous_tangent
            .x
            .mul_add(next_tangent.y, -previous_tangent.y * next_tangent.x);
        let reverses = turn.abs() <= 0.000001 && previous_tangent.dot(next_tangent) < 0.0;
        let side = vertex.normal_distance.signum();
        let outer_side = -turn.signum();
        let mut extrusion =
            next_normal * vertex.normal_distance + next_tangent * vertex.tangent_distance;
        let mut miter_offset = Vec2::ZERO;
        let mut miter_multiple = f32::INFINITY;
        let mut miter_valid = false;
        if combined_normal.length_squared() > 0.000001 {
            let miter = combined_normal.normalized();
            let denominator = miter.dot(next_normal);
            if denominator.abs() > 0.001 {
                miter_multiple = (1.0 / denominator).abs();
                miter_offset = miter * (vertex.normal_distance / denominator);
                miter_valid = true;
            }
        }
        if (1.0..=3.0).contains(&vertex.stroke_role) {
            if reverses {
                extrusion = Vec2::ZERO;
            } else if turn.abs() <= 0.000001 {
                extrusion = next_normal * vertex.normal_distance;
            } else if side * outer_side <= 0.0 {
                extrusion = if miter_valid && miter_multiple <= vertex.miter_limit {
                    miter_offset
                } else {
                    Vec2::ZERO
                };
            } else if vertex.stroke_role == 2.0
                && miter_valid
                && miter_multiple <= vertex.miter_limit
            {
                extrusion = miter_offset;
            } else if vertex.stroke_parameter < 0.0 {
                extrusion = previous_normal * vertex.normal_distance;
            } else {
                extrusion = next_normal * vertex.normal_distance;
            }
            extrusion += next_tangent * vertex.tangent_distance;
        } else if vertex.stroke_role >= 4.0 {
            let inner = matches!(vertex.stroke_role as i32, 5 | 7 | 9);
            let candidate_side = if inner { -side } else { side };
            let mut join_active = turn.abs() > 0.000001 && candidate_side * outer_side > 0.0;
            if matches!(vertex.stroke_role as i32, 6 | 7) {
                join_active &= !miter_valid || miter_multiple > vertex.miter_limit;
            }
            extrusion = if !join_active {
                Vec2::ZERO
            } else if inner {
                if miter_valid && miter_multiple <= vertex.miter_limit {
                    miter_offset
                } else {
                    Vec2::ZERO
                }
            } else if vertex.stroke_role == 8.0 {
                let start = previous_normal * candidate_side;
                let finish = next_normal * candidate_side;
                let angle = start
                    .x
                    .mul_add(finish.y, -start.y * finish.x)
                    .atan2(start.dot(finish))
                    * vertex.stroke_parameter;
                Vec2::new(
                    start.x.mul_add(angle.cos(), -start.y * angle.sin()),
                    start.x.mul_add(angle.sin(), start.y * angle.cos()),
                ) * vertex.normal_distance.abs()
            } else if vertex.stroke_parameter < 0.0 {
                previous_normal * vertex.normal_distance
            } else {
                next_normal * vertex.normal_distance
            };
        } else if miter_valid && miter_multiple <= vertex.miter_limit {
            extrusion = miter_offset + next_tangent * vertex.tangent_distance;
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
    let budget = crate::SceneBudget::new(4, 4, 2_000, 16_000, 120_000, 4);
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
fn large_center_circle_fill_preserves_camera_relative_radius() {
    let center = Vec2::splat(1.0e20);
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
fn positioned_viewport_camera_and_scissor_share_one_logical_origin() {
    let target = LogicalViewport::new(800.0, 600.0).unwrap();
    let local = LogicalViewport::new(320.0, 180.0).unwrap();
    let origin = Vec2::new(40.0, 70.0);
    let camera = Camera2d::new(Vec2::ZERO, 2.0).unwrap();
    let uniform = CameraUniform::new_in_region(camera, local, origin, target).unwrap();

    assert_eq!(
        uniform.world_to_screen(Vec2::ZERO, 0.0),
        Vec2::new(200.0, 160.0)
    );
    for scale in [1.0, 1.25, 1.5, 2.0, 3.0] {
        let scissor = logical_viewport_scissor(
            origin,
            local,
            scale,
            (target.width() * scale) as u32,
            (target.height() * scale) as u32,
        )
        .unwrap();
        assert_eq!(scissor.x, (origin.x * scale).floor() as u32);
        assert_eq!(scissor.y, (origin.y * scale).floor() as u32);
        assert!(scissor.width as f32 / scale >= local.width());
        assert!(scissor.height as f32 / scale >= local.height());
    }
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

async fn assert_gpu_stroke_pixel_matrix(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 248;
    const ROW_BYTES: u32 = WIDTH * 4;
    let sample_count = preferred_sample_count(adapter, format);
    let PipelineResources {
        pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        ..
    } = create_pipeline(device, format, sample_count);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine stroke pixel-matrix resolve target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let multisample = (sample_count > 1).then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine stroke pixel-matrix multisample target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    });
    let multisample_view = multisample
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
    let viewport = LogicalViewport::new(WIDTH as f32, HEIGHT as f32).unwrap();
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let camera_uniform = CameraUniform::new(camera, viewport).unwrap();
    queue.write_buffer(
        &camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );

    let caps = [
        crate::StrokeCap2d::Butt,
        crate::StrokeCap2d::Square,
        crate::StrokeCap2d::Round,
    ];
    let joins = [
        crate::StrokeJoin2d::Bevel,
        crate::StrokeJoin2d::Miter,
        crate::StrokeJoin2d::Round,
    ];
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let screen_to_world =
        |x: f32, y: f32| Vec2::new(x - WIDTH as f32 * 0.5, HEIGHT as f32 * 0.5 - y);
    for width_mode in 0..2 {
        for turn_direction in 0..2 {
            for (cap_index, cap) in caps.iter().copied().enumerate() {
                for (join_index, join) in joins.iter().copied().enumerate() {
                    let row = cap_index * joins.len() + join_index;
                    let center_x = 64.0 + (width_mode * 2 + turn_direction) as f32 * 128.0;
                    let center_y = 12.0 + row as f32 * 24.0;
                    let vertical = if turn_direction == 0 { 6.0 } else { -6.0 };
                    let points = vec![
                        screen_to_world(center_x - 14.0, center_y + vertical),
                        screen_to_world(center_x, center_y - vertical),
                        screen_to_world(center_x + 14.0, center_y + vertical),
                    ];
                    let color = Color::rgba(1.0, 1.0, 1.0, 0.5);
                    let style = if width_mode == 0 {
                        crate::StrokeStyle2d::logical(
                            crate::LogicalPixels::new(10.0).unwrap(),
                            color,
                        )
                    } else {
                        crate::StrokeStyle2d::world(crate::WorldLength::new(10.0).unwrap(), color)
                    }
                    .with_cap(cap)
                    .with_join(join)
                    .with_miter_limit(4.0)
                    .unwrap();
                    scene.try_styled_polyline(points, style).unwrap();
                }
            }
        }
    }
    let marker = crate::StrokeMarker2d::arrow(
        crate::LogicalPixels::new(10.0).unwrap(),
        crate::LogicalPixels::new(12.0).unwrap(),
    );
    scene
        .try_styled_line(
            screen_to_world(254.0, 235.0),
            screen_to_world(258.0, 235.0),
            crate::StrokeStyle2d::logical(
                crate::LogicalPixels::new(10.0).unwrap(),
                Color::rgba(1.0, 1.0, 1.0, 0.5),
            )
            .with_cap(crate::StrokeCap2d::Round)
            .with_start_marker(marker)
            .with_end_marker(marker),
        )
        .unwrap();
    let identity = Arc::new(());
    let prepared = prepare_scene_resources(device, queue, identity, &scene)
        .expect("stroke pixel matrix should prepare");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine stroke pixel-matrix readback"),
        size: u64::from(ROW_BYTES) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine stroke pixel-matrix encoder"),
    });
    {
        let attachment_view = multisample_view.as_ref().unwrap_or(&target_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine stroke pixel-matrix pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target: multisample_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                    store: if multisample_view.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_vertex_buffer(0, prepared.vertex_buffer.slice(..));
        for batch in &prepared.draw_batches {
            pass.draw(batch.vertex_range.clone(), 0..1);
        }
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
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("stroke pixel-matrix submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("stroke pixel-matrix readback should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("stroke pixel-matrix callback")
        .expect("stroke pixel-matrix should map");
    let bytes = slice
        .get_mapped_range()
        .expect("stroke pixel-matrix mapped bytes");
    let pixel = |x: usize, y: usize| &bytes[y * ROW_BYTES as usize + x * 4..][..4];
    let mut covered = [[[0usize; 9]; 2]; 2];
    for (width_mode, turns) in covered.iter_mut().enumerate() {
        for (turn_direction, counts) in turns.iter_mut().enumerate() {
            let center_x = 64usize + (width_mode * 2 + turn_direction) * 128;
            for (row, count) in counts.iter_mut().enumerate() {
                let center_y = 12usize + row * 24;
                let mut maximum = 0u8;
                for y in center_y.saturating_sub(11)..=(center_y + 11).min(HEIGHT as usize - 1) {
                    for x in center_x - 22..=center_x + 22 {
                        let value = pixel(x, y)[0];
                        maximum = maximum.max(value);
                        *count += usize::from(value > 32);
                    }
                }
                assert!(
                    (175..=210).contains(&maximum),
                    "stroke matrix width_mode={width_mode} turn={turn_direction} row={row} has missing or multiply blended pixels: maximum={maximum}, samples={sample_count}"
                );
                assert!(
                    *count > 100,
                    "stroke matrix width_mode={width_mode} turn={turn_direction} row={row} did not rasterize enough pixels: {count}"
                );
            }
            for join in 0..3 {
                assert!(counts[3 + join] > counts[join]);
                assert!(counts[6 + join] > counts[join]);
            }
            for cap in 0..3 {
                assert!(
                    counts[cap * 3 + 1] > counts[cap * 3],
                    "miter/bevel coverage did not differ for width={width_mode} turn={turn_direction} cap={cap}: {counts:?}"
                );
                assert!(
                    counts[cap * 3 + 2] > counts[cap * 3],
                    "round/bevel coverage did not differ for width={width_mode} turn={turn_direction} cap={cap}: {counts:?}"
                );
            }
        }
    }
    for (turn_direction, (logical_rows, world_rows)) in
        covered[0].iter().zip(covered[1].iter()).enumerate()
    {
        for (row, (logical, world)) in logical_rows
            .iter()
            .copied()
            .zip(world_rows.iter().copied())
            .enumerate()
        {
            assert!(
                logical.abs_diff(world) <= 24,
                "logical/world stroke matrix diverged for turn={turn_direction} row={row}: {:?}",
                [logical, world]
            );
        }
    }
    for (width_mode, turns) in covered.iter().enumerate() {
        for (row, (clockwise, counterclockwise)) in turns[0]
            .iter()
            .copied()
            .zip(turns[1].iter().copied())
            .enumerate()
        {
            assert!(
                clockwise.abs_diff(counterclockwise) <= 24,
                "mirrored stroke matrix diverged for width={width_mode} row={row}: {:?}",
                [clockwise, counterclockwise]
            );
        }
    }
    let short_body = pixel(256, 235)[0];
    let short_start_marker = pixel(248, 235)[0];
    let short_end_marker = pixel(264, 235)[0];
    assert!((175..=210).contains(&short_body));
    assert!(short_body.abs_diff(short_start_marker) <= 8);
    assert!(short_body.abs_diff(short_end_marker) <= 8);
    assert!(pixel(241, 235)[0] < 10 && pixel(271, 235)[0] < 10);
    drop(bytes);
    readback.unmap();
    eprintln!(
        "sim-engine stroke pixel matrix: 36 mirrored cap/join/width cells + short dual markers, sample_count={sample_count}"
    );
}

async fn assert_gpu_large_center_circle(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    const EXTENT: u32 = 64;
    const ROW_BYTES: u32 = EXTENT * 4;
    let sample_count = preferred_sample_count(adapter, format);
    let PipelineResources {
        pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        ..
    } = create_pipeline(device, format, sample_count);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine large-center circle resolve target"),
        size: wgpu::Extent3d {
            width: EXTENT,
            height: EXTENT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let multisample = (sample_count > 1).then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine large-center circle multisample target"),
            size: wgpu::Extent3d {
                width: EXTENT,
                height: EXTENT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    });
    let multisample_view = multisample
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

    let center = Vec2::splat(1.0e20);
    let viewport = LogicalViewport::new(EXTENT as f32, EXTENT as f32).unwrap();
    let camera = Camera2d::new(center, 8.0).unwrap();
    let camera_uniform = CameraUniform::new(camera, viewport).unwrap();
    queue.write_buffer(
        &camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            center,
            1.0,
            ShapeStyle::fill_stroke(Color::WHITE, 2.0, Color::rgb8(255, 0, 0)),
        )
        .unwrap();
    let prepared = prepare_scene_resources(device, queue, Arc::new(()), &scene)
        .expect("large-center circle should prepare without degenerating");
    assert_eq!(prepared.tessellation_stats().rendered_command_count(), 1);
    assert_eq!(prepared.tessellation_stats().dropped_command_count(), 0);

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine large-center circle readback"),
        size: u64::from(ROW_BYTES) * u64::from(EXTENT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine large-center circle encoder"),
    });
    {
        let attachment_view = multisample_view.as_ref().unwrap_or(&target_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine large-center circle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target: multisample_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                    store: if multisample_view.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_vertex_buffer(0, prepared.vertex_buffer.slice(..));
        for batch in &prepared.draw_batches {
            pass.draw(batch.vertex_range.clone(), 0..1);
        }
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
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(EXTENT),
            },
        },
        wgpu::Extent3d {
            width: EXTENT,
            height: EXTENT,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("large-center circle submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("large-center circle readback should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("large-center circle callback")
        .expect("large-center circle should map");
    let bytes = slice
        .get_mapped_range()
        .expect("large-center circle mapped bytes");
    let channels = gpu_oracle_channel_indices(format);
    let pixel = |x: usize, y: usize| &bytes[y * ROW_BYTES as usize + x * 4..][..4];
    let center_pixel = pixel(32, 32);
    assert!(
        center_pixel[channels[0]] > 220
            && center_pixel[channels[1]] > 220
            && center_pixel[channels[2]] > 220,
        "camera-relative large-center circle fill did not rasterize: {center_pixel:?}"
    );
    let outside = pixel(48, 32);
    assert!(
        outside[channels[0]] < 10 && outside[channels[1]] < 10 && outside[channels[2]] < 10,
        "large-center circle exceeded its camera-relative bounds: {outside:?}"
    );
    drop(bytes);
    readback.unmap();
}

fn parse_gpu_oracle_surface_format(name: &str) -> Option<wgpu::TextureFormat> {
    match name {
        "Rgba8Unorm" => Some(wgpu::TextureFormat::Rgba8Unorm),
        "Rgba8UnormSrgb" => Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        "Bgra8Unorm" => Some(wgpu::TextureFormat::Bgra8Unorm),
        "Bgra8UnormSrgb" => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        _ => None,
    }
}

fn gpu_oracle_channel_indices(format: wgpu::TextureFormat) -> [usize; 4] {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
        _ => panic!("unsupported byte-channel oracle format: {format:?}"),
    }
}

#[test]
fn gpu_oracle_surface_format_preserves_production_channel_order() {
    assert_eq!(
        parse_gpu_oracle_surface_format("Rgba8UnormSrgb"),
        Some(wgpu::TextureFormat::Rgba8UnormSrgb)
    );
    assert_eq!(
        parse_gpu_oracle_surface_format("Bgra8UnormSrgb"),
        Some(wgpu::TextureFormat::Bgra8UnormSrgb)
    );
    assert_eq!(
        gpu_oracle_channel_indices(wgpu::TextureFormat::Rgba8UnormSrgb),
        [0, 1, 2, 3]
    );
    assert_eq!(
        gpu_oracle_channel_indices(wgpu::TextureFormat::Bgra8UnormSrgb),
        [2, 1, 0, 3]
    );
    assert_eq!(parse_gpu_oracle_surface_format("Rgba16Float"), None);
}

#[test]
fn offscreen_gpu_readback_verifies_camera_depth_and_clip_contract() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = if let Ok(required_pci_bus_id) =
            std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")
        {
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .await
                .into_iter()
                .find(|candidate| candidate.get_info().device_pci_bus_id == required_pci_bus_id)
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
                .ok()
        };
        let Some(adapter) = adapter else {
            assert_ne!(
                std::env::var("SIM_ENGINE_REQUIRE_GPU_TESTS").as_deref(),
                Ok("1"),
                "the required physical GPU adapter is unavailable"
            );
            return;
        };
        let adapter_info = adapter.get_info();
        let required_surface_format = std::env::var("SIM_ENGINE_GPU_SURFACE_FORMAT").ok();
        assert!(
            required_surface_format.is_some()
                || std::env::var("SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT").as_deref()
                    != Ok("1"),
            "SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=1 requires SIM_ENGINE_GPU_SURFACE_FORMAT"
        );
        let format = required_surface_format
            .as_deref()
            .map(|name| {
                parse_gpu_oracle_surface_format(name)
                    .unwrap_or_else(|| panic!("unsupported production surface format: {name}"))
            })
            .unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);
        let sample_count = preferred_sample_count(&adapter, format);
        if let Ok(expected) = std::env::var("SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT") {
            assert_eq!(
                sample_count,
                expected
                    .parse::<u32>()
                    .expect("SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT must be a u32"),
                "offscreen oracle MSAA differs from the production surface selection"
            );
        }
        if std::env::var("SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY").as_deref() == Ok("1") {
            assert_eq!(
                adapter_info.backend.to_str(),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_BACKEND").unwrap(),
            );
            assert_eq!(
                adapter_info.name,
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_NAME").unwrap(),
            );
            assert_eq!(
                format!("{:#06x}", adapter_info.vendor),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_VENDOR").unwrap(),
            );
            assert_eq!(
                format!("{:#06x}", adapter_info.device),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_DEVICE").unwrap(),
            );
            assert_eq!(
                adapter_info.device_pci_bus_id,
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID").unwrap(),
                "semantic oracle selected a different physical adapter instance"
            );
        }
        if let Ok(path) = std::env::var("SIM_ENGINE_GPU_EVIDENCE_PATH") {
            let clean = |value: &str| value.replace(['\n', '\r', '='], " ");
            let revision = std::env::var("SIM_ENGINE_RELEASE_SHA")
                .expect("GPU evidence requires SIM_ENGINE_RELEASE_SHA");
            let evidence = format!(
                "format_version=1\nvcs_sha={}\ncrate_version={}\nbackend={:?}\nname={}\ndevice_type={:?}\ndriver={}\ndriver_info={}\nvendor={:#06x}\ndevice={:#06x}\npci_bus_id={}\noracle_format={:?}\noracle_sample_count={}\n",
                clean(&revision),
                env!("CARGO_PKG_VERSION"),
                adapter_info.backend,
                clean(&adapter_info.name),
                adapter_info.device_type,
                clean(&adapter_info.driver),
                clean(&adapter_info.driver_info),
                adapter_info.vendor,
                adapter_info.device,
                clean(&adapter_info.device_pci_bus_id),
                format,
                sample_count,
            );
            std::fs::write(&path, evidence)
                .unwrap_or_else(|error| panic!("write GPU evidence to {path}: {error}"));
        }
        if std::env::var("SIM_ENGINE_REQUIRE_GPU_TESTS").as_deref() == Ok("1") {
            eprintln!(
                "sim-engine GPU evidence: name={:?}, type={:?}, backend={:?}, driver={:?}, driver_info={:?}, vendor={:#06x}, device={:#06x}, pci_bus_id={:?}, surface_format={:?}, sample_count={}",
                adapter_info.name,
                adapter_info.device_type,
                adapter_info.backend,
                adapter_info.driver,
                adapter_info.driver_info,
                adapter_info.vendor,
                adapter_info.device,
                adapter_info.device_pci_bus_id,
                format,
                sample_count,
            );
            if std::env::var("SIM_ENGINE_REQUIRE_VULKAN").as_deref() == Ok("1") {
                assert_eq!(
                    adapter_info.backend,
                    wgpu::Backend::Vulkan,
                    "SIM_ENGINE_REQUIRE_VULKAN=1 requires a Vulkan adapter"
                );
            }
        }
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
        assert_gpu_large_center_circle(&adapter, &device, &queue, format).await;
        assert_gpu_stroke_pixel_matrix(&adapter, &device, &queue, format).await;
        mesh3d::assert_gpu_depth_contract(&device, &queue, format);
        mesh3d::assert_gpu_scene_recovery_contract(
            &device,
            &queue,
            &recovery_device,
            &recovery_queue,
        );
        let vertex_limit = device.limits().max_buffer_size / std::mem::size_of::<Vertex>() as u64;
        if let Ok(first_invalid_capacity) = usize::try_from(vertex_limit.saturating_add(1)) {
            assert!(!buffer_capacity_fits::<Vertex>(
                &device,
                first_invalid_capacity
            ));
        }
        let PipelineResources {
            pipeline,
            target_pipeline: _,
            dynamic_pipeline: _,
            particle_pipeline: _,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines: _,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout: _,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        } = create_pipeline(&device, format, 1);
        let image_renderer = ImageRenderer::new(&device, format, 1);
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
        let stroke_dash = crate::StrokeDashPattern2d::new(&[4.0, 4.0], 0.0, 8).unwrap();
        scene
            .try_styled_line(
                Vec2::new(-20.0, 20.0),
                Vec2::new(4.0, 20.0),
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(4.0).unwrap(),
                    Color::WHITE,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_dash_pattern(stroke_dash),
            )
            .unwrap();
        let translucent_red = Color::rgba(1.0, 0.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(-26.0, -8.0),
                    Vec2::new(-18.0, -2.0),
                    Vec2::new(-10.0, -8.0),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_red,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Round),
            )
            .unwrap();
        let translucent_green = Color::rgba(0.0, 1.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(2.0, -8.0),
                    Vec2::new(10.0, -2.0),
                    Vec2::new(18.0, -8.0),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_green,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Bevel),
            )
            .unwrap();
        let translucent_yellow = Color::rgba(1.0, 1.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(10.0, 4.5),
                    Vec2::new(18.0, 11.4),
                    Vec2::new(26.0, 4.5),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_yellow,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Miter)
                .with_miter_limit(1.0)
                .unwrap(),
            )
            .unwrap();
        let translucent_blue = Color::rgba(0.0, 0.0, 1.0, 0.5);
        let arrow = crate::StrokeMarker2d::arrow(
            crate::LogicalPixels::new(8.0).unwrap(),
            crate::LogicalPixels::new(12.0).unwrap(),
        );
        scene
            .try_styled_line(
                Vec2::new(-24.0, -24.0),
                Vec2::new(6.0, -24.0),
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_blue,
                )
                .with_cap(crate::StrokeCap2d::Round)
                .with_end_marker(arrow),
            )
            .unwrap();
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
        let triangle_bytes = 3 * std::mem::size_of::<DynamicGpu>();
        let mut dynamic_source = DynamicMesh2d {
            renderer_identity: Arc::clone(&source_identity),
            vertex_buffer: Arc::new(create_dynamic_vertex_buffer(&device, 8)),
            geometry_extents: GeometryExtents::from_dynamic_vertices(&dynamic_vertices),
            vertices: dynamic_vertices,
            vertex_capacity: 8,
            budget: Some(DynamicMeshBudget::new(3, triangle_bytes, triangle_bytes).unwrap()),
        };
        queue.write_buffer(
            &dynamic_source.vertex_buffer,
            0,
            bytemuck::cast_slice(dynamic_source.vertices.as_slice()),
        );
        let original_dynamic_buffer = Arc::clone(&dynamic_source.vertex_buffer);
        let original_dynamic_vertices = dynamic_source.vertices.clone();
        assert_eq!(
            replace_dynamic_mesh_resources(
                &device,
                &queue,
                &mut dynamic_source,
                &[dynamic_vertex; 6],
            ),
            Err(DynamicMeshError::BudgetExceeded {
                resource: DynamicMeshBudgetResource::Vertices,
                limit: 3,
                actual: 6,
            })
        );
        assert!(Arc::ptr_eq(
            &dynamic_source.vertex_buffer,
            &original_dynamic_buffer
        ));
        assert_eq!(dynamic_source.vertices, original_dynamic_vertices);
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
        let heatmap_uniform = HeatmapUniform::new(0.0, 1.0, 2, 2, ScalarFieldSampling::Nearest);
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

        let image_pixels = vec![255, 0, 0, 255, 0, 255, 0, 128];
        let image_source = image::create_image_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            2,
            1,
            image_pixels,
            ImageBudget::new(2, 1, 8).unwrap(),
        )
        .expect("bounded image should upload");
        let sprite_region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(3.0, 4.0),
            LogicalViewport::new(8.0, 6.0).unwrap(),
        )
        .unwrap();
        let image_sprites = vec![
            ImageSprite2d::new(
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
                sprite_region,
                Color::WHITE,
            )
            .unwrap(),
        ];
        let image_batch = image::create_image_batch_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            &image_source,
            image_sprites.clone(),
            ImageBatchBudget::new(2, 1024).unwrap(),
        )
        .expect("bounded image batch should upload");
        let recovered_image = image::create_image_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            image_source.width(),
            image_source.height(),
            image_source.pixels().to_vec(),
            image_source.budget(),
        )
        .expect("image should migrate to a second device");
        let recovered_image_batch = image::create_image_batch_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &recovered_image,
            image_sprites,
            image_batch.budget(),
        )
        .expect("image batch should migrate to a second device");
        assert_eq!(recovered_image.pixels(), image_source.pixels());
        assert_eq!(recovered_image_batch.sprites(), image_batch.sprites());

        let atlas_entries = vec![
            GlyphAtlasEntry::new(
                GlyphId::new('μ' as u32),
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            ),
            GlyphAtlasEntry::new(
                GlyphId::new('Δ' as u32),
                ImageTexelRect::new(1, 0, 1, 1).unwrap(),
            ),
        ];
        let glyph_atlas = glyph::create_glyph_atlas_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            2,
            1,
            image_source.pixels().to_vec(),
            atlas_entries.clone(),
            GlyphAtlasBudget::new(ImageBudget::new(2, 1, 8).unwrap(), 4, 1024).unwrap(),
        )
        .expect("glyph atlas should upload");
        let glyphs = vec![
            PositionedGlyph2d::new(GlyphId::new('μ' as u32), sprite_region, Color::WHITE).unwrap(),
            PositionedGlyph2d::new(
                GlyphId::new('Δ' as u32),
                LogicalViewportRegion::new(
                    LogicalScreenPosition::new(11.0, 4.0),
                    LogicalViewport::new(8.0, 6.0).unwrap(),
                )
                .unwrap(),
                Color::WHITE,
            )
            .unwrap(),
        ];
        let glyph_run = glyph::create_glyph_run_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            &glyph_atlas,
            glyphs.clone(),
            GlyphRunBudget::new(4, 4096).unwrap(),
        )
        .expect("glyph run should upload");
        let recovered_atlas = glyph::create_glyph_atlas_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            2,
            1,
            glyph_atlas.image.pixels().to_vec(),
            atlas_entries,
            glyph_atlas.budget(),
        )
        .expect("glyph atlas should migrate to a second device");
        let recovered_glyph_run = glyph::create_glyph_run_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &recovered_atlas,
            glyphs,
            glyph_run.budget(),
        )
        .expect("glyph run should migrate to a second device");
        assert_eq!(recovered_atlas.entries(), glyph_atlas.entries());
        assert_eq!(recovered_glyph_run.glyphs(), glyph_run.glyphs());
        assert_eq!(recovered_glyph_run.statistics().rendered_quads(), 2);

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
                let scissor = batch.screen_clip.map_or(
                    ScissorRect {
                        x: 0,
                        y: 0,
                        width: 64,
                        height: 64,
                    },
                    |clip| screen_clip_to_scissor(clip, viewport, 1.0).unwrap(),
                );
                pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
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
        let [red, green, blue, _alpha] = gpu_oracle_channel_indices(format);
        assert!(pixel(33, 29)[0] > 200, "camera/depth pixel was not drawn");
        assert!(pixel(33, 34)[0] < 10, "clip failed to remove outside pixel");
        assert!(pixel(14, 14)[0] > 200, "styled dash body was not drawn");
        assert!(pixel(18, 14)[0] < 10, "styled dash gap was not preserved");
        assert!(pixel(22, 14)[0] > 200, "styled dash phase did not repeat");
        let assert_uniform_translucency = |body: &[u8], detail: &[u8], channel: usize, name| {
            assert!(
                (160..=210).contains(&body[channel]),
                "{name} body did not preserve half-alpha linear color: {body:?}"
            );
            assert!(
                body[channel].abs_diff(detail[channel]) <= 8,
                "{name} detail was alpha-blended more than once: body={body:?}, detail={detail:?}"
            );
        };
        assert_uniform_translucency(pixel(10, 37), pixel(14, 32), red, "round join");
        assert_uniform_translucency(pixel(38, 37), pixel(42, 32), green, "bevel join");
        assert_uniform_translucency(pixel(46, 25), pixel(50, 20), red, "miter fallback");
        assert!(
            pixel(50, 17)[red] < 10 && pixel(50, 17)[green] < 10,
            "over-limit miter spike was not replaced by bevel geometry: {:?}",
            pixel(50, 17)
        );
        assert_uniform_translucency(pixel(20, 53), pixel(42, 53), blue, "arrow marker");
        assert_uniform_translucency(pixel(20, 53), pixel(6, 53), blue, "round cap");
        assert!(
            pixel(48, 53)[blue] < 10,
            "the endpoint cap protruded beyond the arrow tip: {:?}",
            pixel(48, 53)
        );
        assert!(
            pixel(50, 48)[red] > 200,
            "instanced particle center was not drawn"
        );
        assert!(
            pixel(50, 48)[green] < 10,
            "instanced particle color was not applied"
        );
        assert!(
            pixel(56, 54)[red] < 10,
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
            bytemuck::bytes_of(&CompositeUniform::full_surface(1.0)),
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

        let composition_region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalViewport::new(1.0, 1.0).unwrap(),
        )
        .unwrap();
        let region_uniform = CompositeUniform::in_region(
            1.0,
            composition_region,
            LogicalViewport::new(2.0, 2.0).unwrap(),
        )
        .unwrap();
        queue.write_buffer(&composition_uniform, 0, bytemuck::bytes_of(&region_uniform));
        let mut region_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen region composition encoder"),
        });
        {
            let mut pass = region_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen region composition pass"),
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
        region_encoder.copy_texture_to_buffer(
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
        queue.submit([region_encoder.finish()]);
        let region_slice = composition_readback.slice(..);
        let (region_sender, region_receiver) = mpsc::channel();
        region_slice.map_async(wgpu::MapMode::Read, move |result| {
            region_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen region composition should complete");
        region_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen region composition callback")
            .expect("offscreen region composition should map");
        let region_bytes = region_slice
            .get_mapped_range()
            .expect("offscreen region composition bytes");
        assert!(region_bytes[0] > 180, "region top-left pixel was not drawn");
        assert!(
            region_bytes[4] < 8 && region_bytes[256] < 8,
            "region composition escaped its logical destination"
        );
        drop(region_bytes);
        composition_readback.unmap();

        queue.write_buffer(
            &target_composition_pipelines.secondary_uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(0.5)),
        );
        queue.write_buffer(
            &target_composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(0.5)),
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
            bytemuck::bytes_of(&HeatmapUniform::new(
                0.0,
                1.0,
                2,
                2,
                ScalarFieldSampling::Linear,
            )),
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

        let image_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let image_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen atlas image"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &image_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 0, 0, 255, 0, 255, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let image_view = image_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image_uniform = ImageUniform {
            destination: [0.5, 1.0, -0.5, 0.0],
            uv_rect: [0.25, 0.5, 0.25, 0.5],
            tint: [1.0, 1.0, 1.0, 0.5],
            world_clip_x: [0.0; 4],
            world_clip_y: [0.0; 4],
            world_mode: [0.0; 4],
        };
        let image_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&image_uniform_buffer, 0, bytemuck::bytes_of(&image_uniform));
        let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen image bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let world_image_uniform = ImageUniform {
            destination: [0.0; 4],
            uv_rect: [0.75, 0.5, 0.75, 0.5],
            tint: [1.0, 1.0, 1.0, 0.5],
            world_clip_x: [0.0, 1.0, 0.0, 1.0],
            world_clip_y: [1.0, 1.0, -1.0, -1.0],
            world_mode: [1.0, 0.0, 0.0, 0.0],
        };
        let world_image_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen world image uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &world_image_uniform_buffer,
            0,
            bytemuck::bytes_of(&world_image_uniform),
        );
        let world_image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen world image bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: world_image_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let image_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen image target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let image_target_view = image_target.create_view(&wgpu::TextureViewDescriptor::default());
        let image_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut image_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen image encoder"),
        });
        {
            let mut pass = image_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen image pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image_target_view,
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
            pass.set_pipeline(&image_renderer.pipeline);
            pass.set_bind_group(0, &image_bind_group, &[]);
            pass.draw(0..6, 0..1);
            pass.set_bind_group(0, &world_image_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        image_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &image_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([image_encoder.finish()]);
        let image_slice = image_readback.slice(..);
        let (image_sender, image_receiver) = mpsc::channel();
        image_slice.map_async(wgpu::MapMode::Read, move |result| {
            image_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen image readback should complete");
        image_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen image callback")
            .expect("offscreen image should map");
        if let Some(error) = image_scope.pop().await {
            panic!("offscreen image validation failed: {error}");
        }
        let image_bytes = image_slice
            .get_mapped_range()
            .expect("offscreen image bytes");
        assert!(image_bytes[red] > 180 && image_bytes[red] < 195);
        assert!(
            image_bytes[green] < 8,
            "screen image atlas sampling bled into the next texel"
        );
        assert!(
            image_bytes[4 + red] < 8,
            "world image atlas sampling bled into the previous texel"
        );
        assert!(image_bytes[4 + green] > 180 && image_bytes[4 + green] < 195);
        drop(image_bytes);
        image_readback.unmap();

        let image_batch_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let image_batch_uniform = ImageUniform {
            destination: [1.0, -2.0, -1.0, 1.0],
            uv_rect: [0.0, 0.0, 0.0, 0.0],
            tint: Color::WHITE.to_array(),
            world_clip_x: [0.0; 4],
            world_clip_y: [0.0; 4],
            world_mode: [0.0; 4],
        };
        let image_batch_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image batch uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &image_batch_uniform_buffer,
            0,
            bytemuck::bytes_of(&image_batch_uniform),
        );
        let image_batch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen image batch bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_batch_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let image_instances = [
            image::ImageInstance {
                destination: [0.0, 0.0, 1.0, 1.0],
                uv_rect: [0.25, 0.5, 0.25, 0.5],
                tint: Color::WHITE.to_array(),
            },
            image::ImageInstance {
                destination: [1.0, 0.0, 1.0, 1.0],
                uv_rect: [0.75, 0.5, 0.75, 0.5],
                tint: Color::WHITE.with_alpha(0.5).to_array(),
            },
        ];
        let image_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image batch instances"),
            size: std::mem::size_of_val(&image_instances) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &image_instance_buffer,
            0,
            bytemuck::cast_slice(&image_instances),
        );
        let mut image_batch_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine offscreen image batch encoder"),
            });
        {
            let mut pass = image_batch_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen image batch pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image_target_view,
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
            pass.set_pipeline(&image_renderer.batch_pipeline);
            pass.set_bind_group(0, &image_batch_bind_group, &[]);
            pass.set_vertex_buffer(0, image_instance_buffer.slice(..));
            pass.draw(0..6, 0..2);
        }
        image_batch_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &image_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([image_batch_encoder.finish()]);
        let image_batch_slice = image_readback.slice(..);
        let (image_batch_sender, image_batch_receiver) = mpsc::channel();
        image_batch_slice.map_async(wgpu::MapMode::Read, move |result| {
            image_batch_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen image batch readback should complete");
        image_batch_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen image batch callback")
            .expect("offscreen image batch should map");
        if let Some(error) = image_batch_scope.pop().await {
            panic!("offscreen image batch validation failed: {error}");
        }
        let image_batch_bytes = image_batch_slice
            .get_mapped_range()
            .expect("offscreen image batch bytes");
        assert!(image_batch_bytes[red] > 247 && image_batch_bytes[green] < 8);
        assert!(image_batch_bytes[4 + red] < 8);
        assert!(image_batch_bytes[4 + green] > 180 && image_batch_bytes[4 + green] < 195);
        drop(image_batch_bytes);
        image_readback.unmap();
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
fn renderer_options_bound_device_recovery_quarantine() {
    let options = WgpuRendererOptions::new(RendererPresentMode::Vsync, 1.0).unwrap();
    assert_eq!(options.max_quarantined_devices(), 4);
    assert_eq!(
        options
            .with_max_quarantined_devices(2)
            .unwrap()
            .max_quarantined_devices(),
        2
    );
    assert_eq!(
        options.with_max_quarantined_devices(0),
        Err(RendererConfigurationError::InvalidRecoveryLimit { limit: 0 })
    );
    assert_eq!(
        options.with_max_quarantined_devices(9),
        Err(RendererConfigurationError::InvalidRecoveryLimit { limit: 9 })
    );
    assert!(recovery_quarantine_has_capacity(3, 4));
    assert!(!recovery_quarantine_has_capacity(4, 4));
    assert!(!recovery_quarantine_has_capacity(usize::MAX, 4));
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
fn renderer_present_modes_select_concrete_surface_fallbacks() {
    assert_eq!(
        select_surface_present_mode(
            RendererPresentMode::Vsync,
            &[wgpu::PresentMode::Immediate, wgpu::PresentMode::Fifo]
        ),
        RendererSurfacePresentMode::Fifo
    );
    assert_eq!(
        select_surface_present_mode(
            RendererPresentMode::NoVsync,
            &[
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ]
        ),
        RendererSurfacePresentMode::Immediate
    );
    assert_eq!(
        select_surface_present_mode(
            RendererPresentMode::NoVsync,
            &[wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox]
        ),
        RendererSurfacePresentMode::Mailbox
    );
    assert_eq!(
        select_surface_present_mode(RendererPresentMode::NoVsync, &[wgpu::PresentMode::Fifo]),
        RendererSurfacePresentMode::Fifo
    );
    assert!(RendererSurfacePresentMode::Mailbox.is_refresh_synchronized());
    assert!(!RendererSurfacePresentMode::Immediate.is_refresh_synchronized());
}

#[test]
fn pre_present_notification_only_paces_synchronized_surface_modes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let notify = move || {
        observed.fetch_add(1, Ordering::Relaxed);
    };

    invoke_pre_present_notify(RendererSurfacePresentMode::Immediate, Some(&notify));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    for mode in [
        RendererSurfacePresentMode::Mailbox,
        RendererSurfacePresentMode::Fifo,
        RendererSurfacePresentMode::FifoRelaxed,
    ] {
        invoke_pre_present_notify(mode, Some(&notify));
    }
    assert_eq!(calls.load(Ordering::Relaxed), 3);

    invoke_pre_present_notify(RendererSurfacePresentMode::Fifo, None);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}
