//! Scene admission, ordering, styles and rollback regressions.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::MAX_STROKE_DASH_SUBSEGMENTS;
use crate::{
    Color, DrawCommand, Fill, Layer, LinearGradient, LogicalScreenPosition, LogicalScreenVector,
    RadialGradient, Rect, Scene, SceneBudget, SceneBudgetResource, SceneCommand, SceneError,
    ScenePrimitive, ScreenClipRect, ShapeStyle, StrokeCap2d, StrokeDashPattern2d, StrokeJoin2d,
    StrokeMarker2d, StrokeStyle2d, StrokeStyleError, Vec2,
};
use crate::{LogicalPixels, WorldLength};

const UNLIMITED: usize = usize::MAX;

const fn budget_for(
    commands: usize,
    points: usize,
    vertices: usize,
    retained_bytes: usize,
    upload_bytes: usize,
    batches: usize,
) -> SceneBudget {
    SceneBudget::new(
        commands,
        points,
        vertices,
        retained_bytes,
        UNLIMITED,
        upload_bytes,
        batches,
    )
}

#[test]
fn scene_collects_visual_commands_only() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

    assert_eq!(scene.commands.len(), 1);
}

#[test]
fn scene_keeps_commands_sorted_by_layer() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.circle_on_layer(
        Layer::FOREGROUND,
        Vec2::ZERO,
        10.0,
        ShapeStyle::filled(Color::WHITE),
    );
    scene.line_on_layer(Layer::BACKGROUND, Vec2::ZERO, Vec2::X, 1.0, Color::WHITE);

    assert!(matches!(scene.commands[0].command, DrawCommand::Line(_)));
    assert!(matches!(scene.commands[1].command, DrawCommand::Circle(_)));
}

#[test]
fn scene_budget_rejection_is_atomic_and_counted() {
    let budget = budget_for(1, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED);
    let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
    scene
        .try_circle(Vec2::new(1.0, 2.0), 3.0, ShapeStyle::filled(Color::WHITE))
        .unwrap();

    let before_order = scene.commands()[0].order();
    let result = scene.try_circle(Vec2::new(4.0, 5.0), 6.0, ShapeStyle::filled(Color::WHITE));

    assert_eq!(
        result,
        Err(SceneError::BudgetExceeded {
            resource: SceneBudgetResource::Commands,
            limit: 1,
            requested: 2,
        })
    );
    assert_eq!(scene.command_count(), 1);
    assert_eq!(scene.commands()[0].order(), before_order);
    assert_eq!(scene.statistics().requested_commands(), 2);
    assert_eq!(scene.statistics().accepted_commands(), 1);
    assert_eq!(scene.statistics().rejected_commands(), 1);
}

#[test]
fn scene_budget_enforces_every_resource_at_insertion() {
    let mut reference = Scene::new(Color::BLACK).unwrap();
    reference
        .try_polyline(
            vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)],
            2.0,
            Color::WHITE,
        )
        .unwrap();
    let usage = reference.statistics();
    let cases = [
        (
            SceneBudgetResource::Points,
            budget_for(
                UNLIMITED,
                usage.retained_points() - 1,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
            ),
            usage.retained_points(),
        ),
        (
            SceneBudgetResource::TessellatedVertices,
            budget_for(
                UNLIMITED,
                UNLIMITED,
                usage.estimated_tessellated_vertices() - 1,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
            ),
            usage.estimated_tessellated_vertices(),
        ),
        (
            SceneBudgetResource::RetainedBytes,
            budget_for(
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                usage.retained_bytes() - 1,
                UNLIMITED,
                UNLIMITED,
            ),
            usage.retained_bytes(),
        ),
        (
            SceneBudgetResource::UploadBytes,
            budget_for(
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                usage.estimated_upload_bytes() - 1,
                UNLIMITED,
            ),
            usage.estimated_upload_bytes(),
        ),
        (
            SceneBudgetResource::DrawBatches,
            budget_for(
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                UNLIMITED,
                usage.estimated_draw_batches() - 1,
            ),
            usage.estimated_draw_batches(),
        ),
    ];

    for (resource, budget, requested) in cases {
        let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
        let result = scene.try_polyline(
            vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)],
            2.0,
            Color::WHITE,
        );
        assert_eq!(
            result,
            Err(SceneError::BudgetExceeded {
                resource,
                limit: requested - 1,
                requested,
            })
        );
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.statistics().accepted_commands(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 1);
    }
}

#[test]
fn exact_scene_budget_is_accepted_and_clear_preserves_limits() {
    let mut reference = Scene::new(Color::BLACK).unwrap();
    reference
        .try_line(Vec2::ZERO, Vec2::X, 2.0, Color::WHITE)
        .unwrap();
    let usage = reference.statistics();
    let budget = budget_for(
        usage.accepted_commands(),
        usage.retained_points(),
        usage.estimated_tessellated_vertices(),
        usage.retained_bytes(),
        usage.estimated_upload_bytes(),
        usage.estimated_draw_batches(),
    );
    let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();

    scene
        .try_line(Vec2::ZERO, Vec2::X, 2.0, Color::WHITE)
        .unwrap();
    assert_eq!(scene.statistics(), usage);
    scene.clear();

    assert_eq!(scene.budget(), Some(budget));
    assert_eq!(scene.statistics(), Default::default());
    assert_eq!(scene.command_count(), 0);
}

#[test]
fn scene_allocation_budget_bounds_spare_command_capacity_atomically() {
    let command_bytes = size_of::<SceneCommand>();
    let too_small = SceneBudget::new(
        1,
        UNLIMITED,
        UNLIMITED,
        command_bytes,
        command_bytes - 1,
        UNLIMITED,
        UNLIMITED,
    );
    let mut rejected = Scene::with_budget(Color::BLACK, too_small).unwrap();
    let error = rejected
        .try_circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE))
        .unwrap_err();
    assert!(matches!(
        error,
        SceneError::BudgetExceeded {
            resource: SceneBudgetResource::AllocationBytes,
            limit,
            requested,
        } if limit == command_bytes - 1 && requested >= command_bytes
    ));
    assert_eq!(rejected.command_count(), 0);
    assert_eq!(rejected.allocation_bytes(), 0);

    let bounded = SceneBudget::new(
        4,
        UNLIMITED,
        UNLIMITED,
        4 * command_bytes,
        4 * command_bytes,
        UNLIMITED,
        UNLIMITED,
    );
    let mut scene = Scene::with_budget(Color::BLACK, bounded).unwrap();
    scene
        .try_circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    assert!(scene.allocation_bytes() <= bounded.max_allocation_bytes());
    scene.clear();
    assert!(scene.allocation_bytes() <= bounded.max_allocation_bytes());
}

#[test]
fn batch_allocation_budget_rejects_before_consuming_unbounded_input() {
    let budget = SceneBudget::new(
        usize::MAX,
        UNLIMITED,
        UNLIMITED,
        UNLIMITED,
        1,
        UNLIMITED,
        UNLIMITED,
    );
    let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
    let consumed = std::cell::Cell::new(0usize);
    let command = DrawCommand::circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)).unwrap();
    let commands = std::iter::from_fn(|| {
        consumed.set(consumed.get().saturating_add(1));
        Some((Layer::DEFAULT, command.clone()))
    });

    assert!(matches!(
        scene.try_extend_to_layers(commands),
        Err(SceneError::BudgetExceeded {
            resource: SceneBudgetResource::AllocationBytes,
            limit: 1,
            ..
        })
    ));
    assert_eq!(consumed.get(), 1);
    assert_eq!(scene.command_count(), 0);
    assert_eq!(scene.allocation_bytes(), 0);
}

#[test]
fn batch_insertion_sorts_once_and_is_atomic_on_budget_failure() {
    let budget = budget_for(4, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED);
    let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
    scene
        .try_extend_to_layers([
            (
                Layer::FOREGROUND,
                DrawCommand::circle(Vec2::new(1.0, 0.0), 1.0, ShapeStyle::filled(Color::WHITE))
                    .unwrap(),
            ),
            (
                Layer::BACKGROUND,
                DrawCommand::line(Vec2::ZERO, Vec2::X, 1.0, Color::WHITE).unwrap(),
            ),
            (
                Layer::FOREGROUND,
                DrawCommand::circle(Vec2::new(2.0, 0.0), 1.0, ShapeStyle::filled(Color::WHITE))
                    .unwrap(),
            ),
        ])
        .unwrap();

    assert_eq!(scene.command_count(), 3);
    assert_eq!(scene.commands()[0].layer(), Layer::BACKGROUND);
    assert_eq!(scene.commands()[1].order(), 0);
    assert_eq!(scene.commands()[2].order(), 2);

    let result = scene.try_extend_to_layers([
        (
            Layer::DEFAULT,
            DrawCommand::circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)).unwrap(),
        ),
        (
            Layer::DEFAULT,
            DrawCommand::circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE)).unwrap(),
        ),
    ]);
    assert_eq!(
        result,
        Err(SceneError::BudgetExceeded {
            resource: SceneBudgetResource::Commands,
            limit: 4,
            requested: 5,
        })
    );
    assert_eq!(scene.command_count(), 3);
    assert_eq!(scene.statistics().requested_commands(), 5);
    assert_eq!(scene.statistics().accepted_commands(), 3);
    assert_eq!(scene.statistics().rejected_commands(), 2);
}

#[test]
fn linear_gradient_samples_along_axis() {
    let gradient = LinearGradient::new(Vec2::ZERO, Vec2::X, Color::BLACK, Color::WHITE);

    let sampled = Fill::LinearGradient(gradient).color_at(Vec2::new(0.5, 4.0));

    assert!((sampled.red() - 0.5).abs() < 0.001);
    assert!((sampled.green() - 0.5).abs() < 0.001);
    assert!((sampled.blue() - 0.5).abs() < 0.001);
}

#[test]
fn extreme_gradients_sample_without_f32_overflow() {
    let gradient = LinearGradient::new(
        Vec2::new(-f32::MAX, 0.0),
        Vec2::new(f32::MAX, 0.0),
        Color::BLACK,
        Color::WHITE,
    );
    let mut scene = Scene::new(Color::BLACK).unwrap();

    assert!(
        scene
            .try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle::filled_with(Fill::LinearGradient(gradient)),
            )
            .is_ok()
    );
    assert_eq!(
        gradient.color_at(Vec2::ZERO),
        Color::rgba(0.5, 0.5, 0.5, 1.0)
    );

    let radial = RadialGradient::new(
        Vec2::splat(f32::MAX),
        0.0,
        f32::MAX,
        Color::BLACK,
        Color::WHITE,
    );
    assert_eq!(radial.color_at(Vec2::splat(-f32::MAX)), Color::WHITE);
}

#[test]
fn overflowing_circle_and_shadow_are_rejected_at_scene_boundary() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert_eq!(
        scene.try_circle(
            Vec2::splat(f32::MAX),
            f32::MAX,
            ShapeStyle::filled(Color::WHITE),
        ),
        Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle))
    );
    assert_eq!(
        scene.try_circle(
            Vec2::ZERO,
            1.0,
            ShapeStyle::filled(Color::WHITE).with_shadow(super::Shadow::new(
                LogicalScreenVector::new(0.0, 0.0),
                f32::MAX,
                Color::WHITE,
            )),
        ),
        Err(SceneError::InvalidShadow(ScenePrimitive::Circle))
    );
    assert_eq!(
        scene.try_line(
            Vec2::new(-f32::MAX, 0.0),
            Vec2::new(f32::MAX, 0.0),
            1.0,
            Color::WHITE,
        ),
        Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line))
    );
    assert_eq!(
        scene.try_rect(
            Rect::new(Vec2::new(-f32::MAX, 0.0), Vec2::new(f32::MAX, 1.0),),
            0.0,
            ShapeStyle::filled(Color::WHITE),
        ),
        Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect))
    );
}

#[test]
fn commands_capture_temporary_screen_clip() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let outer = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(10.0, 20.0),
        LogicalScreenVector::new(100.0, 80.0),
    )
    .unwrap();
    let inner = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(50.0, 0.0),
        LogicalScreenVector::new(100.0, 50.0),
    )
    .unwrap();

    scene
        .with_screen_clip(outer, |scene| {
            scene
                .with_screen_clip(inner, |scene| {
                    scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));
                })
                .unwrap();
        })
        .unwrap();
    scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

    assert_eq!(
        scene.commands[0].screen_clip,
        Some(
            ScreenClipRect::new(
                LogicalScreenPosition::new(50.0, 20.0),
                LogicalScreenPosition::new(110.0, 50.0),
            )
            .unwrap(),
        )
    );
    assert_eq!(scene.commands[1].screen_clip, None);
}

#[test]
fn scene_rejects_invalid_primitives_before_ordering() {
    let mut scene = Scene::new(Color::BLACK).unwrap();

    assert!(!scene.circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)));
    assert!(!scene.line(Vec2::ZERO, Vec2::X, f32::NAN, Color::WHITE));
    assert!(!scene.polyline(vec![Vec2::ZERO], 1.0, Color::WHITE));
    assert_eq!(scene.command_count(), 0);

    assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
    assert_eq!(scene.commands()[0].order(), 0);
}

#[test]
fn bounded_stroke_configuration_rejects_invalid_patterns_and_miters() {
    assert_eq!(
        StrokeDashPattern2d::new(&[1.0], 0.0, 4),
        Err(StrokeStyleError::InvalidDashElementCount)
    );
    assert_eq!(
        StrokeDashPattern2d::new(&[1.0, 0.0], 0.0, 4),
        Err(StrokeStyleError::InvalidDashLength)
    );
    assert_eq!(
        StrokeDashPattern2d::new(&[1.0, 1.0], f32::NAN, 4),
        Err(StrokeStyleError::InvalidDashPhase)
    );
    assert_eq!(
        StrokeDashPattern2d::new(&[1.0, 1.0], 0.0, 0),
        Err(StrokeStyleError::InvalidDashExpansionLimit)
    );
    assert_eq!(
        StrokeDashPattern2d::new(&[1.0, 1.0], 0.0, MAX_STROKE_DASH_SUBSEGMENTS + 1,),
        Err(StrokeStyleError::InvalidDashExpansionLimit)
    );
    assert_eq!(
        StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
            .with_miter_limit(0.5),
        Err(StrokeStyleError::InvalidMiterLimit)
    );
}

#[test]
fn dash_boundaries_must_be_representable_at_the_source_coordinate_scale() {
    let dash = StrokeDashPattern2d::new(&[f32::MIN_POSITIVE, f32::MIN_POSITIVE], 0.0, 8).unwrap();
    let style = StrokeStyle2d::logical(LogicalPixels::new(1.0).unwrap(), Color::WHITE)
        .with_dash_pattern(dash);

    assert!(matches!(
        DrawCommand::styled_line(Vec2::ZERO, Vec2::new(f32::MAX, 0.0), style),
        Err(SceneError::UnrepresentableStrokePattern(
            ScenePrimitive::Line,
        ))
    ));
}

#[test]
fn dash_expansion_is_rejected_atomically_at_the_scene_boundary() {
    let dash = StrokeDashPattern2d::new(&[2.0, 2.0], 0.0, 2).unwrap();
    let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
        .with_cap(StrokeCap2d::Butt)
        .with_dash_pattern(dash);
    let mut scene = Scene::new(Color::BLACK).unwrap();

    assert_eq!(
        scene.try_styled_line(Vec2::ZERO, Vec2::new(10.0, 0.0), style),
        Err(SceneError::StrokeExpansionLimitExceeded {
            primitive: ScenePrimitive::Line,
            limit: 2,
            required: 3,
        })
    );
    assert_eq!(scene.command_count(), 0);
    assert_eq!(scene.statistics().requested_commands(), 1);
    assert_eq!(scene.statistics().rejected_commands(), 1);
}

#[test]
fn scene_statistics_group_requested_accepted_and_rejected_primitives() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE))
        .unwrap();
    assert_eq!(
        scene.try_line(Vec2::ZERO, Vec2::ZERO, 1.0, Color::WHITE),
        Err(SceneError::DegenerateGeometry(ScenePrimitive::Line))
    );
    scene
        .try_polyline(vec![Vec2::ZERO, Vec2::X], 1.0, Color::WHITE)
        .unwrap();

    let statistics = scene.statistics();
    assert_eq!(statistics.requested_by_primitive().circles(), 1);
    assert_eq!(statistics.requested_by_primitive().lines(), 1);
    assert_eq!(statistics.requested_by_primitive().polylines(), 1);
    assert_eq!(statistics.requested_by_primitive().total(), 3);
    assert_eq!(statistics.accepted_by_primitive().circles(), 1);
    assert_eq!(statistics.accepted_by_primitive().polylines(), 1);
    assert_eq!(statistics.accepted_by_primitive().total(), 2);
    assert_eq!(statistics.rejected_by_primitive().lines(), 1);
    assert_eq!(statistics.rejected_by_primitive().total(), 1);
}

#[test]
fn rich_stroke_values_are_reusable_and_world_overflow_is_rejected() {
    let marker = StrokeMarker2d::arrow(
        LogicalPixels::new(8.0).unwrap(),
        LogicalPixels::new(6.0).unwrap(),
    );
    let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
        .with_cap(StrokeCap2d::Square)
        .with_join(StrokeJoin2d::Bevel)
        .with_start_marker(marker)
        .with_end_marker(marker);
    let command =
        DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 1.0)], style)
            .unwrap();
    assert_eq!(command.estimated_tessellated_vertices(), 24);

    let unsafe_world = StrokeStyle2d::world(WorldLength::new(f32::MAX).unwrap(), Color::WHITE);
    assert!(matches!(
        DrawCommand::styled_line(Vec2::ZERO, Vec2::X, unsafe_world),
        Err(SceneError::InvalidStroke(ScenePrimitive::Line))
    ));

    let unsafe_logical =
        StrokeStyle2d::logical(LogicalPixels::new(f32::MAX).unwrap(), Color::WHITE);
    assert!(matches!(
        DrawCommand::styled_line(Vec2::ZERO, Vec2::X, unsafe_logical),
        Err(SceneError::InvalidStroke(ScenePrimitive::Line))
    ));
}

#[test]
fn polyline_rejects_zero_segments_and_numerically_ambiguous_turns() {
    let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE);
    assert!(matches!(
        DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::ZERO, Vec2::X], style),
        Err(SceneError::DegenerateGeometry(ScenePrimitive::Polyline))
    ));
    assert!(matches!(
        DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::ZERO], style),
        Err(SceneError::DegenerateStrokeTurn {
            primitive: ScenePrimitive::Polyline,
            vertex_index: 1,
        })
    ));
    assert!(matches!(
        DrawCommand::styled_polyline(
            vec![Vec2::ZERO, Vec2::X, Vec2::new(0.0, 0.000_000_1)],
            style,
        ),
        Err(SceneError::DegenerateStrokeTurn {
            primitive: ScenePrimitive::Polyline,
            vertex_index: 1,
        })
    ));
    assert!(matches!(
        DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 0.000_05)], style,),
        Err(SceneError::DegenerateStrokeTurn {
            primitive: ScenePrimitive::Polyline,
            vertex_index: 1,
        })
    ));
}

#[test]
fn background_and_clip_are_validated_when_created() {
    assert!(matches!(
        Scene::new(Color::rgba(f32::NAN, 0.0, 0.0, 1.0)),
        Err(SceneError::InvalidBackground)
    ));
    assert!(matches!(
        Scene::new(Color::rgb(1.01, 0.0, 0.0)),
        Err(SceneError::InvalidBackground)
    ));
    assert_eq!(
        ScreenClipRect::new(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(0.0, 10.0),
        ),
        Err(SceneError::InvalidScreenClip)
    );
    assert_eq!(
        ScreenClipRect::new(
            LogicalScreenPosition::new(f32::NAN, 0.0),
            LogicalScreenPosition::new(10.0, 10.0),
        ),
        Err(SceneError::InvalidScreenClip)
    );
}

#[test]
fn render_bound_scene_colors_must_be_normalized() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert_eq!(
        scene.try_circle(
            Vec2::ZERO,
            1.0,
            ShapeStyle::filled(Color::rgb(2.0, 0.0, 0.0)),
        ),
        Err(SceneError::InvalidFill(ScenePrimitive::Circle))
    );
    assert_eq!(
        scene.try_line(Vec2::ZERO, Vec2::X, 1.0, Color::rgba(0.0, 0.0, 0.0, -0.01),),
        Err(SceneError::InvalidStroke(ScenePrimitive::Line))
    );
    assert_eq!(scene.command_count(), 0);
}

#[test]
fn depth_does_not_reorder_commands_within_a_layer() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .with_depth(4.0, |scene| {
            scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));
        })
        .unwrap();
    scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));

    assert_eq!(scene.commands()[0].depth(), 4.0);
    assert_eq!(scene.commands()[1].depth(), 0.0);
}

#[test]
fn allocation_bytes_preserve_spare_command_capacity_after_clear() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_polyline(
            vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)],
            1.0,
            Color::WHITE,
        )
        .unwrap();
    let allocated = scene.allocation_bytes();
    assert!(allocated >= size_of::<SceneCommand>() + 3 * size_of::<Vec2>());
    scene.clear();
    assert_eq!(scene.statistics().retained_bytes(), 0);
    assert_eq!(
        scene.allocation_bytes(),
        scene.commands.capacity() * size_of::<SceneCommand>()
    );
    assert!(scene.allocation_bytes() > 0);
}

#[test]
fn clone_recomputes_capacity_based_polyline_accounting() {
    let mut points = Vec::with_capacity(1_024);
    points.extend([Vec2::ZERO, Vec2::X]);
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene.try_polyline(points, 1.0, Color::WHITE).unwrap();

    let cloned = scene.clone();
    let actual_owned = cloned
        .commands
        .iter()
        .map(|command| command.command.owned_allocation_bytes())
        .fold(0usize, usize::saturating_add);
    assert_eq!(cloned.owned_payload_bytes, actual_owned);
    assert_eq!(
        cloned.statistics.retained_bytes(),
        cloned
            .commands
            .len()
            .saturating_mul(size_of::<SceneCommand>())
            .saturating_add(actual_owned)
    );
    assert_eq!(
        cloned.allocation_bytes(),
        cloned
            .commands
            .capacity()
            .saturating_mul(size_of::<SceneCommand>())
            .saturating_add(actual_owned)
    );
}

#[test]
fn structured_scene_errors_preserve_rejection_reason() {
    let mut scene = Scene::new(Color::BLACK).unwrap();

    assert_eq!(
        scene.try_circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)),
        Err(SceneError::InvalidDimension(ScenePrimitive::Circle))
    );
    assert_eq!(
        scene.try_line(Vec2::ZERO, Vec2::ZERO, 1.0, Color::WHITE),
        Err(SceneError::DegenerateGeometry(ScenePrimitive::Line))
    );
    assert_eq!(
        scene.try_circle(
            Vec2::ZERO,
            1.0,
            ShapeStyle {
                fill: None,
                stroke: None,
                shadow: None,
            },
        ),
        Err(SceneError::MissingStyle(ScenePrimitive::Circle))
    );
    assert_eq!(scene.command_count(), 0);
}

#[test]
fn commands_capture_depth_and_restore_it_after_unwind() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let _ = scene.with_depth(4.5, |scene| {
            scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));
            panic!("intentional depth unwind");
        });
    }));
    assert!(panic_result.is_err());

    scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));

    assert_eq!(scene.commands()[0].depth(), 4.5);
    assert_eq!(scene.commands()[1].depth(), 0.0);
    assert_eq!(
        scene.with_depth(f32::NAN, |_| {}),
        Err(SceneError::NonFiniteDepth)
    );
}

#[test]
fn gradients_sanitize_invalid_and_reversed_configuration() {
    let invalid_linear = LinearGradient::new(
        Vec2::new(f32::NAN, 0.0),
        Vec2::X,
        Color::BLACK,
        Color::WHITE,
    );
    assert_eq!(invalid_linear.color_at(Vec2::ZERO), Color::WHITE);

    let reversed = RadialGradient::new(Vec2::ZERO, 10.0, 5.0, Color::WHITE, Color::BLACK);
    assert_eq!(reversed.inner_radius(), 5.0);
    assert_eq!(reversed.outer_radius(), 10.0);
    assert_eq!(reversed.color_at(Vec2::ZERO), Color::BLACK);
}

#[test]
fn tiny_nonzero_radial_range_preserves_both_color_endpoints() {
    let gradient = RadialGradient::new(Vec2::ZERO, 0.0, 0.000_000_01, Color::BLACK, Color::WHITE);

    assert_eq!(gradient.color_at(Vec2::ZERO), Color::BLACK);
    assert_eq!(
        gradient.color_at(Vec2::new(0.000_000_01, 0.0)),
        Color::WHITE
    );
}

#[test]
fn temporary_screen_clip_restores_state_after_unwind() {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(0.0, 0.0),
        LogicalScreenVector::new(100.0, 100.0),
    )
    .unwrap();

    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let _ = scene.with_screen_clip(clip, |_| panic!("intentional test panic"));
    }));
    assert!(panic_result.is_err());

    assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
    assert_eq!(scene.commands()[0].screen_clip(), None);
}
