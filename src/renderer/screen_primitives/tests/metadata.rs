use super::*;
use crate::LogicalViewport;

#[test]
fn fused_snapshot_preserves_independent_quality_summary_counts_and_instance_bytes() {
    let mut primitives = Vec::new();
    let mut counts = crate::PrimitiveCommandCounts::default();
    for index in 0..256 {
        let x = index as f32 * 2.125 - 100.0;
        primitives.push(
            ScreenPrimitive2d::circle(
                LogicalScreenPosition::new(x, 100.25),
                LogicalPixels::new(4.0).unwrap(),
                Color::WHITE,
            )
            .unwrap(),
        );
        primitives.push(
            ScreenPrimitive2d::round_line(
                LogicalScreenPosition::new(x, 20.0),
                LogicalScreenPosition::new(x + 17.0, 24.25),
                LogicalPixels::new(1.25).unwrap(),
                Color::WHITE,
            )
            .unwrap(),
        );
        primitives.push(
            ScreenPrimitive2d::rect(
                LogicalScreenPosition::new(x, 200.0),
                LogicalScreenVector::new(6.25, 3.75),
                Color::WHITE,
            )
            .unwrap(),
        );
        counts.increment(crate::ScenePrimitive::Circle);
        counts.increment(crate::ScenePrimitive::Line);
        counts.increment(crate::ScenePrimitive::Rect);
    }
    primitives.push(
        ScreenPrimitive2d::circle(
            LogicalScreenPosition::new(-1800.0, 60.0),
            LogicalPixels::new(1810.5).unwrap(),
            Color::WHITE,
        )
        .unwrap(),
    );
    primitives.push(
        ScreenPrimitive2d::round_line(
            LogicalScreenPosition::new(-2000.0, 40.0),
            LogicalScreenPosition::new(110.25, 40.75),
            LogicalPixels::new(1.25).unwrap(),
            Color::WHITE,
        )
        .unwrap(),
    );
    primitives.push(
        ScreenPrimitive2d::rect(
            LogicalScreenPosition::new(100.125, 100.25),
            LogicalScreenVector::new(16_384.0, 300.0),
            Color::WHITE,
        )
        .unwrap(),
    );
    counts.increment(crate::ScenePrimitive::Circle);
    counts.increment(crate::ScenePrimitive::Line);
    counts.increment(crate::ScenePrimitive::Rect);
    let expected_instances: Vec<_> = primitives.iter().map(|primitive| primitive.gpu).collect();
    let expected_quality = quality::Quality::new(&primitives);
    let mut instances = Vec::with_capacity(primitives.len());
    let (metadata, allocations) =
        crate::test_allocations::count(|| resources::fill_instances(&mut instances, &primitives));
    assert_eq!(allocations, 0);
    assert_eq!(metadata.counts, counts);
    assert_eq!(metadata.quality, expected_quality);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&instances),
        bytemuck::cast_slice::<_, u8>(&expected_instances)
    );
    let viewport = LogicalViewport::new(1280.0, 800.0).unwrap();
    let uniform = CameraUniform::new(
        super::super::super::screen_camera(viewport).unwrap(),
        viewport,
    )
    .unwrap();
    assert_eq!(
        validation::instances_are_safe_accelerated(&instances, metadata.certificate, uniform),
        validation::instances_are_safe(&instances, uniform)
    );
    assert!(validation::instances_are_safe(&instances, uniform));
}
