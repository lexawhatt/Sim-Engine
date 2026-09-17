use super::*;

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
