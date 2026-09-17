use super::*;

#[test]
fn background_changes_are_normalized_atomic_and_allocation_free() {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let identity = scene.scene_id;
    let statistics = scene.statistics();
    let lighting = scene.lighting();
    let fog = scene.fog();
    let background = Color::rgba(0.125, 0.25, 0.5, 0.5);
    let (_, allocations) = crate::test_allocations::count(|| {
        scene.set_background(background).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            for channel in 0..4 {
                let mut channels = [0.5; 4];
                channels[channel] = bad;
                assert_eq!(
                    scene.set_background(Color::rgba(
                        channels[0],
                        channels[1],
                        channels[2],
                        channels[3],
                    )),
                    Err(Scene3dError::InvalidBackground)
                );
                assert_eq!(scene.background(), background);
            }
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(scene.scene_id, identity);
    assert_eq!(scene.statistics(), statistics);
    assert_eq!(scene.lighting(), lighting);
    assert_eq!(scene.fog(), fog);
    for alpha in [0.0, f32::from_bits(1), 1.0] {
        scene
            .set_background(Color::rgba(0.5, 0.0, 1.0, alpha))
            .unwrap();
        assert_eq!(scene.background().alpha(), alpha);
    }
}

#[test]
fn background_setter_does_not_change_legacy_constructor_validation() {
    let translucent = Color::rgba(0.5, 0.0, 1.0, 0.5);
    assert!(matches!(
        Scene3d::new(translucent),
        Err(Scene3dError::InvalidBackground)
    ));
    assert_eq!(
        Scene3d::with_alpha_background(translucent)
            .unwrap()
            .background(),
        translucent
    );
    let cause = Texture3dError::MissingTextureCoordinates;
    let error = Scene3dError::Texture(cause);
    assert_eq!(error.source().unwrap().to_string(), cause.to_string());
    assert!(Scene3dError::InvalidBackground.source().is_none());
}
