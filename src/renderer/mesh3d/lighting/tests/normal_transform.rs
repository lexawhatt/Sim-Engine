use super::*;

#[test]
fn inverse_transpose_preserves_uniform_tiny_scales_and_rejects_lost_source_directions() {
    let mesh = |normal| {
        Mesh3d::with_attributes(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            vec![0, 1, 2],
            vec![],
            crate::Mesh3dAttributes::new()
                .with_normals(vec![normal; 3])
                .unwrap(),
        )
        .unwrap()
    };
    let tiny = Transform3d::from_rotation_scale(crate::Rotation3d::IDENTITY, 1e-30).unwrap();
    let transport = SurfaceTransport {
        normal_rows: normal_rows(tiny).unwrap(),
        lit: true,
        ..SurfaceTransport::default()
    };
    assert_eq!(
        transport
            .vertex(&mesh(Vec3::Z), 0, tiny.model_rows().unwrap())
            .unwrap(),
        [0.0, 0.0, 1.0, 0.0]
    );
    let extreme = Transform3d::new(
        Vec3::ZERO,
        crate::Rotation3d::IDENTITY,
        Vec3::new(1e-20, 1e20, 1.0).unwrap(),
    )
    .unwrap();
    let transport = SurfaceTransport {
        normal_rows: normal_rows(extreme).unwrap(),
        lit: true,
        ..SurfaceTransport::default()
    };
    assert_eq!(
        transport.vertex(&mesh(Vec3::Y), 0, extreme.model_rows().unwrap()),
        Err(Mesh3dSurfaceError::NormalTransform)
    );
    assert!(
        transport
            .vertex(&mesh(Vec3::X), 0, extreme.model_rows().unwrap())
            .is_ok()
    );
    assert!(
        SurfaceTransport::default()
            .vertex(&mesh(Vec3::Y), 0, extreme.model_rows().unwrap())
            .is_ok()
    );
    let anisotropic = Transform3d::new(
        Vec3::ZERO,
        crate::Rotation3d::IDENTITY,
        Vec3::new(2.0, 4.0, 8.0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        normal_rows(anisotropic).unwrap(),
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0]
        ]
    );
}
