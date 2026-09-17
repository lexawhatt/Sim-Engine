//! Resources contract regressions.

use super::*;

#[test]
fn scene3d_rejects_out_of_range_clear_colors() {
    assert!(matches!(
        Scene3d::new(Color::rgb(1.01, 0.0, 0.0)),
        Err(Scene3dError::InvalidBackground)
    ));
}

#[test]
fn retained_mesh_and_scene_transform_reject_nonportable_shader_sources() {
    let subnormal = f32::from_bits(1);
    let mesh = Mesh3d::with_display_edges(
        vec![
            Vec3::new(subnormal, 0.0, 0.0).unwrap(),
            Vec3::new(1.0, 0.0, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    assert!(!mesh3d_source_is_portable(&mesh));

    let transform = Transform3d::new(
        Vec3::new(subnormal, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        validate_scene3d_transform(transform),
        Err(Scene3dError::InvalidTransform)
    );
    assert_eq!(validate_scene3d_transform(Transform3d::IDENTITY), Ok(()));
}

#[test]
fn target_pixel_scale_is_independent_of_window_dpi() {
    let viewport = LogicalViewport::new(1000.0, 500.0).unwrap();
    assert_eq!(
        target_pixels_per_logical(2000, 1000, viewport),
        Some(physical_per_logical(2.0))
    );
    assert_eq!(
        target_pixels_per_logical(500, 250, viewport),
        Some(physical_per_logical(0.5))
    );
    assert_eq!(target_pixels_per_logical(500, 300, viewport), None);
    assert!(aspect_matches(16.0 / 9.0, 1920.0 / 1080.0));
    assert!(!aspect_matches(16.0 / 9.0, 4.0 / 3.0));
    assert!(!aspect_matches(1.0 / 8_192.0, (1.0 / 8_192.0) * 1.08));
    assert_eq!(
        target_pixels_per_logical(
            1,
            8_192,
            LogicalViewport::new(100_000.0, 431_157_900.0).unwrap(),
        ),
        None
    );
}

#[test]
fn camera_target_aspect_mismatch_is_rejected_by_render_contract() {
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::perspective(1.0, 16.0 / 9.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let target = LogicalViewport::new(800.0, 600.0).unwrap();
    assert_eq!(
        validate_camera_target_aspect(camera, target),
        Err(Mesh3dRenderError::CameraTargetAspectMismatch)
    );
}

#[test]
fn logical_edge_width_is_equal_for_native_and_half_resolution_target() {
    let logical_width = logical(1.0);
    for pixels_per_logical in [physical_per_logical(2.0), physical_per_logical(0.5)] {
        let physical_width = edge_raster_envelope(logical_width, pixels_per_logical)
            .unwrap()
            .physical_half_width
            * 2.0;
        assert_eq!(
            physical_width / pixels_per_logical.get(),
            logical_width.get()
        );
    }
}

#[test]
fn mesh_upload_preflight_rejects_each_buffer_before_staging_allocation() {
    let max_buffer_size = 48;
    let layout = preflight_mesh3d_upload(4, 12, 2, max_buffer_size).unwrap();
    assert_eq!(layout.vertex_bytes, 48);
    assert_eq!(layout.index_bytes, 48);
    assert_eq!(layout.edge_bytes, 48);
    assert_eq!(layout.total_bytes, 144);

    assert_eq!(
        preflight_mesh3d_upload(5, 0, 0, max_buffer_size),
        Err(Mesh3dResourceError::CapacityTooLarge)
    );
    assert_eq!(
        preflight_mesh3d_upload(1, 13, 0, max_buffer_size),
        Err(Mesh3dResourceError::CapacityTooLarge)
    );
    assert_eq!(
        preflight_mesh3d_upload(1, 0, 3, max_buffer_size),
        Err(Mesh3dResourceError::CapacityTooLarge)
    );

    if usize::BITS > u32::BITS {
        let first_draw_count_overflow = u32::MAX as usize + 1;
        assert_eq!(
            preflight_mesh3d_upload(1, first_draw_count_overflow, 0, u64::MAX),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );
        assert_eq!(
            preflight_mesh3d_upload(1, 0, first_draw_count_overflow, u64::MAX),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );
    }
}
