use super::*;

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
    let minimum = -8.707_754e19_f32;
    let maximum = 2.625_617_7e19_f32;
    let exact =
        LayeredVisualizationOptions::new((minimum, maximum), Color::BLACK, Color::BLACK).unwrap();
    assert_eq!(exact.value_range(), (minimum, maximum));
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
    assert!(matches!(
        WgpuRendererOptions::new(RendererPresentMode::Vsync, f32::MAX as f64),
        Err(RendererConfigurationError::InvalidScaleFactor { .. })
    ));
}

#[test]
fn surface_dimensions_are_rejected_before_configuration() {
    assert_eq!(validate_surface_dimensions(8_192, 4_096, 8_192), Ok(()));
    assert_eq!(
        validate_surface_dimensions(8_193, 4_096, 8_192),
        Err(RendererConfigurationError::SurfaceDimensionsTooLarge {
            width: 8_193,
            height: 4_096,
            limit: 8_192,
        })
    );
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
