//! Exercise the real offscreen submission worker, not a second test renderer.

use super::super::{pipeline, resources};
use crate::renderer::{
    RenderStatus, RenderTarget2d, RenderTargetError, RenderTargetLoad, RendererFrameError,
    ScreenPrimitive2d, ScreenPrimitiveBudget, create_pipeline,
    primitive_targets::PrimitiveTargetRenderer,
};
use crate::{
    Color, LogicalPixels, LogicalScreenPosition, LogicalViewport, LogicalViewportRegion,
    PhysicalPerLogical,
};
use std::{sync::Arc, time::Duration};

pub(super) fn verify(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) {
    let identity = Arc::new(());
    let ordinary = create_pipeline(device, format, 1);
    let pipeline = pipeline::create_pipeline(device, &ordinary.camera_bind_group_layout, format, 1);
    let worker = PrimitiveTargetRenderer {
        renderer_identity: &identity,
        device,
        queue,
        pipeline: &pipeline,
        camera_uniform_buffer: &ordinary.camera_uniform_buffer,
        camera_bind_group: &ordinary.camera_bind_group,
    };
    let extent = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("analytic target submission oracle"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut target = RenderTarget2d {
        content_revision: std::sync::atomic::AtomicU64::new(0),
        renderer_identity: Arc::clone(&identity),
        resource_identity: Arc::new(()),
        view: texture.create_view(&Default::default()),
        _texture: texture,
        width: 64,
        height: 64,
        allocation_bytes: 64 * 64 * 4,
    };
    let circle = |color| {
        ScreenPrimitive2d::circle(
            LogicalScreenPosition::new(12.0, 12.0),
            LogicalPixels::new(4.0).unwrap(),
            color,
        )
        .unwrap()
    };
    let red = resources::create(
        device,
        queue,
        Arc::clone(&identity),
        &[circle(Color::rgba(1.0, 0.0, 0.0, 0.5))],
        ScreenPrimitiveBudget::default(),
    )
    .unwrap();
    let blue = resources::create(
        device,
        queue,
        Arc::clone(&identity),
        &[circle(Color::rgba(0.0, 0.0, 1.0, 0.5))],
        ScreenPrimitiveBudget::default(),
    )
    .unwrap();
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(4.0, 2.0),
        LogicalViewport::new(24.0, 24.0).unwrap(),
    )
    .unwrap();
    for scale in [1.0, 2.0] {
        let density = PhysicalPerLogical::new(scale).unwrap();
        let report = worker
            .draw(
                &target,
                &red,
                density,
                Some(region),
                RenderTargetLoad::Clear(Color::TRANSPARENT),
            )
            .unwrap();
        assert_eq!(report.status(), RenderStatus::Drawn);
        assert_eq!(report.metrics().tessellation_stats().command_count(), 1);
        let first = read(device, queue, &target);
        let capture_state = crate::renderer::capture::CaptureState::default();
        let correlation = crate::CaptureCorrelation {
            frame_id: 7,
            texture_revision: 11,
        };
        crate::renderer::capture::verify_backpressure(device, queue, &target);
        let captured = crate::renderer::capture::capture(
            &capture_state,
            device,
            queue,
            &target,
            correlation,
            64 * 64 * 8,
            Duration::from_secs(15),
        )
        .unwrap();
        assert_eq!(captured.pixels(), first);
        assert_eq!(captured.correlation(), correlation);
        assert_eq!(Some(captured.content_id()), target.content_id().as_ref());
        assert!(matches!(
            crate::renderer::capture::capture(
                &capture_state,
                device,
                queue,
                &target,
                correlation,
                64 * 64 * 8 - 1,
                Duration::from_secs(15)
            ),
            Err(crate::CaptureError::BudgetExceeded)
        ));
        assert_pixel(
            &first,
            format,
            (16.0 * scale) as usize,
            (14.0 * scale) as usize,
            [188, 0, 0, 128],
        );
        assert_pixel(&first, format, 0, 0, [0; 4]);
        worker
            .draw(
                &target,
                &blue,
                density,
                Some(region),
                RenderTargetLoad::Load,
            )
            .unwrap();
        let second = read(device, queue, &target);
        // Linear premultiplied red=.25, blue=.5, alpha=.75. RGB is stored in sRGB.
        assert_pixel(
            &second,
            format,
            (16.0 * scale) as usize,
            (14.0 * scale) as usize,
            [137, 0, 188, 191],
        );
        assert_pixel(&second, format, 0, 0, [0; 4]);

        let invalid_region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(60.0, 60.0),
            LogicalViewport::new(12.0, 12.0).unwrap(),
        )
        .unwrap();
        assert_eq!(
            worker.draw(
                &target,
                &red,
                density,
                Some(invalid_region),
                RenderTargetLoad::Clear(Color::WHITE)
            ),
            Err(RenderTargetError::Frame(
                RendererFrameError::InvalidViewport
            ))
        );
        assert_eq!(
            worker.draw(
                &target,
                &red,
                density,
                Some(region),
                RenderTargetLoad::Clear(Color::rgba(f32::NAN, 0.0, 0.0, 1.0))
            ),
            Err(RenderTargetError::InvalidBackground)
        );
        let foreign = resources::create(
            device,
            queue,
            Arc::new(()),
            &[circle(Color::WHITE)],
            ScreenPrimitiveBudget::default(),
        )
        .unwrap();
        assert_eq!(
            worker.draw(
                &target,
                &foreign,
                density,
                Some(region),
                RenderTargetLoad::Clear(Color::WHITE)
            ),
            Err(RenderTargetError::RendererMismatch)
        );
        target.renderer_identity = Arc::new(());
        assert_eq!(
            worker.draw(
                &target,
                &red,
                density,
                Some(region),
                RenderTargetLoad::Clear(Color::WHITE)
            ),
            Err(RenderTargetError::RendererMismatch)
        );
        target.renderer_identity = Arc::clone(&identity);
        assert_eq!(
            second,
            read(device, queue, &target),
            "rejected draw changed target pixels"
        );
    }
    eprintln!(
        "Analytic offscreen submission: positioned target at scale1/2, transparent clear, premultiplied load, foreign-resource and invalid-input preservation passed"
    );
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, target: &RenderTarget2d) -> Vec<u8> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
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
    queue.submit([encoder.finish()]);
    crate::renderer::test_support::read_buffer(device, &buffer, Duration::from_secs(15))
}

fn assert_pixel(bytes: &[u8], format: wgpu::TextureFormat, x: usize, y: usize, expected: [u8; 4]) {
    let index = (y * 64 + x) * 4;
    let channels = if format == wgpu::TextureFormat::Bgra8UnormSrgb {
        [2, 1, 0, 3]
    } else {
        [0, 1, 2, 3]
    };
    for (component, channel) in channels.into_iter().enumerate() {
        assert!(
            bytes[index + channel].abs_diff(expected[component]) <= 1,
            "target ({x},{y}) component {component}: {} != {}",
            bytes[index + channel],
            expected[component]
        );
    }
}
