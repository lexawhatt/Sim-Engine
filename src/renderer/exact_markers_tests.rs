use super::*;
use crate::{LogicalPixels, StrokeMarker2d, StrokeMarkerAnchor2d, StrokeStyle2d, WorldLength};

fn marker() -> StrokeMarker2d {
    StrokeMarker2d::arrow(
        LogicalPixels::new(30.0).unwrap(),
        LogicalPixels::new(12.0).unwrap(),
    )
    .with_anchor(StrokeMarkerAnchor2d::TipAtEndpoint)
}

fn screen(vertex: Vertex, uniform: CameraUniform) -> Vec2 {
    let (ranges, _) = vertex_screen_ranges(vertex, uniform).unwrap();
    Vec2::new(
        ((ranges[0].0 + ranges[0].1) * 0.5) as f32,
        ((ranges[1].0 + ranges[1].1) * 0.5) as f32,
    )
}

#[test]
fn exact_vector_tips_and_short_dual_markers_survive_camera_transforms() {
    let viewport = LogicalViewport::new(640.0, 480.0).unwrap();
    for world_width in [false, true] {
        for delta in [
            Vec2::new(80.0, 0.0),
            Vec2::new(-80.0, 0.0),
            Vec2::new(36.0, 48.0),
            Vec2::new(0.004, 0.0),
        ] {
            for (zoom, rotation, tilt) in [(1.0, 0.0, 0.0), (2.0, 0.7, 0.5), (0.5, -0.6, -0.7)] {
                let mut camera = Camera2d::new(Vec2::ZERO, zoom).unwrap();
                camera.set_rotation(rotation).unwrap();
                camera.set_projection(crate::Projection2d::new(tilt, 1.0).unwrap());
                let uniform = CameraUniform::new(camera, viewport).unwrap();
                let style = if world_width {
                    StrokeStyle2d::world(WorldLength::new(2.0).unwrap(), Color::WHITE)
                } else {
                    StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                }
                .with_start_marker(marker())
                .with_end_marker(marker());
                let from = -delta * 0.5;
                let to = delta * 0.5;
                let mut scene = Scene::new(Color::BLACK).unwrap();
                scene.try_styled_line(from, to, style).unwrap();
                let mut vertices = Vec::new();
                let mut batches = Vec::new();
                super::super::tessellation::tessellate_scene(&scene, &mut vertices, &mut batches)
                    .unwrap();
                assert_eq!(vertices.len(), 12);
                assert!(vertices.iter().copied().all(Vertex::is_finite));
                assert_eq!(vertices[6].world_position, [from.x, from.y]);
                assert_eq!(vertices[9].world_position, [to.x, to.y]);
                for (tip, endpoint) in [(vertices[6], from), (vertices[9], to)] {
                    assert_eq!(tip.normal_distance, 0.0);
                    assert_eq!(tip.tangent_distance, 0.0);
                    let expected = camera
                        .world_to_screen(endpoint, viewport)
                        .unwrap()
                        .to_vec2();
                    assert!((screen(tip, uniform) - expected).length() < 0.0001);
                }
                let start = screen(vertices[6], uniform);
                let end = screen(vertices[9], uniform);
                let tangent = (end - start).normalized();
                let start_shaft = (screen(vertices[0], uniform) - start).dot(tangent);
                let end_shaft = (screen(vertices[1], uniform) - start).dot(tangent);
                assert!(
                    start_shaft <= end_shaft,
                    "shaft inverted: {start_shaft} > {end_shaft}"
                );
                assert!(vertices.len() <= scene.statistics().estimated_tessellated_vertices());
            }
        }
    }
}

#[test]
fn exact_vector_incompatible_paths_fail_before_scene_mutation() {
    let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
        .with_end_marker(marker());
    let mut scene = Scene::new(Color::BLACK).unwrap();
    assert_eq!(
        scene.try_styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 1.0)], style),
        Err(crate::SceneError::UnsupportedTipMarkerPath(
            crate::ScenePrimitive::Polyline
        ))
    );
    let dashed =
        style.with_dash_pattern(crate::StrokeDashPattern2d::new(&[2.0, 2.0], 0.0, 100).unwrap());
    assert_eq!(
        scene.try_styled_line(Vec2::ZERO, Vec2::X, dashed),
        Err(crate::SceneError::UnsupportedTipMarkerPath(
            crate::ScenePrimitive::Line
        ))
    );
    assert_eq!(scene.command_count(), 0);
    assert_eq!(scene.statistics().rejected_commands(), 2);
    scene
        .try_styled_polyline(vec![Vec2::ZERO, Vec2::X], style)
        .unwrap();
}

pub(in crate::renderer) fn assert_gpu_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    sample_count: u32,
) {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    let viewport = LogicalViewport::new(WIDTH as f32, HEIGHT as f32).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 2.0).unwrap();
    camera.set_rotation(0.63).unwrap();
    camera.set_projection(crate::Projection2d::new(0.61, 1.0).unwrap());
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for world_width in [false, true] {
        for row in 0..4 {
            let center = Vec2::new(
                if world_width { 192.0 } else { 64.0 },
                32.0 + row as f32 * 64.0,
            );
            let direction = match row {
                0 => Vec2::new(36.0, 0.0),
                1 => Vec2::new(-30.0, 16.0),
                2 => Vec2::new(2.0, 0.0),
                _ => Vec2::new(24.0, 12.0),
            };
            let from = camera
                .screen_to_world(
                    LogicalScreenPosition::from_vec2(center - direction),
                    viewport,
                )
                .unwrap();
            let to = camera
                .screen_to_world(
                    LogicalScreenPosition::from_vec2(center + direction),
                    viewport,
                )
                .unwrap();
            let mut style = if world_width {
                StrokeStyle2d::world(
                    WorldLength::new(5.0).unwrap(),
                    Color::rgba(1.0, 1.0, 1.0, 0.5),
                )
            } else {
                StrokeStyle2d::logical(
                    LogicalPixels::new(8.0).unwrap(),
                    Color::rgba(1.0, 1.0, 1.0, 0.5),
                )
            }
            .with_cap(crate::StrokeCap2d::Square);
            if row != 0 {
                style = style.with_start_marker(marker());
            }
            if row != 1 {
                style = style.with_end_marker(if row == 3 {
                    StrokeMarker2d::arrow(
                        LogicalPixels::new(14.0).unwrap(),
                        LogicalPixels::new(12.0).unwrap(),
                    )
                } else {
                    marker()
                });
            }
            scene.try_styled_line(from, to, style).unwrap();
        }
    }
    let mut vertices = Vec::new();
    let mut batches = Vec::new();
    super::super::tessellation::tessellate_scene(&scene, &mut vertices, &mut batches).unwrap();
    assert!(geometry_is_safe_for(
        GeometryExtents::from_vertices(&vertices),
        GeometryValidationSource::Tessellated(&vertices),
        uniform
    ));
    let PipelineResources {
        pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        ..
    } = create_pipeline(device, format, sample_count);
    queue.write_buffer(&camera_uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exact marker pixel vertices"),
        size: std::mem::size_of_val(vertices.as_slice()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    let make_texture = |samples, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exact marker semantic target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let target = make_texture(
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let target_view = target.create_view(&Default::default());
    let multisample = (sample_count > 1)
        .then(|| make_texture(sample_count, wgpu::TextureUsages::RENDER_ATTACHMENT));
    let multisample_view = multisample
        .as_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exact marker pixel readback"),
        size: u64::from(WIDTH * HEIGHT * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exact endpoint marker pixels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisample_view.as_ref().unwrap_or(&target_view),
                depth_slice: None,
                resolve_target: multisample_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
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
                bytes_per_row: Some(WIDTH * 4),
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
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let pixels = slice.get_mapped_range().unwrap();
    let ceiling = if format.is_srgb() { 190 } else { 130 };
    assert!(
        pixels
            .chunks_exact(4)
            .all(|pixel| pixel[0] <= ceiling && pixel[1] <= ceiling && pixel[2] <= ceiling),
        "exact-vector shaft or marker blended twice"
    );
    let floor = if format.is_srgb() { 180 } else { 122 };
    for world_width in 0..2 {
        for row in 0..4 {
            let center_x = if world_width == 1 { 192 } else { 64 };
            let center_y = 32 + row * 64;
            let mut filled = 0;
            for y in center_y - 25..center_y + 25 {
                for x in center_x - 50..center_x + 50 {
                    let pixel = &pixels[(y * WIDTH as usize + x) * 4..][..4];
                    if pixel[0] >= floor {
                        filled += 1;
                    }
                }
            }
            assert!(
                filled >= if row == 2 { 10 } else { 200 },
                "missing exact-marker row {row}, world={world_width}: {filled}"
            );
            if row == 0 || row == 2 {
                let extent = if row == 0 { 36 } else { 2 };
                for x in [center_x - extent - 2, center_x + extent + 1] {
                    for y in center_y - 10..center_y + 10 {
                        assert_eq!(
                            pixels[(y * WIDTH as usize + x) * 4],
                            0,
                            "marker or cap exceeded exact endpoint, row={row} world={world_width} x={x} y={y}"
                        );
                    }
                }
            }
        }
    }
    drop(pixels);
    readback.unmap();
}
