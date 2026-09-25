//! Independent pixel oracle. Shape membership uses host circle/segment distance,
//! not the production quad coordinates, shader predicates, or validation helpers.

use crate::renderer::screen_primitives::{
    ScreenPrimitive2d, ScreenPrimitiveBatch2d, ScreenPrimitiveBudget, ScreenPrimitiveError,
    pipeline, resources,
};
use crate::renderer::{
    CameraUniform, ScreenClipRect, create_pipeline, frame, screen_camera, tessellate_scene,
};
use crate::{
    Color, LogicalPixels, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportRegion, ScreenScene, ShapeStyle,
};
use std::{sync::Arc, time::Duration};
use wgpu::util::DeviceExt;

const LOGICAL_WIDTH: u32 = 128;
const LOGICAL_HEIGHT: u32 = 112;

#[derive(Clone, Copy)]
enum Shape {
    Circle {
        center: [f32; 2],
        radius: f32,
    },
    Capsule {
        from: [f32; 2],
        to: [f32; 2],
        width: f32,
    },
    Rectangle {
        minimum: [f32; 2],
        size: [f32; 2],
    },
}

#[derive(Clone, Copy)]
struct Case {
    shape: Shape,
    color: [f32; 4],
    clip: Option<[f32; 4]>,
}

impl Case {
    fn primitive(self) -> ScreenPrimitive2d {
        let color = Color::rgba(self.color[0], self.color[1], self.color[2], self.color[3]);
        let primitive = match self.shape {
            Shape::Circle { center, radius } => ScreenPrimitive2d::circle(
                position(center),
                LogicalPixels::new(radius).unwrap(),
                color,
            ),
            Shape::Capsule { from, to, width } => ScreenPrimitive2d::round_line(
                position(from),
                position(to),
                LogicalPixels::new(width).unwrap(),
                color,
            ),
            Shape::Rectangle { minimum, size } => ScreenPrimitive2d::rect(
                position(minimum),
                LogicalScreenVector::new(size[0], size[1]),
                color,
            ),
        }
        .unwrap();
        self.clip
            .map_or(primitive, |clip| primitive.with_clip(clip_rect(clip)))
    }

    fn append_legacy(self, scene: &mut ScreenScene) {
        scene.set_screen_clip(self.clip.map(clip_rect)).unwrap();
        let color = Color::rgba(self.color[0], self.color[1], self.color[2], self.color[3]);
        match self.shape {
            Shape::Circle { center, radius } => scene.try_circle(
                position(center),
                LogicalPixels::new(radius).unwrap(),
                ShapeStyle::filled(color),
            ),
            Shape::Capsule { from, to, width } => scene.try_line(
                position(from),
                position(to),
                LogicalPixels::new(width).unwrap(),
                color,
            ),
            Shape::Rectangle { minimum, size } => scene.try_square_rect(
                position(minimum),
                LogicalScreenVector::new(size[0], size[1]),
                ShapeStyle::filled(color),
            ),
        }
        .unwrap();
    }

    fn signed_distance(self, point: [f64; 2]) -> f64 {
        match self.shape {
            Shape::Circle { center, radius } => {
                (point[0] - f64::from(center[0])).hypot(point[1] - f64::from(center[1]))
                    - f64::from(radius)
            }
            Shape::Capsule { from, to, width } => {
                let start = from.map(f64::from);
                let end = to.map(f64::from);
                let direction = [end[0] - start[0], end[1] - start[1]];
                let length_squared = direction[0] * direction[0] + direction[1] * direction[1];
                let amount = (((point[0] - start[0]) * direction[0]
                    + (point[1] - start[1]) * direction[1])
                    / length_squared)
                    .clamp(0.0, 1.0);
                (point[0] - start[0] - amount * direction[0])
                    .hypot(point[1] - start[1] - amount * direction[1])
                    - f64::from(width) * 0.5
            }
            Shape::Rectangle { minimum, size } => {
                let center = [
                    f64::from(minimum[0]) + f64::from(size[0]) * 0.5,
                    f64::from(minimum[1]) + f64::from(size[1]) * 0.5,
                ];
                let edge = [
                    (point[0] - center[0]).abs() - f64::from(size[0]) * 0.5,
                    (point[1] - center[1]).abs() - f64::from(size[1]) * 0.5,
                ];
                edge[0].max(0.0).hypot(edge[1].max(0.0)) + edge[0].max(edge[1]).min(0.0)
            }
        }
    }
}

fn position(point: [f32; 2]) -> LogicalScreenPosition {
    LogicalScreenPosition::new(point[0], point[1])
}
fn clip_rect(rect: [f32; 4]) -> ScreenClipRect {
    ScreenClipRect::new(position([rect[0], rect[1]]), position([rect[2], rect[3]])).unwrap()
}
fn region() -> LogicalViewportRegion {
    LogicalViewportRegion::new(
        position([8.0, 4.0]),
        LogicalViewport::new(112.0, 104.0).unwrap(),
    )
    .unwrap()
}
fn item_clip() -> ScreenClipRect {
    clip_rect([3.25, 2.75, 106.5, 99.5])
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            shape: Shape::Circle {
                center: [62.25, 110.25],
                radius: 44.0,
            },
            color: [0.3, 0.2, 0.8, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Circle {
                center: [22.25, 18.75],
                radius: 10.25,
            },
            color: [1.0, 0.0, 0.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Circle {
                center: [28.25, 18.75],
                radius: 8.5,
            },
            color: [0.0, 1.0, 0.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [8.25, 50.25],
                to: [42.5, 50.25],
                width: 8.0,
            },
            color: [0.0, 0.0, 1.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [88.5, 48.25],
                to: [56.25, 48.25],
                width: 7.5,
            },
            color: [1.0, 0.5, 0.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [62.25, 15.25],
                to: [90.25, 33.25],
                width: 6.5,
            },
            color: [0.0, 1.0, 1.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [102.25, 36.25],
                to: [90.25, 10.25],
                width: 5.25,
            },
            color: [1.0, 0.0, 1.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [10.25, 70.5],
                to: [10.251, 70.5],
                width: 6.0,
            },
            color: [1.0, 1.0, 0.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Circle {
                center: [100.5, 77.5],
                radius: 0.45,
            },
            color: [1.0, 1.0, 1.0, 1.0],
            clip: None,
        },
        Case {
            shape: Shape::Rectangle {
                minimum: [-8.0, -8.0],
                size: [128.0, 120.0],
            },
            color: [0.5, 0.2, 1.0, 0.5],
            clip: Some([18.25, 30.75, 34.5, 42.25]),
        },
        Case {
            shape: Shape::Rectangle {
                minimum: [44.25, 56.75],
                size: [20.5, 9.25],
            },
            color: [0.8, 0.7, 0.1, 0.5],
            clip: Some([48.5, 54.25, 70.25, 64.75]),
        },
    ]
}

pub(in crate::renderer) fn verify_analytic_primitives(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    production_samples: u32,
) {
    super::targets::verify(device, queue, format);
    verify_source_precision_invalidates_cached_proof(device, queue);
    let identity = Arc::new(());
    let cases = cases();
    let primitives: Vec<_> = cases.iter().map(|case| case.primitive()).collect();
    let batch = resources::create(
        device,
        queue,
        Arc::clone(&identity),
        &primitives,
        ScreenPrimitiveBudget::default(),
    )
    .unwrap();
    for samples in std::iter::once(1).chain((production_samples != 1).then_some(production_samples))
    {
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let pixels = render(device, queue, &batch, &cases, format, samples, scale, false);
            let legacy = render(device, queue, &batch, &cases, format, samples, scale, true);
            verify_pixels(&pixels, Some(&legacy), &cases, format, scale, samples);
        }
    }
    // A large offscreen curve is deliberately checked against mathematics,
    // not the legacy fixed 64-sector fan whose boundary error grows with radius.
    let large_cases = [
        Case {
            shape: Shape::Circle {
                center: [-1750.25, 51.75],
                radius: 1810.5,
            },
            color: [0.25, 0.5, 1.0, 0.5],
            clip: None,
        },
        Case {
            shape: Shape::Capsule {
                from: [-1500.25, 74.25],
                to: [96.75, 89.25],
                width: 1.25,
            },
            color: [1.0, 0.25, 0.0, 0.5],
            clip: None,
        },
        cases[8],
    ];
    let large_primitives: Vec<_> = large_cases.iter().map(|case| case.primitive()).collect();
    let large_batch = resources::create(
        device,
        queue,
        Arc::clone(&identity),
        &large_primitives,
        ScreenPrimitiveBudget::default(),
    )
    .unwrap();
    for samples in std::iter::once(1).chain((production_samples != 1).then_some(production_samples))
    {
        for scale in [1.0, 1.25, 2.0] {
            let pixels = render(
                device,
                queue,
                &large_batch,
                &large_cases,
                format,
                samples,
                scale,
                false,
            );
            verify_pixels(&pixels, None, &large_cases, format, scale, samples);
        }
    }
    verify_updates(
        device,
        queue,
        recovery_device,
        recovery_queue,
        format,
        production_samples,
        identity,
        &primitives,
        &cases,
    );
    eprintln!(
        "Analytic primitive oracle: independent circle/capsule/rectangle interiors, large offscreen curves, <=0.25px single-sample boundary error, <=1px legacy boundary differences, fractional placement/clips, ordered half-alpha, scales1/1.25/1.5/2, MSAA1+{production_samples}, source-precision cache invalidation, updates and restoration passed"
    );
}

fn verify_source_precision_invalidates_cached_proof(device: &wgpu::Device, queue: &wgpu::Queue) {
    let rectangle = |x| {
        ScreenPrimitive2d::rect(
            LogicalScreenPosition::new(x, 1.0),
            LogicalScreenVector::new(1024.0, 5.0),
            Color::WHITE,
        )
        .unwrap()
    };
    let exact = rectangle(1.0);
    let rounded = rectangle(f32::from_bits(1.0_f32.to_bits() + 1));
    assert_eq!(
        bytemuck::bytes_of(&exact.gpu),
        bytemuck::bytes_of(&rounded.gpu)
    );
    assert!(rounded.source_error > exact.source_error);
    let viewport = LogicalViewport::new(1280.0, 800.0).unwrap();
    let uniform = CameraUniform::new(screen_camera(viewport).unwrap(), viewport).unwrap();
    let mut batch = resources::create(
        device,
        queue,
        Arc::new(()),
        &[exact],
        ScreenPrimitiveBudget::default(),
    )
    .unwrap();
    let mut accepted = 1.0_f32.to_bits();
    let mut rejected = 64.0_f32.to_bits();
    assert!(batch.is_safe_for(uniform, f32::from_bits(accepted)));
    assert!(!batch.is_safe_for(uniform, f32::from_bits(rejected)));
    while rejected - accepted > 1 {
        let middle = accepted + (rejected - accepted) / 2;
        if batch.is_safe_for(uniform, f32::from_bits(middle)) {
            accepted = middle;
        } else {
            rejected = middle;
        }
    }
    let scale = f32::from_bits(accepted);
    assert!(batch.is_safe_for(uniform, scale));
    let update = resources::update(device, queue, &mut batch, &[rounded]).unwrap();
    assert_eq!(update.uploaded_bytes(), 0);
    assert!(
        !batch.is_safe_for(uniform, scale),
        "source precision reused a stale success"
    );
    let update = resources::update(device, queue, &mut batch, &[exact]).unwrap();
    assert_eq!(update.uploaded_bytes(), 0);
    assert!(
        batch.is_safe_for(uniform, scale),
        "source precision reused a stale rejection"
    );
}

#[allow(clippy::too_many_arguments)]
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: &ScreenPrimitiveBatch2d,
    cases: &[Case],
    format: wgpu::TextureFormat,
    samples: u32,
    scale: f32,
    legacy: bool,
) -> Vec<u8> {
    let width = (LOGICAL_WIDTH as f32 * scale) as u32;
    let height = (LOGICAL_HEIGHT as f32 * scale) as u32;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = |sample_count| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("analytic primitive pixel oracle"),
            size: extent,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | if sample_count == 1 {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        })
    };
    let target = texture(1);
    let view = target.create_view(&Default::default());
    let multisampled = (samples > 1).then(|| texture(samples));
    let multisampled_view = multisampled
        .as_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let ordinary = create_pipeline(device, format, samples);
    let analytic =
        pipeline::create_pipeline(device, &ordinary.camera_bind_group_layout, format, samples);
    let region = region();
    let uniform = CameraUniform::new_in_region(
        screen_camera(region.viewport()).unwrap(),
        region.viewport(),
        region.origin().to_vec2(),
        LogicalViewport::new(LOGICAL_WIDTH as f32, LOGICAL_HEIGHT as f32).unwrap(),
    )
    .unwrap();
    assert!(
        batch.is_safe_for(uniform, scale),
        "valid analytic fixture rejected"
    );
    queue.write_buffer(
        &ordinary.camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniform),
    );
    let mut legacy_vertices = Vec::new();
    let mut legacy_batches = Vec::new();
    let legacy_buffer = if legacy {
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        for case in cases {
            case.append_legacy(&mut scene);
        }
        let stats =
            tessellate_scene(scene.as_scene(), &mut legacy_vertices, &mut legacy_batches).unwrap();
        assert_eq!(
            stats.dropped_command_count(),
            0,
            "legacy fixture silently dropped geometry"
        );
        Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("legacy analytic oracle control"),
                contents: bytemuck::cast_slice(&legacy_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        )
    } else {
        None
    };
    let stride = (width * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("analytic pixel oracle readback"),
        size: u64::from(stride) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("analytic production pixel oracle"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisampled_view.as_ref().unwrap_or(&view),
                resolve_target: multisampled_view.as_ref().map(|_| &view),
                depth_slice: None,
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
        if let Some(buffer) = &legacy_buffer {
            pass.set_pipeline(&ordinary.pipeline);
            pass.set_bind_group(0, &ordinary.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(..));
            for draw in &legacy_batches {
                if let Some(clip) = physical_clip(
                    draw.screen_clip.map(|clip| {
                        [
                            clip.min().to_vec2().x,
                            clip.min().to_vec2().y,
                            clip.max().to_vec2().x,
                            clip.max().to_vec2().y,
                        ]
                    }),
                    scale,
                ) {
                    pass.set_scissor_rect(clip[0], clip[1], clip[2] - clip[0], clip[3] - clip[1]);
                    pass.draw(draw.vertex_range.clone(), 0..1);
                }
            }
        } else {
            frame::encode_screen_primitives_for_test(
                &mut pass,
                &analytic,
                &ordinary.camera_bind_group,
                batch,
                region,
                Some(item_clip()),
                scale,
                width,
                height,
            );
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
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    crate::renderer::test_support::read_buffer(device, &readback, Duration::from_secs(15))
}

// Independent conservative logical-scissor mapping for this fixture's explicit
// positioned viewport. Do not call the production scissor or intersection code.
fn physical_clip(local: Option<[f32; 4]>, scale: f32) -> Option<[u32; 4]> {
    let offset = [(8.0 * scale) as u32, (4.0 * scale) as u32];
    let mut clip = [
        offset[0],
        offset[1],
        offset[0] + (112.0 * scale) as u32,
        offset[1] + (104.0 * scale) as u32,
    ];
    for logical in [Some([3.25, 2.75, 106.5, 99.5]), local]
        .into_iter()
        .flatten()
    {
        let candidate = [
            offset[0] + (logical[0].max(0.0) * scale).floor() as u32,
            offset[1] + (logical[1].max(0.0) * scale).floor() as u32,
            offset[0] + (logical[2].clamp(0.0, 112.0) * scale).ceil() as u32,
            offset[1] + (logical[3].clamp(0.0, 104.0) * scale).ceil() as u32,
        ];
        clip = [
            clip[0].max(candidate[0]),
            clip[1].max(candidate[1]),
            clip[2].min(candidate[2]),
            clip[3].min(candidate[3]),
        ];
    }
    (clip[0] < clip[2] && clip[1] < clip[3]).then_some(clip)
}

fn verify_pixels(
    actual: &[u8],
    legacy: Option<&[u8]>,
    cases: &[Case],
    format: wgpu::TextureFormat,
    scale: f32,
    samples: u32,
) {
    let width = (LOGICAL_WIDTH as f32 * scale) as u32;
    let height = (LOGICAL_HEIGHT as f32 * scale) as u32;
    let stride = ((width * 4).div_ceil(256) * 256) as usize;
    let channels = if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        [2, 1, 0]
    } else {
        [0, 1, 2]
    };
    let encode = |value: f64| {
        let value = if format.is_srgb() {
            if value <= 0.0031308 {
                value * 12.92
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            }
        } else {
            value
        };
        (value * 255.0).round() as u8
    };
    let mut interior_samples = 0;
    let mut colored_samples = 0;
    let mut uncertain_samples = 0;
    let mut thin_circle_peak = 0;
    for y in 0..height {
        for x in 0..width {
            let point = [
                (f64::from(x) + 0.5) / f64::from(scale) - 8.0,
                (f64::from(y) + 0.5) / f64::from(scale) - 4.0,
            ];
            let mut expected = [0.0_f64; 3];
            let mut near_boundary = false;
            let mut near_legacy_boundary = false;
            for case in cases {
                let in_clip = physical_clip(case.clip, scale).is_some_and(|clip| {
                    (clip[0]..clip[2]).contains(&x) && (clip[1]..clip[3]).contains(&y)
                });
                if !in_clip {
                    continue;
                }
                let distance = case.signed_distance(point);
                let physical_distance = distance.abs() * f64::from(scale);
                // Single-sample coverage is evaluated at the pixel center:
                // outside the public 0.25px envelope it must match the exact
                // host shape. MSAA sample locations can straddle a wider band.
                near_boundary |= physical_distance <= if samples == 1 { 0.25 } else { 1.0 };
                near_legacy_boundary |= physical_distance <= 1.0;
                if distance <= 0.0 {
                    for (channel, value) in expected.iter_mut().enumerate() {
                        *value = f64::from(case.color[channel]) * f64::from(case.color[3])
                            + *value * (1.0 - f64::from(case.color[3]));
                    }
                }
            }
            let offset = y as usize * stride + x as usize * 4;
            assert_eq!(actual[offset + 3], 255, "alpha changed over opaque clear");
            if (point[0] - 100.5).hypot(point[1] - 77.5) < 1.5 {
                thin_circle_peak = thin_circle_peak.max(actual[offset + channels[0]]);
            }
            if near_boundary {
                uncertain_samples += 1;
                continue;
            }
            interior_samples += 1;
            colored_samples += usize::from(expected.iter().any(|value| *value != 0.0));
            for channel in 0..3 {
                let index = offset + channels[channel];
                let tolerance = if expected[channel] == 0.0 { 0 } else { 4 };
                assert!(
                    actual[index].abs_diff(encode(expected[channel])) <= tolerance,
                    "analytic shape mismatch scale{scale}, pixel({x},{y}), channel{channel}: actual{} expected{}",
                    actual[index],
                    encode(expected[channel])
                );
                if let Some(legacy) = legacy
                    && !near_legacy_boundary
                {
                    assert!(
                        actual[index].abs_diff(legacy[index]) <= 4,
                        "analytic/legacy difference extends beyond 1 physical pixel: scale{scale}, ({x},{y}), channel{channel}, actual{} legacy{}",
                        actual[index],
                        legacy[index]
                    );
                }
            }
        }
    }
    assert!(
        interior_samples > (width * height) as usize * 4 / 5,
        "boundary tolerance swallowed the image"
    );
    assert!(colored_samples > 1000);
    assert!(uncertain_samples > 0);
    assert!(
        thin_circle_peak > 16,
        "the subpixel white circle disappeared at scale {scale}"
    );
}

#[allow(clippy::too_many_arguments)]
fn verify_updates(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    samples: u32,
    identity: Arc<()>,
    primitives: &[ScreenPrimitive2d],
    cases: &[Case],
) {
    let budget = ScreenPrimitiveBudget::new(primitives.len(), 64 * 1024, 64 * 1024, 64 * 1024);
    let mut batch =
        resources::create(device, queue, Arc::clone(&identity), primitives, budget).unwrap();
    let reference = render(device, queue, &batch, cases, format, samples, 1.25, false);
    let buffer = batch.instance_buffer.clone();
    let same = resources::update(device, queue, &mut batch, primitives).unwrap();
    assert_eq!(same.uploaded_bytes(), 0);
    assert!(same.buffer_reused());
    let mut excess = primitives.to_vec();
    excess.push(primitives[0]);
    assert!(matches!(
        resources::update(device, queue, &mut batch, &excess),
        Err(ScreenPrimitiveError::BudgetExceeded { .. })
    ));
    assert_eq!(batch.instance_buffer, buffer);
    let after = render(device, queue, &batch, cases, format, samples, 1.25, false);
    assert_eq!(after, reference, "rejected update changed pixels");
    let mut moved_cases = cases.to_vec();
    moved_cases[1].shape = Shape::Circle {
        center: [38.25, 18.75],
        radius: 10.25,
    };
    let moved: Vec<_> = moved_cases.iter().map(|case| case.primitive()).collect();
    let moved_report = resources::update(device, queue, &mut batch, &moved).unwrap();
    assert!(moved_report.uploaded_bytes() > 0);
    assert!(moved_report.buffer_reused());
    let moved_pixels = render(
        device,
        queue,
        &batch,
        &moved_cases,
        format,
        samples,
        1.25,
        false,
    );
    assert_ne!(
        moved_pixels, reference,
        "changed geometry did not reach the GPU"
    );
    let moved_legacy = render(
        device,
        queue,
        &batch,
        &moved_cases,
        format,
        samples,
        1.25,
        true,
    );
    verify_pixels(
        &moved_pixels,
        Some(&moved_legacy),
        &moved_cases,
        format,
        1.25,
        samples,
    );
    moved_cases[1].clip = Some([32.0, 12.0, 45.0, 30.0]);
    let clipped: Vec<_> = moved_cases.iter().map(|case| case.primitive()).collect();
    let clip_report = resources::update(device, queue, &mut batch, &clipped).unwrap();
    assert_eq!(
        clip_report.uploaded_bytes(),
        0,
        "clip-only changes uploaded instance geometry"
    );
    let clipped_pixels = render(
        device,
        queue,
        &batch,
        &moved_cases,
        format,
        samples,
        1.25,
        false,
    );
    let clipped_legacy = render(
        device,
        queue,
        &batch,
        &moved_cases,
        format,
        samples,
        1.25,
        true,
    );
    verify_pixels(
        &clipped_pixels,
        Some(&clipped_legacy),
        &moved_cases,
        format,
        1.25,
        samples,
    );
    resources::update(device, queue, &mut batch, primitives).unwrap();
    let independent =
        resources::create(device, queue, Arc::clone(&identity), primitives, budget).unwrap();
    resources::update(device, queue, &mut batch, &[]).unwrap();
    assert_eq!(batch.primitive_count(), 0);
    let unaffected = render(
        device,
        queue,
        &independent,
        cases,
        format,
        samples,
        1.25,
        false,
    );
    assert_eq!(unaffected, reference, "separate owned snapshot was mutated");
    resources::update(device, queue, &mut batch, primitives).unwrap();
    assert_eq!(batch.instance_buffer, buffer);
    assert_eq!(
        render(device, queue, &batch, cases, format, samples, 1.25, false),
        reference
    );
    let restored_identity = Arc::new(());
    assert!(!Arc::ptr_eq(&restored_identity, &batch.renderer_identity));
    let restored = resources::restore(
        recovery_device,
        recovery_queue,
        Arc::clone(&restored_identity),
        &batch,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&restored_identity, &restored.renderer_identity));
    assert!(Arc::ptr_eq(&identity, &batch.renderer_identity));
    assert_eq!(
        render(
            recovery_device,
            recovery_queue,
            &restored,
            cases,
            format,
            samples,
            1.25,
            false
        ),
        reference,
        "device restoration changed pixels"
    );
}
