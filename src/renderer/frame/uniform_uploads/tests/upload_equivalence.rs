//! Transfer oracle: independent per-draw data, exact queue order and bounded
//! reusable scratch. This is not a full surface/cache-collector test.
use super::*;
use std::time::Duration;
use wgpu::util::DeviceExt;

const INITIAL_COUNT: usize = 1000;
const GROWN_COUNT: usize = 2000;
const BYTE_LIMIT: usize = 256 * 1024;

struct Readback {
    buffer: wgpu::Buffer,
    expected: Vec<u8>,
    tolerance: u8,
}

pub(in crate::renderer) fn assert_gpu_uniform_upload_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform transfer oracle layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline = pixel_pipeline(device, &layout);
    let mut values = (0..GROWN_COUNT)
        .map(|index| value(index, 0))
        .collect::<Vec<_>>();
    let bindings = values
        .iter()
        .map(|bytes| binding(device, &layout, bytes))
        .collect::<Vec<_>>();
    let mut uploads = UniformUploadBatch::default();
    let mut pending = Vec::new();

    // Two submissions share the retained scratch buffer, with no poll/map/wait
    // between them. Every frame captures old/new bytes and actual binding pixels.
    for generation in 1..=2 {
        let old_capacity = uploads.allocation_bytes();
        let bytes = values[..INITIAL_COUNT].iter().map(Vec::len).sum::<usize>();
        uploads.begin();
        assert!(uploads.prepare(
            bytes,
            INITIAL_COUNT,
            BYTE_LIMIT,
            device.limits().max_buffer_size
        ));
        for (index, destination) in values[..INITIAL_COUNT].iter_mut().enumerate() {
            *destination = value(index, generation);
            assert!(uploads.stage(index, destination));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let report = uploads.flush(device, queue, &mut encoder, &bindings);
        assert_eq!(report.uploaded_bytes, bytes);
        assert_eq!(report.copy_count, INITIAL_COUNT);
        assert_eq!(report.created_buffers, usize::from(generation == 1));
        assert_eq!(report.buffer_bytes, uploads.allocation_bytes());
        assert!(report.buffer_bytes >= bytes && report.buffer_bytes <= BYTE_LIMIT);
        assert!(report.peak_buffer_bytes >= old_capacity);
        pending.push(capture_uniforms(
            device,
            &mut encoder,
            &bindings[..INITIAL_COUNT],
            &values[..INITIAL_COUNT],
        ));
        pending.push(capture_pixels(
            device,
            &pipeline,
            &mut encoder,
            &bindings[..INITIAL_COUNT],
            &values[..INITIAL_COUNT],
        ));
        queue.submit([encoder.finish()]);
    }
    for readback in pending {
        assert_readback(device, readback);
    }

    // An accepted plan with no changed records must issue no queue transfer.
    uploads.begin();
    assert!(uploads.prepare(
        96 * INITIAL_COUNT,
        INITIAL_COUNT,
        BYTE_LIMIT,
        device.limits().max_buffer_size
    ));
    let mut encoder = device.create_command_encoder(&Default::default());
    let report = uploads.flush(device, queue, &mut encoder, &bindings);
    assert_eq!(
        (
            report.uploaded_bytes,
            report.copy_count,
            report.created_buffers
        ),
        (0, 0, 0)
    );
    queue.submit([encoder.finish()]);

    // Sparse binding indices are not record indices: 32 copies may target index
    // 999. First, middle and last destinations change, all holes stay untouched.
    let indices: Vec<_> = (0..32)
        .map(|index| index * (INITIAL_COUNT - 1) / 31)
        .collect();
    let bytes = indices.iter().map(|&index| values[index].len()).sum();
    uploads.begin();
    assert!(uploads.prepare(
        bytes,
        indices.len(),
        BYTE_LIMIT,
        device.limits().max_buffer_size
    ));
    for &index in &indices {
        values[index] = value(index, 3);
        assert!(uploads.stage(index, &values[index]));
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    let report = uploads.flush(device, queue, &mut encoder, &bindings);
    assert_eq!(
        (
            report.uploaded_bytes,
            report.copy_count,
            report.created_buffers
        ),
        (bytes, 32, 0)
    );
    let readback = capture_uniforms(
        device,
        &mut encoder,
        &bindings[..INITIAL_COUNT],
        &values[..INITIAL_COUNT],
    );
    queue.submit([encoder.finish()]);
    assert_readback(device, readback);

    // Growth replaces scratch once, shrink retains it but uploads/copies only
    // the current prefix. Old trailing records must never be replayed.
    for (count, generation) in [(GROWN_COUNT, 4), (INITIAL_COUNT, 5)] {
        let old_capacity = uploads.allocation_bytes();
        let bytes = values[..count].iter().map(Vec::len).sum::<usize>();
        uploads.begin();
        assert!(uploads.prepare(bytes, count, BYTE_LIMIT, device.limits().max_buffer_size));
        for (index, destination) in values[..count].iter_mut().enumerate() {
            *destination = value(index, generation);
            assert!(uploads.stage(index, destination));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let report = uploads.flush(device, queue, &mut encoder, &bindings);
        assert_eq!(report.uploaded_bytes, bytes);
        assert_eq!(report.copy_count, count);
        assert_eq!(report.created_buffers, usize::from(count == GROWN_COUNT));
        if report.created_buffers > 0 {
            assert!(report.peak_buffer_bytes >= old_capacity + report.buffer_bytes);
        } else {
            assert_eq!(report.buffer_bytes, old_capacity);
        }
        let readback = capture_uniforms(device, &mut encoder, &bindings, &values);
        queue.submit([encoder.finish()]);
        assert_readback(device, readback);
    }

    // A temporary alias has no dirty slot of its own. Stage one independent
    // record per 32 real destinations and read all 64 interleaved aliases.
    let aliases = bindings[..32]
        .iter()
        .flat_map(|binding| [binding.clone(), binding.clone()])
        .collect::<Vec<_>>();
    let mut alias_values = Vec::new();
    uploads.begin();
    let bytes = values[..32].iter().map(Vec::len).sum();
    assert!(uploads.prepare(bytes, 32, BYTE_LIMIT, device.limits().max_buffer_size));
    for (index, destination) in values[..32].iter_mut().enumerate() {
        *destination = value(index, 6);
        assert!(uploads.stage(index * 2, destination));
        alias_values.extend([destination.clone(), destination.clone()]);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    let report = uploads.flush(device, queue, &mut encoder, &aliases);
    assert_eq!((report.uploaded_bytes, report.copy_count), (bytes, 32));
    let readback = capture_uniforms(device, &mut encoder, &aliases, &alias_values);
    queue.submit([encoder.finish()]);
    assert_readback(device, readback);

    // Disabled, too-small and over-device-limit plans fall back to ordinary
    // writes. A failed prepare must not replay an earlier accepted plan.
    for (count, byte_limit, max_buffer_size) in [
        (31, BYTE_LIMIT, device.limits().max_buffer_size),
        (32, 0, device.limits().max_buffer_size),
        (32, 31, device.limits().max_buffer_size),
        (32, BYTE_LIMIT, 31),
    ] {
        uploads.begin();
        let bytes = values[..count].iter().map(Vec::len).sum();
        assert!(!uploads.prepare(bytes, count, byte_limit, max_buffer_size));
        for (index, destination) in values[..count].iter_mut().enumerate() {
            *destination = value(index, 7);
            assert!(!uploads.stage(index, destination));
            queue.write_buffer(&bindings[index]._buffer, 0, destination);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let report = uploads.flush(device, queue, &mut encoder, &bindings);
        assert_eq!((report.uploaded_bytes, report.copy_count), (0, 0));
        let readback = capture_uniforms(device, &mut encoder, &bindings[..count], &values[..count]);
        queue.submit([encoder.finish()]);
        assert_readback(device, readback);
    }
    uploads.clear();
    assert_eq!(uploads.allocation_bytes(), 0);
    assert_eq!(uploads.cpu_bytes(), 0);
}

fn value(index: usize, generation: usize) -> Vec<u8> {
    let size = [64, 96, 32][index % 3];
    let mut bytes = vec![0; size];
    for (word, chunk) in bytes.chunks_exact_mut(4).enumerate() {
        let sentinel = ((index as u32) << 16) ^ ((generation as u32) << 8) ^ word as u32;
        chunk.copy_from_slice(&sentinel.to_le_bytes());
    }
    let color = [
        ((index * 37 + generation * 41) % 251) as f32 / 255.0,
        ((index * 71 + generation * 13) % 251) as f32 / 255.0,
        ((index * 19 + generation * 83) % 251) as f32 / 255.0,
        1.0_f32,
    ];
    bytes[..16].copy_from_slice(bytemuck::bytes_of(&color));
    bytes
}

fn binding(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bytes: &[u8]) -> FrameBinding {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform transfer oracle destination"),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform transfer oracle binding"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    FrameBinding {
        _buffer: buffer,
        bind_group,
    }
}

fn capture_uniforms(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    bindings: &[FrameBinding],
    values: &[Vec<u8>],
) -> Readback {
    let expected = values.iter().flatten().copied().collect::<Vec<_>>();
    let buffer = readback_buffer(device, expected.len());
    let mut offset = 0;
    for (binding, bytes) in bindings.iter().zip(values) {
        encoder.copy_buffer_to_buffer(&binding._buffer, 0, &buffer, offset, bytes.len() as u64);
        offset += bytes.len() as u64;
    }
    Readback {
        buffer,
        expected,
        tolerance: 0,
    }
}

fn pixel_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("uniform transfer oracle shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            r#"
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return color; }
"#,
        )),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("uniform transfer oracle pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("uniform transfer oracle pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn capture_pixels(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    encoder: &mut wgpu::CommandEncoder,
    bindings: &[FrameBinding],
    values: &[Vec<u8>],
) -> Readback {
    let width = 64;
    let height = (bindings.len() as u32).div_ceil(width);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("uniform transfer oracle pixels"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("uniform transfer oracle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_scissor_rect(index as u32 % width, index as u32 / width, 1, 1);
            pass.draw(0..3, 0..1);
        }
    }
    let mut expected = vec![0; (width * height * 4) as usize];
    for (index, bytes) in values.iter().enumerate() {
        for channel in 0..4 {
            let offset = channel * 4;
            let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            expected[index * 4 + channel] = (value * 255.0).round() as u8;
        }
    }
    let buffer = readback_buffer(device, expected.len());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(height),
            },
        },
        size,
    );
    Readback {
        buffer,
        expected,
        tolerance: 1,
    }
}

fn readback_buffer(device: &wgpu::Device, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform transfer oracle readback"),
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn assert_readback(device: &wgpu::Device, readback: Readback) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let slice = readback.buffer.slice(..);
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
    let actual = slice.get_mapped_range().unwrap();
    assert_eq!(actual.len(), readback.expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&readback.expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= readback.tolerance,
            "uniform transfer byte {index}: {actual} != {expected}"
        );
    }
}
