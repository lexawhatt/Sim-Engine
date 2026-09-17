//! Low-level instance, depth and mathematical-edge resources.

use super::*;

pub(super) fn create_instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D instance buffer"),
        size: (capacity.max(1) * std::mem::size_of::<MeshInstanceGpu>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_edge_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    depth_compare: wgpu::CompareFunction,
    fragment_entry: &'static str,
    label: &'static str,
    clip_space: bool,
) -> wgpu::RenderPipeline {
    let (vertex_entry, depth_bias) = if depth_compare == wgpu::CompareFunction::Greater {
        (
            "mesh3d_hidden_edge_vs_main",
            wgpu::DepthBiasState {
                constant: HIDDEN_EDGE_DEPTH_BIAS,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        )
    } else {
        (
            "mesh3d_visible_edge_vs_main",
            wgpu::DepthBiasState {
                constant: VISIBLE_EDGE_DEPTH_BIAS,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        )
    };
    let vertex_entry = if clip_space {
        if depth_compare == wgpu::CompareFunction::Greater {
            "mesh3d_clipped_hidden_edge_vs_main"
        } else {
            "mesh3d_clipped_visible_edge_vs_main"
        }
    } else {
        vertex_entry
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(if clip_space {
                SurfaceClipEdge::LAYOUT
            } else {
                MeshEdgeGpu::LAYOUT
            })],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: depth_bias,
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn align_to(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment).saturating_mul(alignment)
    }
}

pub(super) fn create_edge_object_buffer(
    device: &wgpu::Device,
    stride: usize,
    capacity: usize,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D edge object uniform buffer"),
        size: capacity.max(1).saturating_mul(stride) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(super) fn create_edge_object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine 3D edge object bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<EdgeObjectUniform>() as u64),
            }),
        }],
    })
}

pub(super) fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine 3D depth target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}
