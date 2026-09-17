//! Shared primitive, particle, heatmap and composition pipeline construction.

use std::borrow::Cow;

use super::{
    BlendMode, CameraUniform, CompositeUniform, DynamicGpu, HeatmapUniform, ParticleGpu,
    ParticleUnitVertex, Vertex,
    visualization::{CompositionPipelines, create_composition_pipeline},
};

pub(super) struct PipelineResources {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) target_pipeline: wgpu::RenderPipeline,
    pub(super) dynamic_pipeline: wgpu::RenderPipeline,
    pub(super) particle_pipeline: wgpu::RenderPipeline,
    pub(super) target_particle_pipeline: wgpu::RenderPipeline,
    pub(super) heatmap_pipeline: wgpu::RenderPipeline,
    pub(super) target_heatmap_pipeline: wgpu::RenderPipeline,
    pub(super) composition_pipelines: CompositionPipelines,
    pub(super) target_composition_pipelines: CompositionPipelines,
    pub(super) camera_uniform_buffer: wgpu::Buffer,
    pub(super) camera_bind_group: wgpu::BindGroup,
    pub(super) camera_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) heatmap_uniform_buffer: wgpu::Buffer,
    pub(super) heatmap_bind_group_layout: wgpu::BindGroupLayout,
}

pub(super) fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> PipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine flat color shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("primitive.wgsl"))),
    });
    let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine camera uniform buffer"),
        size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine camera bind group"),
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine shape pipeline layout"),
        bind_group_layouts: &[Some(&camera_bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine shape pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Vertex::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let target_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine target shape pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Vertex::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let dynamic_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine dynamic mesh pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("dynamic_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(DynamicGpu::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine particle pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("particle_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(ParticleUnitVertex::LAYOUT), Some(ParticleGpu::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let target_particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine target particle pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("particle_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(ParticleUnitVertex::LAYOUT), Some(ParticleGpu::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let heatmap_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine heatmap uniform buffer"),
        size: std::mem::size_of::<HeatmapUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let heatmap_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine heatmap bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let heatmap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine heatmap pipeline layout"),
        bind_group_layouts: &[Some(&heatmap_bind_group_layout)],
        immediate_size: 0,
    });
    let heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine heatmap pipeline"),
        layout: Some(&heatmap_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("heatmap_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("heatmap_fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let target_heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine render-target heatmap pipeline"),
        layout: Some(&heatmap_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("heatmap_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("heatmap_fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let composition_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine composition uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composition_secondary_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine composition secondary uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composition_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine composition bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let composition_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine composition pipeline layout"),
            bind_group_layouts: &[Some(&composition_bind_group_layout)],
            immediate_size: 0,
        });
    let composition_pipelines = CompositionPipelines {
        alpha: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Alpha,
        ),
        additive: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Additive,
        ),
        replace: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Replace,
        ),
        uniform_buffer: composition_uniform_buffer,
        secondary_uniform_buffer: composition_secondary_uniform_buffer,
        bind_group_layout: composition_bind_group_layout,
    };
    let target_composition_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine target composition uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let target_composition_secondary_uniform_buffer =
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine target composition secondary uniform buffer"),
            size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    let target_composition_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine target composition bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let target_composition_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine target composition pipeline layout"),
            bind_group_layouts: &[Some(&target_composition_bind_group_layout)],
            immediate_size: 0,
        });
    let target_composition_pipelines = CompositionPipelines {
        alpha: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Alpha,
        ),
        additive: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Additive,
        ),
        replace: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Replace,
        ),
        uniform_buffer: target_composition_uniform_buffer,
        secondary_uniform_buffer: target_composition_secondary_uniform_buffer,
        bind_group_layout: target_composition_bind_group_layout,
    };

    PipelineResources {
        pipeline,
        target_pipeline,
        dynamic_pipeline,
        particle_pipeline,
        target_particle_pipeline,
        heatmap_pipeline,
        target_heatmap_pipeline,
        composition_pipelines,
        target_composition_pipelines,
        camera_uniform_buffer,
        camera_bind_group,
        camera_bind_group_layout,
        heatmap_uniform_buffer,
        heatmap_bind_group_layout,
    }
}
