//! Shared depth, blending and projected winding policy for all surface layouts.

use super::*;

pub(super) struct SurfacePipelines([wgpu::RenderPipeline; 4]);

impl std::ops::Deref for SurfacePipelines {
    type Target = wgpu::RenderPipeline;
    fn deref(&self) -> &Self::Target {
        &self.0[0]
    }
}

impl SurfacePipelines {
    pub(super) fn for_style(&self, style: crate::SurfaceStyle3d) -> &wgpu::RenderPipeline {
        let blend = usize::from(style.alpha_mode() == SurfaceAlphaMode3d::Blend);
        let cull = usize::from(style.sidedness() == SurfaceSidedness3d::FrontOnly);
        &self.0[blend * 2 + cull]
    }
}

#[derive(Clone, Copy)]
pub(super) struct SurfaceLayout {
    pub(super) clipped: bool,
    pub(super) textured: bool,
    pub(super) colored: bool,
}

pub(super) fn create_surface_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    source: SurfaceLayout,
) -> SurfacePipelines {
    let mut buffers = vec![Some(match (source.clipped, source.textured) {
        (false, _) => MeshVertexGpu::LAYOUT,
        (true, false) => SurfaceClipVertex::LAYOUT,
        (true, true) => SurfaceClipVertex::TEXTURED_LAYOUT,
    })];
    if source.textured && !source.clipped {
        buffers.push(Some(MeshUvGpu::LAYOUT));
    }
    buffers.push(Some(MeshInstanceGpu::LAYOUT));
    if source.colored {
        buffers.push(Some(MeshColorGpu::LAYOUT));
    }
    let vertex_entry = match (source.textured, source.clipped, source.colored) {
        (false, false, false) => "mesh3d_vs_main",
        (false, false, true) => "mesh3d_colored_vs_main",
        (false, true, false) => "mesh3d_clipped_surface_vs_main",
        (false, true, true) => "mesh3d_colored_clipped_vs_main",
        (true, false, false) => "retained_vs_main",
        (true, false, true) => "colored_retained_vs_main",
        (true, true, false) => "clipped_vs_main",
        (true, true, true) => "colored_clipped_vs_main",
    };
    SurfacePipelines(std::array::from_fn(|index| {
        let blended = index >= 2;
        let front_only = index % 2 == 1;
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sim-engine 3D surface material pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: front_only.then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(!blended),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(if source.textured {
                    "fs_main"
                } else {
                    "mesh3d_fs_main"
                }),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: blended.then_some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }))
}
