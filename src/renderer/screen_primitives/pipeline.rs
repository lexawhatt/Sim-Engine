use super::ScreenPrimitiveGpu;

pub(in crate::renderer) fn create_pipeline(
    device: &wgpu::Device,
    camera_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine analytic screen primitive shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("primitive.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine analytic screen primitive layout"),
        bind_group_layouts: &[Some(camera_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine analytic screen primitive pipeline"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ScreenPrimitiveGpu>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x2, 4 => Float32x2, 5 => Float32x2, 6 => Float32x4],
            })],
        },
        primitive: wgpu::PrimitiveState::default(), depth_stencil: None,
        multisample: wgpu::MultisampleState { count: sample_count, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
        }), multiview_mask: None, cache: None,
    })
}
