//! Independent color resolution with native-resolution visibility.

use super::{
    Camera3d, LogicalViewport, Mesh3dRenderBudget, Mesh3dRenderError, Mesh3dRenderReport,
    RenderTarget3d, Scene3d, SurfaceAlphaMode3d, WgpuRenderer,
};
use crate::renderer::{RenderTargetBudget, RenderTargetLifecycleError};
use std::{error::Error, fmt};

/// Reusable low-resolution color, native visibility, detail and final targets.
///
/// The base scene is rendered twice: low-resolution shading and native coverage/
/// depth. Detail geometry is rendered at native resolution. A final depth-aware
/// pass selects the nearest surface and writes shared native depth. This costs
/// three scene draws; it does not claim to reduce vertex or visibility work.
/// Supports opaque/masked surfaces; transparent blends and mathematical edges
/// require the ordinary scene path and are rejected explicitly here.
/// All attachments and pipelines belong to one logical renderer generation.
pub struct LayeredTarget3d {
    low: RenderTarget3d,
    base: RenderTarget3d,
    detail: RenderTarget3d,
    output: RenderTarget3d,
    pipeline: wgpu::RenderPipeline,
    bindings: wgpu::BindGroup,
    allocation_bytes: usize,
}
impl LayeredTarget3d {
    /// Last complete output, with native color and depth. Redraw after creation,
    /// resize or device recovery. A rejected render retains the previous output.
    pub const fn output(&self) -> &RenderTarget3d {
        &self.output
    }
    /// Base color resolution in physical texels.
    pub fn low_size(&self) -> (u32, u32) {
        (self.low.width(), self.low.height())
    }
    /// Native output resolution in physical texels.
    pub fn size(&self) -> (u32, u32) {
        (self.output.width(), self.output.height())
    }
    /// Nominal total color/depth storage for all four targets; driver pipeline
    /// metadata and in-flight retirement are excluded.
    pub const fn allocation_bytes(&self) -> usize {
        self.allocation_bytes
    }
}

/// Reports all three submitted scene workloads; final composition adds one draw.
#[derive(Debug)]
pub struct LayeredRenderReport3d {
    /// Low-resolution base shading workload.
    pub low: Mesh3dRenderReport,
    /// Native base visibility workload (currently includes shading).
    pub base: Mesh3dRenderReport,
    /// Native detail workload.
    pub detail: Mesh3dRenderReport,
}

/// Synchronous layered rendering or allocation failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum LayeredRenderError3d {
    /// Target admission or creation failed.
    Target(RenderTargetLifecycleError),
    /// A scene failed ordinary numerical/resource validation.
    Scene(Mesh3dRenderError),
    /// Blended surfaces or mathematical edges cannot be independently depth-composed.
    UnsupportedSurface,
}
impl fmt::Display for LayeredRenderError3d {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target(error) => write!(formatter, "{error}"),
            Self::Scene(error) => write!(formatter, "{error}"),
            Self::UnsupportedSurface => write!(
                formatter,
                "layered scenes require opaque or masked surfaces without mathematical edges"
            ),
        }
    }
}
impl Error for LayeredRenderError3d {}
impl From<RenderTargetLifecycleError> for LayeredRenderError3d {
    fn from(error: RenderTargetLifecycleError) -> Self {
        Self::Target(error)
    }
}
impl From<Mesh3dRenderError> for LayeredRenderError3d {
    fn from(error: Mesh3dRenderError) -> Self {
        Self::Scene(error)
    }
}

fn validate_surface_modes(scene: &Scene3d) -> Result<(), LayeredRenderError3d> {
    for instance in scene
        .instances()
        .iter()
        .filter(|instance| instance.is_visible())
    {
        let style = instance.style();
        if style.wireframe_style().is_some()
            || style
                .surface_style()
                .is_some_and(|surface| surface.alpha_mode() == SurfaceAlphaMode3d::Blend)
        {
            return Err(LayeredRenderError3d::UnsupportedSurface);
        }
    }
    Ok(())
}

impl WgpuRenderer {
    /// Allocates a reusable two-resolution scene compositor. Both physical aspects
    /// must match `viewport`. The budget counts low color/depth plus three native
    /// color/depth targets before any attachment allocation.
    pub fn create_layered_target3d(
        &self,
        low_size: (u32, u32),
        native_size: (u32, u32),
        viewport: LogicalViewport,
        budget: RenderTargetBudget,
    ) -> Result<LayeredTarget3d, LayeredRenderError3d> {
        let low_bytes = self.target_storage_bytes(low_size.0, low_size.1, true)?;
        let native_bytes = self.target_storage_bytes(native_size.0, native_size.1, true)?;
        let allocation_bytes = native_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(low_bytes))
            .ok_or(RenderTargetLifecycleError::SizeOverflow)?;
        budget.admit(0, allocation_bytes)?;
        let low =
            self.create_render_target3d_with_budget(low_size.0, low_size.1, viewport, budget)?;
        let base = self.create_render_target3d_with_budget(
            native_size.0,
            native_size.1,
            viewport,
            budget,
        )?;
        let detail = self.create_render_target3d_with_budget(
            native_size.0,
            native_size.1,
            viewport,
            budget,
        )?;
        let output = self.create_render_target3d_with_budget(
            native_size.0,
            native_size.1,
            viewport,
            budget,
        )?;
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim-engine layered visibility layout"),
                entries: &std::array::from_fn::<_, 6, _>(|index| wgpu::BindGroupLayoutEntry {
                    binding: index as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: if index % 2 == 0 {
                            wgpu::TextureSampleType::Float { filterable: false }
                        } else {
                            wgpu::TextureSampleType::Depth
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }),
            });
        let views = [
            &low.color.view,
            &low.depth_view,
            &base.color.view,
            &base.depth_view,
            &detail.color.view,
            &detail.depth_view,
        ];
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine layered visibility bindings"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 6, _>(|index| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: wgpu::BindingResource::TextureView(views[index]),
            }),
        });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sim-engine layered visibility"),
                source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sim-engine layered visibility composition"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: super::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        Ok(LayeredTarget3d {
            low,
            base,
            detail,
            output,
            pipeline,
            bindings,
            allocation_bytes,
        })
    }

    /// Transactional resize/device restoration, charging all old/new attachments.
    /// Returned errors retain the old output. Success requires a full redraw.
    pub fn replace_layered_target3d(
        &self,
        target: &mut LayeredTarget3d,
        low_size: (u32, u32),
        native_size: (u32, u32),
        viewport: LogicalViewport,
        budget: RenderTargetBudget,
    ) -> Result<crate::RenderTargetReplacementReport, LayeredRenderError3d> {
        let low_bytes = self.target_storage_bytes(low_size.0, low_size.1, true)?;
        let new_bytes = self
            .target_storage_bytes(native_size.0, native_size.1, true)?
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(low_bytes))
            .ok_or(RenderTargetLifecycleError::SizeOverflow)?;
        let report = budget.admit(target.allocation_bytes, new_bytes)?;
        let replacement = self.create_layered_target3d(low_size, native_size, viewport, budget)?;
        *target = replacement;
        Ok(report)
    }

    /// Renders low-resolution base color plus native detail with native mutual
    /// occlusion. The same camera is used for every pass. Ties select the base
    /// (equivalent to Less with base inserted first). Base color uses nearest
    /// texel sampling; native color fills uncovered low-resolution silhouettes.
    /// The output changes only after all three scene submissions succeed.
    /// `budgets[0]` applies to each base draw; `budgets[1]` to detail.
    /// Effects on the base never affect detail colors.
    pub fn render_layered_scene3d(
        &mut self,
        target: &LayeredTarget3d,
        base: &Scene3d,
        detail: &Scene3d,
        camera: Camera3d,
        budgets: [Mesh3dRenderBudget; 2],
    ) -> Result<LayeredRenderReport3d, LayeredRenderError3d> {
        let [base_budget, detail_budget] = budgets.map(|budget| budget.with_existing_depth(false));
        validate_surface_modes(base)?;
        validate_surface_modes(detail)?;
        self.validate_scene3d_for_target(&target.low, base, camera, base_budget)?;
        self.validate_scene3d_for_target(&target.base, base, camera, base_budget)?;
        self.validate_scene3d_for_target(&target.detail, detail, camera, detail_budget)?;
        let low =
            self.render_scene3d_to_target_with_budget(&target.low, base, camera, base_budget)?;
        let base =
            self.render_scene3d_to_target_with_budget(&target.base, base, camera, base_budget)?;
        let detail = self.render_scene3d_to_target_with_budget(
            &target.detail,
            detail,
            camera,
            detail_budget,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine layered frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine native shared visibility"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.output.color.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.output.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target.pipeline);
            pass.set_bind_group(0, &target.bindings, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        target.output.color.mark_written();
        Ok(LayeredRenderReport3d { low, base, detail })
    }
}
