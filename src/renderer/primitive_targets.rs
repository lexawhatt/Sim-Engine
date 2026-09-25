//! Offscreen drawing of the same retained analytic screen primitive resources.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    CameraUniform, LogicalScreenPosition, LogicalViewport, LogicalViewportRegion,
    PhysicalPerLogical, RenderReport, RenderStatus, RenderTarget2d, RenderTargetError,
    RenderTargetLoad, RendererFrameError, ScreenPrimitiveBatch2d, Vec2, frame,
    premultiplied_wgpu_color, prepared_scene_belongs_to, render_report, screen_camera,
};

/// A borrowed offscreen encoder does not require or acquire a window surface.
pub(super) struct PrimitiveTargetRenderer<'renderer> {
    pub(super) renderer_identity: &'renderer Arc<()>,
    pub(super) device: &'renderer wgpu::Device,
    pub(super) queue: &'renderer wgpu::Queue,
    pub(super) pipeline: &'renderer wgpu::RenderPipeline,
    pub(super) camera_uniform_buffer: &'renderer wgpu::Buffer,
    pub(super) camera_bind_group: &'renderer wgpu::BindGroup,
}

impl PrimitiveTargetRenderer<'_> {
    pub(super) fn draw(
        &self,
        target: &RenderTarget2d,
        batch: &ScreenPrimitiveBatch2d,
        pixel_scale: PhysicalPerLogical,
        viewport: Option<LogicalViewportRegion>,
        load: RenderTargetLoad,
    ) -> Result<RenderReport, RenderTargetError> {
        let started = Instant::now();
        if !prepared_scene_belongs_to(self.renderer_identity, &target.renderer_identity)
            || !prepared_scene_belongs_to(self.renderer_identity, &batch.renderer_identity)
        {
            return Err(RenderTargetError::RendererMismatch);
        }
        if let RenderTargetLoad::Clear(color) = load
            && !color.is_normalized()
        {
            return Err(RenderTargetError::InvalidBackground);
        }
        let scale = pixel_scale.get();
        let target_viewport = LogicalViewport::new(
            target.width() as f32 / scale,
            target.height() as f32 / scale,
        )
        .map_err(|_| RenderTargetError::Frame(RendererFrameError::InvalidViewport))?;
        let region = viewport.unwrap_or(
            LogicalViewportRegion::new(
                LogicalScreenPosition::from_vec2(Vec2::ZERO),
                target_viewport,
            )
            .map_err(|_| RenderTargetError::Frame(RendererFrameError::InvalidViewport))?,
        );
        let origin = region.origin().to_vec2();
        let maximum = origin + region.viewport().size();
        if !maximum.is_finite()
            || origin.x < 0.0
            || origin.y < 0.0
            || maximum.x > target_viewport.width()
            || maximum.y > target_viewport.height()
            || super::logical_viewport_scissor(
                origin,
                region.viewport(),
                scale,
                target.width(),
                target.height(),
            )
            .is_none()
        {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidViewport,
            ));
        }
        let camera = screen_camera(region.viewport())
            .map_err(|_| RenderTargetError::Frame(RendererFrameError::InvalidGeometryTransform))?;
        let uniform =
            CameraUniform::new_in_region(camera, region.viewport(), origin, target_viewport)
                .ok_or(RenderTargetError::Frame(
                    RendererFrameError::InvalidGeometryTransform,
                ))?;
        if !batch.is_safe_for(uniform, scale) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidGeometryTransform,
            ));
        }
        let preparation = started.elapsed();
        let uniform_started = Instant::now();
        self.queue
            .write_buffer(self.camera_uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let uniform_upload = uniform_started.elapsed();
        let encode_started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine analytic screen target encoder"),
            });
        {
            let attachment = wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match load {
                        RenderTargetLoad::Load => wgpu::LoadOp::Load,
                        RenderTargetLoad::Clear(color) => {
                            wgpu::LoadOp::Clear(premultiplied_wgpu_color(color))
                        }
                    },
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine analytic screen target pass"),
                color_attachments: &[Some(attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            frame::encode_screen_primitives(
                &mut pass,
                self.pipeline,
                self.camera_bind_group,
                batch,
                region,
                None,
                scale,
                target.width(),
                target.height(),
            );
        }
        self.queue.submit([encoder.finish()]);
        Ok(render_report(
            RenderStatus::Drawn,
            preparation,
            Duration::ZERO,
            uniform_upload,
            Duration::ZERO,
            encode_started.elapsed(),
            started.elapsed(),
            true,
            false,
            batch.tessellation_stats(),
        ))
    }
}
