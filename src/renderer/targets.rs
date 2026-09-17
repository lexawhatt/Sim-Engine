//! Offscreen target ownership, presentation and temporal trail accumulation.

use super::{
    Arc, BlendMode, Color, CompositeUniform, Duration, Error, Instant, RenderReport, RenderStatus,
    RendererFrameError, RendererSurfaceStatus, TessellationStats, WgpuRenderer, fmt,
    premultiplied_wgpu_color, prepared_scene_belongs_to, render_report,
    render_target_allocation_bytes,
};

/// A renderer-owned offscreen color target in physical texture pixels.
///
/// Targets preserve GPU pixels only; a host must redraw them after device
/// recreation. Use the renderer's physical [`WgpuRenderer::size`] when a
/// target should match the presentation surface exactly.
pub struct RenderTarget2d {
    pub(super) renderer_identity: Arc<()>,
    pub(super) resource_identity: Arc<()>,
    pub(super) _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) allocation_bytes: usize,
}

/// A pair of offscreen targets used for bounded temporal accumulation.
///
/// Each accumulation retains a caller-selected amount of the previous frame
/// and then composites a distinct source target over it. The buffer owns no
/// domain state and can therefore serve particles, scalar fields, or any later
/// target-rendering producer.
pub struct TrailBuffer2d {
    renderer_identity: Arc<()>,
    front: RenderTarget2d,
    back: RenderTarget2d,
}

impl TrailBuffer2d {
    /// Returns trail-buffer width in physical texture pixels.
    pub fn width(&self) -> u32 {
        self.front.width()
    }

    /// Returns trail-buffer height in physical texture pixels.
    pub fn height(&self) -> u32 {
        self.front.height()
    }

    /// Returns trail-buffer physical dimensions.
    pub fn size(&self) -> (u32, u32) {
        self.front.size()
    }

    /// Returns nominal texel-storage bytes for both ping-pong textures.
    pub fn allocation_bytes(&self) -> usize {
        self.front
            .allocation_bytes()
            .saturating_add(self.back.allocation_bytes())
    }
}

impl RenderTarget2d {
    /// Returns target width in physical texture pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns target height in physical texture pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the physical texture dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns nominal single-level texel-storage bytes implied by its format.
    /// Backend row/tile/page alignment and resource metadata are not observable
    /// through wgpu and are therefore excluded.
    pub fn allocation_bytes(&self) -> usize {
        self.allocation_bytes
    }
}

/// Failure while creating or composing a [`RenderTarget2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderTargetError {
    /// Target dimensions must both be non-zero.
    ZeroDimension,
    /// Dimensions exceed the current device's supported target size or allocation range.
    DimensionsTooLarge,
    /// The target belongs to another renderer and GPU device.
    RendererMismatch,
    /// Opacity must be finite and within `0.0..=1.0`.
    InvalidOpacity,
    /// A temporal source must not be one of its destination ping-pong targets.
    SourceAliasesDestination,
    /// The destination clear color was not normalized linear RGBA.
    InvalidBackground,
    /// Surface acquisition or presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for RenderTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "render target dimensions must be non-zero"),
            Self::DimensionsTooLarge => {
                write!(formatter, "render target dimensions exceed device limits")
            }
            Self::RendererMismatch => {
                write!(formatter, "render target belongs to a different renderer")
            }
            Self::InvalidOpacity => write!(
                formatter,
                "render target opacity must be finite and in 0.0..=1.0"
            ),
            Self::SourceAliasesDestination => {
                write!(
                    formatter,
                    "trail source must be distinct from its destination targets"
                )
            }
            Self::InvalidBackground => {
                write!(formatter, "render target background must be normalized")
            }
            Self::Frame(error) => write!(formatter, "render target frame failed: {error}"),
        }
    }
}

impl Error for RenderTargetError {}

impl WgpuRenderer {
    /// Creates an offscreen target with physical texture dimensions.
    pub fn create_render_target(
        &self,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget2d, RenderTargetError> {
        if width == 0 || height == 0 {
            return Err(RenderTargetError::ZeroDimension);
        }
        if width > self.device.limits().max_texture_dimension_2d
            || height > self.device.limits().max_texture_dimension_2d
        {
            return Err(RenderTargetError::DimensionsTooLarge);
        }
        let allocation_bytes = render_target_allocation_bytes(self.config.format, width, height)
            .ok_or(RenderTargetError::DimensionsTooLarge)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(RenderTarget2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            resource_identity: Arc::new(()),
            _texture: texture,
            view,
            width,
            height,
            allocation_bytes,
        })
    }

    /// Creates an empty ping-pong target pair for temporal accumulation.
    pub fn create_trail_buffer(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<TrailBuffer2d, RenderTargetError> {
        let front = self.create_render_target(width, height)?;
        let back = self.create_render_target(width, height)?;
        let trails = TrailBuffer2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            front,
            back,
        };
        self.draw_clear_trail_buffer(&trails, Color::rgba(0.0, 0.0, 0.0, 0.0), Instant::now());
        Ok(trails)
    }

    /// Recreates an empty target with the source dimensions after device recovery.
    ///
    /// GPU pixels are deliberately not retained; callers redraw their source
    /// state into the restored target.
    pub fn restore_render_target(
        &self,
        source: &RenderTarget2d,
    ) -> Result<RenderTarget2d, RenderTargetError> {
        self.create_render_target(source.width(), source.height())
    }

    /// Recreates an empty, deterministically cleared trail buffer after recovery.
    pub fn restore_trail_buffer(
        &mut self,
        source: &TrailBuffer2d,
    ) -> Result<TrailBuffer2d, RenderTargetError> {
        self.create_trail_buffer(source.width(), source.height())
    }

    /// Retains one bounded frame of history and composites a fresh source target.
    ///
    /// `retention` and `source_opacity` must be finite values in `0.0..=1.0`.
    /// The source must be a distinct target; use [`WgpuRenderer::clear_trail_buffer`]
    /// to discard accumulated history deterministically.
    pub fn accumulate_trail_buffer(
        &mut self,
        trails: &mut TrailBuffer2d,
        source: &RenderTarget2d,
        retention: f32,
        source_opacity: f32,
        source_blend: BlendMode,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        self.validate_render_target(source)?;
        if Arc::ptr_eq(&source.resource_identity, &trails.front.resource_identity)
            || Arc::ptr_eq(&source.resource_identity, &trails.back.resource_identity)
        {
            return Err(RenderTargetError::SourceAliasesDestination);
        }
        if !retention.is_finite()
            || !(0.0..=1.0).contains(&retention)
            || !source_opacity.is_finite()
            || !(0.0..=1.0).contains(&source_opacity)
        {
            return Err(RenderTargetError::InvalidOpacity);
        }
        Ok(self.draw_trail_accumulation(
            trails,
            source,
            retention,
            source_opacity,
            source_blend,
            Instant::now(),
        ))
    }

    /// Clears both ping-pong targets, resetting all retained temporal history.
    pub fn clear_trail_buffer(
        &mut self,
        trails: &mut TrailBuffer2d,
        color: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        if !color.is_normalized() {
            return Err(RenderTargetError::InvalidBackground);
        }
        Ok(self.draw_clear_trail_buffer(trails, color, Instant::now()))
    }

    /// Presents the most recently accumulated temporal target.
    pub fn compose_trail_buffer(
        &mut self,
        trails: &TrailBuffer2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        self.compose_render_target(&trails.front, blend_mode, opacity, background)
    }

    /// Composes a previously rendered target over the presentation surface.
    pub fn compose_render_target(
        &mut self,
        target: &RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_render_target(target)?;
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(RenderTargetError::InvalidOpacity);
        }
        if !background.is_normalized() {
            return Err(RenderTargetError::InvalidBackground);
        }
        self.draw_composed_render_target(target, blend_mode, opacity, background, Instant::now())
            .map_err(RenderTargetError::Frame)
    }

    fn draw_composed_render_target(
        &mut self,
        target: &RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let _ = self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RendererFrameError::Surface(
                    RendererSurfaceStatus::Validation,
                ));
            }
        };
        let surface_acquire = acquire_started_at.elapsed();
        let upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(opacity)),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine composition bind group"),
            layout: &self.composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(multisample_target) => (&multisample_target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine composition encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(background.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.composition_pipelines.pipeline(blend_mode));
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
        self.queue.present(surface_texture);
        Ok(render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            upload,
            Duration::ZERO,
            surface_acquire,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_trail_accumulation(
        &mut self,
        trails: &mut TrailBuffer2d,
        source: &RenderTarget2d,
        retention: f32,
        source_opacity: f32,
        source_blend: BlendMode,
        frame_started_at: Instant,
    ) -> RenderReport {
        let upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.target_composition_pipelines.secondary_uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(retention)),
        );
        self.queue.write_buffer(
            &self.target_composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(source_opacity)),
        );
        let history_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine trail history bind group"),
            layout: &self.target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&trails.front.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .target_composition_pipelines
                        .secondary_uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let source_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine trail source bind group"),
            layout: &self.target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .target_composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine trail accumulation encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail history pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &trails.back.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.target_composition_pipelines.alpha);
            pass.set_bind_group(0, &history_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail source pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &trails.back.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.target_composition_pipelines.pipeline(source_blend));
            pass.set_bind_group(0, &source_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        std::mem::swap(&mut trails.front, &mut trails.back);
        render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            upload,
            Duration::ZERO,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        )
    }

    fn draw_clear_trail_buffer(
        &self,
        trails: &TrailBuffer2d,
        color: Color,
        frame_started_at: Instant,
    ) -> RenderReport {
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine trail clear encoder"),
            });
        for view in [&trails.front.view, &trails.back.view] {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(color)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit([encoder.finish()]);
        render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        )
    }

    pub(super) fn validate_render_target(
        &self,
        target: &RenderTarget2d,
    ) -> Result<(), RenderTargetError> {
        prepared_scene_belongs_to(&self.renderer_identity, &target.renderer_identity)
            .then_some(())
            .ok_or(RenderTargetError::RendererMismatch)
    }

    fn validate_trail_buffer(&self, trails: &TrailBuffer2d) -> Result<(), RenderTargetError> {
        prepared_scene_belongs_to(&self.renderer_identity, &trails.renderer_identity)
            .then_some(())
            .ok_or(RenderTargetError::RendererMismatch)
    }
}
