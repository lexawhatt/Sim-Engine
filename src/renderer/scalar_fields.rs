//! Scalar-texture updates, recovery, color mapping and heatmap drawing.

use super::*;

/// Renderer-owned scalar texture retaining its validated source field.
pub struct ScalarFieldTexture {
    pub(super) renderer_identity: Arc<()>,
    pub(super) texture: wgpu::Texture,
    pub(super) field: ScalarField,
    pub(super) source_minimum: f32,
    pub(super) source_maximum: f32,
}

impl ScalarFieldTexture {
    /// Returns the current grid width in texels.
    pub fn width(&self) -> usize {
        self.field.width()
    }

    /// Returns the current grid height in texels.
    pub fn height(&self) -> usize {
        self.field.height()
    }

    /// Returns retained CPU scalar data used for recovery and uploads.
    pub fn field(&self) -> &ScalarField {
        &self.field
    }

    /// Returns retained CPU scalar bytes used for device-loss recovery.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.field.value_allocation_bytes()
    }

    /// Returns nominal `R32Float` texel-storage bytes requested from the GPU,
    /// excluding backend row/tile/page alignment and metadata.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.field
            .values()
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

/// Failure while creating or updating a [`ScalarFieldTexture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFieldTextureError {
    /// The texture belongs to another renderer and GPU device.
    RendererMismatch,
    /// Grid dimensions exceed the addressable `wgpu` texture range.
    DimensionsTooLarge,
    /// A region does not fit inside the current scalar texture.
    UpdateRegionOutOfBounds,
    /// Region values did not match the region dimensions.
    InvalidUpdateValueCount,
    /// A region value was NaN or infinite.
    NonFiniteUpdateValue,
    /// A finite scalar source lies outside the portable GPU arithmetic envelope.
    NonPortableValue,
    /// CPU recovery storage could not be reserved.
    AllocationFailed {
        /// Bytes requested for the rejected retained scalar copy.
        requested_bytes: usize,
    },
}

impl fmt::Display for ScalarFieldTextureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => write!(
                formatter,
                "scalar field texture belongs to a different renderer"
            ),
            Self::DimensionsTooLarge => write!(
                formatter,
                "scalar field dimensions exceed GPU texture limits"
            ),
            Self::UpdateRegionOutOfBounds => {
                write!(formatter, "scalar texture update region is out of bounds")
            }
            Self::InvalidUpdateValueCount => {
                write!(
                    formatter,
                    "scalar texture update values do not match the region"
                )
            }
            Self::NonFiniteUpdateValue => {
                write!(formatter, "scalar texture update values must be finite")
            }
            Self::NonPortableValue => write!(
                formatter,
                "scalar values must be normal-or-zero and remain inside the portable GPU arithmetic envelope"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for scalar field recovery data"
            ),
        }
    }
}

impl Error for ScalarFieldTextureError {}

/// Failure while rendering a scalar texture as a heatmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarFieldRenderError {
    /// The texture belongs to another renderer and GPU device.
    RendererMismatch,
    /// The scalar range was non-finite or did not have positive extent.
    InvalidValueRange {
        /// Lower bound supplied by the caller.
        minimum: f32,
        /// Upper bound supplied by the caller.
        maximum: f32,
    },
    /// The clear color is not normalized linear RGBA.
    InvalidBackground,
    /// Surface acquisition or presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for ScalarFieldRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => write!(
                formatter,
                "scalar field texture belongs to a different renderer"
            ),
            Self::InvalidValueRange { minimum, maximum } => {
                write!(formatter, "invalid scalar value range {minimum}..{maximum}")
            }
            Self::InvalidBackground => {
                write!(formatter, "scalar field background must be normalized")
            }
            Self::Frame(error) => write!(formatter, "scalar field frame failed: {error}"),
        }
    }
}

impl Error for ScalarFieldRenderError {}

/// CPU-side outcome of one scalar-field texture upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarFieldUploadReport {
    upload: Duration,
    reallocated: bool,
}

impl ScalarFieldUploadReport {
    /// Returns CPU time spent allocating when needed and enqueuing the upload.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns whether differing dimensions recreated the GPU texture.
    pub fn reallocated(self) -> bool {
        self.reallocated
    }
}

impl WgpuRenderer {
    /// Creates a single-channel floating-point texture from a validated scalar grid.
    pub fn create_scalar_field_texture(
        &self,
        field: ScalarField,
    ) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
        create_scalar_field_texture_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            field,
        )
    }

    /// Replaces scalar data, recreating the texture only when dimensions change.
    pub fn update_scalar_field_texture(
        &self,
        texture: &mut ScalarFieldTexture,
        field: ScalarField,
    ) -> Result<ScalarFieldUploadReport, ScalarFieldTextureError> {
        let upload_started_at = Instant::now();
        self.validate_scalar_field_texture(texture)?;
        validate_scalar_field_device_extent(&self.device, &field)?;
        if !scalar_field_sources_are_portable(&field) {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
        let field = compact_scalar_field_for_retention(field)?;
        let (source_minimum, source_maximum) = field.value_range();
        let reallocated = texture.width() != field.width() || texture.height() != field.height();
        if reallocated {
            texture.texture = create_scalar_field_texture(&self.device, &field)?;
        }
        upload_scalar_field_texture(&self.queue, &texture.texture, &field)?;
        texture.field = field;
        texture.source_minimum = source_minimum;
        texture.source_maximum = source_maximum;
        Ok(ScalarFieldUploadReport {
            upload: upload_started_at.elapsed(),
            reallocated,
        })
    }

    /// Updates a rectangular scalar texture region without recreating it.
    pub fn update_scalar_field_texture_region(
        &self,
        texture: &mut ScalarFieldTexture,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        values: &[f32],
    ) -> Result<ScalarFieldUploadReport, ScalarFieldTextureError> {
        let upload_started_at = Instant::now();
        self.validate_scalar_field_texture(texture)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ScalarFieldTextureError::NonFiniteUpdateValue);
        }
        if values
            .iter()
            .copied()
            .any(|value| !is_portable_shader_source(value))
        {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
        validate_scalar_field_texture_region(
            texture.width(),
            texture.height(),
            x,
            y,
            width,
            height,
            values.len(),
        )?;
        let update_minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let update_maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let overwrites_extremum = (y..y + height).any(|row| {
            let start = row * texture.width() + x;
            texture.field.values()[start..start + width]
                .iter()
                .any(|value| *value == texture.source_minimum || *value == texture.source_maximum)
        });
        let (source_minimum, source_maximum) = if overwrites_extremum {
            scalar_region_result_range(&texture.field, x, y, width, height, values)
        } else {
            (
                texture.source_minimum.min(update_minimum),
                texture.source_maximum.max(update_maximum),
            )
        };
        let source_extent = f64::from(source_maximum) - f64::from(source_minimum);
        if !source_extent.is_finite() || source_extent > f64::from(MAX_PORTABLE_SHADER_VALUE) {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
        texture
            .field
            .replace_region(x, y, width, height, values)
            .map_err(|error| match error {
                crate::ScalarFieldError::InvalidValueCount { .. } => {
                    ScalarFieldTextureError::InvalidUpdateValueCount
                }
                crate::ScalarFieldError::NonFiniteValue => {
                    ScalarFieldTextureError::NonFiniteUpdateValue
                }
                _ => ScalarFieldTextureError::UpdateRegionOutOfBounds,
            })?;
        upload_scalar_field_texture_region(
            &self.queue,
            &texture.texture,
            x,
            y,
            width,
            height,
            values,
        )?;
        texture.source_minimum = source_minimum;
        texture.source_maximum = source_maximum;
        Ok(ScalarFieldUploadReport {
            upload: upload_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Recreates a scalar texture on this renderer from its retained CPU grid.
    pub fn restore_scalar_field_texture(
        &self,
        source: &ScalarFieldTexture,
    ) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
        validate_scalar_field_device_extent(&self.device, &source.field)?;
        let field = clone_scalar_field_for_restore(&source.field)?;
        self.create_scalar_field_texture(field)
    }

    /// Draws a scalar texture across the logical viewport through a color map.
    ///
    /// `value_range` maps `minimum` to the first color-map entry and `maximum`
    /// to the last. Values outside the range clamp to those endpoints.
    pub fn render_scalar_field_texture(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        value_range: (f32, f32),
        background: Color,
    ) -> Result<RenderStatus, ScalarFieldRenderError> {
        self.render_scalar_field_texture_with_metrics(texture, color_map, value_range, background)
            .map(RenderReport::status)
    }

    /// Draws a scalar texture and returns normal CPU-side renderer metrics.
    pub fn render_scalar_field_texture_with_metrics(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            value_extent,
            ScalarFieldSampling::Nearest,
            background,
            Instant::now(),
        )
        .map_err(ScalarFieldRenderError::Frame)
    }

    /// Draws a scalar texture with an explicit texel sampling mode.
    pub fn render_scalar_field_texture_with_sampling(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        background: Color,
    ) -> Result<RenderStatus, ScalarFieldRenderError> {
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            value_extent,
            sampling,
            background,
            Instant::now(),
        )
        .map(RenderReport::status)
        .map_err(ScalarFieldRenderError::Frame)
    }

    /// Renders a scalar heatmap into an offscreen target instead of presenting it.
    pub fn render_scalar_field_texture_to_target(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        value_range: (f32, f32),
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.render_scalar_field_texture_to_target_with_sampling(
            target,
            texture,
            color_map,
            value_range,
            ScalarFieldSampling::Nearest,
            background,
        )
    }

    /// Renders a scalar heatmap into a target with explicit source sampling.
    pub fn render_scalar_field_texture_to_target_with_sampling(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.validate_render_target(target)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        Ok(self.draw_scalar_field_texture_to_target(
            target,
            texture,
            color_map,
            minimum,
            value_extent,
            sampling,
            background,
            Instant::now(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_scalar_field_texture(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        minimum: f32,
        value_extent: f32,
        sampling: ScalarFieldSampling,
        background: Color,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let uniform = HeatmapUniform::new(
            minimum,
            value_extent,
            texture.width(),
            texture.height(),
            sampling,
        );
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
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(
            &self.heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine heatmap bind group"),
            layout: &self.heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.heatmap_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(target) => (&target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine heatmap encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine heatmap pass"),
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
            pass.set_pipeline(&self.heatmap_pipeline);
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
    fn draw_scalar_field_texture_to_target(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        minimum: f32,
        value_extent: f32,
        sampling: ScalarFieldSampling,
        background: Color,
        frame_started_at: Instant,
    ) -> RenderReport {
        let upload_started_at = Instant::now();
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(
            &self.heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&HeatmapUniform::new(
                minimum,
                value_extent,
                texture.width(),
                texture.height(),
                sampling,
            )),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine target heatmap bind group"),
            layout: &self.heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.heatmap_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine render-target heatmap encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine render-target heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(background)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.target_heatmap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
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

    pub(super) fn validate_scalar_field_texture(
        &self,
        texture: &ScalarFieldTexture,
    ) -> Result<(), ScalarFieldTextureError> {
        prepared_scene_belongs_to(&self.renderer_identity, &texture.renderer_identity)
            .then_some(())
            .ok_or(ScalarFieldTextureError::RendererMismatch)
    }

    pub(super) fn color_map_view(&mut self, color_map: &ColorMap) -> wgpu::TextureView {
        let lut = color_map_lut(color_map);
        if let Some(cached) = self
            .color_map_cache
            .as_ref()
            .filter(|cached| cached.lut == lut)
        {
            return cached.view.clone();
        }
        let cached = create_cached_color_map(&self.device, &self.queue, lut);
        let view = cached.view.clone();
        self.color_map_cache = Some(cached);
        view
    }
}
