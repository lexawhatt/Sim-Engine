use super::*;

pub(super) struct CompositionPipelines {
    pub(super) alpha: wgpu::RenderPipeline,
    pub(super) additive: wgpu::RenderPipeline,
    pub(super) replace: wgpu::RenderPipeline,
    pub(super) uniform_buffer: wgpu::Buffer,
    pub(super) secondary_uniform_buffer: wgpu::Buffer,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
}

pub(super) struct CachedColorMap {
    pub(super) lut: [u8; COLOR_MAP_LUT_SIZE as usize * 4],
    pub(super) _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

impl CompositionPipelines {
    pub(super) fn pipeline(&self, blend_mode: BlendMode) -> &wgpu::RenderPipeline {
        match blend_mode {
            BlendMode::Alpha => &self.alpha,
            BlendMode::Additive => &self.additive,
            BlendMode::Replace => &self.replace,
        }
    }
}

pub(super) fn create_composition_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    sample_count: u32,
    blend_mode: BlendMode,
) -> wgpu::RenderPipeline {
    let blend = match blend_mode {
        BlendMode::Alpha => Some(wgpu::BlendState::ALPHA_BLENDING),
        BlendMode::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        BlendMode::Replace => Some(wgpu::BlendState {
            // `composite_fs_main` produces straight RGB. Preserve the target
            // representation by premultiplying on write while replacing every
            // destination component.
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        }),
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine composition pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("composite_vs_main"),
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
            module: shader,
            entry_point: Some("composite_fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Validated presentation settings for a scalar field with a particle overlay.
///
/// This is a rendering composition contract, not a simulation data model. The
/// host remains responsible for producing both the scalar and particle state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayeredVisualizationOptions {
    minimum: f32,
    maximum: f32,
    value_extent: f32,
    sampling: ScalarFieldSampling,
    blend_mode: BlendMode,
    opacity: f32,
    target_background: Color,
    surface_background: Color,
}

impl LayeredVisualizationOptions {
    /// Creates settings with linear field sampling and opaque replacement.
    pub fn new(
        (minimum, maximum): (f32, f32),
        target_background: Color,
        surface_background: Color,
    ) -> Result<Self, LayeredVisualizationError> {
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(LayeredVisualizationError::InvalidValueRange { minimum, maximum })?;
        if !target_background.is_normalized() || !surface_background.is_normalized() {
            return Err(LayeredVisualizationError::InvalidBackground);
        }
        Ok(Self {
            minimum,
            maximum,
            value_extent,
            sampling: ScalarFieldSampling::Linear,
            blend_mode: BlendMode::Replace,
            opacity: 1.0,
            target_background,
            surface_background,
        })
    }

    /// Selects nearest or manual-linear scalar sampling.
    pub fn with_sampling(mut self, sampling: ScalarFieldSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Selects final target composition and validates its opacity.
    pub fn with_composition(
        mut self,
        blend_mode: BlendMode,
        opacity: f32,
    ) -> Result<Self, LayeredVisualizationError> {
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(LayeredVisualizationError::InvalidOpacity);
        }
        self.blend_mode = blend_mode;
        self.opacity = opacity;
        Ok(self)
    }

    /// Returns the finite scalar range represented by the heatmap.
    pub fn value_range(self) -> (f32, f32) {
        (self.minimum, self.maximum)
    }

    /// Returns the selected scalar sampling mode.
    pub fn sampling(self) -> ScalarFieldSampling {
        self.sampling
    }

    /// Returns final target-to-surface blending.
    pub fn blend_mode(self) -> BlendMode {
        self.blend_mode
    }

    /// Returns final target opacity.
    pub fn opacity(self) -> f32 {
        self.opacity
    }
}

/// Status, timings, and encoded-work counters for one fused scalar/particle frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayeredVisualizationReport {
    report: RenderReport,
    render_pass_count: usize,
    draw_call_count: usize,
}

impl LayeredVisualizationReport {
    /// Returns whether the surface frame was drawn or temporarily skipped.
    pub fn status(self) -> RenderStatus {
        self.report.status()
    }

    /// Returns CPU-side renderer stage timings.
    pub fn metrics(self) -> RendererFrameMetrics {
        self.report.metrics()
    }

    /// Returns render passes actually encoded for this call.
    pub const fn render_pass_count(self) -> usize {
        self.render_pass_count
    }

    /// Returns draw calls actually encoded for this call.
    pub const fn draw_call_count(self) -> usize {
        self.draw_call_count
    }
}

/// Failure while drawing a bounded scalar-field and particle composition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayeredVisualizationError {
    /// At least one resource belongs to another renderer/device.
    RendererMismatch,
    /// Scalar endpoints are invalid or their subtraction overflows `f32`.
    InvalidValueRange {
        /// Lower scalar endpoint supplied by the host.
        minimum: f32,
        /// Upper scalar endpoint supplied by the host.
        maximum: f32,
    },
    /// A clear color is not normalized finite linear RGBA.
    InvalidBackground,
    /// Final composition opacity is outside the normalized finite range.
    InvalidOpacity,
    /// Camera arithmetic cannot produce finite particle geometry.
    InvalidGeometryTransform,
    /// Particle preparation, surface acquisition, or frame presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for LayeredVisualizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "visual resources belong to another renderer")
            }
            Self::InvalidValueRange { minimum, maximum } => {
                write!(formatter, "invalid scalar value range {minimum}..{maximum}")
            }
            Self::InvalidBackground => {
                write!(
                    formatter,
                    "visualization colors must be normalized linear RGBA"
                )
            }
            Self::InvalidOpacity => write!(formatter, "visualization opacity must be in 0..=1"),
            Self::InvalidGeometryTransform => {
                write!(formatter, "particle camera transform is invalid")
            }
            Self::Frame(error) => write!(formatter, "visualization frame failed: {error}"),
        }
    }
}

impl Error for LayeredVisualizationError {}

impl WgpuRenderer {
    /// Draws a heatmap, overlays a budgeted particle field, and presents the
    /// target with one command encoder and one queue submission.
    ///
    /// This fused path avoids synchronously submitting each layer and leaves
    /// substantially more CPU time for an expensive host simulation. The
    /// target should have the presentation surface's aspect ratio; lowering
    /// both dimensions proportionally bounds raster cost without changing the
    /// logical camera contract.
    pub fn render_layered_visualization(
        &mut self,
        target: &RenderTarget2d,
        scalar: &ScalarFieldTexture,
        color_map: &ColorMap,
        particles: &mut ParticleField2d,
        camera: &Camera2d,
        options: LayeredVisualizationOptions,
    ) -> Result<LayeredVisualizationReport, LayeredVisualizationError> {
        let frame_started_at = Instant::now();
        let preparation_started_at = Instant::now();
        self.validate_render_target(target)
            .map_err(|_| LayeredVisualizationError::RendererMismatch)?;
        self.validate_scalar_field_texture(scalar)
            .map_err(|_| LayeredVisualizationError::RendererMismatch)?;
        self.validate_particle_field(particles)
            .map_err(|_| LayeredVisualizationError::RendererMismatch)?;
        if !scalar_normalization_is_portable(scalar, options.minimum, options.value_extent) {
            return Err(LayeredVisualizationError::InvalidValueRange {
                minimum: options.minimum,
                maximum: options.maximum,
            });
        }

        // Particle validation and all fallible host reservation complete before
        // any heatmap cache or queue mutation.
        let preparation = self
            .prepare_particle_draw(particles, *camera)
            .map_err(|error| match error {
                RendererFrameError::InvalidGeometryTransform => {
                    LayeredVisualizationError::InvalidGeometryTransform
                }
                other => LayeredVisualizationError::Frame(other),
            })?;
        let preparation_duration = preparation_started_at.elapsed();

        let acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                particles.statistics = preparation.statistics;
                return Ok(skipped_layered_report(layered_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                )));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                particles.statistics = preparation.statistics;
                return Ok(skipped_layered_report(layered_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                )));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let _ = self.resize(self.config.width, self.config.height);
                particles.statistics = preparation.statistics;
                return Ok(skipped_layered_report(layered_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                )));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(LayeredVisualizationError::Frame(
                    RendererFrameError::Surface(RendererSurfaceStatus::Lost),
                ));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(LayeredVisualizationError::Frame(
                    RendererFrameError::Surface(RendererSurfaceStatus::Validation),
                ));
            }
        };
        let surface_acquire = acquire_started_at.elapsed();
        let (particle_upload, camera_uniform_upload) =
            self.upload_particle_draw(particles, preparation);
        let upload_started_at = Instant::now();
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = scalar
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(
            &self.heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&HeatmapUniform::new(
                options.minimum,
                options.value_extent,
                scalar.width(),
                scalar.height(),
                options.sampling,
            )),
        );
        let heatmap_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine layered heatmap bind group"),
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
        self.queue.write_buffer(
            &self.composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(options.opacity)),
        );
        let composition_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine layered composition bind group"),
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
        let upload = combined_layered_upload(particle_upload, upload_started_at.elapsed());
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
                label: Some("sim-engine layered visualization encoder"),
            });
        let mut render_pass_count = 0_usize;
        let mut draw_call_count = 0_usize;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine layered heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(
                            options.target_background,
                        )),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.target_heatmap_pipeline);
            pass.set_bind_group(0, &heatmap_bind_group, &[]);
            pass.draw(0..6, 0..1);
            render_pass_count += 1;
            draw_call_count += 1;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine layered particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
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
            if preparation.visible_count > 0 {
                pass.set_pipeline(&self.target_particle_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.particle_unit_buffer.slice(..));
                pass.set_vertex_buffer(1, particles.instance_buffer.slice(..));
                pass.draw(0..6, 0..preparation.visible_count as u32);
                draw_call_count += 1;
            }
            render_pass_count += 1;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine layered composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(options.surface_background.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.composition_pipelines.pipeline(options.blend_mode));
            pass.set_bind_group(0, &composition_bind_group, &[]);
            pass.draw(0..6, 0..1);
            render_pass_count += 1;
            draw_call_count += 1;
        }
        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
        self.queue.present(surface_texture);
        particles.statistics = ParticleStatistics {
            rendered: preparation.visible_count,
            ..preparation.statistics
        };
        Ok(LayeredVisualizationReport {
            report: layered_report(
                RenderStatus::Drawn,
                preparation_duration,
                upload,
                camera_uniform_upload,
                surface_acquire,
                encode_started_at.elapsed(),
                frame_started_at.elapsed(),
            ),
            render_pass_count,
            draw_call_count,
        })
    }
}

fn skipped_layered_report(report: RenderReport) -> LayeredVisualizationReport {
    LayeredVisualizationReport {
        report,
        render_pass_count: 0,
        draw_call_count: 0,
    }
}

fn layered_report(
    status: RenderStatus,
    preparation: Duration,
    upload: Duration,
    camera_uniform_upload: Duration,
    surface_acquire: Duration,
    encode_submit_present: Duration,
    total_cpu: Duration,
) -> RenderReport {
    render_report(
        status,
        preparation,
        upload,
        camera_uniform_upload,
        surface_acquire,
        encode_submit_present,
        total_cpu,
        false,
        true,
        TessellationStats::default(),
    )
}

fn combined_layered_upload(particle_upload: Duration, remaining_upload: Duration) -> Duration {
    particle_upload.saturating_add(remaining_upload)
}

#[cfg(test)]
mod tests {
    use super::{combined_layered_upload, layered_report, skipped_layered_report};
    use crate::{RenderStatus, RendererSurfaceStatus};
    use std::time::Duration;

    #[test]
    fn layered_upload_includes_particle_and_composition_uploads() {
        assert_eq!(
            combined_layered_upload(Duration::from_millis(7), Duration::from_millis(3)),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn layered_reports_completed_preparation_for_drawn_and_skipped_frames() {
        let preparation = Duration::from_millis(4);
        let drawn = layered_report(
            RenderStatus::Drawn,
            preparation,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            preparation,
        );
        let skipped = layered_report(
            RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
            preparation,
            Duration::ZERO,
            Duration::ZERO,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::from_millis(5),
        );

        assert_eq!(drawn.metrics().tessellation(), preparation);
        assert_eq!(skipped.metrics().tessellation(), preparation);
        let skipped = skipped_layered_report(skipped);
        assert_eq!(skipped.render_pass_count(), 0);
        assert_eq!(skipped.draw_call_count(), 0);
    }
}
