//! Ordinary and prepared scene drawing through the shared geometry pipeline.

use super::{
    Arc, Camera2d, CameraUniform, Color, Duration, GeometryExtents, GeometryValidationCache,
    GeometryValidationSource, Instant, LogicalViewport, LogicalViewportRegion, PhysicalPerLogical,
    PreparedDrawBatch, PreparedScene, PreparedSceneError, PreparedSceneRenderError,
    PreparedScreenScene, RenderReport, RenderStatus, RenderTarget2d, RenderTargetError,
    RenderTargetLoad, RendererFrameError, RendererSurfaceStatus, Scene, ScissorRect, ScreenScene,
    TessellationStats, Vec2, Vertex, WgpuRenderer, buffer_capacity_fits, create_vertex_buffer,
    dynamic_vertex_capacity, geometry_is_safe_for, geometry_is_safe_for_cached,
    logical_viewport_scissor, offset_scissor, premultiplied_wgpu_color, prepare_scene_resources,
    prepared_scene_belongs_to, render_report, restore_prepared_scene_resources,
    scene_estimate_fits_streaming_device, screen_camera, screen_clip_to_scissor, tessellate_scene,
};

impl WgpuRenderer {
    /// Draws a scene using the supplied camera.
    ///
    /// Scene positions and sizes are in world units unless a style explicitly
    /// says logical screen pixels. The renderer converts world coordinates to
    /// logical screen coordinates through [`Camera2d`], then to normalized device
    /// coordinates for `wgpu`. Scissor rectangles are converted to physical
    /// surface pixels at the final backend boundary.
    pub fn render(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_with_metrics(scene, camera)
            .map(RenderReport::status)
    }

    /// Draws fixed logical-screen geometry independently of any world camera.
    pub fn render_screen_scene(
        &mut self,
        scene: &ScreenScene,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_screen_scene_with_metrics(scene)
            .map(RenderReport::status)
    }

    /// Draws fixed logical-screen geometry and returns CPU-side stage timings.
    pub fn render_screen_scene_with_metrics(
        &mut self,
        scene: &ScreenScene,
    ) -> Result<RenderReport, RendererFrameError> {
        let viewport = LogicalViewport::new(self.logical_size().0, self.logical_size().1)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let camera =
            screen_camera(viewport).map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        self.render_with_metrics(scene.as_scene(), &camera)
    }

    /// Draws a scene and returns a CPU-side timing breakdown.
    ///
    /// Metrics separate tessellation, buffer upload, surface acquisition, and
    /// encode/submit/present dispatch. They do not measure GPU completion or the
    /// monitor scanout timestamp. Use this for diagnostics; [`WgpuRenderer::render`]
    /// is the simpler equivalent when stage timings are not needed.
    pub fn render_with_metrics(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
    ) -> Result<RenderReport, RendererFrameError> {
        self.render_with_metrics_in_region(scene, camera, None)
    }

    /// Draws a world scene through a camera into one bounded logical viewport.
    ///
    /// Scene-local clips are interpreted relative to the viewport and are
    /// intersected with its physical bounds. The surface outside the viewport
    /// is cleared but receives no scene geometry.
    pub fn render_scene_in_viewport(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: LogicalViewportRegion,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_scene_in_viewport_with_metrics(scene, camera, viewport)
            .map(RenderReport::status)
    }

    /// Draws one bounded scene viewport and returns CPU-side stage timings.
    pub fn render_scene_in_viewport_with_metrics(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: LogicalViewportRegion,
    ) -> Result<RenderReport, RendererFrameError> {
        self.render_with_metrics_in_region(scene, camera, Some(viewport))
    }

    /// Renders an ordinary world scene into an offscreen target.
    ///
    /// Target dimensions are physical texels. `pixel_scale` defines their
    /// logical viewport density independently of window DPI. Scene-local clips
    /// are relative to `viewport`, or to the complete logical target when no
    /// region is supplied. `load` explicitly controls preservation or clearing.
    #[allow(clippy::too_many_arguments)]
    pub fn render_scene_to_target(
        &mut self,
        target: &RenderTarget2d,
        scene: &Scene,
        camera: &Camera2d,
        pixel_scale: PhysicalPerLogical,
        viewport: Option<LogicalViewportRegion>,
        load: RenderTargetLoad,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_render_target(target)?;
        if let RenderTargetLoad::Clear(color) = load
            && !color.is_normalized()
        {
            return Err(RenderTargetError::InvalidBackground);
        }

        let frame_started_at = Instant::now();
        let scale = pixel_scale.get();
        let target_viewport = LogicalViewport::new(
            target.width() as f32 / scale,
            target.height() as f32 / scale,
        )
        .map_err(|_| RenderTargetError::Frame(RendererFrameError::InvalidViewport))?;
        let (local_viewport, origin) = match viewport {
            Some(region) => (region.viewport(), region.origin().to_vec2()),
            None => (target_viewport, Vec2::ZERO),
        };
        let max = origin + local_viewport.size();
        if !max.is_finite()
            || origin.x < 0.0
            || origin.y < 0.0
            || max.x > target_viewport.width()
            || max.y > target_viewport.height()
        {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidViewport,
            ));
        }
        let target_scissor = logical_viewport_scissor(
            origin,
            local_viewport,
            scale,
            target.width(),
            target.height(),
        )
        .ok_or(RenderTargetError::Frame(
            RendererFrameError::InvalidViewport,
        ))?;
        let camera_uniform =
            CameraUniform::new_in_region(*camera, local_viewport, origin, target_viewport).ok_or(
                RenderTargetError::Frame(RendererFrameError::InvalidGeometryTransform),
            )?;

        let tessellation_started_at = Instant::now();
        if !scene_estimate_fits_streaming_device(scene, &self.device, 0, self.vertex_capacity) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::GeometryCapacityTooLarge,
            ));
        }
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)
                .map_err(RendererFrameError::from)
                .map_err(RenderTargetError::Frame)?;
        let tessellation = tessellation_started_at.elapsed();

        let extents = GeometryExtents::from_vertices(&self.vertices);
        if !geometry_is_safe_for(
            extents,
            GeometryValidationSource::Tessellated(&self.vertices),
            camera_uniform,
        ) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidGeometryTransform,
            ));
        }
        let upload_started_at = Instant::now();
        self.ensure_vertex_capacity(self.vertices.len())
            .map_err(RenderTargetError::Frame)?;
        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
        let upload = upload_started_at.elapsed();
        let uniform_started_at = Instant::now();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = uniform_started_at.elapsed();

        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine scene target encoder"),
            });
        {
            let load = match load {
                RenderTargetLoad::Load => wgpu::LoadOp::Load,
                RenderTargetLoad::Clear(color) => {
                    wgpu::LoadOp::Clear(premultiplied_wgpu_color(color))
                }
            };
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine scene target pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !self.vertices.is_empty() {
                pass.set_pipeline(&self.target_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                for batch in &self.draw_batches {
                    let scissor = match batch.screen_clip {
                        Some(clip) => {
                            let Some(local) = screen_clip_to_scissor(clip, local_viewport, scale)
                            else {
                                continue;
                            };
                            let Some(scissor) = offset_scissor(local, target_scissor) else {
                                continue;
                            };
                            scissor
                        }
                        None => target_scissor,
                    };
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        let encode = encode_started_at.elapsed();

        Ok(render_report(
            RenderStatus::Drawn,
            tessellation,
            upload,
            camera_uniform_upload,
            Duration::ZERO,
            encode,
            frame_started_at.elapsed(),
            false,
            true,
            tessellation_stats,
        ))
    }

    fn render_with_metrics_in_region(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: Option<LogicalViewportRegion>,
    ) -> Result<RenderReport, RendererFrameError> {
        let frame_started_at = Instant::now();
        let (_, _, camera_uniform) = self.surface_geometry_context(*camera, viewport)?;
        let tessellation_started_at = Instant::now();
        if !scene_estimate_fits_streaming_device(scene, &self.device, 0, self.vertex_capacity) {
            return Err(RendererFrameError::GeometryCapacityTooLarge);
        }
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)?;
        let tessellation = tessellation_started_at.elapsed();

        let geometry_extents = GeometryExtents::from_vertices(&self.vertices);
        if !geometry_is_safe_for(
            geometry_extents,
            GeometryValidationSource::Tessellated(&self.vertices),
            camera_uniform,
        ) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }

        let upload_started_at = Instant::now();
        self.ensure_vertex_capacity(self.vertices.len())?;
        let upload = upload_started_at.elapsed();

        let vertex_buffer = Arc::clone(&self.vertex_buffer);
        let draw_batches = std::mem::take(&mut self.draw_batches);
        let vertices = std::mem::take(&mut self.vertices);
        let result = self.draw_geometry(
            scene.background(),
            &vertex_buffer,
            vertices.len(),
            geometry_extents,
            GeometryValidationSource::Tessellated(&vertices),
            None,
            &draw_batches,
            *camera,
            tessellation,
            upload,
            false,
            true,
            false,
            Some(&vertices),
            tessellation_stats,
            frame_started_at,
            viewport,
        );
        self.vertices = vertices;
        self.draw_batches = draw_batches;
        result
    }

    /// Tessellates a scene once and uploads immutable geometry to a dedicated GPU buffer.
    ///
    /// Preparing is appropriate for geometry that will be drawn repeatedly while
    /// only the camera or target dimensions change. Any shape, style, gradient,
    /// ordering, or clipping change requires preparing a replacement scene.
    pub fn prepare_scene(&self, scene: &Scene) -> Result<PreparedScene, PreparedSceneError> {
        prepare_scene_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            scene,
        )
    }

    /// Tessellates fixed logical-screen geometry once for repeated composition.
    pub fn prepare_screen_scene(
        &self,
        scene: &ScreenScene,
    ) -> Result<PreparedScreenScene, PreparedSceneError> {
        self.prepare_scene(scene.as_scene())
            .map(|scene| PreparedScreenScene { scene })
    }

    /// Restores prepared logical-screen geometry on a recreated renderer.
    pub fn restore_prepared_screen_scene(
        &self,
        source: &PreparedScreenScene,
    ) -> Result<PreparedScreenScene, PreparedSceneError> {
        self.restore_prepared_scene(&source.scene)
            .map(|scene| PreparedScreenScene { scene })
    }

    /// Recreates prepared GPU resources on this renderer from a retained CPU snapshot.
    ///
    /// This supports renderer recreation after device loss without requiring the
    /// original high-level [`Scene`]. The returned snapshot belongs to this renderer.
    pub fn restore_prepared_scene(
        &self,
        source: &PreparedScene,
    ) -> Result<PreparedScene, PreparedSceneError> {
        restore_prepared_scene_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }

    /// Draws geometry previously uploaded by [`WgpuRenderer::prepare_scene`].
    ///
    /// Camera and viewport changes are applied by the vertex shader and do not
    /// rebuild or re-upload the prepared geometry.
    pub fn render_prepared(
        &mut self,
        scene: &PreparedScene,
        camera: &Camera2d,
    ) -> Result<RenderStatus, PreparedSceneRenderError> {
        self.render_prepared_with_metrics(scene, camera)
            .map(RenderReport::status)
    }

    /// Draws prepared logical-screen geometry independently of a world camera.
    ///
    /// The prepared resource must have been built from a [`ScreenScene`]. The
    /// renderer cannot infer the original coordinate space from raw prepared
    /// vertices, so passing world-space geometry is a caller contract error.
    pub fn render_prepared_screen_scene(
        &mut self,
        scene: &PreparedScreenScene,
    ) -> Result<RenderStatus, PreparedSceneRenderError> {
        let (width, height) = self.logical_size();
        let viewport = LogicalViewport::new(width, height).map_err(|_| {
            PreparedSceneRenderError::Frame(RendererFrameError::InvalidGeometryTransform)
        })?;
        let camera = screen_camera(viewport).map_err(|_| {
            PreparedSceneRenderError::Frame(RendererFrameError::InvalidGeometryTransform)
        })?;
        self.render_prepared(&scene.scene, &camera)
    }

    /// Draws prepared geometry and reports per-frame CPU timing.
    ///
    /// Tessellation and geometry upload durations are zero because both happened
    /// in [`WgpuRenderer::prepare_scene`]. The camera uniform is still updated
    /// once per frame.
    pub fn render_prepared_with_metrics(
        &mut self,
        scene: &PreparedScene,
        camera: &Camera2d,
    ) -> Result<RenderReport, PreparedSceneRenderError> {
        if !prepared_scene_belongs_to(&self.renderer_identity, &scene.renderer_identity) {
            return Err(PreparedSceneRenderError::RendererMismatch);
        }

        self.draw_geometry(
            scene.background,
            &scene.vertex_buffer,
            scene.vertex_count,
            scene.geometry_extents,
            GeometryValidationSource::Tessellated(&scene.vertices),
            Some(&scene.geometry_validation_cache),
            &scene.draw_batches,
            *camera,
            Duration::ZERO,
            Duration::ZERO,
            true,
            false,
            false,
            None,
            scene.tessellation,
            Instant::now(),
            None,
        )
        .map_err(PreparedSceneRenderError::Frame)
    }

    fn surface_geometry_context(
        &self,
        camera: Camera2d,
        viewport_region: Option<LogicalViewportRegion>,
    ) -> Result<(LogicalViewport, ScissorRect, CameraUniform), RendererFrameError> {
        let (logical_width, logical_height) = self.logical_size();
        let target_viewport = LogicalViewport::new(logical_width, logical_height)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let (viewport, viewport_origin) = match viewport_region {
            Some(region) => (region.viewport(), region.origin().to_vec2()),
            None => (target_viewport, Vec2::ZERO),
        };
        let viewport_max = viewport_origin + viewport.size();
        if !viewport_origin.is_finite()
            || !viewport_max.is_finite()
            || viewport_origin.x < 0.0
            || viewport_origin.y < 0.0
            || viewport_max.x > target_viewport.width()
            || viewport_max.y > target_viewport.height()
        {
            return Err(RendererFrameError::InvalidViewport);
        }
        let viewport_scissor = logical_viewport_scissor(
            viewport_origin,
            viewport,
            self.scale_factor as f32,
            self.config.width,
            self.config.height,
        )
        .ok_or(RendererFrameError::InvalidViewport)?;
        let camera_uniform =
            CameraUniform::new_in_region(camera, viewport, viewport_origin, target_viewport)
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
        Ok((viewport, viewport_scissor, camera_uniform))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_geometry(
        &mut self,
        background: Color,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: usize,
        geometry_extents: GeometryExtents,
        geometry_validation: GeometryValidationSource<'_>,
        geometry_validation_cache: Option<&GeometryValidationCache>,
        draw_batches: &[PreparedDrawBatch],
        camera: Camera2d,
        tessellation: Duration,
        mut upload: Duration,
        geometry_reused: bool,
        geometry_streamed: bool,
        uses_dynamic_pipeline: bool,
        pending_vertex_upload: Option<&[Vertex]>,
        tessellation_stats: TessellationStats,
        frame_started_at: Instant,
        viewport_region: Option<LogicalViewportRegion>,
    ) -> Result<RenderReport, RendererFrameError> {
        let (viewport, viewport_scissor, camera_uniform) =
            self.surface_geometry_context(camera, viewport_region)?;
        if !geometry_is_safe_for_cached(
            geometry_validation_cache,
            geometry_extents,
            geometry_validation,
            camera_uniform,
        ) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    tessellation,
                    upload,
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    tessellation,
                    upload,
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let _ = self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    tessellation,
                    upload,
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
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
        let surface_acquire = surface_acquire_started_at.elapsed();
        let vertex_upload_started_at = Instant::now();
        if let Some(vertices) = pending_vertex_upload
            && !vertices.is_empty()
        {
            self.queue
                .write_buffer(vertex_buffer, 0, bytemuck::cast_slice(vertices));
        }
        upload = upload.saturating_add(vertex_upload_started_at.elapsed());
        let camera_uniform_upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();

        let encode_submit_present_started_at = Instant::now();
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
                label: Some("sim-engine render encoder"),
            });

        {
            let color_attachment = wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background.to_wgpu()),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine render pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if vertex_count > 0 {
                let pipeline = if uses_dynamic_pipeline {
                    &self.dynamic_pipeline
                } else {
                    &self.pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                for batch in draw_batches {
                    let scissor = match batch.screen_clip {
                        Some(screen_clip) => {
                            let Some(local_scissor) = screen_clip_to_scissor(
                                screen_clip,
                                viewport,
                                self.scale_factor as f32,
                            ) else {
                                continue;
                            };
                            let Some(scissor) = offset_scissor(local_scissor, viewport_scissor)
                            else {
                                continue;
                            };
                            scissor
                        }
                        None => viewport_scissor,
                    };
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
        }

        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
        self.queue.present(surface_texture);
        let encode_submit_present = encode_submit_present_started_at.elapsed();

        Ok(render_report(
            RenderStatus::Drawn,
            tessellation,
            upload,
            camera_uniform_upload,
            surface_acquire,
            encode_submit_present,
            frame_started_at.elapsed(),
            geometry_reused,
            geometry_streamed,
            tessellation_stats,
        ))
    }

    pub(super) fn ensure_vertex_capacity(
        &mut self,
        vertex_count: usize,
    ) -> Result<(), RendererFrameError> {
        if vertex_count <= self.vertex_capacity {
            return Ok(());
        }

        self.vertex_capacity = dynamic_vertex_capacity(vertex_count)
            .filter(|capacity| buffer_capacity_fits::<Vertex>(&self.device, *capacity))
            .ok_or(RendererFrameError::GeometryCapacityTooLarge)?;
        self.vertex_buffer = Arc::new(create_vertex_buffer(&self.device, self.vertex_capacity));
        Ok(())
    }
}
