use std::time::Duration;
use std::time::Instant;

use super::{
    FrameBinding, FrameBudget, FrameCache, FrameComposer, FrameComposerError, FrameItem,
    FrameReport, FrameSourceStatistics, FrameStatistics, ReadyItem, ReadySource, cache, encoding,
    frame_report, preflight_frame_items, prepare_particle_item, prepare_retained_geometry,
    prepare_streaming_scene, ready_uniform_bytes, resolve_viewport, set_particle_rendered,
    streaming, validate_frame_budget, with_streaming_batches,
};
use crate::renderer::{
    COLOR_MAP_LUT_SIZE, CameraUniform, Color, CompositeUniform, GeometryValidationSource,
    GpuTimingSource, HeatmapUniform, ImageUniform, LogicalScreenPosition, LogicalViewportRegion,
    PreparedDrawBatch, RenderStatus, RendererFrameError, RendererSurfaceStatus, TessellationStats,
    Vec2, Vertex, WgpuRenderer, color_map_lut, image, is_portable_shader_source,
    scalar_normalization_is_portable, scene_estimate_fits_streaming_device, screen_camera,
};

pub(super) fn present_frame(
    mut composer: FrameComposer<'_>,
) -> Result<FrameReport, FrameComposerError> {
    let renderer = &mut *composer.renderer;
    // Always return the transient allocation, including structured error
    // paths before surface acquisition.
    let mut streaming_vertices = std::mem::take(&mut renderer.vertices);
    streaming_vertices.clear();
    let mut ready = std::mem::take(&mut composer.cache.ready);
    let mut bindings = std::mem::take(&mut composer.cache.bindings);
    let result = present_frame_with_vertices(
        renderer,
        composer.background,
        composer.budget,
        &mut composer.items,
        composer.planned,
        &mut streaming_vertices,
        &mut ready,
        &mut bindings,
        &mut composer.cache,
    );
    renderer.vertices = streaming_vertices;
    composer.cache.ready = cache::recycle_ready_with_batches(ready, &mut composer.cache.batches);
    bindings.clear();
    composer.cache.bindings = bindings;
    result
}

#[allow(clippy::too_many_arguments)]
fn present_frame_with_vertices<'frame>(
    renderer: &mut WgpuRenderer,
    background: Color,
    budget: FrameBudget,
    items: &mut Vec<FrameItem<'frame>>,
    planned: FrameStatistics,
    streaming_vertices: &mut Vec<Vertex>,
    ready: &mut Vec<ReadyItem<'frame>>,
    bindings: &mut Vec<FrameBinding>,
    cache: &mut FrameCache,
) -> Result<FrameReport, FrameComposerError> {
    let frame_started_at = Instant::now();
    items.sort_unstable_by_key(FrameItem::sort_key);

    let target_viewport = renderer
        .logical_viewport()
        .map_err(|_| RendererFrameError::InvalidViewport)?;
    let tessellation_started_at = Instant::now();
    preflight_frame_items(renderer, target_viewport, items)?;
    ready
        .try_reserve(items.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: items
                .len()
                .saturating_mul(std::mem::size_of::<ReadyItem<'_>>()),
        })?;
    cache
        .batches
        .try_reserve(items.len().saturating_sub(cache.batches.len()))
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: items
                .len()
                .saturating_mul(std::mem::size_of::<Vec<PreparedDrawBatch>>()),
        })?;
    let mut statistics = FrameStatistics::default();
    let mut tessellation_stats = TessellationStats::default();
    let mut geometry_reused = false;
    let mut geometry_streamed = false;
    let mut streaming_proofs = streaming::StreamingSceneProofs::default();
    let mut simulated_color_map_lut = renderer.color_map_cache.as_ref().map(|cache| cache.lut);

    for item in items.drain(..) {
        match item {
            FrameItem::Scene {
                scene,
                camera,
                options,
                ..
            } => {
                with_streaming_batches(&mut cache.batches, |batches| {
                    prepare_streaming_scene(
                        scene,
                        camera,
                        options,
                        renderer,
                        target_viewport,
                        &mut *streaming_vertices,
                        ready,
                        &mut statistics,
                        &mut tessellation_stats,
                        batches,
                        &mut streaming_proofs,
                    )
                })?;
                geometry_streamed = true;
            }
            FrameItem::ScreenScene { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                let camera_uniform = CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                if !scene_estimate_fits_streaming_device(
                    scene.as_scene(),
                    &renderer.device,
                    streaming_vertices.len(),
                    renderer.vertex_capacity,
                ) {
                    return Err(RendererFrameError::GeometryCapacityTooLarge.into());
                }
                with_streaming_batches(&mut cache.batches, |batches| {
                    streaming_proofs.prepare(
                        scene.as_scene(),
                        camera_uniform,
                        viewport,
                        &mut *streaming_vertices,
                        ready,
                        &mut statistics,
                        &mut tessellation_stats,
                        batches,
                    )
                })?;
                geometry_streamed = true;
            }
            FrameItem::Prepared {
                scene,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                prepare_retained_geometry(
                    ReadySource::Prepared(&scene.vertex_buffer),
                    scene.vertex_count,
                    scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.vertices),
                    Some(&scene.geometry_validation_cache),
                    &scene.draw_batches,
                    scene.command_count,
                    scene.tessellation,
                    camera,
                    viewport,
                    target_viewport,
                    ready,
                    &mut statistics,
                    &mut tessellation_stats,
                    cache.batches.pop().unwrap_or_default(),
                )?;
                geometry_reused = true;
            }
            FrameItem::PreparedScreen { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                prepare_retained_geometry(
                    ReadySource::Prepared(&scene.scene.vertex_buffer),
                    scene.scene.vertex_count,
                    scene.scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.scene.vertices),
                    Some(&scene.scene.geometry_validation_cache),
                    &scene.scene.draw_batches,
                    scene.scene.command_count,
                    scene.scene.tessellation,
                    camera,
                    viewport,
                    target_viewport,
                    ready,
                    &mut statistics,
                    &mut tessellation_stats,
                    cache.batches.pop().unwrap_or_default(),
                )?;
                geometry_reused = true;
            }
            FrameItem::Dynamic {
                mesh,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let batch = (!mesh.vertices.is_empty()).then_some(PreparedDrawBatch {
                    vertex_range: 0..mesh.vertices.len() as u32,
                    screen_clip: None,
                });
                prepare_retained_geometry(
                    ReadySource::Dynamic(&mesh.vertex_buffer),
                    mesh.vertices.len(),
                    mesh.geometry_extents,
                    GeometryValidationSource::Dynamic(&mesh.vertices),
                    Some(&mesh.geometry_validation_cache),
                    batch.as_slice(),
                    usize::from(!mesh.vertices.is_empty()),
                    TessellationStats::default(),
                    camera,
                    viewport,
                    target_viewport,
                    ready,
                    &mut statistics,
                    &mut tessellation_stats,
                    cache.batches.pop().unwrap_or_default(),
                )?;
                geometry_streamed = true;
            }
            FrameItem::Particle {
                field,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                prepare_particle_item(
                    field,
                    camera,
                    viewport,
                    target_viewport,
                    ready,
                    &mut statistics,
                )?;
                geometry_streamed = true;
            }
            FrameItem::Scalar {
                texture,
                color_map,
                minimum,
                maximum,
                value_extent,
                sampling,
                options,
                ..
            } => {
                if !scalar_normalization_is_portable(texture, minimum, value_extent) {
                    return Err(FrameComposerError::InvalidValueRange { minimum, maximum });
                }
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let uniform = HeatmapUniform::new(
                    minimum,
                    value_extent,
                    texture.width(),
                    texture.height(),
                    sampling,
                )
                .in_region(region, target_viewport)
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                let color_map_bytes = COLOR_MAP_LUT_SIZE as usize * 4;
                let lut = color_map_lut(color_map);
                let color_map_upload_bytes = if simulated_color_map_lut == Some(lut) {
                    0
                } else {
                    simulated_color_map_lut = Some(lut);
                    color_map_bytes
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<HeatmapUniform>()
                        .saturating_add(color_map_upload_bytes),
                    streaming_upload_bytes: 0,
                    texture_bytes: 0,
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Scalar {
                    texture,
                    color_map,
                    uniform,
                    viewport,
                });
                geometry_streamed = true;
            }
            FrameItem::Image {
                image,
                source,
                tint,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let destination = CompositeUniform::in_region(1.0, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?
                    .destination;
                let image_width = image.width() as f32;
                let image_height = image.height() as f32;
                let uv_min_x = (source.x() as f32 + 0.5) / image_width;
                let uv_min_y = (source.y() as f32 + 0.5) / image_height;
                let uv_max_x = (source.x() as f32 + source.width() as f32 - 0.5) / image_width;
                let uv_max_y = (source.y() as f32 + source.height() as f32 - 0.5) / image_height;
                let uniform = ImageUniform {
                    destination,
                    uv_rect: [uv_min_x, uv_min_y, uv_max_x, uv_max_y],
                    tint: tint.to_array(),
                    world_clip_x: [0.0; 4],
                    world_clip_y: [0.0; 4],
                    world_mode: [0.0; 4],
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Image {
                    image,
                    sampling,
                    uniform,
                    viewport,
                });
            }
            FrameItem::WorldImage {
                image,
                source,
                rectangle,
                depth,
                camera,
                tint,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera_uniform = CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                let minimum = rectangle.min();
                let maximum = rectangle.max();
                let world_corners = [
                    Vec2::new(minimum.x, maximum.y),
                    maximum,
                    minimum,
                    Vec2::new(maximum.x, minimum.y),
                ];
                let mut clip_corners = [Vec2::ZERO; 4];
                for (clip, world) in clip_corners.iter_mut().zip(world_corners) {
                    let screen = camera_uniform.world_to_screen(world, depth);
                    *clip = Vec2::new(
                        screen.x.mul_add(
                            camera_uniform.screen_to_clip[0],
                            camera_uniform.screen_to_clip[2],
                        ),
                        screen.y.mul_add(
                            camera_uniform.screen_to_clip[1],
                            camera_uniform.screen_to_clip[3],
                        ),
                    );
                    if ![clip.x, clip.y].into_iter().all(is_portable_shader_source) {
                        return Err(RendererFrameError::InvalidGeometryTransform.into());
                    }
                }
                let image_width = image.width() as f32;
                let image_height = image.height() as f32;
                let uniform = ImageUniform {
                    destination: [0.0; 4],
                    uv_rect: [
                        (source.x() as f32 + 0.5) / image_width,
                        (source.y() as f32 + 0.5) / image_height,
                        (source.x() as f32 + source.width() as f32 - 0.5) / image_width,
                        (source.y() as f32 + source.height() as f32 - 0.5) / image_height,
                    ],
                    tint: tint.to_array(),
                    world_clip_x: [
                        clip_corners[0].x,
                        clip_corners[1].x,
                        clip_corners[2].x,
                        clip_corners[3].x,
                    ],
                    world_clip_y: [
                        clip_corners[0].y,
                        clip_corners[1].y,
                        clip_corners[2].y,
                        clip_corners[3].y,
                    ],
                    world_mode: [1.0, 0.0, 0.0, 0.0],
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Image {
                    image,
                    sampling,
                    uniform,
                    viewport,
                });
            }
            FrameItem::ImageBatch {
                image,
                batch,
                placement,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let uniform = image::batch_uniform(target_viewport, viewport.origin, placement)?;
                // Preflight already proved every sprite against this exact
                // uniform. Items keep immutable batch borrows, and neither the
                // target viewport nor the surface configuration changes between
                // preflight and this ready-item construction. Do not repeat the
                // per-glyph interval proof here (it dominates large HUD frames).
                let vertex_count = batch.sprite_count().saturating_mul(6);
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: usize::from(batch.sprite_count() > 0),
                    vertex_count,
                    streaming_vertex_count: 0,
                    reused_vertex_count: vertex_count,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: usize::from(batch.sprite_count() > 0),
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::ImageBatch {
                    image,
                    batch,
                    sampling,
                    uniform,
                    viewport,
                });
                geometry_reused = true;
            }
            FrameItem::Target {
                target,
                blend_mode,
                opacity,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let uniform = CompositeUniform::in_region(opacity, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<CompositeUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: target.allocation_bytes,
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Target {
                    target,
                    blend_mode,
                    uniform,
                    viewport,
                });
            }
        }
    }
    statistics.texture_bytes = planned.texture_bytes;
    let (particle_cpu_before, particle_cpu_after) =
        ready
            .iter()
            .fold((0usize, 0usize), |(before, after), item| match item {
                ReadyItem::Particle {
                    field,
                    initial_cpu_allocation_bytes,
                    ..
                } => (
                    before.saturating_add(*initial_cpu_allocation_bytes),
                    after.saturating_add(field.cpu_allocation_bytes()),
                ),
                _ => (before, after),
            });
    statistics.retained_cpu_bytes = planned
        .retained_cpu_bytes
        .saturating_sub(particle_cpu_before)
        .saturating_add(particle_cpu_after);
    statistics.retained_buffer_bytes = planned.retained_buffer_bytes;
    statistics.source_counts = planned.source_counts;
    validate_frame_budget(budget, statistics)?;
    renderer.ensure_vertex_capacity(streaming_vertices.len())?;

    bindings
        .try_reserve(ready.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: ready
                .len()
                .saturating_mul(std::mem::size_of::<FrameBinding>()),
        })?;
    cache.reserve_slots(ready.len())?;
    cache.prepare_uniform_uploads(ready, renderer.device.limits().max_buffer_size);
    let tessellation = tessellation_started_at.elapsed();

    // Surface availability is resolved after all fallible CPU preparation but
    // before bind-group creation or queue writes. Repeated skipped frames must
    // not accumulate wgpu's deferred upload staging allocations.
    let acquire_started_at = Instant::now();
    let surface_texture = match renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        wgpu::CurrentSurfaceTexture::Timeout => {
            set_particle_rendered(ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Occluded => {
            set_particle_rendered(ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            let _ = renderer.resize(renderer.config.width, renderer.config.height);
            set_particle_rendered(ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost).into());
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            return Err(RendererFrameError::Surface(RendererSurfaceStatus::Validation).into());
        }
    };
    let surface_acquire = acquire_started_at.elapsed();

    let mut binding_upload = Duration::ZERO;
    let mut camera_uniform_upload = Duration::ZERO;
    let planned_uniform_bytes = ready.iter().map(ready_uniform_bytes).sum::<usize>();
    let mut sharing = cache::FrameBindingSharing::default();
    for (slot, item) in ready.iter().enumerate() {
        let binding_started_at = Instant::now();
        bindings.push(cache.binding(renderer, item, slot, &mut sharing, bindings));
        let elapsed = binding_started_at.elapsed();
        match item {
            ReadyItem::Geometry(_) | ReadyItem::Particle { .. } => {
                camera_uniform_upload += elapsed;
            }
            ReadyItem::Scalar { .. }
            | ReadyItem::Image { .. }
            | ReadyItem::ImageBatch { .. }
            | ReadyItem::Target { .. } => {
                binding_upload += elapsed;
            }
        }
    }
    statistics.upload_bytes = statistics
        .upload_bytes
        .saturating_sub(planned_uniform_bytes)
        .saturating_add(cache.uploaded_uniform_bytes());

    let upload_started_at = Instant::now();
    if !streaming_vertices.is_empty() {
        renderer.queue.write_buffer(
            &renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(streaming_vertices),
        );
    }
    for item in ready.iter() {
        if let ReadyItem::Particle {
            field,
            visible_count,
            ..
        } = item
        {
            let visible_instances =
                if field.visible_instances.is_empty() && *visible_count == field.instances.len() {
                    field.instances.as_slice()
                } else {
                    field.visible_instances.as_slice()
                };
            if !visible_instances.is_empty() {
                renderer.queue.write_buffer(
                    &field.instance_buffer,
                    0,
                    bytemuck::cast_slice(visible_instances),
                );
            }
        }
    }
    let mut upload = upload_started_at.elapsed() + binding_upload;
    let encode_started_at = Instant::now();
    let surface_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let (view, resolve_target) = match &renderer.multisample_target {
        Some(target) => (&target.view, Some(&surface_view)),
        None => (&surface_view, None),
    };
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine composed frame encoder"),
        });
    let uniform_batch_started_at = Instant::now();
    cache.flush_uniform_uploads(&renderer.device, &renderer.queue, &mut encoder, bindings);
    let uniform_batch_upload = uniform_batch_started_at.elapsed();
    upload += uniform_batch_upload;
    let timing = renderer.gpu_timing.reserve(GpuTimingSource::FrameComposer);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine composed frame pass"),
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
            timestamp_writes: timing.as_ref().map(|timing| timing.timestamp_writes()),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        encoding::encode_items(renderer, &mut pass, ready, bindings);
    }
    if let Some(timing) = &timing {
        timing.resolve(&mut encoder);
    }
    renderer.queue.submit([encoder.finish()]);
    let gpu_timing_id = timing.map(|timing| renderer.gpu_timing.submitted(timing));
    renderer.notify_before_present();
    renderer.queue.present(surface_texture);
    set_particle_rendered(ready, true);
    let encode_submit_present = encode_started_at
        .elapsed()
        .saturating_sub(uniform_batch_upload);
    let mut report = frame_report(
        RenderStatus::Drawn,
        statistics,
        tessellation,
        upload,
        camera_uniform_upload,
        surface_acquire,
        encode_submit_present,
        frame_started_at.elapsed(),
        geometry_reused,
        geometry_streamed,
        tessellation_stats,
    );
    report.gpu_timing_id = gpu_timing_id;
    Ok(report)
}
