//! Compact retained screen primitives under the ordinary frame's ordering/budget.

use std::time::Instant;

use super::{
    FrameComposer, FrameComposerError, FrameItem, FramePassOptions, FrameSourceKind,
    FrameSourceStatistics, FrameStatistics, ReadyGeometry, ReadyItem, ReadySource,
    ResolvedViewport, RetainedResourceAccounting, add_tessellation_stats, retained_key,
};
use crate::renderer::{
    CameraUniform, LogicalViewport, PreparedDrawBatch, RendererFrameError,
    RendererPreparationMetrics, ScreenPrimitiveBatch2d, TessellationStats,
    prepared_scene_belongs_to, screen_camera,
};

impl<'frame> FrameComposer<'frame> {
    /// Adds compact retained solid circles, round lines and rectangles in local
    /// logical screen pixels. World-camera motion does not affect this source.
    ///
    /// Input primitive order and captured logical clips are preserved. Adjacent
    /// equal clips share an instanced draw. The batch is one frame item: split
    /// it where other sources must interleave. One six-vertex quad is referenced
    /// per primitive; analytic coverage replaces polygonal round boundaries.
    /// Resource updates happen before composition and have their own reports.
    /// This path is classified as prepared geometry in source statistics.
    pub fn draw_screen_primitive_batch(
        &mut self,
        batch: &'frame ScreenPrimitiveBatch2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if !prepared_scene_belongs_to(&self.renderer.renderer_identity, &batch.renderer_identity) {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::PreparedScene,
            });
        }
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_key(batch, 36),
                    batch.retained_cpu_bytes(),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(&batch.instance_buffer, 37),
                    0,
                    batch.gpu_buffer_bytes(),
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: batch.primitive_count(),
                vertex_count: batch.vertex_count(),
                reused_vertex_count: batch.vertex_count(),
                upload_bytes: std::mem::size_of::<CameraUniform>(),
                draw_calls: batch.draw_batches.len(),
                source_counts: FrameSourceStatistics::single(FrameSourceKind::PreparedScene),
                ..FrameStatistics::default()
            },
            FrameItem::ScreenPrimitives {
                batch,
                options,
                insertion: self.next_insertion,
            },
        )
    }
}

pub(super) fn validate(
    batch: &ScreenPrimitiveBatch2d,
    viewport: ResolvedViewport,
    target: LogicalViewport,
    scale_factor: f32,
    preparation: &mut RendererPreparationMetrics,
) -> Result<CameraUniform, FrameComposerError> {
    let camera = screen_camera(viewport.viewport)
        .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
    let uniform = CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target)
        .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    let started = Instant::now();
    let safe = batch.is_safe_for(uniform, scale_factor);
    preparation.geometry_validation += started.elapsed();
    if !safe {
        return Err(RendererFrameError::InvalidGeometryTransform.into());
    }
    Ok(uniform)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'frame>(
    batch: &'frame ScreenPrimitiveBatch2d,
    viewport: ResolvedViewport,
    target: LogicalViewport,
    scale_factor: f32,
    preparation: &mut RendererPreparationMetrics,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
    mut batches: Vec<PreparedDrawBatch>,
) -> Result<(), FrameComposerError> {
    let camera_uniform = validate(batch, viewport, target, scale_factor, preparation)?;
    batches.try_reserve(batch.draw_batches.len()).map_err(|_| {
        FrameComposerError::AllocationFailed {
            requested_bytes: batch
                .draw_batches
                .len()
                .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
        }
    })?;
    batches.extend_from_slice(&batch.draw_batches);
    *statistics = statistics.adding(FrameStatistics {
        pass_count: 1,
        command_count: batch.primitive_count(),
        vertex_count: batch.vertex_count(),
        reused_vertex_count: batch.vertex_count(),
        upload_bytes: std::mem::size_of::<CameraUniform>(),
        draw_calls: batches.len(),
        ..FrameStatistics::default()
    });
    add_tessellation_stats(aggregate, batch.tessellation_stats());
    ready.push(ReadyItem::Geometry(ReadyGeometry {
        source: ReadySource::ScreenPrimitives(&batch.instance_buffer),
        vertex_count: batch.vertex_count(),
        batches,
        camera_uniform,
        viewport,
    }));
    Ok(())
}
