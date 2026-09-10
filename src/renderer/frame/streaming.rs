//! Reuse identical streaming work within one immutable frame, never across frames.

use super::*;

#[derive(Clone, Copy)]
struct SceneProof<'frame> {
    scene: &'frame Scene,
    uniform: CameraUniform,
    ready_index: usize,
    statistics: TessellationStats,
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;

/// Fixed stack storage avoids adding heap retention or an unbounded identity map.
/// References keep each key alive for this present; a mutable/next-frame scene
/// cannot inherit a proof through address reuse. Uniforms compare exact bytes.
#[derive(Default)]
pub(super) struct StreamingSceneProofs<'frame> {
    entries: [Option<SceneProof<'frame>>; 8],
    next: usize,
}

impl<'frame> StreamingSceneProofs<'frame> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        &mut self,
        scene: &'frame Scene,
        uniform: CameraUniform,
        viewport: ResolvedViewport,
        vertices: &mut Vec<Vertex>,
        ready: &mut Vec<ReadyItem<'frame>>,
        statistics: &mut FrameStatistics,
        aggregate: &mut TessellationStats,
        batches: &mut Vec<PreparedDrawBatch>,
    ) -> Result<(), FrameComposerError> {
        let cached = self.entries.iter().flatten().find(|proof| {
            std::ptr::eq(proof.scene, scene)
                && bytemuck::bytes_of(&proof.uniform) == bytemuck::bytes_of(&uniform)
        });
        if let Some(proof) = cached {
            let Some(ReadyItem::Geometry(source)) = ready.get(proof.ready_index) else {
                // Only successful appends publish proofs, and ready is append-only
                // for the lifetime of this helper. Fail closed if that changes.
                return Err(RendererFrameError::InvalidGeometryTransform.into());
            };
            let additional = proof.statistics.vertex_count;
            batches.try_reserve(source.batches.len()).map_err(|_| {
                FrameComposerError::AllocationFailed {
                    requested_bytes: source
                        .batches
                        .len()
                        .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
                }
            })?;
            // The stream is uploaded once before encoding and never mutated
            // between draws. Reuse its ranges, not just its numeric proof: each
            // ordered draw reads the same vertex bytes without a duplicate copy
            // or upload. New frames own a fresh memo and cannot alias this one.
            batches.extend_from_slice(&source.batches);
            *statistics = statistics.adding(FrameStatistics {
                pass_count: 1,
                command_count: scene.command_count(),
                vertex_count: additional,
                streaming_vertex_count: additional,
                upload_bytes: std::mem::size_of::<CameraUniform>(),
                draw_calls: batches.len(),
                ..FrameStatistics::default()
            });
            add_tessellation_stats(aggregate, proof.statistics);
            ready.push(ReadyItem::Geometry(ReadyGeometry {
                source: ReadySource::Streaming,
                vertex_count: additional,
                batches: std::mem::take(batches),
                camera_uniform: uniform,
                // Clips/scissors remain per item, not a property of its proof.
                viewport,
            }));
            return Ok(());
        }
        let ready_index = ready.len();
        let mut source_statistics = TessellationStats::default();
        prepare_streaming_scene_resolved(
            scene,
            uniform,
            viewport,
            vertices,
            ready,
            statistics,
            &mut source_statistics,
            batches,
        )?;
        add_tessellation_stats(aggregate, source_statistics);
        self.entries[self.next] = Some(SceneProof {
            scene,
            uniform,
            ready_index,
            statistics: source_statistics,
        });
        self.next = (self.next + 1) % self.entries.len();
        Ok(())
    }
}
