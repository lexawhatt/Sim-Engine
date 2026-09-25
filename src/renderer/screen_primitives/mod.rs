//! Compact retained analytic circles, capsules and rectangles in logical pixels.

use super::{CameraUniform, PreparedDrawBatch, WgpuRenderer};
use std::sync::{Arc, Mutex};

type CachedProof = (Option<(CameraUniform, u32)>, bool);

mod certificate;
mod pipeline;
mod quality;
mod resources;
mod types;
mod validation;

pub(super) use pipeline::create_pipeline;
pub(super) use types::ScreenPrimitiveGpu;

pub use types::{
    ScreenPrimitive2d, ScreenPrimitiveBudget, ScreenPrimitiveError, ScreenPrimitiveUploadReport,
};

/// An ordered immutable draw snapshot of solid screen primitives.
///
/// Each circle or independent round line is one analytic quad, not a polygonal
/// fan. Curves have no segmentation error. Coverage is evaluated per MSAA sample;
/// a single-sample target has binary edge coverage. Rectangles are axis-aligned.
/// Input order and clips are preserved, including normalized translucent colors.
/// The batch owns CPU recovery data and cannot be cloned into mutable aliases.
/// Camera transforms cannot be supplied: use fixed-screen frame composition.
/// A conservative construction/fragment/transform precision envelope must fit
/// within 0.25 physical pixels at the destination scale; otherwise drawing is
/// rejected before submission. MSAA coverage remains sample-dependent.
pub struct ScreenPrimitiveBatch2d {
    pub(super) renderer_identity: Arc<()>,
    pub(super) instance_buffer: wgpu::Buffer,
    pub(super) instances: Vec<ScreenPrimitiveGpu>,
    pub(super) draw_batches: Vec<PreparedDrawBatch>,
    capacity: usize,
    budget: ScreenPrimitiveBudget,
    primitive_counts: crate::PrimitiveCommandCounts,
    certificate: certificate::Certificate,
    quality: quality::Quality,
    proof_cache: Mutex<[CachedProof; 8]>,
}

impl ScreenPrimitiveBatch2d {
    /// Number of retained primitives, without truncation or visibility quotas.
    pub fn primitive_count(&self) -> usize {
        self.instances.len()
    }
    /// Six shader-generated quad vertices are submitted per instance.
    pub fn vertex_count(&self) -> usize {
        self.instances.len().saturating_mul(6)
    }
    /// Owned instance and clip-span CPU capacities, excluding this fixed object.
    pub fn retained_cpu_bytes(&self) -> usize {
        self.instances
            .capacity()
            .saturating_mul(std::mem::size_of::<ScreenPrimitiveGpu>())
            .saturating_add(
                self.draw_batches
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
            )
    }
    /// Actual retained GPU instance-buffer capacity, excluding driver overhead.
    pub fn gpu_buffer_bytes(&self) -> usize {
        self.capacity * std::mem::size_of::<ScreenPrimitiveGpu>()
    }
    /// Limits used for creation and every subsequent full update.
    pub const fn budget(&self) -> ScreenPrimitiveBudget {
        self.budget
    }

    pub(super) fn tessellation_stats(&self) -> super::TessellationStats {
        super::TessellationStats {
            command_count: self.primitive_count(),
            rendered_command_count: self.primitive_count(),
            command_counts: self.primitive_counts,
            rendered_counts: self.primitive_counts,
            vertex_count: self.vertex_count(),
            draw_batch_count: self.draw_batches.len(),
            ..Default::default()
        }
    }

    pub(super) fn is_safe_for(&self, uniform: CameraUniform, scale_factor: f32) -> bool {
        {
            let cache = self
                .proof_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some((_, valid)) = cache.iter().find(|(key, _)| {
                key.as_ref().is_some_and(|(key, scale)| {
                    bytemuck::bytes_of(key) == bytemuck::bytes_of(&uniform)
                        && *scale == scale_factor.to_bits()
                })
            }) {
                return *valid;
            }
        }
        let valid = self.quality.accepts(uniform, scale_factor)
            && validation::instances_are_safe_accelerated(
                &self.instances,
                self.certificate,
                uniform,
            );
        let mut cache = self
            .proof_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cache.rotate_right(1);
        cache[0] = (Some((uniform, scale_factor.to_bits())), valid);
        valid
    }
}

impl WgpuRenderer {
    /// Uploads ordered analytic screen primitives under explicit residency/work limits.
    /// No draw or presentation occurs. The input slice is snapshotted.
    pub fn create_screen_primitive_batch(
        &self,
        primitives: &[ScreenPrimitive2d],
        budget: ScreenPrimitiveBudget,
    ) -> Result<ScreenPrimitiveBatch2d, ScreenPrimitiveError> {
        resources::create(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            primitives,
            budget,
        )
    }

    /// Replaces the entire snapshot transactionally, reusing CPU/GPU capacity.
    /// Rejected input or allocation failure leaves the previous snapshot intact.
    /// Changed instance data invalidates all camera/viewport proofs. Clip-only
    /// changes preserve numeric proofs because they cannot change shader positions. GPU queue order
    /// protects prior submitted draws; no synchronous readback or wait is added.
    pub fn update_screen_primitive_batch(
        &self,
        batch: &mut ScreenPrimitiveBatch2d,
        primitives: &[ScreenPrimitive2d],
    ) -> Result<ScreenPrimitiveUploadReport, ScreenPrimitiveError> {
        if !Arc::ptr_eq(&self.renderer_identity, &batch.renderer_identity) {
            return Err(ScreenPrimitiveError::RendererMismatch);
        }
        resources::update(&self.device, &self.queue, batch, primitives)
    }

    /// Recreates a device-local batch from its complete retained CPU snapshot.
    /// The old batch remains valid for its old renderer and is not mutated.
    pub fn restore_screen_primitive_batch(
        &self,
        source: &ScreenPrimitiveBatch2d,
    ) -> Result<ScreenPrimitiveBatch2d, ScreenPrimitiveError> {
        resources::restore(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) use tests::verify_analytic_primitives;
