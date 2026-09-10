//! Exact, bounded reuse of immutable sprite geometry proofs.

use super::*;
use std::sync::Mutex;

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) use tests::verify_revision_invalidation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedProof {
    key: [u32; 6],
    accepted: bool,
}

pub(super) fn destination_bits(sprite: ImageSprite2d) -> [u32; 4] {
    let destination = sprite.destination();
    let origin = destination.origin().to_vec2();
    let size = destination.viewport().size();
    [origin.x, origin.y, size.x, size.y].map(f32::to_bits)
}

/// One fixed inline entry keeps `ImageBatch2d: Send + Sync` without introducing
/// a global registry, heap allocation, address-based identity or unbounded cache.
#[derive(Default)]
pub(super) struct BatchProofCache {
    entry: Mutex<Option<CachedProof>>,
}

impl BatchProofCache {
    pub(super) fn invalidate(&mut self) {
        *self
            .entry
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    pub(super) fn validate(
        &self,
        sprites: &[ImageSprite2d],
        viewport_origin: Vec2,
        clip_transform: [f32; 4],
    ) -> bool {
        let key = [
            viewport_origin.x,
            viewport_origin.y,
            clip_transform[0],
            clip_transform[1],
            clip_transform[2],
            clip_transform[3],
        ]
        .map(f32::to_bits);
        {
            let entry = self
                .entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(proof) = *entry
                && proof.key == key
            {
                return proof.accepted;
            }
        }
        // Evaluate the original proof outside the lock. Concurrent immutable
        // readers may race to replace the one entry; each result is still keyed
        // by its exact transform. A destination/count mutation requires &mut
        // ImageBatch2d and clears this entry before publication, so no numeric
        // revision can wrap. UV/tint-only changes do not enter this proof.
        let accepted = sprites_are_safe_for_target(sprites, viewport_origin, clip_transform);
        *self
            .entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(CachedProof { key, accepted });
        accepted
    }
}

// Keep the original interval/FTZ checks unchanged. In particular, widening all
// sprites into a single extrema interval is not used: endpoint-sensitive
// subnormal rejection is not generally monotone under such widening.
pub(in crate::renderer) fn sprites_are_safe_for_target(
    sprites: &[ImageSprite2d],
    viewport_origin: Vec2,
    clip_transform: [f32; 4],
) -> bool {
    if ![viewport_origin.x, viewport_origin.y]
        .into_iter()
        .chain(clip_transform)
        .all(is_portable_shader_source)
    {
        return false;
    }
    sprites.iter().all(|sprite| {
        let destination = sprite.destination();
        let origin = destination.origin().to_vec2();
        let size = destination.viewport().size();
        if ![origin.x, origin.y, size.x, size.y]
            .into_iter()
            .all(is_portable_shader_source)
        {
            return false;
        }
        let horizontal = shader_interval_sum_range([
            (f64::from(viewport_origin.x), f64::from(viewport_origin.x)),
            (f64::from(origin.x), f64::from(origin.x)),
            (0.0, f64::from(size.x)),
        ]);
        let vertical = shader_interval_sum_range([
            (f64::from(viewport_origin.y), f64::from(viewport_origin.y)),
            (f64::from(origin.y), f64::from(origin.y)),
            (0.0, f64::from(size.y)),
        ]);
        horizontal.is_some_and(|horizontal| {
            shader_clip_interval_is_safe(
                horizontal.0,
                horizontal.1,
                clip_transform[0],
                clip_transform[2],
            )
        }) && vertical.is_some_and(|vertical| {
            shader_clip_interval_is_safe(
                vertical.0,
                vertical.1,
                clip_transform[1],
                clip_transform[3],
            )
        })
    })
}
