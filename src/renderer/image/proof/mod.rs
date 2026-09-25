//! Exact, bounded reuse of immutable sprite geometry proofs.

use super::ImageSprite2d;
use crate::renderer::{
    Vec2, is_portable_shader_source, shader_clip_interval_is_safe, shader_interval_sum_range,
};
use std::sync::Mutex;

const CERTIFICATE_MAXIMUM: f32 = 1_048_576.0; // 2^20
const CERTIFICATE_MINIMUM_SCALE: f32 = 1.0 / CERTIFICATE_MAXIMUM;

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
        // Evaluate the source proof outside the lock. Concurrent immutable
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

// A narrow sufficient certificate avoids interval work for ordinary positive
// screen destinations. Every uncertain sprite keeps the original interval/FTZ
// checks below. Widening sprites into a shared extrema interval is not used:
// endpoint-sensitive subnormal rejection is not monotone under such widening.
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
    let positive_target = positive_target_certificate(viewport_origin, clip_transform);
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
        if positive_target && positive_sprite_certificate(origin, size) {
            return true;
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

fn positive_target_certificate(viewport_origin: Vec2, clip: [f32; 4]) -> bool {
    [viewport_origin.x, viewport_origin.y]
        .into_iter()
        .all(|value| (0.0..=CERTIFICATE_MAXIMUM).contains(&value))
        && clip[..2]
            .iter()
            .all(|value| (CERTIFICATE_MINIMUM_SCALE..=1.0).contains(&value.abs()))
        && clip[2..].iter().all(|value| value.abs() <= 4.0)
}

fn positive_sprite_certificate(origin: Vec2, size: Vec2) -> bool {
    // Sources passed their original finite/normal checks above. In each axis
    // let s=viewport_origin+origin>=1 and 0<size<=2^20. The three-term interval
    // uses at most gamma(3): lower >= s*(1-gamma3)-gamma3*2^20-3*MIN > 0.624,
    // while upper < 2^22. At scales [2^-20,1], both clip products stay normal
    // with magnitude [2^-21,2^22]. The remaining bounded offset and rounding
    // cannot approach 2^120. Thus every original interval check returns Some.
    [origin.x, origin.y]
        .into_iter()
        .all(|value| (1.0..=CERTIFICATE_MAXIMUM).contains(&value))
        && size.x <= CERTIFICATE_MAXIMUM
        && size.y <= CERTIFICATE_MAXIMUM
}
