//! A conservative sufficient proof for ordinary fixed-screen analytic quads.
//! Uncertain or extreme inputs retain the independent per-corner interval path.

use super::{CameraUniform, ScreenPrimitiveGpu};

#[derive(Clone, Copy, Debug)]
pub(super) struct Certificate {
    eligible: bool,
    empty: bool,
    anchor_maximum: [f64; 2],
    corner_maximum: [f64; 2],
    minimum_area_ratio: f64,
    minimum_nonzero_edge: f64,
    maximum_edge: f64,
}

impl Certificate {
    #[cfg(test)]
    pub(super) fn edge_separation_ratio(self, uniform: CameraUniform) -> f64 {
        let mut error = 0.0_f64;
        for axis in 0..2 {
            let translation = if axis == 0 {
                uniform.world_to_screen_x[3]
            } else {
                uniform.world_to_screen_y[3]
            };
            let magnitude = self.anchor_maximum[axis]
                + f64::from(uniform.camera_center[axis]).abs()
                + f64::from(translation).abs()
                + self.corner_maximum[axis]
                + f64::from(uniform.screen_to_clip[axis + 2]).abs()
                    / f64::from(uniform.screen_to_clip[axis]).abs()
                + 1.0;
            error = error.max(128.0 * f64::from(f32::EPSILON) * magnitude);
        }
        self.minimum_nonzero_edge / error
    }

    pub(super) fn new(instances: &[ScreenPrimitiveGpu]) -> Self {
        let mut result = Self::empty();
        for instance in instances {
            result.include(instance);
        }
        result
    }

    pub(super) fn empty() -> Self {
        Self {
            eligible: true,
            empty: true,
            anchor_maximum: [0.0; 2],
            corner_maximum: [0.0; 2],
            minimum_area_ratio: f64::INFINITY,
            minimum_nonzero_edge: f64::INFINITY,
            maximum_edge: 0.0,
        }
    }

    pub(super) fn coordinate_maxima(self) -> ([f64; 2], [f64; 2]) {
        (self.anchor_maximum, self.corner_maximum)
    }

    #[inline]
    pub(super) fn include(&mut self, instance: &ScreenPrimitiveGpu) {
        self.empty = false;
        // This is a proof domain, not an input limit: anything outside it goes
        // through the original interval proof and keeps the same error result.
        let in_domain = |value: f32| {
            value == 0.0
                || (value.is_finite()
                    && (2.0_f32.powi(-16)..=2.0_f32.powi(20)).contains(&value.abs()))
        };
        self.eligible &= instance
            .anchor
            .into_iter()
            .chain(instance.corners.into_iter().flatten())
            .all(in_domain);
        for axis in 0..2 {
            self.anchor_maximum[axis] =
                self.anchor_maximum[axis].max(f64::from(instance.anchor[axis]).abs());
            for corner in instance.corners {
                self.corner_maximum[axis] =
                    self.corner_maximum[axis].max(f64::from(corner[axis]).abs());
            }
        }
        for indices in [[0, 1, 2], [0, 2, 3]] {
            let points = indices.map(|index| instance.corners[index].map(f64::from));
            let first = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
            let second = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
            let positive = first[0] * second[1];
            let negative = first[1] * second[0];
            // Account for f64 preprocessing instead of treating this host
            // determinant as exact real arithmetic.
            let lower_area = (positive - negative).abs()
                - 64.0 * f64::EPSILON * (positive.abs() + negative.abs());
            let perimeter = first.into_iter().chain(second).map(f64::abs).sum::<f64>();
            self.minimum_area_ratio = self.minimum_area_ratio.min(lower_area / (perimeter + 1.0));
            self.eligible &= lower_area > 0.0 && lower_area.is_finite();
            for edge in first.into_iter().chain(second).map(f64::abs) {
                self.maximum_edge = self.maximum_edge.max(edge);
                if edge > 0.0 {
                    self.minimum_nonzero_edge = self.minimum_nonzero_edge.min(edge);
                }
            }
        }
    }

    pub(super) fn accepts(self, uniform: CameraUniform) -> bool {
        if !self.eligible
            || !uniform.sources_are_portable()
            || uniform.world_to_screen_x[..3] != [1.0, 0.0, 0.0]
            || uniform.world_to_screen_y[..3] != [0.0, -1.0, 0.0]
        {
            return false;
        }
        if self.empty {
            return true;
        }
        if uniform.camera_center[0] != uniform.world_to_screen_x[3]
            || uniform.camera_center[1] != -uniform.world_to_screen_y[3]
        {
            return false;
        }
        let mut maximum_error = 0.0_f64;
        for axis in 0..2 {
            let center = f64::from(uniform.camera_center[axis]).abs();
            let translation = f64::from(if axis == 0 {
                uniform.world_to_screen_x[3]
            } else {
                uniform.world_to_screen_y[3]
            })
            .abs();
            let scale = f64::from(uniform.screen_to_clip[axis]).abs();
            let offset = f64::from(uniform.screen_to_clip[axis + 2]).abs();
            if !(8.0..=65_536.0).contains(&center)
                || !(2.0_f64.powi(-20)..=1.0).contains(&scale)
                || !(0.25..=4.0).contains(&offset)
            {
                return false;
            }
            let magnitude = self.anchor_maximum[axis]
                + center
                + translation
                + self.corner_maximum[axis]
                + offset / scale
                + 1.0;
            // For u=EPSILON, the canonical interval pipeline has gamma(1)
            // for anchor-center, gamma(2) for translation and local offset,
            // at most 2u for directed f32 neighbours, and gamma(2) for clip.
            // Using gamma(n)<(n+0.01)u gives position error <16u*M, in logical
            // units. Subtracting two such ranges adds gamma(3) on at most 2M,
            // hence the complete edge-difference error is <48u*M. E=128u*M
            // covers both, including f64 evaluation and tiny normal margins.
            maximum_error = maximum_error.max(128.0 * f64::from(f32::EPSILON) * magnitude);
        }
        // Separated nonzero components cannot produce denormal determinant
        // products. Exactly zero edges have symmetric subtraction envelopes;
        // the nonzero clip offset keeps their uncertainty normal (or exact 0).
        // E already includes the *complete edge-difference* error, rather
        // than just one vertex error. Therefore delta>2E proves every nonzero
        // edge interval stays at least delta/2 from zero. With this domain's
        // minimum M and clip scale, determinant products remain normal.
        if self.minimum_nonzero_edge <= 2.0 * maximum_error {
            return false;
        }
        // If vertex uncertainty is E on each axis, determinant uncertainty is
        // <= 2E*edge_L1 + 8E^2. The final term bounds the interval determinant's
        // own product/add rounding after normalization by edge_L1 + 1.
        let uncertainty = 2.0 * maximum_error
            + 8.0 * maximum_error * maximum_error
            + 64.0 * f64::from(f32::EPSILON) * self.maximum_edge;
        self.minimum_area_ratio > uncertainty
    }
}
