//! Source conversion/fragment and target-specific subpixel precision envelope.
use super::CameraUniform;
#[cfg(test)]
use super::ScreenPrimitive2d;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Quality {
    source_error: f64,
    anchor_maximum: [f64; 2],
    corner_maximum: [f64; 2],
}

impl Quality {
    pub(super) fn from_certificate(
        source_error: f64,
        certificate: super::certificate::Certificate,
    ) -> Self {
        let (anchor_maximum, corner_maximum) = certificate.coordinate_maxima();
        Self {
            source_error,
            anchor_maximum,
            corner_maximum,
        }
    }

    #[cfg(test)]
    pub(super) fn new(primitives: &[ScreenPrimitive2d]) -> Self {
        let mut result = Self {
            source_error: 0.0,
            anchor_maximum: [0.0; 2],
            corner_maximum: [0.0; 2],
        };
        for primitive in primitives {
            result.source_error = result.source_error.max(primitive.source_error);
            for axis in 0..2 {
                result.anchor_maximum[axis] =
                    result.anchor_maximum[axis].max(f64::from(primitive.gpu.anchor[axis]).abs());
                for corner in primitive.gpu.corners {
                    result.corner_maximum[axis] =
                        result.corner_maximum[axis].max(f64::from(corner[axis]).abs());
                }
            }
        }
        result
    }

    pub(super) fn accepts(self, uniform: CameraUniform, scale_factor: f32) -> bool {
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return false;
        }
        let mut affine_error = 0.0_f64;
        for axis in 0..2 {
            let scale = f64::from(uniform.screen_to_clip[axis]).abs();
            if !scale.is_finite() || scale == 0.0 {
                return false;
            }
            let translation = if axis == 0 {
                uniform.world_to_screen_x[3]
            } else {
                uniform.world_to_screen_y[3]
            };
            let magnitude = self.anchor_maximum[axis]
                + self.corner_maximum[axis]
                + f64::from(uniform.camera_center[axis]).abs()
                + f64::from(translation).abs()
                + f64::from(uniform.screen_to_clip[axis + 2]).abs() / scale
                + 1.0;
            // Canonical screen shader: center subtraction, two sparse dots,
            // offset addition, clip multiplication/addition. Include the
            // inverse-clip coordinate error in logical pixel units.
            affine_error = affine_error.max(32.0 * f64::from(f32::EPSILON) * magnitude);
        }
        let error = (self.source_error + 2.0 * affine_error) * f64::from(scale_factor);
        error.is_finite() && error <= 0.25
    }
}
