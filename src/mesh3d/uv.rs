//! Validated material UV mapping, without changing normalized source UVs.

use crate::Vec2;
use std::{error::Error, fmt};

/// Addressing of a complete independently filtered texture, not a packed tile.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureAddressMode3d {
    /// Extends the outer texel to coordinates beyond the image boundary.
    #[default]
    Clamp,
    /// Repeats the complete image using hardware addressing and continuous LOD derivatives.
    /// Crop packed atlas cells into independent textures before selecting this mode.
    Repeat,
}

/// Affine texture coordinates: `sample_uv = source_uv * scale + offset`.
///
/// Source coordinates retain their normalized `0..=1` contract. Zero scale is
/// allowed and selects a constant coordinate; negative scale reverses an axis.
/// Coefficients must be zero or normal finite values, with absolute values at
/// most 65536; both transformed axis endpoints must also lie in that range.
/// These bounds prevent unsafe shader products/sums but do not promise subtexel
/// precision for arbitrarily large repeat counts. Reduce offset modulo the
/// intended repeat period on the host when high phase precision matters.
///
/// ```
/// use sim_engine::{TextureAddressMode3d, TextureUvTransform3d, Vec2};
///
/// let mapping = TextureUvTransform3d::new(
///     Vec2::new(-4.0, 2.0),
///     Vec2::new(0.25, 0.0),
/// )?;
/// assert_eq!(mapping.scale(), Vec2::new(-4.0, 2.0));
/// let addressing = TextureAddressMode3d::Repeat;
/// # let _ = addressing;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureUvTransform3d {
    scale: Vec2,
    offset: Vec2,
}

impl TextureUvTransform3d {
    /// Identity mapping of normalized source coordinates.
    pub const IDENTITY: Self = Self {
        scale: Vec2::new(1.0, 1.0),
        offset: Vec2::ZERO,
    };
    /// Largest accepted coefficient and transformed endpoint magnitude.
    pub const MAX_MAGNITUDE: f32 = 65_536.0;

    /// Validates coefficients and all endpoint-derived arithmetic before use.
    pub fn new(scale: Vec2, offset: Vec2) -> Result<Self, TextureUvTransformError> {
        for (scale, offset) in [(scale.x(), offset.x()), (scale.y(), offset.y())] {
            for value in [scale, offset] {
                if !value.is_finite() {
                    return Err(TextureUvTransformError::NonFinite);
                }
                if value != 0.0 && !value.is_normal() {
                    return Err(TextureUvTransformError::Subnormal);
                }
                if value.abs() > Self::MAX_MAGNITUDE {
                    return Err(TextureUvTransformError::OutOfRange);
                }
            }
            if (f64::from(scale) + f64::from(offset)).abs() > f64::from(Self::MAX_MAGNITUDE) {
                return Err(TextureUvTransformError::OutOfRange);
            }
        }
        Ok(Self { scale, offset })
    }
    /// Returns the componentwise scale applied before offset.
    pub const fn scale(self) -> Vec2 {
        self.scale
    }
    /// Returns the post-scale offset in complete-image UV units.
    pub const fn offset(self) -> Vec2 {
        self.offset
    }
}
impl Default for TextureUvTransform3d {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Rejected UV mapping, never silently clamped into another coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureUvTransformError {
    /// At least one coefficient was NaN or infinite.
    NonFinite,
    /// A nonzero coefficient was subnormal and could flush to zero on a backend.
    Subnormal,
    /// A coefficient or transformed endpoint exceeds the bounded UV range.
    OutOfRange,
}
impl fmt::Display for TextureUvTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFinite => "3D UV scale and offset must be finite",
            Self::Subnormal => "3D UV coefficients must be zero or normal finite values",
            Self::OutOfRange => "3D UV coefficients and endpoints must remain within +/-65536",
        })
    }
}
impl Error for TextureUvTransformError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_uv_bounds_cover_negative_zero_and_extreme_coefficients() {
        assert!(TextureUvTransform3d::new(Vec2::new(-8.0, 0.0), Vec2::new(3.0, -2.0)).is_ok());
        for invalid in [f32::MAX, f32::INFINITY, f32::NAN, f32::from_bits(1)] {
            assert!(TextureUvTransform3d::new(Vec2::new(invalid, 1.0), Vec2::ZERO).is_err());
        }
        assert!(TextureUvTransform3d::new(Vec2::new(65536.0, 1.0), Vec2::new(1.0, 0.0)).is_err());
        assert!(
            TextureUvTransform3d::new(Vec2::new(65536.0, 1.0), Vec2::new(-65536.0, 0.0)).is_ok()
        );
        assert_eq!(
            TextureUvTransform3d::default(),
            TextureUvTransform3d::IDENTITY
        );
    }
}
