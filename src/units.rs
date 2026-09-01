use std::{error::Error, fmt};

/// Positive finite scalar measured in logical screen pixels.
///
/// Construction is explicit so physical pixels, world lengths, and DPI ratios
/// cannot enter logical-width APIs through an unlabelled `f32`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct LogicalPixels(f32);

impl LogicalPixels {
    /// Labels a positive finite scalar as logical screen pixels.
    pub fn new(value: f32) -> Result<Self, UnitError> {
        positive_finite(value)
            .then_some(Self(value))
            .ok_or(UnitError::InvalidLogicalPixels { value })
    }

    /// Returns the scalar value in logical screen pixels.
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Positive finite distance measured in caller-defined world units.
///
/// This type labels scalar distances such as 2D world-space stroke widths and
/// 3D camera near/far ranges. Positions and directions remain vector types.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WorldLength(f32);

impl WorldLength {
    /// Labels a positive finite scalar as a caller-defined 2D or 3D
    /// world-space distance.
    pub fn new(value: f32) -> Result<Self, UnitError> {
        positive_finite(value)
            .then_some(Self(value))
            .ok_or(UnitError::InvalidWorldLength { value })
    }

    /// Returns the scalar value in caller-defined world units.
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Backend-stable physical texels per logical screen pixel.
///
/// A value below one represents a downsampled target and a value above one a
/// native HiDPI or supersampled target. It is not a line width.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PhysicalPerLogical(f32);

pub(crate) const MIN_STABLE_PHYSICAL_PER_LOGICAL: f64 =
    u32::MAX as f64 * f32::MIN_POSITIVE as f64 / 2.0;
// A one-texel viewport has a half-width/half-height translation in the camera
// uniform. Keep that translation normal as well as the viewport dimension and
// reciprocal clip scale.
pub(crate) const MAX_STABLE_PHYSICAL_PER_LOGICAL: f64 = 0.5 / f32::MIN_POSITIVE as f64;

impl PhysicalPerLogical {
    /// Labels a backend-stable physical-to-logical pixel ratio.
    ///
    /// The bounded range keeps logical dimensions and their reciprocal clip
    /// scales and half-viewport translations normal (not subnormal) for every
    /// non-empty `u32` target.
    pub fn new(value: f32) -> Result<Self, UnitError> {
        stable_physical_per_logical(f64::from(value))
            .then_some(Self(value))
            .ok_or(UnitError::InvalidPhysicalPerLogical { value })
    }

    /// Returns physical texels per logical screen pixel.
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Invalid scalar supplied to an explicitly typed rendering unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnitError {
    /// Logical pixel lengths must be finite and strictly positive.
    InvalidLogicalPixels {
        /// Rejected untyped scalar.
        value: f32,
    },
    /// World-space lengths must be finite and strictly positive.
    InvalidWorldLength {
        /// Rejected untyped scalar.
        value: f32,
    },
    /// Physical-to-logical ratios must keep `u32` target transforms in the
    /// normal finite `f32` range.
    InvalidPhysicalPerLogical {
        /// Rejected untyped scalar.
        value: f32,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogicalPixels { value } => {
                write!(
                    formatter,
                    "logical pixel length must be positive and finite, got {value}"
                )
            }
            Self::InvalidWorldLength { value } => {
                write!(
                    formatter,
                    "world length must be positive and finite, got {value}"
                )
            }
            Self::InvalidPhysicalPerLogical { value } => write!(
                formatter,
                "physical-per-logical ratio must keep u32 target transforms in the normal finite f32 range, got {value}"
            ),
        }
    }
}

impl Error for UnitError {}

fn positive_finite(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

pub(crate) fn stable_physical_per_logical(value: f64) -> bool {
    value.is_finite()
        && (MIN_STABLE_PHYSICAL_PER_LOGICAL..=MAX_STABLE_PHYSICAL_PER_LOGICAL).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::{LogicalPixels, PhysicalPerLogical, UnitError, WorldLength};

    #[test]
    fn scalar_units_reject_unlabelled_invalid_lengths() {
        assert_eq!(LogicalPixels::new(2.0).unwrap().get(), 2.0);
        assert_eq!(WorldLength::new(0.1).unwrap().get(), 0.1);
        assert_eq!(PhysicalPerLogical::new(0.5).unwrap().get(), 0.5);
        assert_eq!(
            LogicalPixels::new(0.0),
            Err(UnitError::InvalidLogicalPixels { value: 0.0 })
        );
        assert_eq!(
            WorldLength::new(f32::INFINITY),
            Err(UnitError::InvalidWorldLength {
                value: f32::INFINITY,
            })
        );
        assert_eq!(
            PhysicalPerLogical::new(-1.0),
            Err(UnitError::InvalidPhysicalPerLogical { value: -1.0 })
        );
        assert!(matches!(
            PhysicalPerLogical::new(f32::MIN_POSITIVE),
            Err(UnitError::InvalidPhysicalPerLogical { .. })
        ));
        assert!(matches!(
            PhysicalPerLogical::new(f32::MAX),
            Err(UnitError::InvalidPhysicalPerLogical { .. })
        ));
    }

    #[test]
    fn physical_per_logical_keeps_one_texel_half_viewport_normal() {
        let maximum = 0.5 / f32::MIN_POSITIVE;
        let scale = PhysicalPerLogical::new(maximum).unwrap().get();
        let logical_extent = 1.0_f32 / scale;

        assert!(logical_extent.is_normal());
        assert!((logical_extent * 0.5).is_normal());
        assert!(PhysicalPerLogical::new(f32::from_bits(maximum.to_bits() + 1)).is_err());
    }
}
