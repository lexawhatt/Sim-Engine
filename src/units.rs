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

/// Positive finite distance measured in caller-defined 3D world units.
///
/// This type labels scalar distances such as camera near/far ranges. Positions
/// and directions remain [`crate::Vec3`] because their three components have
/// vector rather than scalar semantics.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WorldLength(f32);

impl WorldLength {
    /// Labels a positive finite scalar as a 3D world-space distance.
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

/// Positive finite physical texels per logical screen pixel.
///
/// A value below one represents a downsampled target and a value above one a
/// native HiDPI or supersampled target. It is not a line width.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PhysicalPerLogical(f32);

impl PhysicalPerLogical {
    /// Labels a positive finite physical-to-logical pixel ratio.
    pub fn new(value: f32) -> Result<Self, UnitError> {
        positive_finite(value)
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
    /// Physical-to-logical ratios must be finite and strictly positive.
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
                "physical-per-logical ratio must be positive and finite, got {value}"
            ),
        }
    }
}

impl Error for UnitError {}

fn positive_finite(value: f32) -> bool {
    value.is_finite() && value > 0.0
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
    }
}
