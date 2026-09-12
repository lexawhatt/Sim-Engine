//! Bounded, application-supplied lighting and distance fog.

use crate::{Color, Vec3};
use std::{error::Error, fmt};

/// Surface RGB illumination; alpha and mathematical display edges are unaffected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceLighting3d {
    /// Preserves the surface/vertex/texture color product without reading normals.
    #[default]
    Unlit,
    /// Ambient plus one directional Lambert term, saturated to normalized LDR RGB.
    Lambert,
}

/// Normalized opaque RGB and bounded ambient intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientLight3d {
    color: Color,
    intensity: f32,
}

impl AmbientLight3d {
    /// Accepts normalized opaque linear RGB and a finite intensity in `0..=1`.
    pub fn new(color: Color, intensity: f32) -> Result<Self, Lighting3dError> {
        validate_light(color, intensity)?;
        Ok(Self { color, intensity })
    }
    /// Returns normalized opaque linear RGB.
    pub const fn color(self) -> Color {
        self.color
    }
    /// Returns the bounded linear intensity.
    pub const fn intensity(self) -> f32 {
        self.intensity
    }
}

/// One directional source whose direction points from a surface toward the light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight3d {
    direction: Vec3,
    color: Color,
    intensity: f32,
}

impl DirectionalLight3d {
    /// Normalizes a nonzero finite direction robustly, without generating normals.
    /// Color is normalized opaque linear RGB; intensity must be finite in `0..=1`.
    pub fn new(
        direction_toward_light: Vec3,
        color: Color,
        intensity: f32,
    ) -> Result<Self, Lighting3dError> {
        validate_light(color, intensity)?;
        let direction = direction_toward_light
            .normalized()
            .map_err(|_| Lighting3dError::InvalidDirection)?;
        Ok(Self {
            direction,
            color,
            intensity,
        })
    }
    /// Returns the normalized world-space direction toward the light.
    pub const fn direction(self) -> Vec3 {
        self.direction
    }
    /// Returns normalized opaque linear RGB.
    pub const fn color(self) -> Color {
        self.color
    }
    /// Returns bounded linear intensity; zero disables directional evaluation.
    pub const fn intensity(self) -> f32 {
        self.intensity
    }
}

/// Ready visual lighting state: ambient and at most one directional source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting3d {
    ambient: AmbientLight3d,
    directional: Option<DirectionalLight3d>,
}

impl Default for Lighting3d {
    fn default() -> Self {
        Self::new(AmbientLight3d {
            color: Color::WHITE,
            intensity: 1.0,
        })
    }
}

impl Lighting3d {
    /// Starts with ambient illumination and no directional source.
    pub const fn new(ambient: AmbientLight3d) -> Self {
        Self {
            ambient,
            directional: None,
        }
    }
    /// Replaces or disables the single directional source.
    pub const fn with_directional(mut self, light: Option<DirectionalLight3d>) -> Self {
        self.directional = light;
        self
    }
    /// Returns ambient illumination.
    pub const fn ambient(self) -> AmbientLight3d {
        self.ambient
    }
    /// Returns the optional directional source.
    pub const fn directional(self) -> Option<DirectionalLight3d> {
        self.directional
    }
}

/// Invalid source lighting state, rejected before any scene mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Lighting3dError {
    /// Light color must be normalized opaque linear RGB.
    InvalidColor,
    /// Intensity must be finite in `0..=1`.
    InvalidIntensity,
    /// Direction must be finite and nonzero.
    InvalidDirection,
}

impl fmt::Display for Lighting3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidColor => "3D light color must be normalized opaque linear RGB",
            Self::InvalidIntensity => "3D light intensity must be finite in 0..=1",
            Self::InvalidDirection => "3D light direction must be finite and nonzero",
        })
    }
}
impl Error for Lighting3dError {}

fn validate_light(color: Color, intensity: f32) -> Result<(), Lighting3dError> {
    if !color.is_normalized() || color.alpha() != 1.0 {
        return Err(Lighting3dError::InvalidColor);
    }
    if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
        return Err(Lighting3dError::InvalidIntensity);
    }
    Ok(())
}

/// Exponential RGB fog using camera-forward world distance, never a culling rule.
///
/// Participating surfaces use `1 - exp(-density * max(view_depth - start, 0))`.
/// Depth is distance along camera forward, not radial distance or projection Z;
/// perspective and orthographic cameras share this contract. Alpha, depth writes,
/// mathematical edges and scene clear color are unchanged. Every surface opts in
/// with `SurfaceStyle3d::with_fog(true)`, independently of illumination mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fog3d {
    color: Color,
    start: f32,
    density: f32,
}

impl Fog3d {
    /// Accepts normalized opaque fog RGB, start in world-distance units and density
    /// in inverse world-distance units. Scalars must be zero or normal positive
    /// finite values; zero density disables fog without evaluating its arithmetic.
    pub fn new(color: Color, start_world_distance: f32, density: f32) -> Result<Self, Fog3dError> {
        if !color.is_normalized() || color.alpha() != 1.0 {
            return Err(Fog3dError::InvalidColor);
        }
        if !nonnegative_normal(start_world_distance) {
            return Err(Fog3dError::InvalidStart);
        }
        if !nonnegative_normal(density) {
            return Err(Fog3dError::InvalidDensity);
        }
        Ok(Self {
            color,
            start: start_world_distance,
            density,
        })
    }
    /// Returns normalized opaque linear fog RGB.
    pub const fn color(self) -> Color {
        self.color
    }
    /// Returns the nonnegative camera-forward distance where fog begins.
    pub const fn start(self) -> f32 {
        self.start
    }
    /// Returns inverse world-distance density; zero disables fog.
    pub const fn density(self) -> f32 {
        self.density
    }
}

fn nonnegative_normal(value: f32) -> bool {
    value == 0.0 || (value.is_normal() && value > 0.0)
}

/// Invalid source fog state, rejected before scene mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fog3dError {
    /// Fog color must be normalized opaque linear RGB.
    InvalidColor,
    /// Start must be zero or positive normal finite world distance.
    InvalidStart,
    /// Density must be zero or positive normal finite inverse world distance.
    InvalidDensity,
}
impl fmt::Display for Fog3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidColor => "3D fog color must be normalized opaque linear RGB",
            Self::InvalidStart => {
                "3D fog start must be zero or positive normal finite world distance"
            }
            Self::InvalidDensity => {
                "3D fog density must be zero or positive normal finite inverse world distance"
            }
        })
    }
}
impl Error for Fog3dError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_rejects_invalid_values_and_normalizes_large_directions() {
        for value in [f32::NAN, f32::INFINITY, -1.0, 1.1] {
            assert_eq!(
                AmbientLight3d::new(Color::WHITE, value),
                Err(Lighting3dError::InvalidIntensity)
            );
        }
        assert_eq!(
            DirectionalLight3d::new(Vec3::ZERO, Color::WHITE, 0.0),
            Err(Lighting3dError::InvalidDirection)
        );
        let large = DirectionalLight3d::new(
            Vec3::new(f32::MAX, f32::MAX, 0.0).unwrap(),
            Color::WHITE,
            1.0,
        )
        .unwrap();
        assert!((large.direction().x() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        for value in [f32::NAN, f32::INFINITY, -1.0, f32::from_bits(1)] {
            assert_eq!(
                Fog3d::new(Color::WHITE, value, 1.0),
                Err(Fog3dError::InvalidStart)
            );
            assert_eq!(
                Fog3d::new(Color::WHITE, 0.0, value),
                Err(Fog3dError::InvalidDensity)
            );
        }
        assert!(Fog3d::new(Color::WHITE, f32::MAX, f32::MAX).is_ok());
        assert!(Fog3d::new(Color::WHITE, 0.0, 0.0).is_ok());
    }
}
