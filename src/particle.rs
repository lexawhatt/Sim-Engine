use std::{error::Error, fmt};

use crate::{Color, Vec2};

/// One validated visual particle used by the optional instanced GPU renderer.
///
/// The type itself is renderer-independent so hosts can generate, validate,
/// serialize, or benchmark particle visual state without enabling `wgpu`.
/// `world_position` is measured in caller-defined world units, `radius` is in
/// logical screen pixels, `color` is linear RGBA, and `depth` is the scalar
/// consumed by [`crate::Projection2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleInstance2d {
    world_position: Vec2,
    radius: f32,
    color: Color,
    depth: f32,
}

impl ParticleInstance2d {
    /// Builds a finite particle with a strictly positive logical-pixel radius.
    pub fn new(
        world_position: Vec2,
        radius: f32,
        color: Color,
        depth: f32,
    ) -> Result<Self, ParticleInstanceError> {
        if !world_position.is_finite()
            || !radius.is_finite()
            || !color.is_finite()
            || !depth.is_finite()
        {
            return Err(ParticleInstanceError::NonFinite);
        }
        if radius <= 0.0 {
            return Err(ParticleInstanceError::InvalidRadius);
        }
        Ok(Self {
            world_position,
            radius,
            color,
            depth,
        })
    }

    /// Returns the world-space particle center.
    pub fn world_position(self) -> Vec2 {
        self.world_position
    }

    /// Returns the radius in logical screen pixels.
    pub fn radius(self) -> f32 {
        self.radius
    }

    /// Returns the linear RGBA particle color.
    pub fn color(self) -> Color {
        self.color
    }

    /// Returns caller-defined pseudo-depth.
    pub fn depth(self) -> f32 {
        self.depth
    }
}

/// Rejection reason for particle visual state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleInstanceError {
    /// Position, radius, color, or pseudo-depth contains NaN or infinity.
    NonFinite,
    /// Radius must be strictly positive in logical screen pixels.
    InvalidRadius,
}

impl fmt::Display for ParticleInstanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(formatter, "particle instance values must be finite"),
            Self::InvalidRadius => write!(formatter, "particle radius must be finite and positive"),
        }
    }
}

impl Error for ParticleInstanceError {}

#[cfg(test)]
mod tests {
    use super::{ParticleInstance2d, ParticleInstanceError};
    use crate::{Color, Vec2};

    #[test]
    fn particle_instances_validate_without_the_renderer_feature() {
        let particle =
            ParticleInstance2d::new(Vec2::new(3.0, -2.0), 4.5, Color::WHITE, 1.0).unwrap();
        assert_eq!(particle.world_position(), Vec2::new(3.0, -2.0));
        assert_eq!(particle.radius(), 4.5);
        assert_eq!(particle.depth(), 1.0);
        assert_eq!(particle.color(), Color::WHITE);
        assert_eq!(
            ParticleInstance2d::new(Vec2::ZERO, 0.0, Color::WHITE, 0.0),
            Err(ParticleInstanceError::InvalidRadius)
        );
        assert_eq!(
            ParticleInstance2d::new(Vec2::ZERO, 1.0, Color::WHITE, f32::NAN),
            Err(ParticleInstanceError::NonFinite)
        );
    }
}
