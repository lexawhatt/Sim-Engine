//! Explicit surface alpha and face-visibility policy, independent of wireframes.

use super::{Color, Mesh3dStyleError};

/// Coverage/depth behavior for a retained surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceAlphaMode3d {
    /// Ignores combined alpha, writes opaque coverage and depth.
    #[default]
    Opaque,
    /// Discards alpha below the cutoff; surviving fragments write alpha one/depth.
    Mask,
    /// Straight-alpha blending after opaque/masked surfaces, without depth writes.
    Blend,
}

/// Surface face visibility after camera projection; display edges are unaffected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceSidedness3d {
    /// Preserves both source windings and disables face culling.
    #[default]
    TwoSided,
    /// Keeps projected counter-clockwise front faces and culls their back faces.
    /// This is viewer-relative winding, not a proof of globally outward topology.
    FrontOnly,
}

/// Validated surface tint, alpha policy and projected face visibility.
///
/// Blend objects are sorted back-to-front by transformed model-bounds-center
/// camera-forward distance, with insertion-order ties. This does not solve
/// intersecting/cyclic overlap or triangle ordering inside one mesh. Blend
/// surfaces do not occlude mathematical hidden-line edges; edges render last.
/// Positive-scale transforms preserve source orientation; reflections are not
/// accepted by `Transform3d`. No lighting or physically based glass is implied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceStyle3d {
    color: Color,
    alpha_mode: SurfaceAlphaMode3d,
    mask_cutoff: f32,
    sidedness: SurfaceSidedness3d,
}

impl SurfaceStyle3d {
    /// Creates the legacy normalized opaque tint, rejecting alpha other than one.
    /// Vertex/texture alpha is ignored by this surface mode, with depth writes.
    pub fn opaque(color: Color) -> Result<Self, Mesh3dStyleError> {
        if !color.is_normalized() || color.alpha() != 1.0 {
            return Err(Mesh3dStyleError::InvalidSurfaceColor);
        }
        Ok(Self {
            color,
            alpha_mode: SurfaceAlphaMode3d::Opaque,
            mask_cutoff: 0.0,
            sidedness: SurfaceSidedness3d::TwoSided,
        })
    }

    /// Creates alpha-cutout coverage with a cutoff that is zero or a normal finite value in `0..=1`.
    /// Combined surface/vertex/texture/material alpha below cutoff is discarded;
    /// equality survives and writes alpha one/depth. Filtering happens before
    /// the comparison; no alpha-to-coverage or automatic coverage preservation.
    pub fn mask(color: Color, cutoff: f32) -> Result<Self, Mesh3dStyleError> {
        if !color.is_normalized() {
            return Err(Mesh3dStyleError::InvalidMaterialColor);
        }
        if !cutoff.is_finite()
            || !(0.0..=1.0).contains(&cutoff)
            || (cutoff != 0.0 && !cutoff.is_normal())
        {
            return Err(Mesh3dStyleError::InvalidMaskCutoff);
        }
        Ok(Self {
            color,
            alpha_mode: SurfaceAlphaMode3d::Mask,
            mask_cutoff: cutoff,
            sidedness: SurfaceSidedness3d::TwoSided,
        })
    }

    /// Creates normalized straight-linear RGBA blending with depth-test/no-write.
    /// Combined alpha zero produces no color coverage and never writes depth.
    pub fn blend(color: Color) -> Result<Self, Mesh3dStyleError> {
        if !color.is_normalized() {
            return Err(Mesh3dStyleError::InvalidMaterialColor);
        }
        Ok(Self {
            color,
            alpha_mode: SurfaceAlphaMode3d::Blend,
            mask_cutoff: 0.0,
            sidedness: SurfaceSidedness3d::TwoSided,
        })
    }

    /// Selects projected face culling without changing mathematical display edges.
    pub const fn with_sidedness(mut self, sidedness: SurfaceSidedness3d) -> Self {
        self.sidedness = sidedness;
        self
    }
    /// Returns the normalized straight-linear multiplicative RGBA tint.
    pub const fn color(self) -> Color {
        self.color
    }
    /// Returns the surface coverage/depth mode.
    pub const fn alpha_mode(self) -> SurfaceAlphaMode3d {
        self.alpha_mode
    }
    /// Returns the alpha cutoff only for masked surfaces.
    pub const fn mask_cutoff(self) -> Option<f32> {
        match self.alpha_mode {
            SurfaceAlphaMode3d::Mask => Some(self.mask_cutoff),
            _ => None,
        }
    }
    /// Returns viewer-relative face visibility; defaults to both sides.
    pub const fn sidedness(self) -> SurfaceSidedness3d {
        self.sidedness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_surface_alpha_preserves_opaque_validation_and_cutoff_endpoints() {
        assert_eq!(
            SurfaceStyle3d::opaque(Color::rgba(1.0, 0.0, 0.0, 0.5)),
            Err(Mesh3dStyleError::InvalidSurfaceColor)
        );
        for alpha in [0.0, 0.5, 1.0] {
            let color = Color::rgba(0.25, 0.5, 1.0, alpha);
            assert_eq!(
                SurfaceStyle3d::blend(color).unwrap().alpha_mode(),
                SurfaceAlphaMode3d::Blend
            );
            for cutoff in [0.0, 0.5, 1.0] {
                let surface = SurfaceStyle3d::mask(color, cutoff).unwrap();
                assert_eq!(surface.mask_cutoff(), Some(cutoff));
                assert_eq!(surface.sidedness(), SurfaceSidedness3d::TwoSided);
                assert_eq!(
                    surface
                        .with_sidedness(SurfaceSidedness3d::FrontOnly)
                        .color(),
                    color
                );
            }
        }
        for cutoff in [f32::NAN, f32::INFINITY, f32::from_bits(1), -0.1, 1.1] {
            assert_eq!(
                SurfaceStyle3d::mask(Color::WHITE, cutoff),
                Err(Mesh3dStyleError::InvalidMaskCutoff)
            );
        }
        for color in [
            Color::rgba(f32::NAN, 0.0, 0.0, 1.0),
            Color::rgba(1.1, 0.0, 0.0, 0.5),
        ] {
            assert_eq!(
                SurfaceStyle3d::blend(color),
                Err(Mesh3dStyleError::InvalidMaterialColor)
            );
            assert_eq!(
                SurfaceStyle3d::mask(color, 0.5),
                Err(Mesh3dStyleError::InvalidMaterialColor)
            );
        }
    }
}
