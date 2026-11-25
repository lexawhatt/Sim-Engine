//! Explicit opt-in raster style for a complete surface pass.

/// Bounded post-lighting RGB controls and UV interpolation for a surface pass.
/// Defaults preserve the original output. Quantization operates in linear light
/// after texture/tint, lighting, fog and scanline attenuation. Alpha is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SurfaceEffects3d {
    levels: u16,
    dither: u8,
    scanline: u8,
    affine_uv: bool,
    snap_grid: [u16; 2],
}
impl SurfaceEffects3d {
    /// Identity effects, with perspective-correct texture interpolation.
    pub const fn new() -> Self {
        Self {
            levels: 0,
            dither: 0,
            scanline: 0,
            affine_uv: false,
            snap_grid: [0, 0],
        }
    }
    /// Snaps projected surface vertices to a normalized screen grid. Both cell
    /// counts must be 1 through 32768, or both zero to disable. Counts span the
    /// complete viewport and stay consistent across independently sized targets.
    /// Requires Native policy, surface-only objects and bounded positive clip W;
    /// unsupported geometry is rejected before target mutation. Snapping may
    /// deliberately collapse sub-cell triangles. Depth is not quantized.
    pub const fn with_vertex_snap(mut self, grid: [u16; 2]) -> Result<Self, SurfaceEffectsError3d> {
        if (grid[0] == 0) != (grid[1] == 0) || grid[0] > 32768 || grid[1] > 32768 {
            return Err(SurfaceEffectsError3d);
        }
        self.snap_grid = grid;
        Ok(self)
    }
    /// Viewport-wide snap cell counts, or `[0,0]` for identity geometry.
    pub const fn vertex_snap_grid(self) -> [u16; 2] {
        self.snap_grid
    }
    /// Selects 2 through 256 evenly spaced linear RGB levels; zero disables it.
    /// A value of one or greater than 256 is rejected.
    pub const fn with_quantization(mut self, levels: u16) -> Result<Self, SurfaceEffectsError3d> {
        if levels == 1 || levels > 256 {
            return Err(SurfaceEffectsError3d);
        }
        self.levels = levels;
        Ok(self)
    }
    /// Ordered 4x4 Bayer dither strength in 1/255 units of one quantization step.
    /// Has no effect when quantization is disabled. Coordinates are target texels.
    pub const fn with_dither(mut self, strength: u8) -> Self {
        self.dither = strength;
        self
    }
    /// Attenuation of odd physical target rows in 1/255 units (255 is black).
    pub const fn with_scanlines(mut self, strength: u8) -> Self {
        self.scanline = strength;
        self
    }
    /// Chooses affine UV interpolation. Requires the Native surface policy;
    /// StrictPortable rejects it before rendering to preserve clipping contracts.
    pub const fn with_affine_uv(mut self, enabled: bool) -> Self {
        self.affine_uv = enabled;
        self
    }
    /// Quantization level count; zero means disabled.
    pub const fn quantization_levels(self) -> u16 {
        self.levels
    }
    /// Ordered dither strength in 1/255 units.
    pub const fn dither_strength(self) -> u8 {
        self.dither
    }
    /// Odd-row attenuation in 1/255 units.
    pub const fn scanline_strength(self) -> u8 {
        self.scanline
    }
    /// Whether UVs interpolate affinely in screen space.
    pub const fn affine_uv(self) -> bool {
        self.affine_uv
    }
}
/// Invalid quantization level count or vertex-snap grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceEffectsError3d;
impl std::fmt::Display for SurfaceEffectsError3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid surface quantization levels or vertex-snap grid")
    }
}
impl std::error::Error for SurfaceEffectsError3d {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snap_grid_requires_both_axes_and_bounded_counts() {
        for grid in [[0, 0], [1, 1], [384, 240], [32768, 32768]] {
            assert_eq!(
                SurfaceEffects3d::new()
                    .with_vertex_snap(grid)
                    .unwrap()
                    .vertex_snap_grid(),
                grid
            );
        }
        for grid in [[0, 1], [1, 0], [32769, 2], [2, u16::MAX]] {
            assert!(SurfaceEffects3d::new().with_vertex_snap(grid).is_err());
        }
    }
    #[test]
    fn quantization_domain_and_identity() {
        assert_eq!(SurfaceEffects3d::new(), SurfaceEffects3d::default());
        for levels in [0, 2, 16, 256] {
            assert_eq!(
                SurfaceEffects3d::new()
                    .with_quantization(levels)
                    .unwrap()
                    .quantization_levels(),
                levels
            );
        }
        for levels in [1, 257, u16::MAX] {
            assert!(SurfaceEffects3d::new().with_quantization(levels).is_err());
        }
    }
}
