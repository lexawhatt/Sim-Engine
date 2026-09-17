use super::*;

/// Explicit creation policy for independently filtered 3D RGBA8 images.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Texture3dOptions {
    mipmaps: TextureMipmaps3d,
    preserve_alpha: bool,
}
impl Texture3dOptions {
    /// Preserves legacy opaque source pixels and a single mip level.
    pub const fn new() -> Self {
        Self {
            mipmaps: TextureMipmaps3d::None,
            preserve_alpha: false,
        }
    }
    /// Selects bounded complete-chain generation; packed tiles are not isolated automatically.
    pub const fn with_mipmaps(mut self, mipmaps: TextureMipmaps3d) -> Self {
        self.mipmaps = mipmaps;
        self
    }
    /// Accepts arbitrary straight alpha rather than requiring alpha 255.
    pub const fn with_alpha_preservation(mut self, preserve: bool) -> Self {
        self.preserve_alpha = preserve;
        self
    }
    /// Returns the retained complete-image mip policy.
    pub const fn mipmaps(self) -> TextureMipmaps3d {
        self.mipmaps
    }
    /// Whether alpha values other than 255 are accepted and retained.
    pub const fn preserves_alpha(self) -> bool {
        self.preserve_alpha
    }
}
