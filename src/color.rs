use crate::tween::Interpolate;

/// Linear RGBA color with channels in the `0.0..=1.0` range.
///
/// Constructors do not clamp so intermediate animation values can overshoot.
/// Use [`Color::clamp`] before sending untrusted colors to a renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel, where `0.0` is transparent and `1.0` is opaque.
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);

    /// Builds an opaque color from floating-point RGB channels.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::rgba(r, g, b, 1.0)
    }

    /// Builds a color from floating-point RGBA channels.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Builds an opaque color from `0..=255` RGB channels.
    pub fn rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::rgba8(r, g, b, 255)
    }

    /// Builds a color from `0..=255` RGBA channels.
    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::rgba(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    /// Returns the same RGB color with a replaced alpha channel.
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self { a: alpha, ..self }
    }

    /// Multiplies only the alpha channel by `factor`.
    pub fn scale_alpha(self, factor: f32) -> Self {
        Self {
            a: self.a * factor,
            ..self
        }
    }

    /// Clamps every channel to `0.0..=1.0`.
    pub fn clamp(self) -> Self {
        Self::rgba(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
            self.a.clamp(0.0, 1.0),
        )
    }

    /// Returns clamped channels in RGBA order for GPU vertex data.
    pub fn to_array(self) -> [f32; 4] {
        let color = self.clamp();
        [color.r, color.g, color.b, color.a]
    }

    /// Converts to the `wgpu` clear color type.
    #[cfg(feature = "wgpu")]
    pub fn to_wgpu(self) -> wgpu::Color {
        let color = self.clamp();
        wgpu::Color {
            r: color.r as f64,
            g: color.g as f64,
            b: color.b as f64,
            a: color.a as f64,
        }
    }
}

impl Interpolate for Color {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        Self::rgba(
            self.r.interpolate(end.r, amount),
            self.g.interpolate(end.g, amount),
            self.b.interpolate(end.b, amount),
            self.a.interpolate(end.a, amount),
        )
    }
}

/// Default visual palette for Sim;Engine demos and early integrations.
///
/// Applications can ignore this and provide their own colors. The palette exists
/// so examples start from a polished baseline instead of bare grayscale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    /// Main background color.
    pub background: Color,
    /// Elevated surface color for panels or overlays.
    pub surface: Color,
    /// Low-contrast grid line color.
    pub grid: Color,
    /// Higher-contrast axis line color.
    pub axis: Color,
    /// Primary data color.
    pub primary: Color,
    /// Secondary data color.
    pub secondary: Color,
    /// Accent data color.
    pub accent: Color,
    /// Warning or high-energy data color.
    pub warning: Color,
}

impl Palette {
    /// Returns the default Sim;Engine color palette.
    pub fn sim() -> Self {
        Self {
            background: Color::rgb8(12, 14, 18),
            surface: Color::rgb8(27, 31, 39),
            grid: Color::rgba8(140, 158, 180, 38),
            axis: Color::rgba8(214, 224, 238, 100),
            primary: Color::rgb8(86, 195, 255),
            secondary: Color::rgb8(128, 228, 167),
            accent: Color::rgb8(255, 190, 94),
            warning: Color::rgb8(255, 105, 120),
        }
    }
}
