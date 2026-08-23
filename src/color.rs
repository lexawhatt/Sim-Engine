use crate::tween::Interpolate;

/// Linear RGBA color with channels in the `0.0..=1.0` range.
///
/// Constructors do not clamp so intermediate animation values can overshoot.
/// Use [`Color::clamp`] before sending untrusted colors to a renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
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

    /// Returns the red linear-light channel.
    pub const fn red(self) -> f32 {
        self.r
    }

    /// Returns the green linear-light channel.
    pub const fn green(self) -> f32 {
        self.g
    }

    /// Returns the blue linear-light channel.
    pub const fn blue(self) -> f32 {
        self.b
    }

    /// Returns alpha coverage, where `0.0` is transparent and `1.0` is opaque.
    pub const fn alpha(self) -> f32 {
        self.a
    }

    /// Builds an opaque linear color from `0..=255` sRGB channels.
    pub fn rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::rgba8(r, g, b, 255)
    }

    /// Builds a linear color from `0..=255` sRGB and alpha channels.
    ///
    /// RGB channels are converted from sRGB to linear light. Alpha is linear
    /// coverage and is only normalized to `0.0..=1.0`.
    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::rgba(
            srgb_channel_to_linear(r as f32 / 255.0),
            srgb_channel_to_linear(g as f32 / 255.0),
            srgb_channel_to_linear(b as f32 / 255.0),
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

    /// Returns true when every channel is finite.
    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite() && self.a.is_finite()
    }

    /// Returns true when every channel is finite and lies in `0.0..=1.0`.
    pub fn is_normalized(self) -> bool {
        self.is_finite()
            && [self.r, self.g, self.b, self.a]
                .into_iter()
                .all(|channel| (0.0..=1.0).contains(&channel))
    }

    /// Sanitizes every channel to `0.0..=1.0`.
    ///
    /// NaN channels become `0.0`. Infinite values clamp to the nearest bound.
    pub fn clamp(self) -> Self {
        Self::rgba(
            sanitize_channel(self.r),
            sanitize_channel(self.g),
            sanitize_channel(self.b),
            sanitize_channel(self.a),
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

fn sanitize_channel(channel: f32) -> f32 {
    if channel.is_nan() {
        0.0
    } else {
        channel.clamp(0.0, 1.0)
    }
}

fn srgb_channel_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
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

    fn is_valid_interpolation_value(self) -> bool {
        self.is_finite()
    }
}

/// Default visual palette for Sim;Engine demos and early integrations.
///
/// Applications can ignore this and provide their own colors. The palette exists
/// so examples start from a polished baseline instead of bare grayscale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    background: Color,
    surface: Color,
    grid: Color,
    axis: Color,
    primary: Color,
    secondary: Color,
    accent: Color,
    warning: Color,
}

impl Palette {
    /// Builds a palette from colors ordered as background, surface, grid, axis,
    /// primary, secondary, accent, and warning.
    pub const fn from_colors(colors: [Color; 8]) -> Self {
        Self {
            background: colors[0],
            surface: colors[1],
            grid: colors[2],
            axis: colors[3],
            primary: colors[4],
            secondary: colors[5],
            accent: colors[6],
            warning: colors[7],
        }
    }

    /// Returns the main background color.
    pub const fn background(self) -> Color {
        self.background
    }
    /// Returns the elevated surface color.
    pub const fn surface(self) -> Color {
        self.surface
    }
    /// Returns the grid-line color.
    pub const fn grid(self) -> Color {
        self.grid
    }
    /// Returns the axis color.
    pub const fn axis(self) -> Color {
        self.axis
    }
    /// Returns the primary data color.
    pub const fn primary(self) -> Color {
        self.primary
    }
    /// Returns the secondary data color.
    pub const fn secondary(self) -> Color {
        self.secondary
    }
    /// Returns the accent color.
    pub const fn accent(self) -> Color {
        self.accent
    }
    /// Returns the warning color.
    pub const fn warning(self) -> Color {
        self.warning
    }
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

#[cfg(test)]
mod tests {
    use super::Color;

    #[test]
    fn clamp_sanitizes_non_finite_channels() {
        let color = Color::rgba(f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.5).clamp();

        assert_eq!(color, Color::rgba(0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn rgb8_converts_srgb_to_linear_light() {
        let middle_gray = Color::rgb8(128, 128, 128);

        assert!((middle_gray.red() - 0.21586).abs() < 0.0001);
        assert_eq!(middle_gray.red(), middle_gray.green());
        assert_eq!(middle_gray.green(), middle_gray.blue());
    }
}
