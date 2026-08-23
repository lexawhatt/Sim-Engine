use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector used for world-space and screen-space coordinates.
///
/// The type is intentionally unitless. Callers should document whether a value
/// is in simulation world units, logical screen pixels, physical surface pixels,
/// or another coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub(crate) x: f32,
    pub(crate) y: f32,
}

impl Vec2 {
    /// Vector with both components set to `0.0`.
    pub const ZERO: Self = Self::new(0.0, 0.0);
    /// Vector with both components set to `1.0`.
    pub const ONE: Self = Self::new(1.0, 1.0);
    /// Positive unit vector on the x axis.
    pub const X: Self = Self::new(1.0, 0.0);
    /// Positive unit vector on the y axis.
    pub const Y: Self = Self::new(0.0, 1.0);

    /// Builds a vector from explicit components.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the horizontal component.
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the vertical component.
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Builds a vector with both components set to the same value.
    pub fn splat(value: f32) -> Self {
        Self::new(value, value)
    }

    /// Returns the squared vector length.
    ///
    /// Use this instead of [`Vec2::length`] when only relative distance matters.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Returns the Euclidean vector length.
    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    /// Returns the dot product between two vectors.
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Returns a unit-length vector with the same direction.
    ///
    /// Zero-length and near-zero vectors return [`Vec2::ZERO`] instead of NaN.
    pub fn normalized(self) -> Self {
        if !self.is_finite() {
            Self::ZERO
        } else {
            let maximum_component = self.x.abs().max(self.y.abs());
            if maximum_component <= f32::EPSILON {
                return Self::ZERO;
            }
            let scaled = self / maximum_component;
            let length = scaled.length();
            if length <= f32::EPSILON {
                Self::ZERO
            } else {
                scaled / length
            }
        }
    }

    /// Returns a perpendicular vector rotated 90 degrees counter-clockwise.
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    /// Linearly interpolates toward `end`.
    ///
    /// `amount` is not clamped so callers can intentionally extrapolate.
    pub fn lerp(self, end: Self, amount: f32) -> Self {
        if !self.is_finite() || !end.is_finite() || !amount.is_finite() {
            return self;
        }
        Self::new(
            stable_lerp_component(self.x, end.x, amount),
            stable_lerp_component(self.y, end.y, amount),
        )
    }

    /// Returns true when both components are finite values.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::{Rect, Vec2};

    #[test]
    fn normalization_preserves_large_finite_direction() {
        let normalized = Vec2::new(3.0e38, 3.0e38).normalized();

        assert!(normalized.is_finite());
        assert!((normalized.length() - 1.0).abs() < 0.0001);
        assert!(normalized.x > 0.0 && normalized.y > 0.0);
    }

    #[test]
    fn interpolation_and_rect_center_stay_finite_at_extremes() {
        let midpoint = Vec2::splat(f32::MAX).lerp(Vec2::splat(-f32::MAX), 0.5);
        assert_eq!(midpoint, Vec2::ZERO);

        let rect = Rect::new(Vec2::splat(f32::MAX), Vec2::splat(f32::MAX));
        assert_eq!(rect.center(), Vec2::splat(f32::MAX));
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

/// Axis-aligned rectangle represented by minimum and maximum corners.
///
/// The rectangle may be constructed with inverted coordinates. Call
/// [`Rect::normalized`] before using width or height when input ordering is not
/// trusted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub(crate) min: Vec2,
    pub(crate) max: Vec2,
}

impl Rect {
    /// Builds a rectangle from two corners in the caller's coordinate space.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Returns the first stored corner, which is the minimum after normalization.
    pub const fn min(self) -> Vec2 {
        self.min
    }

    /// Returns the second stored corner, which is the maximum after normalization.
    pub const fn max(self) -> Vec2 {
        self.max
    }

    /// Builds a rectangle from its center and size.
    ///
    /// Negative size components are allowed and can be normalized later.
    pub fn from_center_size(center: Vec2, size: Vec2) -> Self {
        let half = size * 0.5;
        Self::new(center - half, center + half)
    }

    /// Builds a rectangle from its minimum corner and size.
    ///
    /// Negative size components are allowed and can be normalized later.
    pub fn from_min_size(min: Vec2, size: Vec2) -> Self {
        Self::new(min, min + size)
    }

    /// Returns `max.x - min.x` in the rectangle's coordinate space.
    pub fn width(self) -> f32 {
        self.max.x - self.min.x
    }

    /// Returns `max.y - min.y` in the rectangle's coordinate space.
    pub fn height(self) -> f32 {
        self.max.y - self.min.y
    }

    /// Returns width and height as a vector in the rectangle's coordinate space.
    pub fn size(self) -> Vec2 {
        Vec2::new(self.width(), self.height())
    }

    /// Returns the midpoint between `min` and `max`.
    pub fn center(self) -> Vec2 {
        Vec2::new(
            stable_midpoint(self.min.x, self.max.x),
            stable_midpoint(self.min.y, self.max.y),
        )
    }

    /// Expands all edges outward by `amount`.
    ///
    /// Negative values shrink the rectangle.
    pub fn expand(self, amount: f32) -> Self {
        Self::new(
            self.min - Vec2::splat(amount),
            self.max + Vec2::splat(amount),
        )
    }

    /// Returns a rectangle whose `min` components are not greater than `max`.
    pub fn normalized(self) -> Self {
        Self::new(
            Vec2::new(self.min.x.min(self.max.x), self.min.y.min(self.max.y)),
            Vec2::new(self.min.x.max(self.max.x), self.min.y.max(self.max.y)),
        )
    }
}

fn stable_lerp_component(start: f32, end: f32, amount: f32) -> f32 {
    let value = f64::from(start) + (f64::from(end) - f64::from(start)) * f64::from(amount);
    value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
}

fn stable_midpoint(first: f32, second: f32) -> f32 {
    ((f64::from(first) + f64::from(second)) * 0.5) as f32
}
