/// Easing curve used to remap normalized tween progress.
///
/// Curves receive progress in `0.0..=1.0` and return a remapped progress value.
/// Some curves may overshoot above `1.0` to create a spring-like visual feel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// Constant speed with no acceleration.
    Linear,
    /// Smooth polynomial ramp with zero slope at both ends.
    SmoothStep,
    /// Cubic acceleration from rest.
    EaseInCubic,
    /// Cubic deceleration into rest.
    EaseOutCubic,
    /// Cubic acceleration and deceleration around the midpoint.
    #[default]
    EaseInOutCubic,
    /// Fast exponential movement that settles near the end.
    EaseOutExpo,
    /// Overshoots the target, then settles back.
    EaseOutBack,
}

impl Easing {
    /// Samples the curve at normalized progress.
    ///
    /// Input is clamped to `0.0..=1.0`. Output is usually in that same range,
    /// except for overshooting curves such as [`Easing::EaseOutBack`].
    pub fn sample(self, amount: f32) -> f32 {
        let t = amount.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => 1.0 - (1.0 - t).powi(3),
            Self::EaseInOutCubic if t < 0.5 => 4.0 * t * t * t,
            Self::EaseInOutCubic => 1.0 - (-2.0 * t + 2.0).powi(3) * 0.5,
            Self::EaseOutExpo if t >= 1.0 => 1.0,
            Self::EaseOutExpo => 1.0 - 2.0_f32.powf(-10.0 * t),
            Self::EaseOutBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Easing;

    #[test]
    fn easing_clamps_input() {
        assert_eq!(Easing::Linear.sample(-1.0), 0.0);
        assert_eq!(Easing::Linear.sample(2.0), 1.0);
    }
}
