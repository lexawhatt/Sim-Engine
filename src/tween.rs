use std::{error::Error, fmt, time::Duration};

use crate::{easing::Easing, math::Vec2};

/// Value type that can be interpolated by [`Tween`].
///
/// Implementations should avoid producing NaN for normal finite inputs. `amount`
/// is already eased by the caller and is usually in `0.0..=1.0`, but overshoot
/// curves may pass values above `1.0`.
pub trait Interpolate: Copy {
    /// Interpolates between `self` and `end`.
    fn interpolate(self, end: Self, amount: f32) -> Self;

    /// Returns whether this value may be stored by a [`Tween`].
    ///
    /// Implementations must reject any invalid representation that interpolation
    /// could otherwise propagate into renderer or camera state.
    fn is_valid_interpolation_value(self) -> bool;
}

impl Interpolate for f32 {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        finite_f32_lerp(self, end, amount)
    }

    fn is_valid_interpolation_value(self) -> bool {
        self.is_finite()
    }
}

impl Interpolate for Vec2 {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        Self::new(
            finite_f32_lerp(self.x, end.x, amount),
            finite_f32_lerp(self.y, end.y, amount),
        )
    }

    fn is_valid_interpolation_value(self) -> bool {
        self.is_finite()
    }
}

fn finite_f32_lerp(start: f32, end: f32, amount: f32) -> f32 {
    if !start.is_finite() || !end.is_finite() || !amount.is_finite() {
        return start;
    }
    let value = f64::from(start) + (f64::from(end) - f64::from(start)) * f64::from(amount);
    value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
}

/// Rejection reason for tween construction, retargeting, or interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweenError {
    /// An initial, target, or snapped value violates its interpolation contract.
    InvalidValue,
    /// A custom interpolation implementation produced an invalid intermediate value.
    InvalidInterpolatedValue,
}

impl fmt::Display for TweenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue => write!(
                formatter,
                "tween values must satisfy their interpolation contract"
            ),
            Self::InvalidInterpolatedValue => {
                write!(formatter, "tween interpolation produced an invalid value")
            }
        }
    }
}

impl Error for TweenError {}

/// Time-based interpolation from one value to another.
///
/// The tween stores its current value and can be retargeted while active. Time is
/// supplied by the caller, which lets each host drive rendering from its own loop.
#[derive(Debug, Clone)]
pub struct Tween<T: Interpolate> {
    start: T,
    current: T,
    target: T,
    elapsed: Duration,
    duration: Duration,
    easing: Easing,
    active: bool,
}

impl<T: Interpolate> Tween<T> {
    /// Creates an inactive tween whose current value and target are `initial`.
    pub fn new(initial: T) -> Result<Self, TweenError> {
        if !initial.is_valid_interpolation_value() {
            return Err(TweenError::InvalidValue);
        }
        Ok(Self {
            start: initial,
            current: initial,
            target: initial,
            elapsed: Duration::ZERO,
            duration: Duration::ZERO,
            easing: Easing::Linear,
            active: false,
        })
    }

    /// Retargets the tween and returns it for builder-style setup.
    pub fn to(mut self, target: T, duration: Duration, easing: Easing) -> Result<Self, TweenError> {
        self.set_target(target, duration, easing)?;
        Ok(self)
    }

    /// Returns the current interpolated value.
    pub fn value(&self) -> T {
        self.current
    }

    /// Returns the current target value.
    pub fn target(&self) -> T {
        self.target
    }

    /// Returns true while the tween is still moving toward its target.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Starts moving from the current value toward `target`.
    ///
    /// A zero duration snaps immediately to the target.
    pub fn set_target(
        &mut self,
        target: T,
        duration: Duration,
        easing: Easing,
    ) -> Result<(), TweenError> {
        if !target.is_valid_interpolation_value() {
            return Err(TweenError::InvalidValue);
        }
        self.start = self.current;
        self.target = target;
        self.elapsed = Duration::ZERO;
        self.duration = duration;
        self.easing = easing;
        self.active = duration > Duration::ZERO;
        if !self.active {
            self.current = target;
        }
        Ok(())
    }

    /// Replaces the current value and target without animation.
    pub fn snap(&mut self, value: T) -> Result<(), TweenError> {
        if !value.is_valid_interpolation_value() {
            return Err(TweenError::InvalidValue);
        }
        self.start = value;
        self.current = value;
        self.target = value;
        self.elapsed = Duration::ZERO;
        self.duration = Duration::ZERO;
        self.active = false;
        Ok(())
    }

    /// Advances the tween by `dt` and returns the new current value.
    ///
    /// Large `dt` values are allowed and clamp the tween to its target.
    pub fn update(&mut self, dt: Duration) -> Result<T, TweenError> {
        if !self.active {
            return Ok(self.current);
        }

        let elapsed = self.elapsed.saturating_add(dt);
        let amount = if elapsed >= self.duration {
            1.0
        } else {
            (elapsed.as_secs_f64() / self.duration.as_secs_f64()) as f32
        };
        let current = self
            .start
            .interpolate(self.target, self.easing.sample(amount));
        if !current.is_valid_interpolation_value() {
            return Err(TweenError::InvalidInterpolatedValue);
        }

        self.elapsed = elapsed;
        self.current = current;
        if amount >= 1.0 {
            self.current = self.target;
            self.active = false;
        }

        Ok(self.current)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{Easing, Interpolate, Tween, TweenError};

    #[test]
    fn tween_reaches_target() {
        let mut tween = Tween::new(0.0)
            .unwrap()
            .to(10.0, Duration::from_millis(100), Easing::Linear)
            .unwrap();

        assert_eq!(tween.update(Duration::from_millis(50)), Ok(5.0));
        assert_eq!(tween.update(Duration::from_millis(50)), Ok(10.0));
        assert!(!tween.is_active());
    }

    #[test]
    fn maximum_duration_completes_with_maximum_update() {
        let mut tween = Tween::new(2.0)
            .unwrap()
            .to(9.0, Duration::MAX, Easing::Linear)
            .unwrap();

        assert_eq!(tween.update(Duration::MAX), Ok(9.0));
        assert!(!tween.is_active());
    }

    #[test]
    fn retarget_uses_current_value_as_new_start() {
        let mut tween = Tween::new(0.0)
            .unwrap()
            .to(10.0, Duration::from_secs(2), Easing::Linear)
            .unwrap();
        assert_eq!(tween.update(Duration::from_secs(1)), Ok(5.0));

        tween
            .set_target(9.0, Duration::from_secs(1), Easing::Linear)
            .unwrap();

        assert_eq!(tween.update(Duration::from_millis(500)), Ok(7.0));
        assert_eq!(tween.update(Duration::from_millis(500)), Ok(9.0));
    }

    #[test]
    fn extreme_finite_values_do_not_overflow_during_interpolation() {
        let mut scalar = Tween::new(f32::MAX)
            .unwrap()
            .to(-f32::MAX, Duration::from_secs(2), Easing::Linear)
            .unwrap();
        let midpoint = scalar.update(Duration::from_secs(1)).unwrap();
        assert!(midpoint.is_finite());
        assert_eq!(midpoint, 0.0);

        let mut vector = Tween::new(crate::Vec2::splat(f32::MAX))
            .unwrap()
            .to(
                crate::Vec2::splat(-f32::MAX),
                Duration::from_secs(2),
                Easing::Linear,
            )
            .unwrap();
        assert!(vector.update(Duration::from_secs(1)).unwrap().is_finite());
    }

    #[derive(Clone, Copy)]
    struct InvalidIntermediate(f32);

    impl Interpolate for InvalidIntermediate {
        fn interpolate(self, _end: Self, _amount: f32) -> Self {
            Self(f32::NAN)
        }

        fn is_valid_interpolation_value(self) -> bool {
            self.0.is_finite()
        }
    }

    #[test]
    fn tween_rejects_invalid_values_and_is_atomic_on_bad_interpolation() {
        assert!(matches!(
            Tween::new(f32::NAN),
            Err(TweenError::InvalidValue)
        ));

        let mut tween = Tween::new(InvalidIntermediate(1.0))
            .unwrap()
            .to(
                InvalidIntermediate(2.0),
                Duration::from_secs(2),
                Easing::Linear,
            )
            .unwrap();
        assert!(matches!(
            tween.update(Duration::from_secs(1)),
            Err(TweenError::InvalidInterpolatedValue)
        ));
        assert_eq!(tween.value().0, 1.0);
        assert_eq!(tween.elapsed, Duration::ZERO);
        assert!(tween.is_active());
        assert_eq!(
            tween.set_target(
                InvalidIntermediate(f32::NAN),
                Duration::ZERO,
                Easing::Linear,
            ),
            Err(TweenError::InvalidValue)
        );
    }
}
