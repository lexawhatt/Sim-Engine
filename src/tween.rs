use std::time::Duration;

use crate::{easing::Easing, math::Vec2};

/// Value type that can be interpolated by [`Tween`].
///
/// Implementations should avoid producing NaN for normal finite inputs. `amount`
/// is already eased by the caller and is usually in `0.0..=1.0`, but overshoot
/// curves may pass values above `1.0`.
pub trait Interpolate: Copy {
    /// Interpolates between `self` and `end`.
    fn interpolate(self, end: Self, amount: f32) -> Self;
}

impl Interpolate for f32 {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        self + (end - self) * amount
    }
}

impl Interpolate for Vec2 {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        self.lerp(end, amount)
    }
}

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
    pub fn new(initial: T) -> Self {
        Self {
            start: initial,
            current: initial,
            target: initial,
            elapsed: Duration::ZERO,
            duration: Duration::ZERO,
            easing: Easing::Linear,
            active: false,
        }
    }

    /// Retargets the tween and returns it for builder-style setup.
    pub fn to(mut self, target: T, duration: Duration, easing: Easing) -> Self {
        self.set_target(target, duration, easing);
        self
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
    pub fn set_target(&mut self, target: T, duration: Duration, easing: Easing) {
        self.start = self.current;
        self.target = target;
        self.elapsed = Duration::ZERO;
        self.duration = duration;
        self.easing = easing;
        self.active = duration > Duration::ZERO;
        if !self.active {
            self.current = target;
        }
    }

    /// Replaces the current value and target without animation.
    pub fn snap(&mut self, value: T) {
        self.start = value;
        self.current = value;
        self.target = value;
        self.elapsed = Duration::ZERO;
        self.duration = Duration::ZERO;
        self.active = false;
    }

    /// Advances the tween by `dt` and returns the new current value.
    ///
    /// Large `dt` values are allowed and clamp the tween to its target.
    pub fn update(&mut self, dt: Duration) -> T {
        if !self.active {
            return self.current;
        }

        self.elapsed = self.elapsed.saturating_add(dt);
        let duration = self.duration.as_secs_f32();
        let raw_amount = if duration <= f32::EPSILON {
            1.0
        } else {
            self.elapsed.as_secs_f32() / duration
        };
        let amount = raw_amount.clamp(0.0, 1.0);
        self.current = self
            .start
            .interpolate(self.target, self.easing.sample(amount));

        if amount >= 1.0 {
            self.current = self.target;
            self.active = false;
        }

        self.current
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{Easing, Tween};

    #[test]
    fn tween_reaches_target() {
        let mut tween = Tween::new(0.0).to(10.0, Duration::from_millis(100), Easing::Linear);

        assert_eq!(tween.update(Duration::from_millis(50)), 5.0);
        assert_eq!(tween.update(Duration::from_millis(50)), 10.0);
        assert!(!tween.is_active());
    }
}
