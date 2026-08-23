use std::{error::Error, fmt};

use crate::Color;

/// A finite, rectangular scalar grid for dense simulation data.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl ScalarField {
    /// Creates a field from row-major values, validating dimensions and finiteness.
    pub fn new(width: usize, height: usize, values: Vec<f32>) -> Result<Self, ScalarFieldError> {
        let expected_len = checked_len(width, height)?;
        if values.len() != expected_len {
            return Err(ScalarFieldError::InvalidValueCount {
                expected: expected_len,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ScalarFieldError::NonFiniteValue);
        }
        Ok(Self {
            width,
            height,
            values,
        })
    }

    /// Creates a finite field filled with one finite scalar value.
    pub fn filled(width: usize, height: usize, value: f32) -> Result<Self, ScalarFieldError> {
        let len = checked_len(width, height)?;
        if !value.is_finite() {
            return Err(ScalarFieldError::NonFiniteValue);
        }
        Ok(Self {
            width,
            height,
            values: vec![value; len],
        })
    }

    /// Returns the horizontal cell count.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the vertical cell count.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns row-major scalar values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns a cell value, or `None` outside field bounds.
    pub fn value_at(&self, x: usize, y: usize) -> Option<f32> {
        self.index(x, y).map(|index| self.values[index])
    }

    /// Replaces one cell with a finite value.
    pub fn set(&mut self, x: usize, y: usize, value: f32) -> Result<(), ScalarFieldError> {
        if !value.is_finite() {
            return Err(ScalarFieldError::NonFiniteValue);
        }
        let index = self
            .index(x, y)
            .ok_or(ScalarFieldError::OutOfBounds { x, y })?;
        self.values[index] = value;
        Ok(())
    }

    /// Replaces all values while preserving the field dimensions.
    pub fn replace_values(&mut self, values: Vec<f32>) -> Result<(), ScalarFieldError> {
        if values.len() != self.values.len() {
            return Err(ScalarFieldError::InvalidValueCount {
                expected: self.values.len(),
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ScalarFieldError::NonFiniteValue);
        }
        self.values = values;
        Ok(())
    }

    /// Replaces a contiguous rectangular region with finite row-major values.
    pub fn replace_region(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        values: &[f32],
    ) -> Result<(), ScalarFieldError> {
        let expected = checked_len(width, height)?;
        if values.len() != expected {
            return Err(ScalarFieldError::InvalidValueCount {
                expected,
                actual: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ScalarFieldError::NonFiniteValue);
        }
        let end_x = x
            .checked_add(width)
            .ok_or(ScalarFieldError::OutOfBounds { x, y })?;
        let end_y = y
            .checked_add(height)
            .ok_or(ScalarFieldError::OutOfBounds { x, y })?;
        if end_x > self.width || end_y > self.height {
            return Err(ScalarFieldError::OutOfBounds { x, y });
        }
        for row in 0..height {
            let destination = (y + row) * self.width + x;
            let source = row * width;
            self.values[destination..destination + width]
                .copy_from_slice(&values[source..source + width]);
        }
        Ok(())
    }

    /// Returns finite minimum and maximum scalar values.
    pub fn value_range(&self) -> (f32, f32) {
        let mut minimum = self.values[0];
        let mut maximum = minimum;
        for value in &self.values[1..] {
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
        }
        (minimum, maximum)
    }

    fn index(&self, x: usize, y: usize) -> Option<usize> {
        (x < self.width && y < self.height).then_some(y * self.width + x)
    }
}

fn checked_len(width: usize, height: usize) -> Result<usize, ScalarFieldError> {
    if width == 0 || height == 0 {
        return Err(ScalarFieldError::ZeroDimension);
    }
    width
        .checked_mul(height)
        .ok_or(ScalarFieldError::DimensionsOverflow)
}

/// Rejection reason for scalar-field construction or mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFieldError {
    /// Width and height must both be non-zero.
    ZeroDimension,
    /// Width multiplied by height overflowed `usize`.
    DimensionsOverflow,
    /// Input values did not match the grid dimensions.
    InvalidValueCount {
        /// Required row-major cell count.
        expected: usize,
        /// Supplied value count.
        actual: usize,
    },
    /// A scalar value was NaN or infinite.
    NonFiniteValue,
    /// A requested cell lies outside the field dimensions.
    OutOfBounds {
        /// Horizontal cell index.
        x: usize,
        /// Vertical cell index.
        y: usize,
    },
}

impl fmt::Display for ScalarFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "scalar field dimensions must be non-zero"),
            Self::DimensionsOverflow => write!(formatter, "scalar field dimensions overflow"),
            Self::InvalidValueCount { expected, actual } => {
                write!(
                    formatter,
                    "scalar field needs {expected} values, got {actual}"
                )
            }
            Self::NonFiniteValue => write!(formatter, "scalar field values must be finite"),
            Self::OutOfBounds { x, y } => {
                write!(formatter, "scalar field cell ({x}, {y}) is out of bounds")
            }
        }
    }
}

impl Error for ScalarFieldError {}

/// One normalized color-map control point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorStop {
    position: f32,
    color: Color,
}

impl ColorStop {
    /// Creates a stop with normalized position and normalized linear RGBA color.
    ///
    /// Normalized colors keep the CPU sampler consistent with the renderer's
    /// `Rgba8Unorm` heatmap lookup texture instead of silently clipping HDR values.
    pub fn new(position: f32, color: Color) -> Result<Self, ColorMapError> {
        if !position.is_finite() || !color.is_finite() {
            return Err(ColorMapError::NonFiniteStop);
        }
        if !(0.0..=1.0).contains(&position) {
            return Err(ColorMapError::PositionOutOfRange);
        }
        if !color.is_normalized() {
            return Err(ColorMapError::ColorOutOfRange);
        }
        Ok(Self { position, color })
    }

    /// Returns the normalized stop position.
    pub fn position(self) -> f32 {
        self.position
    }

    /// Returns the linear RGBA stop color.
    pub fn color(self) -> Color {
        self.color
    }
}

/// A piecewise-linear map from normalized scalar values to linear RGBA colors.
///
/// [`Self::sample`] evaluates the control points exactly on the CPU. The `wgpu`
/// heatmap renderer samples a documented 256-entry, 8-bit lookup table, so
/// transitions narrower than one LUT interval can be quantized away.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorMap {
    stops: Vec<ColorStop>,
}

impl ColorMap {
    /// Builds a color map from at least two strictly increasing stops.
    pub fn new(stops: Vec<ColorStop>) -> Result<Self, ColorMapError> {
        if stops.len() < 2 {
            return Err(ColorMapError::TooFewStops);
        }
        if stops
            .windows(2)
            .any(|pair| pair[0].position >= pair[1].position)
        {
            return Err(ColorMapError::StopsNotIncreasing);
        }
        Ok(Self { stops })
    }

    /// Builds a two-color normalized gradient.
    pub fn linear(start: Color, end: Color) -> Result<Self, ColorMapError> {
        Self::new(vec![ColorStop::new(0.0, start)?, ColorStop::new(1.0, end)?])
    }

    /// Returns the immutable control points.
    pub fn stops(&self) -> &[ColorStop] {
        &self.stops
    }

    /// Samples the map, clamping a finite normalized input to its endpoints.
    pub fn sample(&self, value: f32) -> Result<Color, ColorMapError> {
        if !value.is_finite() {
            return Err(ColorMapError::NonFiniteSample);
        }
        Ok(self.sample_normalized(value))
    }

    pub(crate) fn sample_normalized(&self, value: f32) -> Color {
        let value = value.clamp(0.0, 1.0);
        let upper = self.stops.partition_point(|stop| stop.position < value);
        if upper == 0 {
            return self.stops[0].color;
        }
        if upper == self.stops.len() {
            return self.stops[upper - 1].color;
        }
        let start = self.stops[upper - 1];
        let end = self.stops[upper];
        let amount = (value - start.position) / (end.position - start.position);
        Color::rgba(
            start.color.red() + (end.color.red() - start.color.red()) * amount,
            start.color.green() + (end.color.green() - start.color.green()) * amount,
            start.color.blue() + (end.color.blue() - start.color.blue()) * amount,
            start.color.alpha() + (end.color.alpha() - start.color.alpha()) * amount,
        )
    }
}

/// Rejection reason for color-map construction or sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMapError {
    /// A stop position or color channel was NaN or infinite.
    NonFiniteStop,
    /// A stop position was outside `0.0..=1.0`.
    PositionOutOfRange,
    /// A linear RGBA stop color had a channel outside `0.0..=1.0`.
    ColorOutOfRange,
    /// A map needs at least two stops.
    TooFewStops,
    /// Stops must have strictly increasing positions.
    StopsNotIncreasing,
    /// A sampled normalized value was NaN or infinite.
    NonFiniteSample,
}

impl fmt::Display for ColorMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ColorMapError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_field_validates_shape_finiteness_and_updates() {
        let mut field = ScalarField::new(2, 2, vec![1.0, -3.0, 2.0, 4.0]).unwrap();
        assert_eq!(field.value_range(), (-3.0, 4.0));
        field.set(1, 0, 6.0).unwrap();
        assert_eq!(field.value_at(1, 0), Some(6.0));
        assert_eq!(
            field.set(2, 0, 0.0),
            Err(ScalarFieldError::OutOfBounds { x: 2, y: 0 })
        );
        assert_eq!(
            ScalarField::filled(0, 1, 0.0),
            Err(ScalarFieldError::ZeroDimension)
        );
        assert_eq!(
            ScalarField::new(2, 2, vec![0.0; 3]),
            Err(ScalarFieldError::InvalidValueCount {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            ScalarField::filled(1, 1, f32::NAN),
            Err(ScalarFieldError::NonFiniteValue)
        );
        field.replace_region(0, 1, 2, 1, &[7.0, 8.0]).unwrap();
        assert_eq!(field.values(), &[1.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            field.replace_region(1, 1, 2, 1, &[0.0, 1.0]),
            Err(ScalarFieldError::OutOfBounds { x: 1, y: 1 })
        );
        assert_eq!(
            field.replace_region(0, 0, 1, 2, &[0.0]),
            Err(ScalarFieldError::InvalidValueCount {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            field.replace_region(0, 0, 1, 1, &[f32::INFINITY]),
            Err(ScalarFieldError::NonFiniteValue)
        );
    }

    #[test]
    fn color_map_interpolates_and_rejects_invalid_contracts() {
        let map = ColorMap::linear(Color::BLACK, Color::WHITE).unwrap();
        assert_eq!(map.sample(-1.0).unwrap(), Color::BLACK);
        assert_eq!(map.sample(2.0).unwrap(), Color::WHITE);
        assert_eq!(map.sample(0.5).unwrap(), Color::rgba(0.5, 0.5, 0.5, 1.0));
        assert_eq!(map.sample(f32::NAN), Err(ColorMapError::NonFiniteSample));
        let first = ColorStop::new(0.75, Color::WHITE).unwrap();
        let second = ColorStop::new(0.5, Color::BLACK).unwrap();
        assert_eq!(
            ColorMap::new(vec![first, second]),
            Err(ColorMapError::StopsNotIncreasing)
        );
        assert_eq!(
            ColorStop::new(0.5, Color::rgba(1.1, 0.0, 0.0, 1.0)),
            Err(ColorMapError::ColorOutOfRange)
        );
    }
}
