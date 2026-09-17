//! Source-ordered portable shader dot-product envelopes.

use super::{
    MAX_PORTABLE_SHADER_VALUE, Mesh3dRenderError, interval_products_f64, is_portable_shader_source,
    shader_interval_sum_range,
};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy)]
pub(super) struct ShaderValueRange {
    pub(super) fixed: f32,
    pub(super) minimum: f64,
    pub(super) maximum: f64,
}

impl ShaderValueRange {
    pub(super) fn exact(value: f32) -> Self {
        Self {
            fixed: value,
            minimum: f64::from(value),
            maximum: f64::from(value),
        }
    }
}

pub(super) fn shader_dot_range(
    row: [f32; 4],
    point: [ShaderValueRange; 4],
) -> Result<ShaderValueRange, Mesh3dRenderError> {
    for axis in 0..4 {
        if !is_portable_shader_source(row[axis])
            || !is_portable_shader_source(point[axis].fixed)
            || point[axis].minimum < -f64::from(MAX_PORTABLE_SHADER_VALUE)
            || point[axis].maximum > f64::from(MAX_PORTABLE_SHADER_VALUE)
            || (point[axis].minimum != 0.0
                && point[axis].minimum.abs() < f64::from(f32::MIN_POSITIVE))
            || (point[axis].maximum != 0.0
                && point[axis].maximum.abs() < f64::from(f32::MIN_POSITIVE))
        {
            return Err(Mesh3dRenderError::InvalidGeometryTransform);
        }
    }
    let terms: [(f64, f64); 4] = std::array::from_fn(|axis| {
        interval_products_f64(row[axis], point[axis].minimum, point[axis].maximum)
    });
    let (minimum, maximum) =
        shader_interval_sum_range(terms).ok_or(Mesh3dRenderError::InvalidGeometryTransform)?;
    let mut result = 0.0_f32;
    for axis in 0..4 {
        let product = row[axis] * point[axis].fixed;
        result += product;
        if !product.is_finite() || !result.is_finite() {
            return Err(Mesh3dRenderError::InvalidGeometryTransform);
        }
    }
    Ok(ShaderValueRange {
        fixed: result,
        minimum,
        maximum,
    })
}

#[cfg(test)]
pub(super) fn shader_dot(row: [f32; 4], point: [f32; 4]) -> Result<f32, Mesh3dRenderError> {
    shader_dot_range(row, point.map(ShaderValueRange::exact)).map(|value| value.fixed)
}

pub(super) fn shader_range_minimum_magnitude(value: ShaderValueRange) -> f64 {
    if value.minimum <= 0.0 && value.maximum >= 0.0 {
        0.0
    } else {
        value.minimum.abs().min(value.maximum.abs())
    }
}

pub(super) fn rounded_f32_product_range(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let products = [
        left.0 * right.0,
        left.0 * right.1,
        left.1 * right.0,
        left.1 * right.1,
    ];
    rounded_f32_range(
        products.into_iter().fold(f64::INFINITY, f64::min),
        products.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )
}

pub(super) fn rounded_f32_add_range(
    left: (f64, f64),
    right: (f64, f64),
    subtract: bool,
) -> Option<(f64, f64)> {
    let (minimum, maximum) = if subtract {
        (left.0 - right.1, left.1 - right.0)
    } else {
        (left.0 + right.0, left.1 + right.1)
    };
    rounded_f32_range(minimum, maximum)
}

pub(super) fn rounded_f32_range(minimum: f64, maximum: f64) -> Option<(f64, f64)> {
    if !minimum.is_finite()
        || !maximum.is_finite()
        || minimum < -f64::from(MAX_PORTABLE_SHADER_VALUE)
        || maximum > f64::from(MAX_PORTABLE_SHADER_VALUE)
    {
        return None;
    }
    let mut minimum = f64::from(next_f32_down(minimum as f32)?);
    let mut maximum = f64::from(next_f32_up(maximum as f32)?);
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    if maximum > 0.0 && minimum < minimum_normal {
        minimum = minimum.min(0.0);
    }
    if minimum < 0.0 && maximum > -minimum_normal {
        maximum = maximum.max(0.0);
    }
    Some((minimum, maximum))
}

pub(super) fn next_f32_down(value: f32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    let next = if value == 0.0 {
        -f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits().checked_sub(1)?)
    } else {
        f32::from_bits(value.to_bits().checked_add(1)?)
    };
    next.is_finite().then_some(next)
}

pub(super) fn next_f32_up(value: f32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    let next = if value == 0.0 {
        f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits().checked_add(1)?)
    } else {
        f32::from_bits(value.to_bits().checked_sub(1)?)
    };
    next.is_finite().then_some(next)
}

pub(super) fn wgsl_division_range(
    numerator: (f64, f64),
    denominator: (f64, f64),
) -> Option<(f64, f64)> {
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    let maximum_divisor = 2.0_f64.powi(126);
    if denominator.0 < minimum_normal || denominator.1 > maximum_divisor {
        return None;
    }
    let bounds = if denominator == (1.0, 1.0)
        && numerator.0.is_finite()
        && numerator.1.is_finite()
        && numerator.0 <= numerator.1
    {
        // Orthographic W is often exactly one. Keep the full WGSL division
        // envelope below, including all three outward ULPs, even in this case.
        numerator
    } else {
        let quotients = [
            numerator.0 / denominator.0,
            numerator.0 / denominator.1,
            numerator.1 / denominator.0,
            numerator.1 / denominator.1,
        ];
        (
            quotients.into_iter().fold(f64::INFINITY, f64::min),
            quotients.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let mut range = rounded_f32_range(bounds.0, bounds.1)?;
    // Division permits 2.5 ULP. rounded_f32_range already contributed one
    // outward neighbor, so add two more on each side.
    for _ in 0..2 {
        range.0 = f64::from(next_f32_down(range.0 as f32)?);
        range.1 = f64::from(next_f32_up(range.1 as f32)?);
    }
    Some(range)
}

pub(super) fn wgsl_signed_division_range(
    numerator: (f64, f64),
    denominator: (f64, f64),
) -> Option<(f64, f64)> {
    if denominator.0 > 0.0 {
        wgsl_division_range(numerator, denominator)
    } else if denominator.1 < 0.0 {
        wgsl_division_range(
            (-numerator.1, -numerator.0),
            (-denominator.1, -denominator.0),
        )
    } else {
        None
    }
}

pub(super) fn interval_lerp_range(
    start: (f64, f64),
    end: (f64, f64),
    amount: (f64, f64),
) -> Option<(f64, f64)> {
    let delta = rounded_f32_add_range(end, start, true)?;
    let scaled_delta = rounded_f32_product_range(delta, amount)?;
    rounded_f32_add_range(start, scaled_delta, false)
}
