use crate::renderer::MAX_PORTABLE_SHADER_VALUE;

pub(in crate::renderer) fn range_vector_add(
    left: [(f64, f64); 2],
    right: [(f64, f64); 2],
) -> Option<[(f64, f64); 2]> {
    Some([
        rounded_f32_add_range(left[0], right[0], false)?,
        rounded_f32_add_range(left[1], right[1], false)?,
    ])
}

pub(in crate::renderer) fn range_vector_scale(
    vector: [(f64, f64); 2],
    scalar: (f64, f64),
) -> Option<[(f64, f64); 2]> {
    Some([
        rounded_f32_product_range(vector[0], scalar)?,
        rounded_f32_product_range(vector[1], scalar)?,
    ])
}

pub(in crate::renderer) fn range_dot(
    left: [(f64, f64); 2],
    right: [(f64, f64); 2],
) -> Option<(f64, f64)> {
    rounded_f32_add_range(
        rounded_f32_product_range(left[0], right[0])?,
        rounded_f32_product_range(left[1], right[1])?,
        false,
    )
}

pub(in crate::renderer) fn range_negate(value: (f64, f64)) -> (f64, f64) {
    (-value.1, -value.0)
}

pub(in crate::renderer) fn range_abs(value: (f64, f64)) -> (f64, f64) {
    if value.0 <= 0.0 && value.1 >= 0.0 {
        (0.0, value.0.abs().max(value.1.abs()))
    } else {
        (
            value.0.abs().min(value.1.abs()),
            value.0.abs().max(value.1.abs()),
        )
    }
}

pub(super) fn rounded_f32_division_range(
    numerator: (f64, f64),
    denominator: (f64, f64),
) -> Option<(f64, f64)> {
    if denominator.0 <= 0.0 && denominator.1 >= 0.0 {
        return None;
    }
    let quotients = [
        numerator.0 / denominator.0,
        numerator.0 / denominator.1,
        numerator.1 / denominator.0,
        numerator.1 / denominator.1,
    ];
    let mut range = rounded_f32_range(
        quotients.into_iter().fold(f64::INFINITY, f64::min),
        quotients.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )?;
    // WGSL permits 2.5 ULP division error. Three outward neighbours cover it;
    // an inexact endpoint may already carry one additional rounding neighbour.
    for _ in 0..3 {
        range = (
            f64::from(next_f32_down(range.0 as f32)?),
            f64::from(next_f32_up(range.1 as f32)?),
        );
    }
    Some(range)
}

pub(in crate::renderer) fn rounded_f32_product_range(
    left: (f64, f64),
    right: (f64, f64),
) -> Option<(f64, f64)> {
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

pub(in crate::renderer) fn rounded_f32_add_range(
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
    if minimum == maximum && f64::from(minimum as f32) == minimum {
        return Some((minimum, maximum));
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

pub(super) fn shader_interval_difference(
    left: (f64, f64),
    right: (f64, f64),
) -> Option<(f64, f64)> {
    shader_interval_sum_range([left, (-right.1, -right.0)])
}

pub(super) fn shader_interval_product(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let products = [
        left.0 * right.0,
        left.0 * right.1,
        left.1 * right.0,
        left.1 * right.1,
    ];
    shader_interval_sum_range([(
        products.into_iter().fold(f64::INFINITY, f64::min),
        products.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )])
}

#[cfg(test)]
pub(in crate::renderer) fn shader_interval_sum_is_safe<const N: usize>(
    terms: [(f64, f64); N],
) -> bool {
    shader_interval_sum_range(terms).is_some()
}

#[cfg(test)]
mod tests;

pub(in crate::renderer) fn shader_interval_sum_range<const N: usize>(
    terms: [(f64, f64); N],
) -> Option<(f64, f64)> {
    // WGSL's `dot` may be lowered with a backend-selected association or FMA
    // pattern. Bound all multiplication/addition rounding as well as every
    // same-sign partial sum, then return the complete finite output interval.
    // Keeping that interval is essential when a later shader stage multiplies
    // a cancellation residual by another large coefficient.
    let maximum = f64::from(MAX_PORTABLE_SHADER_VALUE);
    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    let mut minimum_sum = 0.0;
    let mut maximum_sum = 0.0;
    let mut magnitude_sum = 0.0;
    let mut active_terms = 0usize;
    let mut inexact_products = 0usize;
    let mut active_term = (0.0, 0.0);
    for term in terms {
        // WGSL permits implementations to flush subnormal arithmetic to zero.
        // A flushed product can become visually significant after a later
        // large camera or clip multiplier, so reject it structurally.
        if is_nonzero_subnormal_f64(term.0)
            || is_nonzero_subnormal_f64(term.1)
            || !term.0.is_finite()
            || !term.1.is_finite()
            || term.0 < -maximum
            || term.1 > maximum
        {
            return None;
        }
        positive_sum += term.1.max(0.0);
        negative_sum += term.0.min(0.0);
        minimum_sum += term.0;
        maximum_sum += term.1;
        magnitude_sum += term.0.abs().max(term.1.abs());
        if term.0 != 0.0 || term.1 != 0.0 {
            active_terms += 1;
            active_term = term;
        }
        inexact_products += usize::from(term.0 != term.1 || f64::from(term.0 as f32) != term.0);
    }

    // Exact zero terms do not participate in any association, and an isolated
    // exactly representable product needs no rounding allowance. This keeps
    // component-selection rows tight inside the portability envelope while
    // retaining a conservative margin for operations which can round.
    if active_terms == 1 {
        let term = active_term;
        // A single interval term has no association ambiguity. When both
        // endpoints are already representable f32 values, they also prove
        // that the producing multiply rounded exactly at the extrema; adding
        // the remaining exact zero terms cannot enlarge the range. This is
        // important for a later dot row which merely selects or negates one
        // component from an earlier association envelope.
        if [term.0, term.1]
            .into_iter()
            .all(|value| f64::from(value as f32) == value)
        {
            return Some(term);
        }
    }
    let operation_count = inexact_products + active_terms.saturating_sub(1);
    // WGSL correctly-rounded operations may select either adjacent f32 value;
    // use the directed-rounding bound rather than Rust's round-to-nearest.
    // Compute the original formula at compile time for all N<=8 operation
    // counts. No decimal approximation or reduced rounding allowance is used.
    const GAMMA: [f64; 16] = {
        let unit_roundoff = f32::EPSILON as f64;
        let mut values = [0.0; 16];
        let mut index = 0;
        while index < values.len() {
            let count = index as f64;
            values[index] = count * unit_roundoff / (1.0 - count * unit_roundoff);
            index += 1;
        }
        values
    };
    let gamma = GAMMA.get(operation_count).copied().unwrap_or_else(|| {
        let count = operation_count as f64;
        let unit_roundoff = f64::from(f32::EPSILON);
        count * unit_roundoff / (1.0 - count * unit_roundoff)
    });
    let operation_count = operation_count as f64;
    let subnormal_margin = if magnitude_sum > 0.0 {
        operation_count * f64::from(f32::MIN_POSITIVE)
    } else {
        0.0
    };
    let rounding_margin = gamma * magnitude_sum + subnormal_margin;
    let minimum_output = minimum_sum - rounding_margin;
    let maximum_output = maximum_sum + rounding_margin;
    if !positive_sum.is_finite()
        || !negative_sum.is_finite()
        || !minimum_output.is_finite()
        || !maximum_output.is_finite()
        || positive_sum + rounding_margin > maximum
        || negative_sum - rounding_margin < -maximum
        || minimum_output < -maximum
        || maximum_output > maximum
    {
        return None;
    }
    Some((minimum_output, maximum_output))
}

pub(in crate::renderer) fn is_nonzero_subnormal(value: f32) -> bool {
    value != 0.0 && value.abs() < f32::MIN_POSITIVE
}

pub(super) fn is_nonzero_subnormal_f64(value: f64) -> bool {
    value != 0.0 && value.abs() < f64::from(f32::MIN_POSITIVE)
}

pub(super) fn interval_products(coefficient: f32, minimum: f32, maximum: f32) -> (f64, f64) {
    interval_products_f64(coefficient, f64::from(minimum), f64::from(maximum))
}

pub(in crate::renderer) fn interval_products_f64(
    coefficient: f32,
    minimum: f64,
    maximum: f64,
) -> (f64, f64) {
    let first = f64::from(coefficient) * minimum;
    let second = f64::from(coefficient) * maximum;
    (first.min(second), first.max(second))
}
