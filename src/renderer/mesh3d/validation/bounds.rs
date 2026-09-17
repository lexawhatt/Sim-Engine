//! Sufficient whole-mesh proofs for finite model and final-row arithmetic.
//!
//! This never rejects geometry. An inconclusive proof delegates to the original
//! source-ordered vertex validation, preserving errors and their attribution.

use super::*;

const MINIMUM_NORMAL: f64 = f32::MIN_POSITIVE as f64;
const MAXIMUM: f64 = MAX_PORTABLE_SHADER_VALUE as f64;
// Four products and three additions dominate every per-vertex dot envelope.
// The extra normal margin also bounds permitted subnormal intermediate sums.
const GAMMA: f64 = 7.0 * f32::EPSILON as f64 / (1.0 - 7.0 * f32::EPSILON as f64);
const SUBNORMAL_MARGIN: f64 = 7.0 * MINIMUM_NORMAL;

#[derive(Clone, Copy)]
struct OperandBounds {
    minimum: f64,
    maximum: f64,
    // Each actual fixed value AND each per-vertex interval endpoint is either
    // zero or at least this magnitude. Infinity denotes an identically zero
    // component. Unlike a plain interval, this can prove products do not flush
    // even when an axis includes zero or has vertices of both signs.
    minimum_nonzero: f64,
}

impl OperandBounds {
    const ZERO: Self = Self {
        minimum: 0.0,
        maximum: 0.0,
        minimum_nonzero: f64::INFINITY,
    };
    const ONE: Self = Self {
        minimum: 1.0,
        maximum: 1.0,
        minimum_nonzero: 1.0,
    };
}

pub(super) fn native_transform_is_proven(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> bool {
    final_rows_are_proven(mesh, model_rows, camera_rows)
}

pub(super) fn fog_transform_is_proven(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    depth_row: [f32; 4],
) -> bool {
    final_rows_are_proven(mesh, model_rows, [depth_row])
}

fn final_rows_are_proven<const N: usize>(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    final_rows: [[f32; 4]; N],
) -> bool {
    // The reference validates every source/row operand, even if multiplied by
    // zero. Do not let aggregate zero rows bypass this contractual rejection.
    if !model_rows
        .iter()
        .chain(final_rows.iter())
        .flatten()
        .copied()
        .all(is_portable_shader_source)
    {
        return false;
    }
    let minimum = mesh.bounds_min();
    let maximum = mesh.bounds_max();
    let minimum = [minimum.x(), minimum.y(), minimum.z()];
    let maximum = [maximum.x(), maximum.y(), maximum.z()];
    let minimum_nonzero = mesh.minimum_nonzero_components();
    let mut source = [OperandBounds::ONE; 4];
    for axis in 0..3 {
        if !is_portable_shader_source(minimum[axis])
            || !is_portable_shader_source(maximum[axis])
            || minimum_nonzero[axis] < f32::MIN_POSITIVE
        {
            return false;
        }
        source[axis] = OperandBounds {
            minimum: f64::from(minimum[axis]),
            maximum: f64::from(maximum[axis]),
            minimum_nonzero: f64::from(minimum_nonzero[axis]),
        };
    }
    let mut world = [OperandBounds::ONE; 4];
    for axis in 0..3 {
        let Some(bounds) = model_component_bounds(model_rows[axis], source) else {
            return false;
        };
        world[axis] = bounds;
    }
    // Native clip coordinates and fog distances are not inputs to another dot,
    // so a final interval may include zero. This proves the vertex evaluator's
    // arithmetic contract, not frustum classification or generated attributes.
    final_rows
        .into_iter()
        .all(|row| dot_bounds(row, world).is_some())
}

fn model_component_bounds(row: [f32; 4], source: [OperandBounds; 4]) -> Option<OperandBounds> {
    let mut active = None;
    let mut active_count = 0;
    for axis in 0..4 {
        if row[axis] != 0.0 && source[axis].minimum_nonzero.is_finite() {
            active = Some(axis);
            active_count += 1;
        }
    }
    if active_count == 0 {
        return Some(OperandBounds::ZERO);
    }
    if active_count == 1 {
        let axis = active?;
        let coefficient = row[axis];
        let product_gap = f64::from(coefficient).abs() * source[axis].minimum_nonzero;
        let product =
            interval_products_f64(coefficient, source[axis].minimum, source[axis].maximum);
        if product_gap < MINIMUM_NORMAL || product.0 < -MAXIMUM || product.1 > MAXIMUM {
            return None;
        }
        // Normal f32 values scaled by a power of two stay exact provided the
        // product is normal and bounded. This keeps identity/permutation rows
        // tight all the way down to MIN_POSITIVE, including isolated zeros.
        if coefficient.to_bits() & 0x007f_ffff == 0 {
            return Some(OperandBounds {
                minimum: product.0,
                maximum: product.1,
                minimum_nonzero: product_gap,
            });
        }
        let (minimum, maximum) = dot_bounds(row, source)?;
        // A nonzero single product may round, but cannot cancel. Its own
        // magnitude gives a uniform gap independent of exact zero vertices.
        let minimum_nonzero = product_gap - (GAMMA * product_gap + SUBNORMAL_MARGIN);
        return (minimum_nonzero >= MINIMUM_NORMAL).then_some(OperandBounds {
            minimum,
            maximum,
            minimum_nonzero,
        });
    }
    let (minimum, maximum) = dot_bounds(row, source)?;
    let minimum_nonzero = if minimum >= MINIMUM_NORMAL {
        minimum
    } else if maximum <= -MINIMUM_NORMAL {
        -maximum
    } else {
        // Interior cancellation can create a subnormal fixed value even if
        // every AABB corner is safe. Never infer a gap from corners alone.
        return None;
    };
    Some(OperandBounds {
        minimum,
        maximum,
        minimum_nonzero,
    })
}

fn dot_bounds(row: [f32; 4], point: [OperandBounds; 4]) -> Option<(f64, f64)> {
    let mut minimum_sum = 0.0;
    let mut maximum_sum = 0.0;
    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    let mut magnitude_sum = 0.0;
    for axis in 0..4 {
        if row[axis] == 0.0 || !point[axis].minimum_nonzero.is_finite() {
            continue;
        }
        if f64::from(row[axis]).abs() * point[axis].minimum_nonzero < MINIMUM_NORMAL {
            return None;
        }
        let (minimum, maximum) =
            interval_products_f64(row[axis], point[axis].minimum, point[axis].maximum);
        if minimum < -MAXIMUM || maximum > MAXIMUM {
            return None;
        }
        minimum_sum += minimum;
        maximum_sum += maximum;
        positive_sum += maximum.max(0.0);
        negative_sum += minimum.min(0.0);
        magnitude_sum += minimum.abs().max(maximum.abs());
    }
    // Do not use the reference helper's single-active-term exact-endpoints
    // shortcut here: aggregate representable extrema do not prove every
    // interior product exact. This deliberately wider envelope bounds the
    // reference analytical intervals, not only actual GPU rounded values.
    let margin = if magnitude_sum == 0.0 {
        0.0
    } else {
        GAMMA * magnitude_sum + SUBNORMAL_MARGIN
    };
    let minimum = minimum_sum - margin;
    let maximum = maximum_sum + margin;
    (positive_sum + margin <= MAXIMUM
        && negative_sum - margin >= -MAXIMUM
        && minimum >= -MAXIMUM
        && maximum <= MAXIMUM)
        .then_some((minimum, maximum))
}
