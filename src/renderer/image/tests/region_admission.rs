//! Equivalence oracle frozen before the bounded admission shortcut.

use super::*;

fn original(region: LogicalViewportRegion) -> bool {
    let origin = region.origin().to_vec2();
    let size = region.viewport().size();
    [origin.x, origin.y, size.x, size.y]
        .into_iter()
        .all(is_portable_shader_source)
        && shader_interval_sum_range([
            (f64::from(origin.x), f64::from(origin.x)),
            (0.0, f64::from(size.x)),
        ])
        .is_some()
        && shader_interval_sum_range([
            (f64::from(origin.y), f64::from(origin.y)),
            (0.0, f64::from(size.y)),
        ])
        .is_some()
}

fn check(x: f32, y: f32, width: f32, height: f32) {
    let Ok(viewport) = LogicalViewport::new(width, height) else {
        return;
    };
    let Ok(region) = LogicalViewportRegion::new(LogicalScreenPosition::new(x, y), viewport) else {
        return;
    };
    assert_eq!(
        logical_image_region_is_portable(region),
        original(region),
        "region bits: {:?}",
        [x, y, width, height].map(f32::to_bits)
    );
}

#[test]
fn admission_certificate_matches_original_at_signed_exponent_boundaries() {
    let limits = [
        0.0_f32,
        f32::MIN_POSITIVE,
        1.0,
        MAX_PORTABLE_SHADER_VALUE * 0.25,
        MAX_PORTABLE_SHADER_VALUE * 0.5,
        MAX_PORTABLE_SHADER_VALUE,
        f32::MAX,
    ];
    let mut values = Vec::new();
    for value in limits {
        for bits in [
            value.to_bits().saturating_sub(1),
            value.to_bits(),
            value.to_bits() + 1,
        ] {
            for sign in [0, 0x8000_0000] {
                values.push(f32::from_bits(bits | sign));
            }
        }
    }
    for &x in &values {
        for &width in &values {
            // Both axes, same/opposite signs and exact cancellation endpoints.
            check(x, -x, width, width);
            check(x, 0.0, width, f32::MIN_POSITIVE);
            check(0.0, x, 1.0, width);
        }
    }
    for exponent in 0..=255_u32 {
        for mantissa in [0, 1, (1 << 22) - 1, 1 << 22, (1 << 23) - 1] {
            let value = f32::from_bits((exponent << 23) | mantissa);
            check(value, -value, value, value);
            check(-value, value, 1.0, 1.0);
        }
    }
}

#[test]
fn admission_certificate_matches_original_on_random_full_f32_bit_patterns() {
    let mut state = 0x02a4_e8c6_u32;
    for _ in 0..100_000 {
        let mut values = [0.0; 4];
        for value in &mut values {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = f32::from_bits(state);
        }
        check(values[0], values[1], values[2].abs(), values[3].abs());
    }
}
