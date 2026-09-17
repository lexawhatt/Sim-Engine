use std::time::Instant;

use super::*;

// Frozen dev.2 corner reduction. Keep separate from the shortcut so changes
// in acceptance and outward rounding are compared against the old behavior.
fn reference_division(numerator: (f64, f64), denominator: (f64, f64)) -> Option<(f64, f64)> {
    if denominator.0 < f64::from(f32::MIN_POSITIVE) || denominator.1 > 2.0_f64.powi(126) {
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
    for _ in 0..2 {
        range.0 = f64::from(next_f32_down(range.0 as f32)?);
        range.1 = f64::from(next_f32_up(range.1 as f32)?);
    }
    Some(range)
}

fn bits(range: Option<(f64, f64)>) -> Option<(u64, u64)> {
    range.map(|(minimum, maximum)| (minimum.to_bits(), maximum.to_bits()))
}

#[test]
fn exact_one_division_preserves_three_ulp_envelope_and_fallback_bits() {
    let endpoints = [
        f64::NAN,
        f64::NEG_INFINITY,
        -f64::MAX,
        -f64::from(f32::MAX),
        -f64::from(MAX_PORTABLE_SHADER_VALUE),
        -1.000_000_03,
        -1.0,
        -f64::from(f32::MIN_POSITIVE),
        -f64::from(f32::from_bits(1)),
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::from(f32::from_bits(1)),
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1)),
        1.0,
        1.000_000_03,
        f64::from(f32::MAX),
        f64::from(MAX_PORTABLE_SHADER_VALUE),
        f64::MAX,
        f64::INFINITY,
    ];
    let denominators = [
        (1.0, 1.0),
        (1.0, f64::from(f32::from_bits(1.0f32.to_bits() + 1))),
        (0.5, 1.5),
        (f64::from(f32::MIN_POSITIVE), 1.0),
        (0.0, 1.0),
        (1.0, f64::INFINITY),
        (f64::NAN, 1.0),
    ];
    for minimum in endpoints {
        for maximum in endpoints {
            for denominator in denominators {
                assert_eq!(
                    bits(wgsl_division_range((minimum, maximum), denominator)),
                    bits(reference_division((minimum, maximum), denominator)),
                    "numerator={minimum:?},{maximum:?}; denominator={denominator:?}"
                );
            }
            assert_eq!(
                bits(wgsl_signed_division_range((minimum, maximum), (-1.0, -1.0))),
                bits(reference_division((-maximum, -minimum), (1.0, 1.0))),
            );
        }
    }
    let mut state = 0x243f_6a88u32;
    for _ in 0..20_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let left = f64::from(f32::from_bits(state));
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let right = f64::from(f32::from_bits(state));
        for numerator in [(left, right), (left.min(right), left.max(right))] {
            assert_eq!(
                bits(wgsl_division_range(numerator, (1.0, 1.0))),
                bits(reference_division(numerator, (1.0, 1.0)))
            );
        }
    }
    assert_eq!(
        wgsl_division_range((1.0, 1.0), (1.0, 1.0)),
        Some((
            f64::from(f32::from_bits(1.0f32.to_bits() - 3)),
            f64::from(f32::from_bits(1.0f32.to_bits() + 3)),
        ))
    );
}

#[test]
#[ignore = "paired CPU diagnostic; run serially without other timed workloads"]
fn division_shortcut_paired_cpu_diagnostic() {
    use std::hint::black_box;
    let measure = |fast: bool, denominator| {
        let start = Instant::now();
        for index in 0..300_000 {
            let minimum = f64::from(index % 8192) / 4096.0 - 1.0;
            let numerator = black_box((minimum, minimum + 0.000_001));
            let denominator = black_box(denominator);
            black_box(if fast {
                wgsl_division_range(numerator, denominator)
            } else {
                reference_division(numerator, denominator)
            });
        }
        start.elapsed().as_secs_f64() * 1000.0
    };
    for denominator in [(1.0, 1.0), (0.75, 1.25)] {
        for trial in 0..6 {
            let (reference, candidate) = if trial % 2 == 0 {
                (measure(false, denominator), measure(true, denominator))
            } else {
                let candidate = measure(true, denominator);
                (measure(false, denominator), candidate)
            };
            eprintln!(
                "division denominator={denominator:?} trial={trial} reference_ms={reference:.3} candidate_ms={candidate:.3}"
            );
        }
    }
}
