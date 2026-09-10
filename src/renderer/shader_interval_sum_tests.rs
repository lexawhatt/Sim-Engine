//! Frozen pre-optimization interval oracle and bitwise differential regressions.

use super::*;

// Keep this independent implementation unchanged when optimizing production.
// It is the pre-0.3 performance change's original arithmetic and branch order.
fn legacy<const N: usize>(terms: [(f64, f64); N]) -> Option<(f64, f64)> {
    let maximum = f64::from(MAX_PORTABLE_SHADER_VALUE);
    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    let mut minimum_sum = 0.0;
    let mut maximum_sum = 0.0;
    let mut magnitude_sum = 0.0;
    for term in terms {
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
    }
    let active_terms = terms
        .iter()
        .filter(|term| term.0 != 0.0 || term.1 != 0.0)
        .count();
    if active_terms == 1 {
        let term = terms
            .into_iter()
            .find(|term| term.0 != 0.0 || term.1 != 0.0)?;
        if [term.0, term.1]
            .into_iter()
            .all(|value| f64::from(value as f32) == value)
        {
            return Some(term);
        }
    }
    let inexact_products = terms
        .iter()
        .filter(|term| {
            term.0 != term.1 || !term.0.is_finite() || f64::from(term.0 as f32) != term.0
        })
        .count();
    let operation_count = inexact_products + active_terms.saturating_sub(1);
    let operation_count = operation_count as f64;
    let unit_roundoff = 2.0_f64.powi(-23);
    let gamma = operation_count * unit_roundoff / (1.0 - operation_count * unit_roundoff);
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

fn bits(result: Option<(f64, f64)>) -> Option<(u64, u64)> {
    result.map(|(minimum, maximum)| (minimum.to_bits(), maximum.to_bits()))
}

fn compare<const N: usize>(terms: [(f64, f64); N]) {
    assert_eq!(
        bits(shader_interval_sum_range(terms)),
        bits(legacy(terms)),
        "N={N} terms={terms:?}"
    );
}

fn values() -> [f64; 28] {
    let normal = f64::from(f32::MIN_POSITIVE);
    let maximum = f64::from(MAX_PORTABLE_SHADER_VALUE);
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.0 / 3.0,
        -1.0 / 3.0,
        1.0 + f64::EPSILON,
        -1.0 - f64::EPSILON,
        normal,
        -normal,
        normal.next_down(),
        normal.next_up(),
        normal * 0.5,
        -normal * 0.5,
        maximum,
        -maximum,
        maximum.next_down(),
        maximum.next_up(),
        f64::from(f32::MAX),
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(1),
        f64::from(f32::from_bits(1)),
        f64::from(f32::from_bits(0x3f80_0001)),
    ]
}

fn adversarial<const N: usize>() {
    compare([(0.0, 0.0); N]);
    compare([(-0.0, -0.0); N]);
    for active in 1..=N {
        for inexact in 0..=active {
            let mut terms = [(0.0, 0.0); N];
            for (index, term) in terms[..active].iter_mut().enumerate() {
                let value = if index < inexact {
                    1.0 + 2.0_f64.powi(-30)
                } else {
                    1.0
                };
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                *term = (value * sign, value * sign);
            }
            compare(terms);
            for term in &mut terms[..active] {
                term.1 = term.1.next_up();
            }
            compare(terms);
        }
    }
    for first in values() {
        for second in values() {
            let mut terms = [(0.0, -0.0); N];
            for (index, term) in terms.iter_mut().enumerate() {
                *term = match index % 4 {
                    0 => (first, second),
                    1 => (-second, -first),
                    2 => (first, first),
                    _ => (second, second),
                };
            }
            compare(terms);
        }
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn randomized<const N: usize>() {
    let mut seed = 0x9812_ab67_3210_fed1_u64;
    for case in 0..25_000 {
        let terms = std::array::from_fn(|index| {
            let first = f64::from(f32::from_bits(next(&mut seed) as u32));
            let second = f64::from(f32::from_bits(next(&mut seed) as u32));
            match (case + index) % 7 {
                0 => (0.0, -0.0),
                1 => (first, first),
                2 => (first.min(second), first.max(second)),
                3 => (first * second, first * second),
                4 => (first * 1.000_000_000_1, first * 1.000_000_001),
                5 => {
                    let value = (first - second) * 0.5;
                    (-value.abs(), value.abs())
                }
                _ => (
                    f64::from_bits(next(&mut seed)),
                    f64::from_bits(next(&mut seed)),
                ),
            }
        });
        compare::<N>(terms);
    }
}

#[test]
fn interval_sum_preserves_all_output_bits_and_rejections() {
    macro_rules! check { ($($count:literal),* $(,)?) => { $(adversarial::<$count>(); randomized::<$count>();)* }; }
    // N=17 proves generic fallback outside the optimized 0..15 operation table.
    check!(0, 1, 2, 3, 4, 5, 6, 7, 8, 17);
}

#[test]
#[ignore = "opt-in paired release CPU diagnostic; not an FPS benchmark"]
fn interval_sum_paired_cpu_diagnostic() {
    use std::{hint::black_box, time::Instant};
    type IntervalSum<const N: usize> = fn([(f64, f64); N]) -> Option<(f64, f64)>;
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    fn measure<const N: usize>() {
        let cases: Vec<[(f64, f64); N]> = (0..1024)
            .map(|case| {
                std::array::from_fn(|index| {
                    let value = (case + index + 1) as f64 * 0.125;
                    match (case + index) % 4 {
                        0 => (0.0, 0.0),
                        1 => (value, value),
                        2 => (-value, value),
                        _ => (value / 3.0, value / 3.0),
                    }
                })
            })
            .collect();
        let functions: [IntervalSum<N>; 2] = [legacy::<N>, shader_interval_sum_range::<N>];
        let mut samples = [Vec::new(), Vec::new()];
        for trial in 0..12 {
            for index in [trial % 2, 1 - trial % 2] {
                let function = black_box(functions[index]);
                let started = Instant::now();
                for _ in 0..64 {
                    for terms in &cases {
                        black_box(function(black_box(*terms)));
                    }
                }
                samples[index]
                    .push(started.elapsed().as_secs_f64() * 1e9 / (64.0 * cases.len() as f64));
            }
        }
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "interval_sum N={N} legacy_median_ns={:.3} candidate_median_ns={:.3} speedup={:.3}",
            samples[0][6],
            samples[1][6],
            samples[0][6] / samples[1][6]
        );
    }
    measure::<2>();
    measure::<3>();
    measure::<4>();
    measure::<8>();
}
