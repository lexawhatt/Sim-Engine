use super::*;

// Frozen pre-certificate algorithm: keep this independent of production helpers
// so accepted/rejected equivalence includes endpoint-sensitive FTZ rejection.
fn original(sprites: &[ImageSprite2d], viewport_origin: Vec2, clip: [f32; 4]) -> bool {
    if ![viewport_origin.x, viewport_origin.y]
        .into_iter()
        .chain(clip)
        .all(is_portable_shader_source)
    {
        return false;
    }
    sprites.iter().all(|sprite| {
        let destination = sprite.destination();
        let origin = destination.origin().to_vec2();
        let size = destination.viewport().size();
        if ![origin.x, origin.y, size.x, size.y]
            .into_iter()
            .all(is_portable_shader_source)
        {
            return false;
        }
        let horizontal = shader_interval_sum_range([
            (f64::from(viewport_origin.x), f64::from(viewport_origin.x)),
            (f64::from(origin.x), f64::from(origin.x)),
            (0.0, f64::from(size.x)),
        ]);
        let vertical = shader_interval_sum_range([
            (f64::from(viewport_origin.y), f64::from(viewport_origin.y)),
            (f64::from(origin.y), f64::from(origin.y)),
            (0.0, f64::from(size.y)),
        ]);
        horizontal
            .is_some_and(|range| shader_clip_interval_is_safe(range.0, range.1, clip[0], clip[2]))
            && vertical.is_some_and(|range| {
                shader_clip_interval_is_safe(range.0, range.1, clip[1], clip[3])
            })
    })
}

fn compare(values: [f32; 10]) {
    let [x, y, width, height, vx, vy, sx, sy, ox, oy] = values;
    let Ok(viewport) = LogicalViewport::new(width, height) else {
        return;
    };
    let Ok(destination) = LogicalViewportRegion::new(LogicalScreenPosition::new(x, y), viewport)
    else {
        return;
    };
    let Ok(sprite) = ImageSprite2d::new(
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        destination,
        Color::WHITE,
    ) else {
        return;
    };
    let origin = Vec2::new(vx, vy);
    let clip = [sx, sy, ox, oy];
    assert_eq!(
        sprites_are_safe_for_target(&[sprite], origin, clip),
        original(&[sprite], origin, clip),
        "input bits {:?}",
        values.map(f32::to_bits)
    );
}

#[test]
fn target_certificate_preserves_signed_ftz_and_domain_boundaries() {
    let boundaries = [
        0.0_f32,
        f32::MIN_POSITIVE,
        0.000_000_953_674_3,
        0.25,
        1.0,
        4.0,
        1_048_576.0,
        MAX_PORTABLE_SHADER_VALUE,
        f32::MAX,
    ];
    let baseline = [
        12.5,
        32.25,
        6.0,
        8.0,
        0.0,
        0.0,
        2.0 / 1280.0,
        -2.0 / 800.0,
        -1.0,
        1.0,
    ];
    for boundary in boundaries {
        for bits in [
            boundary.to_bits().saturating_sub(1),
            boundary.to_bits(),
            boundary.to_bits() + 1,
        ] {
            for sign in [0, 0x8000_0000] {
                let value = f32::from_bits(bits | sign);
                for index in 0..10 {
                    let mut values = baseline;
                    values[index] = value;
                    compare(values);
                }
                // Both origin signs, maximum size and exact cancellation.
                compare([
                    value,
                    value,
                    1_048_576.0,
                    1_048_576.0,
                    0.0,
                    0.0,
                    2.0_f32.powi(-20),
                    -1.0,
                    0.0,
                    f32::MIN_POSITIVE,
                ]);
                compare([
                    value,
                    value,
                    1.0,
                    1.0,
                    -value,
                    -value,
                    2.0_f32.powi(-20),
                    -1.0,
                    -1.0,
                    1.0,
                ]);
            }
        }
    }
    for origin in [0.0, f32::MIN_POSITIVE, 1.0, 1_048_576.0] {
        for position in [1.0, 1_048_576.0] {
            for size in [f32::MIN_POSITIVE, 1.0, 1_048_576.0] {
                for scale in [2.0_f32.powi(-20), 1.0] {
                    compare([
                        position, position, size, size, origin, origin, scale, -scale, -4.0, 4.0,
                    ]);
                }
            }
        }
    }
}

#[test]
fn target_certificate_matches_original_for_random_f32_and_normal_destinations() {
    let mut seed = 0x789d_aebb_u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..100_000 {
        let mut values = [0.0; 10];
        for value in &mut values {
            *value = f32::from_bits(next());
        }
        values[2] = values[2].abs();
        values[3] = values[3].abs();
        compare(values);
        let mut positive = [0.0; 10];
        for value in &mut positive[..6] {
            *value = 1.0 + (next() % 1_000_000) as f32 * 0.125;
        }
        positive[6..].copy_from_slice(&[2.0 / 1280.0, -2.0 / 800.0, -1.0, 1.0]);
        compare(positive);
    }
}

#[test]
fn positive_sprite_does_not_hide_another_sprites_ftz_rejection() {
    let safe = sprite(8.0, 8.0, 6.0, 8.0);
    let tiny = sprite(
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
    );
    let origin = Vec2::ZERO;
    let clip = [2.0 / 1280.0, -2.0 / 800.0, -1.0, 1.0];
    assert!(positive_target_certificate(origin, clip));
    assert!(positive_sprite_certificate(
        safe.destination().origin().to_vec2(),
        safe.destination().viewport().size()
    ));
    assert!(!original(&[tiny], origin, clip));
    for sprites in [[safe, tiny], [tiny, safe]] {
        assert!(!sprites_are_safe_for_target(&sprites, origin, clip));
        assert_eq!(
            sprites_are_safe_for_target(&sprites, origin, clip),
            original(&sprites, origin, clip)
        );
    }
}

#[test]
#[ignore = "paired release CPU diagnostic; no GPU or presentation"]
fn paired_24000_glyph_target_certificate_diagnostic() {
    assert!(!black_box(cfg!(debug_assertions)));
    let sprites = (0..24_000)
        .map(|index| {
            sprite(
                4.25 + (index % 100) as f32 * 8.0,
                8.5 + (index / 100 % 60) as f32 * 12.0,
                6.0,
                8.0,
            )
        })
        .collect::<Vec<_>>();
    let origin = Vec2::ZERO;
    let clip = [2.0 / 1280.0, -2.0 / 800.0, -1.0, 1.0];
    let operation = |fast| {
        if fast {
            sprites_are_safe_for_target(black_box(&sprites), black_box(origin), black_box(clip))
        } else {
            original(black_box(&sprites), black_box(origin), black_box(clip))
        }
    };
    let mut times = [Vec::new(), Vec::new()];
    for index in 0..36 {
        for fast in if index % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = Instant::now();
            assert!(operation(fast));
            if index >= 6 {
                times[usize::from(fast)].push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    assert_eq!(crate::test_allocations::count(|| operation(true)).1, 0);
    eprintln!(
        "glyph_target_proof glyphs=24000 samples=30 original_p50_ms={:.6} original_p95_ms={:.6} original_p99_ms={:.6} certificate_p50_ms={:.6} certificate_p95_ms={:.6} certificate_p99_ms={:.6}",
        times[0][14], times[0][28], times[0][29], times[1][14], times[1][28], times[1][29]
    );
}
