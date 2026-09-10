use super::*;
use crate::{LogicalPixels, PhysicalPerLogical};

const FONT: &[u8] = include_bytes!("../../../examples/assets/fonts/DejaVuSans.ttf");

fn font() -> FontFace {
    FontFace::from_bytes(FONT.to_vec(), FontBudget::default()).unwrap()
}
fn style(size: f32, scale: f32) -> TextStyle {
    TextStyle::new(
        LogicalPixels::new(size).unwrap(),
        PhysicalPerLogical::new(scale).unwrap(),
    )
    .unwrap()
}

#[test]
fn font_load_limits_cover_source_capacity_and_declared_glyphs() {
    assert_eq!(
        FontFace::from_bytes(vec![0; 64], FontBudget::default()).unwrap_err(),
        FontError::InvalidFont
    );
    let mut oversized = Vec::with_capacity(FONT.len() + 512);
    oversized.extend_from_slice(FONT);
    assert!(matches!(
        FontFace::from_bytes(oversized, FontBudget::new(FONT.len(), 65535)),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::FontBytes,
            ..
        })
    ));
    assert!(matches!(
        FontFace::from_bytes(FONT.to_vec(), FontBudget::new(FONT.len(), 1)),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::FontGlyphs,
            ..
        })
    ));
    let face = font();
    let cloned = face.clone();
    assert!(std::ptr::eq(face.font_data(), cloned.font_data()));
    assert_eq!(face.allocation_bytes(), FONT.len());
    assert_eq!(face.font_data(), FONT);
    assert!(face.glyph_count() > 1000);
}

#[test]
fn shaping_has_real_cyrillic_ligatures_kerning_and_whitespace_advances() {
    let face = font();
    let style = style(24.0, 1.0);
    let budget = TextLayoutBudget::default();
    let cyrillic = face.shape_line("Привет", &style, &budget).unwrap();
    assert_eq!(cyrillic.glyphs().len(), 6);
    for glyph in cyrillic.glyphs() {
        assert_ne!(glyph.glyph_id(), 0);
        assert_eq!(glyph.cluster() % 2, 0);
    }
    let a = face.shape_line("A", &style, &budget).unwrap();
    let v = face.shape_line("V", &style, &budget).unwrap();
    assert!(face.shape_line("AV", &style, &budget).unwrap().advance() < a.advance() + v.advance());
    assert!(
        face.shape_line("ffi", &style, &budget)
            .unwrap()
            .glyphs()
            .len()
            < 3
    );
    let spaces = face.shape_line("   ", &style, &budget).unwrap();
    assert!(spaces.advance() > 0.0);
    for glyph in spaces.glyphs() {
        let raster = face
            .rasterize_glyph(glyph.glyph_id(), &style, &budget)
            .unwrap();
        assert_eq!(
            (raster.width(), raster.height(), raster.coverage().len()),
            (0, 0, 0)
        );
    }
    let empty = face
        .shape_line("", &style, &TextLayoutBudget::new(0, 0, 0, 0))
        .unwrap();
    assert!(empty.glyphs().is_empty());
    assert_eq!(empty.advance(), 0.0);
}

#[test]
fn line_rejects_unsupported_input_and_explicit_limits() {
    let face = font();
    let style = style(24.0, 1.0);
    for text in ["a\nb", "a\tb", "a\rb", "\u{2028}", "\0"] {
        assert!(matches!(
            face.shape_line(text, &style, &TextLayoutBudget::default()),
            Err(FontError::UnsupportedText { .. })
        ));
    }
    assert_eq!(
        face.shape_line("\u{10ffff}", &style, &TextLayoutBudget::default())
            .unwrap_err(),
        FontError::MissingGlyph { byte_index: 0 }
    );
    assert!(matches!(
        face.shape_line("hello", &style, &TextLayoutBudget::new(4, 10, 1000, 1000)),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::TextBytes,
            required: 5,
            limit: 4
        })
    ));
    assert!(matches!(
        face.shape_line("hello", &style, &TextLayoutBudget::new(5, 1, 1000, 1000)),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::ShapedGlyphs,
            ..
        })
    ));
    assert!(
        TextStyle::new(
            LogicalPixels::new(f32::MAX).unwrap(),
            PhysicalPerLogical::new(2.0).unwrap()
        )
        .is_err()
    );
}

#[test]
fn dpi_changes_raster_resolution_not_logical_shaping() {
    let face = font();
    let budget = TextLayoutBudget::default();
    let low = face
        .shape_line("AV Привет", &style(24.0, 1.0), &budget)
        .unwrap();
    for scale in [1.25, 1.5, 2.0] {
        let high = face
            .shape_line("AV Привет", &style(24.0, scale), &budget)
            .unwrap();
        assert_eq!(low.glyphs(), high.glyphs());
        assert_eq!(low.advance(), high.advance());
        assert_eq!(low.ascent(), high.ascent());
    }
    let glyph = low.glyphs()[0].glyph_id();
    let low = face
        .rasterize_glyph(glyph, &style(24.0, 1.0), &budget)
        .unwrap();
    let high = face
        .rasterize_glyph(glyph, &style(24.0, 2.0), &budget)
        .unwrap();
    assert!(high.width() >= low.width() * 2 - 2);
    assert!(high.height() >= low.height() * 2 - 2);
    assert!(low.bearing_y() < 0);
    assert_eq!(
        low.coverage().len(),
        low.width() as usize * low.height() as usize
    );
    assert!(
        low.coverage()
            .iter()
            .any(|value| *value > 0 && *value < 255),
        "real grayscale edge coverage"
    );
}

#[test]
fn raster_em_scale_and_coverage_match_independent_ab_glyph_reference() {
    let face = font();
    let units = face.font.units_per_em().unwrap();
    let height = face.font.height_unscaled();
    assert_ne!(
        units, height,
        "the fixture must expose em versus font-height confusion"
    );
    assert_raster_reference(&face, &['A', 'j', 'é', 'Ж', 'д']);
}

#[test]
fn opentype_cff_cubic_outlines_shape_and_rasterize_at_logical_em_scale() {
    let bytes = include_bytes!("../../../examples/assets/fonts/Inter-Regular.otf");
    assert_eq!(&bytes[..4], b"OTTO");
    let parsed = rustybuzz::ttf_parser::Face::parse(bytes, 0).unwrap();
    assert!(parsed.tables().cff.is_some());
    assert!(parsed.tables().glyf.is_none());
    let face = FontFace::from_bytes(bytes.to_vec(), FontBudget::default()).unwrap();
    let outline = face.font.outline(face.font.glyph_id('S')).unwrap();
    assert!(
        outline
            .curves
            .iter()
            .any(|curve| matches!(curve, ab_glyph::OutlineCurve::Cubic(..))),
        "the OTF fixture must exercise cubic rather than only quadratic/line contours"
    );
    let style = style(24.0, 1.25);
    let budget = TextLayoutBudget::default();
    let line = face.shape_line("Привет", &style, &budget).unwrap();
    assert_eq!(line.glyphs().len(), 6);
    assert!(line.advance() > 0.0);
    assert!(line.glyphs().iter().all(|glyph| glyph.glyph_id() != 0));
    let space = face.shape_line(" ", &style, &budget).unwrap();
    assert!(space.advance() > 0.0);
    assert!(
        face.rasterize_glyph(space.glyphs()[0].glyph_id(), &style, &budget)
            .unwrap()
            .coverage()
            .is_empty()
    );
    assert_raster_reference(&face, &['S', 'g', 'j', '@', 'Ж']);
}

fn assert_raster_reference(face: &FontFace, characters: &[char]) {
    let units = face.font.units_per_em().unwrap();
    let height = face.font.height_unscaled();
    for &text in characters {
        for (size, scale) in [(16.0, 1.0), (24.0, 1.25), (48.0, 2.0)] {
            let style = style(size, scale);
            let glyph_id = face.font.glyph_id(text);
            let actual = face
                .rasterize_glyph(glyph_id.0, &style, &TextLayoutBudget::default())
                .unwrap();
            let glyph = glyph_id.with_scale(size * scale * height / units);
            let reference = face.font.outline_glyph(glyph).unwrap();
            let bounds = reference.px_bounds();
            assert_eq!(
                (actual.bearing_x(), actual.bearing_y()),
                (bounds.min.x as i32, bounds.min.y as i32)
            );
            assert_eq!(
                (actual.width(), actual.height()),
                (bounds.width() as u32, bounds.height() as u32)
            );
            let mut expected = vec![0; actual.coverage().len()];
            reference.draw(|x, y, value| {
                expected[y as usize * actual.width() as usize + x as usize] =
                    (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            });
            assert_eq!(
                actual.coverage(),
                expected,
                "{text:?} at em={size}, dpi={scale}"
            );
        }
    }
}

#[test]
fn raster_limits_are_checked_before_curve_or_bitmap_allocation() {
    let face = font();
    let style = style(24.0, 1.0);
    let glyph = face.font.glyph_id('W').0;
    let required_segments =
        match face.rasterize_glyph(glyph, &style, &TextLayoutBudget::new(10, 10, 1000000, 0)) {
            Err(FontError::BudgetExceeded {
                resource: FontBudgetResource::OutlineSegments,
                required,
                ..
            }) => required,
            result => panic!("expected preflighted outline limit, got {result:?}"),
        };
    assert!(required_segments > 0);
    assert!(
        face.rasterize_glyph(
            glyph,
            &style,
            &TextLayoutBudget::new(10, 10, 1000000, required_segments)
        )
        .is_ok()
    );
    assert!(matches!(
        face.rasterize_glyph(
            glyph,
            &style,
            &TextLayoutBudget::new(10, 10, 1000000, required_segments - 1)
        ),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::OutlineSegments,
            ..
        })
    ));
    let actual = face
        .rasterize_glyph(glyph, &style, &TextLayoutBudget::default())
        .unwrap();
    let pixels = actual.coverage().len();
    assert!(
        face.rasterize_glyph(glyph, &style, &TextLayoutBudget::new(10, 10, pixels, 16384))
            .is_ok()
    );
    assert!(matches!(
        face.rasterize_glyph(
            glyph,
            &style,
            &TextLayoutBudget::new(10, 10, pixels - 1, 16384)
        ),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::GlyphPixels,
            ..
        })
    ));
    assert_eq!(
        face.rasterize_glyph(u16::MAX, &style, &TextLayoutBudget::default())
            .unwrap_err(),
        FontError::InvalidGlyph { glyph_id: u16::MAX }
    );
    assert!(
        face.rasterize_glyph(
            glyph,
            &super::tests::style(1e30, 1.0),
            &TextLayoutBudget::default()
        )
        .is_err()
    );
}

#[test]
fn explicit_rtl_is_a_directional_run_not_paragraph_layout() {
    let face = font();
    let style = style(24.0, 1.0).with_direction(TextDirection::Rtl);
    let line = face
        .shape_line("אבג", &style, &TextLayoutBudget::default())
        .unwrap();
    assert_eq!(
        line.glyphs()
            .iter()
            .map(|glyph| glyph.cluster())
            .collect::<Vec<_>>(),
        vec![4, 2, 0]
    );
    assert!(line.advance() > 0.0);
    assert!(
        line.glyphs()
            .windows(2)
            .all(|pair| pair[1].logical_x() >= pair[0].logical_x())
    );
}

#[test]
fn combining_accents_preserve_clusters_and_canonical_layout() {
    let face = font();
    let style = style(32.0, 1.25);
    let budget = TextLayoutBudget::default();
    for (composed, decomposed) in [("Café", "Cafe\u{301}"), ("Å", "A\u{30a}")] {
        let composed = face.shape_line(composed, &style, &budget).unwrap();
        let decomposed = face.shape_line(decomposed, &style, &budget).unwrap();
        assert_eq!(composed.advance(), decomposed.advance());
        assert_eq!(composed.glyphs(), decomposed.glyphs());
    }
    // No precomposed q-with-acute exists: this exercises a separate positioned
    // mark rather than only the shaper's canonical-composition fast path.
    let plain = face.shape_line("q", &style, &budget).unwrap();
    let accented = face.shape_line("q\u{301}", &style, &budget).unwrap();
    assert_eq!(accented.glyphs().len(), 2);
    assert_eq!(accented.advance(), plain.advance());
    assert!(accented.glyphs().iter().all(|glyph| glyph.cluster() == 0));
    let ink: Vec<_> = accented
        .glyphs()
        .iter()
        .map(|glyph| {
            assert!(glyph.logical_x().is_finite() && glyph.logical_y().is_finite());
            let raster = face
                .rasterize_glyph(glyph.glyph_id(), &style, &budget)
                .unwrap();
            assert!(raster.coverage().iter().any(|alpha| *alpha != 0));
            let left = glyph.logical_x() + raster.bearing_x() as f32 / style.scale().get();
            let top = glyph.logical_y() + raster.bearing_y() as f32 / style.scale().get();
            let right = left + raster.width() as f32 / style.scale().get();
            (left, top, right)
        })
        .collect();
    // A mark's bearing already includes vertical offset; the baseline offset's
    // sign alone does not prove whether it is visually above the base letter.
    assert!(ink[1].1 < ink[0].1);
    assert!(ink[1].0 < ink[0].2 && ink[1].2 > ink[0].0);
}

#[test]
fn mathematical_supplementary_characters_preserve_unicode_and_utf8_clusters() {
    const TEXT: &str = "𝓣𝔂𝓹𝓮 𝓼𝓸𝓶𝓮𝓽𝓱𝓲𝓷𝓰 𝓽𝓸 𝓼𝓽𝓪𝓻𝓽";
    let bytes = include_bytes!("../../../examples/assets/fonts/DejaVuMathTeXGyre.ttf");
    let face = FontFace::from_bytes(bytes.to_vec(), FontBudget::default()).unwrap();
    let style = style(32.0, 1.25);
    let budget = TextLayoutBudget::default();
    assert!(
        TEXT.chars()
            .filter(|character| *character != ' ')
            .all(|character| u32::from(character) > 65535)
    );
    let shaped = face.shape_line(TEXT, &style, &budget).unwrap();
    assert_eq!(shaped.glyphs().len(), TEXT.chars().count());
    assert!(shaped.advance() > 0.0);
    for (glyph, (byte_index, character)) in shaped.glyphs().iter().zip(TEXT.char_indices()) {
        assert_eq!(glyph.cluster(), byte_index);
        assert!(TEXT.is_char_boundary(glyph.cluster()));
        assert_eq!(glyph.glyph_id(), face.font.glyph_id(character).0);
        assert_ne!(glyph.glyph_id(), 0);
        let raster = face
            .rasterize_glyph(glyph.glyph_id(), &style, &budget)
            .unwrap();
        assert_eq!(
            raster.coverage().iter().any(|alpha| *alpha != 0),
            character != ' '
        );
    }
    assert_ne!(shaped.glyphs()[0].glyph_id(), face.font.glyph_id('T').0);
    assert!(matches!(
        face.shape_line(TEXT, &style, &TextLayoutBudget::new(TEXT.len() - 1, 100, 1000000, 16384)),
        Err(FontError::BudgetExceeded {
            resource: FontBudgetResource::TextBytes,
            required,
            ..
        }) if required == TEXT.len()
    ));
    // No fallback is selected, even if another loaded font covers this glyph.
    // The missing cluster begins after one four-byte supplementary character.
    assert_eq!(face.font.glyph_id('日').0, 0);
    assert_eq!(
        face.shape_line("𝓣日", &style, &budget).unwrap_err(),
        FontError::MissingGlyph { byte_index: 4 }
    );
}

#[test]
fn japanese_subset_shapes_han_hiragana_and_katakana_without_fallback() {
    let bytes = include_bytes!("../../../examples/assets/fonts/SimEngineJapaneseSubset.otf");
    let face = FontFace::from_bytes(bytes.to_vec(), FontBudget::default()).unwrap();
    let budget = TextLayoutBudget::default();
    let low_style = style(28.0, 1.0);
    let high_style = style(28.0, 1.5);
    for text in [
        "こんにちは、せかい！",
        "カタカナ・テキスト",
        "日本語 漢字 東京 世界",
    ] {
        let low = face.shape_line(text, &low_style, &budget).unwrap();
        let high = face.shape_line(text, &high_style, &budget).unwrap();
        assert_eq!(low.glyphs().len(), text.chars().count());
        assert_eq!(low.glyphs(), high.glyphs());
        assert_eq!(low.advance(), high.advance());
        assert!(low.advance() > 0.0);
        for (glyph, (byte_index, character)) in high.glyphs().iter().zip(text.char_indices()) {
            assert_eq!(glyph.cluster(), byte_index);
            assert!(text.is_char_boundary(glyph.cluster()));
            assert_ne!(glyph.glyph_id(), 0);
            let raster = face
                .rasterize_glyph(glyph.glyph_id(), &high_style, &budget)
                .unwrap();
            assert_eq!(
                raster.coverage().iter().any(|alpha| *alpha != 0),
                character != ' '
            );
        }
    }
    // Voiced kana exercise canonical composition/mark shaping, not only Han
    // cmap lookup. Both spellings must produce the same visible layout.
    let composed = face.shape_line("が", &high_style, &budget).unwrap();
    let decomposed = face.shape_line("か\u{3099}", &high_style, &budget).unwrap();
    assert_eq!(composed.glyphs(), decomposed.glyphs());
    assert_eq!(composed.advance(), decomposed.advance());
    assert_eq!(font().font.glyph_id('日').0, 0);
    assert_eq!(
        font().shape_line("日", &high_style, &budget).unwrap_err(),
        FontError::MissingGlyph { byte_index: 0 }
    );
    assert_eq!(face.font.glyph_id('𝓣').0, 0);
    assert_eq!(
        face.shape_line("日𝓣", &high_style, &budget).unwrap_err(),
        FontError::MissingGlyph { byte_index: 3 }
    );
}
