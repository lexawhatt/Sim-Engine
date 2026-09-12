use super::*;

const FONT: &[u8] = include_bytes!("../../../examples/assets/fonts/DejaVuSans.ttf");

fn font() -> FontFace {
    FontFace::from_bytes(FONT.to_vec(), FontBudget::default()).unwrap()
}

fn style(direction: TextDirection) -> TextStyle {
    TextStyle::new(
        crate::LogicalPixels::new(24.0).unwrap(),
        crate::PhysicalPerLogical::new(1.25).unwrap(),
    )
    .unwrap()
    .with_direction(direction)
}

fn assert_same(left: &ShapedLine, right: &ShapedLine) {
    assert_eq!(left.text(), right.text());
    assert_eq!(left.style(), right.style());
    assert_eq!(left.glyphs(), right.glyphs());
    assert_eq!(left.advance(), right.advance());
    assert_eq!(left.ascent(), right.ascent());
    assert_eq!(left.descent(), right.descent());
    assert_eq!(left.line_height(), right.line_height());
}

#[test]
fn session_matches_fresh_shaping_across_scripts_directions_and_changing_lengths() {
    let font = font();
    let budget = TextLayoutBudget::new(256, 256, 4096, 4096);
    for direction in [TextDirection::Auto, TextDirection::Ltr, TextDirection::Rtl] {
        let style = style(direction);
        let mut session = TextShapingSession::new(&font, style, budget).unwrap();
        let mut line = session.shape_line("").unwrap();
        for _ in 0..3 {
            for text in [
                "AV ffi office",
                "Привет, мир",
                "Cafe\u{301} q\u{301}",
                "مرحبا بالعالم",
                "AV",
                "   ",
                "",
            ] {
                let fresh = font.shape_line(text, &style, &budget).unwrap();
                session.update_line(&mut line, text).unwrap();
                assert_same(&line, &fresh);
                assert_eq!(session.cached_plan_count(), 1);
                assert!(
                    session.allocation_bytes() <= 256 * (1 + std::mem::size_of::<ShapedGlyph>())
                );

                // Independently compare the reused-plan path with rustybuzz's
                // ordinary fresh-plan path, including auto-script transitions.
                let face = rustybuzz::Face::from_slice(font.font_data(), 0).unwrap();
                let mut input = rustybuzz::UnicodeBuffer::new();
                input.push_str(text);
                match direction {
                    TextDirection::Auto => {}
                    TextDirection::Ltr => input.set_direction(rustybuzz::Direction::LeftToRight),
                    TextDirection::Rtl => input.set_direction(rustybuzz::Direction::RightToLeft),
                }
                let expected = rustybuzz::shape(&face, &[], input);
                assert_eq!(line.glyphs().len(), expected.len());
                let factor =
                    f64::from(style.logical_em_size().get()) / f64::from(face.units_per_em());
                let mut pen_x = 0.0f64;
                let mut pen_y = 0.0f64;
                for ((actual, info), position) in line
                    .glyphs()
                    .iter()
                    .zip(expected.glyph_infos())
                    .zip(expected.glyph_positions())
                {
                    assert_eq!(u32::from(actual.glyph_id()), info.glyph_id);
                    assert_eq!(actual.cluster(), info.cluster as usize);
                    assert_eq!(
                        actual.logical_x(),
                        ((pen_x + f64::from(position.x_offset)) * factor) as f32
                    );
                    assert_eq!(
                        actual.logical_y(),
                        (-(pen_y + f64::from(position.y_offset)) * factor) as f32
                    );
                    pen_x += f64::from(position.x_advance);
                    pen_y += f64::from(position.y_advance);
                }
                assert_eq!(line.advance(), (pen_x * factor) as f32);
            }
        }
        session.clear_scratch();
        assert_eq!(session.allocation_bytes(), 0);
        assert_eq!(session.cached_plan_count(), 0);
        session.update_line(&mut line, "After clear").unwrap();
    }
}

#[test]
fn shaped_provenance_checks_font_clones_dpi_em_direction_and_limits() {
    let font = font();
    let style = style(TextDirection::Auto);
    let budget = TextLayoutBudget::default();
    let line = font.shape_line("AV", &style, &budget).unwrap();
    assert_eq!(line.validate_for(&font.clone(), &style, &budget), Ok(()));
    assert_eq!(
        line.validate_for(&super::session_tests::font(), &style, &budget),
        Err(ShapedLineError::FontMismatch)
    );
    for changed in [
        style.with_direction(TextDirection::Ltr),
        TextStyle::new(crate::LogicalPixels::new(25.0).unwrap(), style.scale()).unwrap(),
        TextStyle::new(
            style.logical_em_size(),
            crate::PhysicalPerLogical::new(2.0).unwrap(),
        )
        .unwrap(),
    ] {
        assert_eq!(
            line.validate_for(&font, &changed, &budget),
            Err(ShapedLineError::StyleMismatch)
        );
    }
    assert!(matches!(
        line.validate_for(&font, &style, &TextLayoutBudget::new(1, 2, 0, 0)),
        Err(ShapedLineError::Font(FontError::BudgetExceeded {
            resource: FontBudgetResource::TextBytes,
            ..
        }))
    ));
    assert!(matches!(
        line.validate_for(&font, &style, &TextLayoutBudget::new(2, 1, 0, 0)),
        Err(ShapedLineError::Font(FontError::BudgetExceeded {
            resource: FontBudgetResource::ShapedGlyphs,
            ..
        }))
    ));
    assert_eq!(
        line.validate_for(&font, &style, &TextLayoutBudget::new(2, 2, 0, 0)),
        Ok(())
    );
    assert_eq!(
        line.allocation_bytes(),
        line.text.capacity() + line.glyphs.capacity() * std::mem::size_of::<ShapedGlyph>()
    );
}

#[test]
fn session_failed_updates_preserve_text_geometry_metrics_and_allocation() {
    let font = font();
    let style = style(TextDirection::Auto);
    let budget = TextLayoutBudget::new(12, 4, 0, 0);
    let mut session = TextShapingSession::new(&font, style, budget).unwrap();
    let mut line = session.shape_line("AV").unwrap();
    let original = font.shape_line("AV", &style, &budget).unwrap();
    let text_pointer = line.text().as_ptr();
    let glyph_pointer = line.glyphs().as_ptr();
    let bytes = line.allocation_bytes();
    for text in ["\u{10ffff}", "A\nB", "1234567890123", "ABCDE"] {
        assert!(session.update_line(&mut line, text).is_err());
        assert_same(&line, &original);
        assert_eq!(line.text().as_ptr(), text_pointer);
        assert_eq!(line.glyphs().as_ptr(), glyph_pointer);
        assert_eq!(line.allocation_bytes(), bytes);
    }
    assert!(session.update_line(&mut line, "VA").unwrap());
    let (unchanged, allocations) =
        crate::test_allocations::count(|| session.update_line(&mut line, "VA").unwrap());
    assert!(!unchanged);
    assert_eq!(allocations, 0);

    let other = super::session_tests::font();
    let mut foreign_session = TextShapingSession::new(&other, style, budget).unwrap();
    assert_eq!(
        foreign_session.update_line(&mut line, "VA"),
        Err(ShapedLineError::FontMismatch)
    );
    let mut wrong_style =
        TextShapingSession::new(&font, style.with_direction(TextDirection::Ltr), budget).unwrap();
    assert_eq!(
        wrong_style.update_line(&mut line, "VA"),
        Err(ShapedLineError::StyleMismatch)
    );
}

#[test]
fn session_rejects_oversized_imported_capacity_and_finite_position_overflow() {
    let font = font();
    let style = style(TextDirection::Auto);
    let mut line = font
        .shape_line("A", &style, &TextLayoutBudget::default())
        .unwrap();
    line.text.try_reserve_exact(32).unwrap();
    let mut session =
        TextShapingSession::new(&font, style, TextLayoutBudget::new(4, 4, 0, 0)).unwrap();
    assert!(matches!(
        session.update_line(&mut line, "B"),
        Err(ShapedLineError::Font(FontError::BudgetExceeded {
            resource: FontBudgetResource::TextBytes,
            ..
        }))
    ));
    assert_eq!(line.text(), "A");

    let huge = TextStyle::new(
        crate::LogicalPixels::new(f32::MAX / 2.0).unwrap(),
        crate::PhysicalPerLogical::new(1.0).unwrap(),
    )
    .unwrap();
    let mut session = TextShapingSession::new(&font, huge, TextLayoutBudget::default()).unwrap();
    let mut line = session.shape_line("A").unwrap();
    let advance = line.advance();
    assert_eq!(
        session.update_line(&mut line, "WWWWWWWW"),
        Err(ShapedLineError::Font(FontError::InvalidGeometry))
    );
    assert_eq!(line.text(), "A");
    assert_eq!(line.advance(), advance);
}

#[test]
fn warmed_sessions_measure_fewer_allocations_than_fresh_public_shaping() {
    let font = font();
    let budget = TextLayoutBudget::new(256, 256, 4096, 4096);
    for (name, strings, direction) in [
        (
            "latin32",
            [
                "0123456789abcdefghijklmnopqrstuv",
                "ABCDEFGHIJKLMNOPQRSTUV0123456789",
            ],
            TextDirection::Ltr,
        ),
        (
            "cyrillic",
            ["Привет, мир!", "Состояние: готово"],
            TextDirection::Auto,
        ),
        (
            "combining",
            ["Cafe\u{301} q\u{301}", "A\u{30a} q\u{301} e\u{301}"],
            TextDirection::Auto,
        ),
        (
            "ligatures",
            ["office ffi AV", "affinity AV ffi office"],
            TextDirection::Ltr,
        ),
        ("rtl", ["مرحبا بالعالم", "العالم"], TextDirection::Rtl),
    ] {
        let style = style(direction);
        let mut session = TextShapingSession::new(&font, style, budget).unwrap();
        let mut line = session.shape_line(strings[0]).unwrap();
        for index in 0..16 {
            session.update_line(&mut line, strings[index % 2]).unwrap();
        }
        let (_, fresh) = crate::test_allocations::count(|| {
            for index in 0..100 {
                std::hint::black_box(
                    font.shape_line(strings[index % 2], &style, &budget)
                        .unwrap(),
                );
            }
        });
        let (_, reused) = crate::test_allocations::count(|| {
            for index in 0..100 {
                std::hint::black_box(session.update_line(&mut line, strings[index % 2]).unwrap());
            }
        });
        eprintln!("text session allocations {name}: fresh={fresh}, reused={reused}");
        assert!(reused < fresh / 4, "{name}: reused={reused}, fresh={fresh}");
        assert_eq!(session.cached_plan_count(), 1);
        assert!(session.allocation_bytes() <= 256 * (1 + std::mem::size_of::<ShapedGlyph>()));
    }
}

#[test]
fn japanese_session_reuses_dakuten_scripts_and_recovers_after_missing_glyphs() {
    let font = FontFace::from_bytes(
        include_bytes!("../../../examples/assets/fonts/SimEngineJapaneseSubset.otf").to_vec(),
        FontBudget::default(),
    )
    .unwrap();
    let style = style(TextDirection::Auto);
    let budget = TextLayoutBudget::new(128, 64, 4096, 4096);
    let mut session = TextShapingSession::new(&font, style, budget).unwrap();
    let mut line = session.shape_line("日本語").unwrap();
    for _ in 0..4 {
        for text in [
            "こんにちは、せかい！",
            "カタカナ・テキスト",
            "が",
            "か\u{3099}",
            "A漢字",
            "東京 世界",
            "",
            "日本語",
        ] {
            session.update_line(&mut line, text).unwrap();
            assert_same(&line, &font.shape_line(text, &style, &budget).unwrap());
            assert_eq!(session.cached_plan_count(), 1);
            assert!(session.allocation_bytes() <= 128 + 64 * std::mem::size_of::<ShapedGlyph>());
        }
        let previous = font.shape_line(line.text(), &style, &budget).unwrap();
        assert_eq!(
            session.update_line(&mut line, "語\u{10ffff}"),
            Err(ShapedLineError::Font(FontError::MissingGlyph {
                byte_index: 3
            }))
        );
        assert_same(&line, &previous);
        // A failed shape must not leak stale text or inferred segment properties
        // into the next script, regardless of retained scratch capacity.
        session.update_line(&mut line, "カタカナ").unwrap();
        assert_same(
            &line,
            &font.shape_line("カタカナ", &style, &budget).unwrap(),
        );
        session.clear_scratch();
        assert_eq!(session.allocation_bytes(), 0);
        assert_eq!(session.cached_plan_count(), 0);
    }
}
