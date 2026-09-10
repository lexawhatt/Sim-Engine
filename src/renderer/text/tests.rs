use super::*;

#[test]
fn shelf_packing_is_non_overlapping_and_failed_insert_preserves_cursor() {
    let mut shelf = Shelf::default();
    let first = shelf.insert(6, 8, (16, 16)).unwrap();
    let second = shelf.insert(9, 5, (16, 16)).unwrap();
    let third = shelf.insert(5, 7, (16, 16)).unwrap();
    assert_eq!((first.x(), first.y()), (0, 0));
    assert_eq!((second.x(), second.y()), (6, 0));
    assert_eq!((third.x(), third.y()), (0, 8));
    let before = (shelf.x, shelf.y, shelf.height);
    assert!(matches!(
        shelf.insert(12, 9, (16, 16)),
        Err(TextError::AtlasFull)
    ));
    assert_eq!((shelf.x, shelf.y, shelf.height), before);
    let last = shelf.insert(11, 8, (16, 16)).unwrap();
    assert_eq!((last.x(), last.y()), (5, 8));
    assert!(matches!(
        shelf.insert(1, 1, (16, 16)),
        Err(TextError::AtlasFull)
    ));
}

#[test]
fn text_atlas_budget_checks_count_and_padding_dimensions() {
    for (width, height, count) in [(4, 20, 1), (20, 4, 1), (20, 20, 0), (20, 20, 65536)] {
        assert!(matches!(
            TextAtlasBudget::new(width, height, count, GlyphRunBudget::default()),
            Err(TextError::InvalidBudget)
        ));
    }
    let budget = TextAtlasBudget::default();
    assert_eq!(budget.image_budget().unwrap().max_bytes(), 4 * 1024 * 1024);
    let small = GlyphRunBudget::new(1000, GlyphRunBudget::RETAINED_BYTES_PER_GLYPH).unwrap();
    assert!(matches!(
        glyph::validate_glyph_run_budget(2, small),
        Err(GlyphError::MetadataBudgetExceeded { .. })
    ));
}

#[test]
fn font_bitmap_padding_is_white_transparent_and_geometry_preserves_texel_scale() {
    let font = FontFace::from_bytes(
        include_bytes!("../../../examples/assets/fonts/DejaVuSans.ttf").to_vec(),
        crate::FontBudget::default(),
    )
    .unwrap();
    for scale in [1.0, 1.25, 2.0] {
        let style = TextStyle::new(
            crate::LogicalPixels::new(24.0).unwrap(),
            crate::PhysicalPerLogical::new(scale).unwrap(),
        )
        .unwrap();
        let shaped = font
            .shape_line("j", &style, &TextLayoutBudget::default())
            .unwrap();
        let raster = font
            .rasterize_glyph(
                shaped.glyphs()[0].glyph_id(),
                &style,
                &TextLayoutBudget::default(),
            )
            .unwrap();
        let pixels = padded_rgba(&raster).unwrap();
        let width = raster.width() as usize + 4;
        let height = raster.height() as usize + 4;
        assert_eq!(pixels.len(), width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let pixel = &pixels[(y * width + x) * 4..][..4];
                assert_eq!(&pixel[..3], &[255; 3]);
                let expected = if x < 2 || y < 2 || x >= width - 2 || y >= height - 2 {
                    0
                } else {
                    raster.coverage()[(y - 2) * (width - 4) + x - 2]
                };
                assert_eq!(pixel[3], expected);
            }
        }
        let destination = glyph_destination(&raster, 0.0, 0.0, scale).unwrap();
        assert!(
            (destination.viewport().width() * scale - (raster.width() + 3) as f32).abs() < 1.0e-5
        );
        assert!(
            (destination.origin().to_vec2().x * scale - (raster.bearing_x() as f32 - 1.5)).abs()
                < 1.0e-5
        );
    }
}
