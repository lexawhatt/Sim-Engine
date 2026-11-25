use super::*;
use std::{collections::BTreeMap, ops::Range};

fn ranges(line: &ShapedLine) -> Vec<Range<usize>> {
    line.clusters()
        .map(|cluster| cluster.byte_range())
        .collect()
}

// The oracle uses raw font-unit advances from a fresh shaper invocation, indexed
// by logical source bytes. It never reads glyph origins or Engine pen metadata.
fn assert_pen_oracle(line: &ShapedLine) {
    let face = rustybuzz::Face::from_slice(line.font().font_data(), 0).unwrap();
    let mut input = rustybuzz::UnicodeBuffer::new();
    input.push_str(line.text());
    match line.style().direction() {
        TextDirection::Auto => {}
        TextDirection::Ltr => input.set_direction(rustybuzz::Direction::LeftToRight),
        TextDirection::Rtl => input.set_direction(rustybuzz::Direction::RightToLeft),
    }
    let output = rustybuzz::shape(&face, &[], input);
    let factor = f64::from(line.style().logical_em_size().get()) / f64::from(face.units_per_em());
    let mut pens = Vec::new();
    let mut pen = [0i64; 2];
    let mut logical_groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, (info, position)) in output
        .glyph_infos()
        .iter()
        .zip(output.glyph_positions())
        .enumerate()
    {
        let before = pen;
        pen[0] += i64::from(position.x_advance);
        pen[1] -= i64::from(position.y_advance);
        pens.push((before, pen));
        logical_groups
            .entry(info.cluster as usize)
            .or_default()
            .push(index);
    }
    let mut source_ends: Vec<_> = logical_groups.keys().copied().skip(1).collect();
    source_ends.push(line.text().len());
    let expected_ranges: Vec<_> = logical_groups
        .keys()
        .copied()
        .zip(source_ends)
        .map(|(a, b)| a..b)
        .collect();
    let mut actual_ranges = ranges(line);
    actual_ranges.sort_by_key(|range| range.start);
    assert_eq!(actual_ranges, expected_ranges);
    for (visual_index, cluster) in line.clusters().enumerate() {
        let indices = &logical_groups[&cluster.byte_range().start];
        let first = *indices.first().unwrap();
        let last = *indices.last().unwrap();
        assert_eq!(cluster.visual_index(), visual_index);
        assert_eq!(cluster.glyph_range(), first..last + 1);
        assert!(std::ptr::eq(cluster.line(), line));
        let expected_start = [
            (pens[first].0[0] as f64 * factor) as f32,
            (pens[first].0[1] as f64 * factor) as f32,
        ];
        let expected_end = [
            (pens[last].1[0] as f64 * factor) as f32,
            (pens[last].1[1] as f64 * factor) as f32,
        ];
        assert_eq!(
            [
                cluster.pen_start().logical_x(),
                cluster.pen_start().logical_y()
            ],
            expected_start
        );
        assert_eq!(
            [cluster.pen_end().logical_x(), cluster.pen_end().logical_y()],
            expected_end
        );
        for caret in [cluster.start_caret(), cluster.end_caret()] {
            assert!(std::ptr::eq(caret.line(), line));
            assert!(line.text().is_char_boundary(caret.byte_offset()));
            assert_eq!(
                line.caret_at(caret.byte_offset()).unwrap().position(),
                caret.position()
            );
        }
        assert_eq!(
            cluster.advance(),
            [
                f64::from(expected_end[0]) - f64::from(expected_start[0]),
                f64::from(expected_end[1]) - f64::from(expected_start[1])
            ]
        );
    }
}

#[test]
fn cluster_ranges_and_pens_match_font_advances_across_scripts() {
    let face = font();
    let budget = TextLayoutBudget::default();
    for text in [
        "Привет",
        "AV To",
        "office ffi",
        "q\u{301}X",
        "и\u{306}а",
        "אבג",
        "مرحبا",
        "a\u{200d}b",
        "\u{200b}",
        "   ",
    ] {
        for direction in [TextDirection::Auto, TextDirection::Ltr, TextDirection::Rtl] {
            let line = face
                .shape_line(text, &style(24.0, 1.25).with_direction(direction), &budget)
                .unwrap();
            assert_pen_oracle(&line);
            assert_eq!(line.caret_policy(), LigatureCaretPolicy::ClusterBoundaries);
        }
    }
    let line = face
        .shape_line("Привет", &style(24.0, 1.0), &budget)
        .unwrap();
    assert_eq!(ranges(&line), vec![0..2, 2..4, 4..6, 6..8, 8..10, 10..12]);
    assert_eq!(
        line.selection(2..8)
            .unwrap()
            .map(|c| c.byte_range())
            .collect::<Vec<_>>(),
        vec![2..4, 4..6, 6..8]
    );
    assert_eq!(
        line.caret_at(1).unwrap_err(),
        ShapedGeometryError::InvalidByteOffset { byte_offset: 1 }
    );
}

#[test]
fn kerning_moves_the_cluster_boundary_and_spaces_keep_selection_advance() {
    let face = font();
    let style = style(32.0, 1.0);
    let budget = TextLayoutBudget::default();
    let a = face.shape_line("A", &style, &budget).unwrap();
    let av = face.shape_line("AV", &style, &budget).unwrap();
    assert_eq!(ranges(&av), vec![0..1, 1..2]);
    let boundary = av.caret_at(1).unwrap().position().logical_x();
    assert!(
        boundary < a.advance(),
        "the caret must include pair kerning"
    );
    assert_eq!(av.caret_at(2).unwrap().position().logical_x(), av.advance());
    let spaces = face.shape_line("   ", &style, &budget).unwrap();
    assert!(
        spaces
            .selection(0..3)
            .unwrap()
            .all(|cluster| cluster.advance()[0] > 0.0)
    );
    assert_eq!(
        spaces.caret_at(3).unwrap().position().logical_x(),
        spaces.advance()
    );
}

#[test]
fn combining_offsets_are_not_carets_and_ligatures_have_no_invented_interior() {
    let face = font();
    let style = style(32.0, 1.0);
    let budget = TextLayoutBudget::default();
    let line = face.shape_line("q\u{301}X", &style, &budget).unwrap();
    assert_eq!(ranges(&line), vec![0..3, 3..4]);
    let first = line.clusters().next().unwrap();
    assert_eq!(first.glyph_range(), 0..2);
    let mark = line.glyphs()[1];
    assert_ne!(mark.logical_x(), mark.pen_start().logical_x());
    assert_eq!(mark.pen_start(), mark.pen_end());
    assert_eq!(first.start_caret().position().logical_x(), 0.0);
    assert_eq!(first.end_caret().position(), mark.pen_end());
    assert_eq!(
        line.caret_at(1).unwrap_err(),
        ShapedGeometryError::InsideCluster {
            byte_offset: 1,
            cluster_range: 0..3
        }
    );
    assert_eq!(
        line.caret_at(2).unwrap_err(),
        ShapedGeometryError::InvalidByteOffset { byte_offset: 2 }
    );
    assert_eq!(
        line.selection(0..1).unwrap_err(),
        line.caret_at(1).unwrap_err()
    );

    let ligature = face.shape_line("office", &style, &budget).unwrap();
    assert_eq!(ranges(&ligature), vec![0..1, 1..4, 4..5, 5..6]);
    let ffi = ligature.selection(1..4).unwrap().next().unwrap();
    assert_eq!(ffi.glyph_range().len(), 1);
    for byte_offset in [2, 3] {
        assert_eq!(
            ligature.caret_at(byte_offset).unwrap_err(),
            ShapedGeometryError::InsideCluster {
                byte_offset,
                cluster_range: 1..4
            }
        );
        assert!(ligature.selection(byte_offset..4).is_err());
    }
    assert!(ligature.selection(1..1).unwrap().next().is_none());
    assert!(ligature.selection(2..2).is_err());
}

#[test]
fn rtl_preserves_logical_ranges_visual_order_and_reversed_endpoint_positions() {
    let face = font();
    let line = face
        .shape_line("אבג", &style(24.0, 1.0), &TextLayoutBudget::default())
        .unwrap();
    assert_eq!(line.resolved_direction(), TextDirection::Rtl);
    assert_eq!(line.style().direction(), TextDirection::Auto);
    assert_eq!(ranges(&line), vec![4..6, 2..4, 0..2]);
    assert_eq!(
        line.caret_at(0).unwrap().position().logical_x(),
        line.advance()
    );
    assert_eq!(line.caret_at(6).unwrap().position().logical_x(), 0.0);
    let selected: Vec<_> = line.selection(0..4).unwrap().collect();
    assert_eq!(
        selected.iter().map(|c| c.byte_range()).collect::<Vec<_>>(),
        vec![2..4, 0..2]
    );
    assert_eq!(
        selected
            .iter()
            .map(|c| c.visual_index())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    for cluster in selected {
        assert!(
            cluster.start_caret().position().logical_x()
                > cluster.end_caret().position().logical_x()
        );
    }
    let reversed = Range { start: 4, end: 0 };
    assert_eq!(
        line.selection(reversed).unwrap_err(),
        ShapedGeometryError::ReversedRange
    );
}

#[test]
fn empty_lines_and_invalid_offsets_have_explicit_geometry() {
    let face = font();
    for direction in [TextDirection::Auto, TextDirection::Ltr, TextDirection::Rtl] {
        let line = face
            .shape_line(
                "",
                &style(24.0, 1.0).with_direction(direction),
                &TextLayoutBudget::new(0, 0, 0, 0),
            )
            .unwrap();
        assert_eq!(line.clusters().count(), 0);
        assert_eq!(line.selection(0..0).unwrap().count(), 0);
        assert_eq!(line.caret_at(0).unwrap().position().logical_x(), 0.0);
        for byte_offset in [1, usize::MAX] {
            assert_eq!(
                line.caret_at(byte_offset).unwrap_err(),
                ShapedGeometryError::InvalidByteOffset { byte_offset }
            );
        }
    }
}

#[test]
fn geometry_queries_allocate_nothing_and_transactional_reuse_preserves_provenance() {
    let face = font();
    let style = style(24.0, 1.0);
    let budget = TextLayoutBudget::new(64, 32, 0, 0);
    let mut session = TextShapingSession::new(&face, style, budget).unwrap();
    let mut line = session.shape_line("AV office q\u{301}").unwrap();
    let (_, allocations) = crate::test_allocations::count(|| {
        for _ in 0..32 {
            for cluster in line.clusters() {
                std::hint::black_box(cluster.advance());
                std::hint::black_box(line.caret_at(cluster.byte_range().start).unwrap());
            }
            std::hint::black_box(line.selection(0..line.text().len()).unwrap().count());
            std::hint::black_box(line.caret_at(5).unwrap_err());
        }
    });
    assert_eq!(allocations, 0);
    let before = ranges(&line);
    let bytes = line.allocation_bytes();
    let end = line.caret_at(line.text().len()).unwrap().position();
    for text in ["a\nb", "\u{10ffff}", "abcdefghijklmnopqrstuvwxyzABCDEFG"] {
        assert!(session.update_line(&mut line, text).is_err());
        assert_eq!(ranges(&line), before);
        assert_eq!(line.caret_at(line.text().len()).unwrap().position(), end);
        assert_eq!(line.allocation_bytes(), bytes);
        assert_eq!(line.resolved_direction(), TextDirection::Ltr);
    }
    let other = font();
    let mut wrong = TextShapingSession::new(&other, style, budget).unwrap();
    assert_eq!(
        wrong.update_line(&mut line, "AV"),
        Err(ShapedLineError::FontMismatch)
    );
    session.update_line(&mut line, "אבג").unwrap();
    assert_eq!(line.resolved_direction(), TextDirection::Rtl);
    assert_pen_oracle(&line);
    session.clear_scratch();
    assert_eq!(session.allocation_bytes(), 0);
    assert_pen_oracle(&line);
}

#[test]
fn deleted_default_ignorables_keep_zero_output_text_editable() {
    // A sparse, valid cmap maps only 'a'. No U+0020 exists, so the shaper must
    // delete default ignorables instead of substituting an invisible space.
    // All outline/metric tables remain those of the independently bundled font.
    let mut bytes = FONT.to_vec();
    let glyph = font().font.glyph_id('a').0;
    let table_count = u16::from_be_bytes([bytes[4], bytes[5]]) as usize;
    let entry = (0..table_count)
        .map(|index| 12 + index * 16)
        .find(|&entry| &bytes[entry..entry + 4] == b"cmap")
        .unwrap();
    let offset = u32::from_be_bytes(bytes[entry + 8..entry + 12].try_into().unwrap()) as usize;
    let mut cmap = Vec::new();
    for word in [
        0u16,
        1,
        3,
        1,
        0,
        12,
        4,
        32,
        0,
        4,
        4,
        1,
        0,
        0x61,
        0xffff,
        0,
        0x61,
        0xffff,
        glyph.wrapping_sub(0x61),
        1,
        0,
        0,
    ] {
        cmap.extend_from_slice(&word.to_be_bytes());
    }
    assert_eq!(cmap.len(), 44);
    bytes[offset..offset + cmap.len()].copy_from_slice(&cmap);
    bytes[entry + 12..entry + 16].copy_from_slice(&(cmap.len() as u32).to_be_bytes());
    let checksum = cmap.chunks_exact(4).fold(0u32, |sum, word| {
        sum.wrapping_add(u32::from_be_bytes(word.try_into().unwrap()))
    });
    bytes[entry + 4..entry + 8].copy_from_slice(&checksum.to_be_bytes());
    let head_entry = (0..table_count)
        .map(|index| 12 + index * 16)
        .find(|&entry| &bytes[entry..entry + 4] == b"head")
        .unwrap();
    let head_offset =
        u32::from_be_bytes(bytes[head_entry + 8..head_entry + 12].try_into().unwrap()) as usize;
    bytes[head_offset + 8..head_offset + 12].fill(0);
    let sum = bytes.chunks_exact(4).fold(0u32, |sum, word| {
        sum.wrapping_add(u32::from_be_bytes(word.try_into().unwrap()))
    });
    bytes[head_offset + 8..head_offset + 12]
        .copy_from_slice(&0xB1B0_AFBAu32.wrapping_sub(sum).to_be_bytes());
    let face = FontFace::from_bytes(bytes, FontBudget::default()).unwrap();
    let reference = rustybuzz::Face::from_slice(face.font_data(), 0).unwrap();
    assert!(reference.glyph_index(' ').is_none());
    let mut input = rustybuzz::UnicodeBuffer::new();
    input.push_str("\u{200b}\u{200b}");
    assert!(rustybuzz::shape(&reference, &[], input).is_empty());
    for direction in [TextDirection::Ltr, TextDirection::Rtl] {
        let mut session = TextShapingSession::new(
            &face,
            style(24.0, 1.0).with_direction(direction),
            TextLayoutBudget::default(),
        )
        .unwrap();
        let mut line = session.shape_line("a").unwrap();
        session.update_line(&mut line, "\u{200b}\u{200b}").unwrap();
        assert!(line.glyphs().is_empty());
        assert_eq!(line.advance(), 0.0);
        assert_eq!(ranges(&line), vec![0..6]);
        let cluster = line.clusters().next().unwrap();
        assert_eq!(cluster.glyph_range(), 0..0);
        assert_eq!(cluster.advance(), [0.0, 0.0]);
        assert_eq!(
            line.caret_at(0).unwrap().position(),
            line.caret_at(6).unwrap().position()
        );
        assert_eq!(
            line.caret_at(3).unwrap_err(),
            ShapedGeometryError::InsideCluster {
                byte_offset: 3,
                cluster_range: 0..6,
            }
        );
        assert_eq!(line.selection(0..6).unwrap().count(), 1);
        assert_eq!(line.selection(6..6).unwrap().count(), 0);
        let mut clusters = line.clusters();
        assert_eq!(clusters.size_hint(), (0, Some(1)));
        clusters.next().unwrap();
        assert!(clusters.next().is_none());
        assert_eq!(clusters.size_hint(), (0, Some(0)));
        session.update_line(&mut line, "a").unwrap();
        assert!(!line.glyphs().is_empty());
    }
}
