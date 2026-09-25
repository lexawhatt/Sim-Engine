use super::*;
use crate::{LogicalScreenVector, LogicalViewport};

mod gpu;
mod update_diagnostic;
pub(in crate::renderer) use gpu::verify_glyph_batches;

#[test]
fn budget_requires_space_for_a_glyph_and_clip_span() {
    let minimum =
        GlyphBatchBudget::RETAINED_BYTES_PER_GLYPH + GlyphBatchBudget::RETAINED_BYTES_PER_PLACEMENT;
    assert_eq!(
        GlyphBatchBudget::new(1, 1, minimum - 1),
        Err(GlyphError::InvalidBudget)
    );
    assert!(GlyphBatchBudget::new(1, 1, minimum).is_ok());
    assert_eq!(
        GlyphBatchBudget::new(0, 1, minimum),
        Err(GlyphError::InvalidBudget)
    );
    assert_eq!(
        GlyphBatchBudget::new(1, 0, minimum),
        Err(GlyphError::InvalidBudget)
    );
    assert!(image_budget(GlyphBatchBudget::new(1, 1, minimum).unwrap(), 2).is_err());
}

#[test]
fn placement_preserves_uv_size_and_multiplies_tint() {
    let source = super::super::ImageTexelRect::new(2, 3, 8, 7).unwrap();
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(1.25, -3.5),
        LogicalViewport::new(8.0, 7.0).unwrap(),
    )
    .unwrap();
    let sprite = ImageSprite2d::new(source, region, Color::rgba(0.5, 1.0, 0.25, 0.5)).unwrap();
    let placed = placed_sprite(
        sprite,
        ImageBatchPlacement::new(
            LogicalScreenVector::new(-2.5, 9.0),
            Color::rgba(0.5, 0.25, 1.0, 0.5),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(placed.source(), source);
    assert_eq!(
        placed.destination().origin(),
        LogicalScreenPosition::new(-1.25, 5.5)
    );
    assert_eq!(placed.destination().viewport(), region.viewport());
    assert_eq!(placed.tint(), Color::rgba(0.25, 0.25, 0.25, 0.25));
}

#[test]
fn translated_overflow_is_rejected_before_publication() {
    let source = super::super::ImageTexelRect::new(0, 0, 1, 1).unwrap();
    let largest = crate::renderer::MAX_PORTABLE_SHADER_VALUE;
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(largest, 0.0),
        LogicalViewport::new(1.0, 1.0).unwrap(),
    )
    .unwrap();
    let sprite = ImageSprite2d::new(source, region, Color::WHITE).unwrap();
    let placement =
        ImageBatchPlacement::new(LogicalScreenVector::new(largest, 0.0), Color::WHITE).unwrap();
    assert_eq!(
        placed_sprite(sprite, placement),
        Err(GlyphError::InvalidGlyph)
    );
}
