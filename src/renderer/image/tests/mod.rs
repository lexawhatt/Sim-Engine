use super::*;
use crate::renderer::*;

mod retained_updates;
pub(in crate::renderer) use retained_updates::verify_retained_ui_updates;

#[test]
fn image_shape_checks_budget_device_and_exact_pixels_before_allocation() {
    let budget = ImageBudget::new(8, 8, 8 * 8 * 4).unwrap();
    assert_eq!(validate_image_shape(8, 8, 256, budget, 8), Ok(256));
    assert_eq!(
        validate_image_shape(8, 8, 255, budget, 8),
        Err(ImageError::InvalidPixelCount)
    );
    assert_eq!(
        validate_image_shape(9, 1, 36, budget, 16),
        Err(ImageError::DimensionsTooLarge)
    );
    let strict = ImageBudget::new(8, 8, 64).unwrap();
    assert_eq!(
        validate_image_shape(8, 8, 256, strict, 8),
        Err(ImageError::BudgetExceeded {
            limit: 64,
            actual: 256,
        })
    );
}

#[test]
fn atlas_rect_rejects_empty_and_overflowing_ranges() {
    assert_eq!(
        ImageTexelRect::new(0, 0, 0, 1),
        Err(ImageError::ZeroDimension)
    );
    assert_eq!(
        ImageTexelRect::new(u32::MAX, 0, 2, 1),
        Err(ImageError::DimensionsTooLarge)
    );
    assert!(ImageTexelRect::new(1, 2, 3, 4).unwrap().fits(4, 6));
}

#[test]
fn retained_sprite_region_must_fit_portable_shader_sources() {
    let subnormal = LogicalViewportRegion::new(
        LogicalScreenPosition::new(f32::from_bits(1), 0.0),
        LogicalViewport::new(1.0, 1.0).unwrap(),
    )
    .unwrap();
    let overflowing = LogicalViewportRegion::new(
        LogicalScreenPosition::new(2.0e38, 0.0),
        LogicalViewport::new(2.0e38, 1.0).unwrap(),
    )
    .unwrap();
    assert!(!logical_image_region_is_portable(subnormal));
    assert!(!logical_image_region_is_portable(overflowing));
}

#[test]
fn sprite_rejects_finite_inputs_whose_logical_extent_overflows() {
    let source = ImageTexelRect::new(0, 0, 1, 1).unwrap();
    let destination = LogicalViewportRegion::new(
        LogicalScreenPosition::new(f32::MAX, 0.0),
        LogicalViewport::new(f32::MAX, 1.0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        ImageSprite2d::new(source, destination, Color::WHITE),
        Err(ImageError::InvalidSprite)
    );
}

#[test]
fn texture_and_batch_upload_reports_have_distinct_resource_semantics() {
    let replacement = ImageUploadReport {
        uploaded_bytes: 64,
        replaced_texture: true,
    };
    let region = ImageUploadReport {
        uploaded_bytes: 16,
        replaced_texture: false,
    };
    let batch = ImageBatchUploadReport {
        uploaded_instance_bytes: 96,
        replaced_instance_buffer: true,
        sprite_count: 2,
        retained_capacity: 2,
        gpu_capacity: 2,
        retained_bytes: batch_retained_bytes(2),
        peak_retained_bytes: batch_retained_bytes(2),
        peak_gpu_bytes: 96,
    };

    assert!(replacement.replaced_texture());
    assert!(!region.replaced_texture());
    assert_eq!(replacement.uploaded_bytes(), 64);
    assert_eq!(region.uploaded_bytes(), 16);
    assert!(batch.replaced_instance_buffer());
    assert_eq!(batch.uploaded_instance_bytes(), 96);
}
