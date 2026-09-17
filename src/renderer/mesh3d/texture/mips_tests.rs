use super::*;

fn mip_chain(width: u32, height: u32, pixels: Vec<u8>) -> CpuMipChain {
    CpuMipChain::new(
        width,
        height,
        pixels,
        ImageBudget::default(),
        4096,
        TextureMipmaps3d::Generate,
    )
    .unwrap()
}

fn last_pixels(chain: &CpuMipChain) -> &[u8] {
    chain.level_pixels(*chain.levels().last().unwrap())
}

#[test]
fn mip_zero_default_keeps_source_and_allocation_without_generating_levels() {
    assert_eq!(TextureMipmaps3d::default(), TextureMipmaps3d::None);
    let pixels = vec![255, 13, 88, 0, 31, 41, 59, 26];
    let pointer = pixels.as_ptr();
    let chain = CpuMipChain::new(
        2,
        1,
        pixels,
        ImageBudget::new(2, 1, 8).unwrap(),
        2,
        TextureMipmaps3d::None,
    )
    .unwrap();
    assert_eq!(chain.pixels(), [255, 13, 88, 0, 31, 41, 59, 26]);
    assert_eq!(chain.pixels().as_ptr(), pointer);
    assert_eq!(chain.levels().len(), 1);
    assert_eq!(chain.gpu_allocation_bytes(), 8);
}

#[test]
fn mip_rgb_averaging_happens_in_linear_light() {
    let chain = mip_chain(2, 1, vec![0, 0, 0, 255, 255, 255, 255, 255]);
    assert_eq!(last_pixels(&chain), [188, 188, 188, 255]);
    assert_eq!(chain.size(), (2, 1));
    assert_eq!(chain.mode(), TextureMipmaps3d::Generate);
    assert_eq!(chain.budget(), ImageBudget::default());
}

#[test]
fn region_patch_regenerates_every_level_without_changing_untouched_base_texels() {
    let original = mip_chain(5, 3, [24, 48, 96, 255].repeat(15));
    let mut patched = original.try_clone().unwrap();
    let region = crate::ImageTexelRect::new(3, 1, 2, 2).unwrap();
    patched.patch_base(region, &[255, 0, 0, 128].repeat(4));
    patched.regenerate();
    let mut expected_base = original.pixels().to_vec();
    for y in 1..3 {
        for x in 3..5 {
            expected_base[(y * 5 + x) * 4..(y * 5 + x + 1) * 4].copy_from_slice(&[255, 0, 0, 128]);
        }
    }
    let rebuilt = mip_chain(5, 3, expected_base);
    for (actual, expected) in patched.levels().iter().zip(rebuilt.levels()) {
        assert_eq!(
            patched.level_pixels(*actual),
            rebuilt.level_pixels(*expected)
        );
    }
    assert_eq!(original.pixels(), [24, 48, 96, 255].repeat(15));
    let pointer = patched.pixels.as_ptr();
    let capacity = patched.recovery_memory_bytes();
    let (_, allocations) = crate::test_allocations::count(|| patched.regenerate());
    assert_eq!(allocations, 0);
    assert_eq!(patched.pixels.as_ptr(), pointer);
    assert_eq!(patched.recovery_memory_bytes(), capacity);
}

#[test]
fn transparent_color_does_not_bleed_into_filtered_mip() {
    let chain = mip_chain(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 0]);
    assert_eq!(last_pixels(&chain), [255, 0, 0, 128]);
    let partial = mip_chain(2, 1, vec![255, 0, 0, 128, 0, 255, 0, 64]);
    assert_eq!(last_pixels(&partial), [213, 156, 0, 96]);
}

#[test]
fn zero_alpha_generated_texels_have_zero_rgb_but_base_is_unchanged() {
    let source = vec![255, 30, 50, 0, 100, 200, 250, 0];
    let chain = mip_chain(2, 1, source.clone());
    assert_eq!(chain.pixels(), source);
    assert_eq!(last_pixels(&chain), [0, 0, 0, 0]);
    let tiny_alpha = mip_chain(
        2,
        2,
        vec![255, 0, 0, 1, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0],
    );
    assert_eq!(last_pixels(&tiny_alpha), [0, 0, 0, 0]);
}

#[test]
fn odd_source_extent_contributes_the_last_row_and_column() {
    let chain = mip_chain(3, 1, vec![0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255]);
    assert_eq!(last_pixels(&chain), [156, 156, 156, 255]);
    let mut source = vec![0; 3 * 5 * 4];
    for pixel in source.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    source[(4 * 3 + 2) * 4..][..3].fill(255);
    let chain = mip_chain(3, 5, source);
    assert_eq!(
        chain
            .levels()
            .iter()
            .map(|level| (level.width, level.height))
            .collect::<Vec<_>>(),
        [(3, 5), (1, 2), (1, 1)]
    );
    assert_eq!(last_pixels(&chain), [73, 73, 73, 255]);
}

#[test]
fn mip_budget_counts_every_level_and_rejects_before_allocation() {
    let source = vec![255; 4 * 4 * 4];
    let exact = ImageBudget::new(4, 4, 84).unwrap();
    let too_small = ImageBudget::new(4, 4, 83).unwrap();
    let ((result, source), allocations) = crate::test_allocations::count(|| {
        let result = MipLayout::new(4, 4, source.len(), too_small, 4, TextureMipmaps3d::Generate);
        (result, source)
    });
    assert!(matches!(
        result,
        Err(ImageError::BudgetExceeded {
            limit: 83,
            actual: 84
        })
    ));
    assert_eq!(allocations, 0);
    let chain = CpuMipChain::new(4, 4, source, exact, 4, TextureMipmaps3d::Generate).unwrap();
    assert_eq!(chain.gpu_allocation_bytes(), 84);
    assert!(chain.recovery_memory_bytes() <= 84);
    assert_eq!(
        chain
            .levels()
            .iter()
            .map(|level| level.offset)
            .collect::<Vec<_>>(),
        [0, 64, 80]
    );
}

#[test]
fn device_and_source_limits_fail_without_allocation() {
    for (width, height, bytes, maximum, expected) in [
        (0, 2, 0, 8, ImageError::ZeroDimension),
        (8, 2, 64, 4, ImageError::DimensionsTooLarge),
        (2, 8, 64, 4, ImageError::DimensionsTooLarge),
        (2, 2, 15, 8, ImageError::InvalidPixelCount),
        (u32::MAX, 1, 4, u32::MAX, ImageError::DimensionsTooLarge),
    ] {
        let (result, allocations) = crate::test_allocations::count(|| {
            MipLayout::new(
                width,
                height,
                bytes,
                ImageBudget::default(),
                maximum,
                TextureMipmaps3d::Generate,
            )
        });
        assert_eq!(result.unwrap_err(), expected);
        assert_eq!(allocations, 0);
    }
}

#[test]
fn one_texel_and_skinny_images_have_complete_bounded_chains() {
    let pixel = vec![11, 22, 33, 44];
    let chain = mip_chain(1, 1, pixel.clone());
    assert_eq!(chain.levels().len(), 1);
    assert_eq!(last_pixels(&chain), pixel);
    let chain = mip_chain(1, 8, [11, 22, 33, 255].repeat(8));
    assert_eq!(chain.levels().len(), 4);
    assert_eq!(chain.gpu_allocation_bytes(), 60);
    assert_eq!(last_pixels(&chain), [11, 22, 33, 255]);
}

#[test]
fn over_capacity_input_is_compacted_and_clone_preserves_every_level() {
    let mut source = Vec::with_capacity(4096);
    source.extend_from_slice(&[255, 0, 0, 255, 0, 0, 255, 255]);
    let chain = CpuMipChain::new(
        2,
        1,
        source,
        ImageBudget::new(2, 1, 12).unwrap(),
        2,
        TextureMipmaps3d::Generate,
    )
    .unwrap();
    assert!(chain.recovery_memory_bytes() <= 12);
    let copy = chain.try_clone().unwrap();
    assert_eq!(copy.pixels, chain.pixels);
    assert_eq!(copy.levels(), chain.levels());
    assert_ne!(copy.pixels.as_ptr(), chain.pixels.as_ptr());
    assert_eq!(
        copy.validate_device_limit(1),
        Err(ImageError::DimensionsTooLarge)
    );
    assert_eq!(copy.validate_device_limit(2), Ok(()));
}
