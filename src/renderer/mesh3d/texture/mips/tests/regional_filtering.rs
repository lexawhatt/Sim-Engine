//! Full-filter differential checks independent of the regional footprint mapper.

use super::*;
use crate::ImageTexelRect;

use crate::renderer::mesh3d::texture::tests::full_mip_reference;
use full_mip_reference::Level;

fn pixels(width: u32, height: u32, alpha: bool, seed: usize) -> Vec<u8> {
    (0..width as usize * height as usize)
        .flat_map(|index| {
            let value = index.wrapping_mul(0x045d_9f3b) ^ seed.wrapping_mul(0x119d_e1f3);
            [
                value as u8,
                (value >> 8) as u8,
                (value >> 16) as u8,
                if alpha {
                    [0, 1, 2, 127, 254, 255][(index + seed) % 6]
                } else {
                    255
                },
            ]
        })
        .collect()
}

fn reference(width: u32, height: u32, base: Vec<u8>, mipmaps: TextureMipmaps3d) -> Vec<Level> {
    if mipmaps == TextureMipmaps3d::None {
        vec![Level {
            width,
            height,
            pixels: base,
        }]
    } else {
        full_mip_reference::rebuild(width, height, base)
    }
}

fn affected_axis(source: u32, destination: u32, start: u32, length: u32) -> (u32, u32) {
    let ratio = f64::from(source) / f64::from(destination);
    let mut first = None;
    let mut last = 0;
    // Enumerate the original floating filter footprints, without using the
    // production binary searches or a rounded integer-halving approximation.
    for texel in 0..destination {
        let left = f64::from(texel) * ratio;
        let right = (f64::from(texel) + 1.0) * ratio;
        if right > f64::from(start) && left < f64::from(start + length) {
            first.get_or_insert(texel);
            last = texel;
        }
    }
    let first = first.unwrap();
    (first, last - first + 1)
}

fn footprints(levels: &[MipLevel], region: ImageTexelRect) -> Vec<[u32; 4]> {
    let mut expected = vec![[region.x(), region.y(), region.width(), region.height()]];
    for pair in levels.windows(2) {
        let [x, y, width, height] = *expected.last().unwrap();
        let (x, width) = affected_axis(pair[0].width, pair[1].width, x, width);
        let (y, height) = affected_axis(pair[0].height, pair[1].height, y, height);
        expected.push([x, y, width, height]);
    }
    expected
}

fn assert_complete_chain(chain: &CpuMipChain, expected: &[Level]) {
    assert_eq!(chain.levels().len(), expected.len());
    for (index, (level, expected)) in chain.levels().iter().zip(expected).enumerate() {
        assert_eq!(
            (level.width, level.height),
            (expected.width, expected.height)
        );
        assert_eq!(
            chain.level_pixels(*level),
            expected.pixels,
            "complete mip {index}"
        );
    }
}

fn verify_update(previous: &CpuMipChain, region: ImageTexelRect, patch: &[u8]) -> CpuMipChain {
    let mut candidate = previous.try_clone().unwrap();
    let capacity = candidate.recovery_memory_bytes();
    let storage = candidate.pixels.as_ptr();
    let (plan, allocations) = crate::test_allocations::count(|| {
        let plan = candidate.region_update_plan(region).unwrap();
        candidate.patch_base(region, patch);
        candidate.regenerate_regions(&plan);
        plan
    });
    assert_eq!(
        allocations, 0,
        "plan, patch and regional filtering allocated"
    );
    assert_eq!(candidate.recovery_memory_bytes(), capacity);
    assert_eq!(candidate.pixels.as_ptr(), storage);
    let expected_regions = footprints(previous.levels(), region);
    assert_eq!(plan.regions().len(), expected_regions.len());
    let mut lower_bytes = 0;
    for (index, (actual, expected)) in plan.regions().iter().zip(&expected_regions).enumerate() {
        assert_eq!(
            [actual.x, actual.y, actual.width, actual.height],
            *expected,
            "mip {index} footprint"
        );
        if index > 0 {
            lower_bytes += expected[2] as usize * expected[3] as usize * 4;
        }
    }
    assert_eq!(plan.lower_upload_bytes(), lower_bytes);

    let (width, height) = previous.size();
    let mut base = previous.pixels().to_vec();
    let row_bytes = region.width() as usize * 4;
    for row in 0..region.height() as usize {
        let start = ((region.y() as usize + row) * width as usize + region.x() as usize) * 4;
        base[start..start + row_bytes]
            .copy_from_slice(&patch[row * row_bytes..(row + 1) * row_bytes]);
    }
    assert_complete_chain(&candidate, &reference(width, height, base, previous.mode()));
    for (level, [x, y, width, height]) in previous.levels().iter().zip(expected_regions) {
        let before = previous.level_pixels(*level);
        let after = candidate.level_pixels(*level);
        for row in 0..level.height {
            for column in 0..level.width {
                if column < x || column >= x + width || row < y || row >= y + height {
                    let index = (row as usize * level.width as usize + column as usize) * 4;
                    assert_eq!(
                        &after[index..index + 4],
                        &before[index..index + 4],
                        "outside the affected footprint"
                    );
                }
            }
        }
    }
    candidate
}

#[test]
fn every_small_rectangle_matches_frozen_full_filter_and_brute_force_footprints() {
    for mipmaps in [TextureMipmaps3d::None, TextureMipmaps3d::Generate] {
        for alpha in [false, true] {
            for height in 1..=9 {
                for width in 1..=9 {
                    let base = pixels(width, height, alpha, (width + height) as usize);
                    let chain = CpuMipChain::new(
                        width,
                        height,
                        base.clone(),
                        ImageBudget::default(),
                        1024,
                        mipmaps,
                    )
                    .unwrap();
                    assert_complete_chain(&chain, &reference(width, height, base, mipmaps));
                    for y in 0..height {
                        for x in 0..width {
                            for patch_height in 1..=height - y {
                                for patch_width in 1..=width - x {
                                    let region =
                                        ImageTexelRect::new(x, y, patch_width, patch_height)
                                            .unwrap();
                                    let patch = pixels(
                                        patch_width,
                                        patch_height,
                                        alpha,
                                        (x * 17 + y * 13 + patch_width * 3 + patch_height) as usize,
                                    );
                                    verify_update(&chain, region, &patch);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn repeated_overlapping_separated_and_identical_edits_keep_complete_mips_exact() {
    for (width, height) in [(17, 11), (1, 63), (65, 1), (31, 33)] {
        let mut chain = CpuMipChain::new(
            width,
            height,
            pixels(width, height, true, 4),
            ImageBudget::default(),
            1024,
            TextureMipmaps3d::Generate,
        )
        .unwrap();
        for step in 0..24 {
            let patch_width = (1 + step % 7).min(width);
            let patch_height = (1 + step % 5).min(height);
            let x = if step % 2 == 0 {
                0
            } else {
                width - patch_width
            };
            let y = if step % 3 == 0 {
                0
            } else {
                height - patch_height
            };
            let region = ImageTexelRect::new(x, y, patch_width, patch_height).unwrap();
            let patch = pixels(patch_width, patch_height, true, step as usize + 99);
            chain = verify_update(&chain, region, &patch);
            chain = verify_update(&chain, region, &patch);
        }
        let region = ImageTexelRect::new(0, 0, width, height).unwrap();
        verify_update(&chain, region, &pixels(width, height, true, 201));
    }
}

#[test]
fn invalid_region_planning_is_atomic_and_complete_clone_capacity_is_unchanged() {
    let mut pixels = Vec::with_capacity(4096);
    pixels.extend([255; 4].repeat(9 * 7));
    let chain = CpuMipChain::new(
        9,
        7,
        pixels,
        ImageBudget::default(),
        1024,
        TextureMipmaps3d::Generate,
    )
    .unwrap();
    let expected = reference(9, 7, chain.pixels().to_vec(), TextureMipmaps3d::Generate);
    assert_eq!(chain.recovery_memory_bytes(), 4096);
    for region in [
        ImageTexelRect::new(9, 0, 1, 1).unwrap(),
        ImageTexelRect::new(0, 6, 2, 2).unwrap(),
    ] {
        let (result, allocations) =
            crate::test_allocations::count(|| chain.region_update_plan(region));
        assert!(result.is_err());
        assert_eq!(allocations, 0);
        assert_eq!(chain.recovery_memory_bytes(), 4096);
        assert_complete_chain(&chain, &expected);
    }
    let candidate = verify_update(
        &chain,
        ImageTexelRect::new(8, 6, 1, 1).unwrap(),
        &[255, 17, 5, 1],
    );
    assert_eq!(chain.recovery_memory_bytes(), 4096);
    assert!(candidate.recovery_memory_bytes() >= candidate.gpu_allocation_bytes());
    assert!(candidate.recovery_memory_bytes() < chain.recovery_memory_bytes());
}
