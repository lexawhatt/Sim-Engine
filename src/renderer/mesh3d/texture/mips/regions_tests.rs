//! Numeric footprint and fixed-capacity planning boundaries without image allocation.

use super::*;

fn scanned_axis(source: u32, destination: u32, start: u32, end: u32) -> (u32, u32) {
    let ratio = f64::from(source) / f64::from(destination);
    let mut first = None;
    let mut last = 0;
    for index in 0..destination {
        let left = f64::from(index) * ratio;
        let right = (f64::from(index) + 1.0) * ratio;
        if right.min(f64::from(end)) - left.max(f64::from(start)) > 0.0 {
            first.get_or_insert(index);
            last = index + 1;
        }
    }
    (first.unwrap(), last)
}

#[test]
fn binary_footprints_match_exhaustive_positive_overlap() {
    for source in 1..=65 {
        let destination = (source / 2).max(1);
        for start in 0..source {
            for end in start + 1..=source {
                assert_eq!(
                    affected_axis(source, destination, start, end).unwrap(),
                    scanned_axis(source, destination, start, end),
                    "source {source}, destination {destination}, dirty [{start}, {end})"
                );
            }
        }
    }
}

#[test]
fn odd_shared_boundaries_and_single_texel_axes_are_inclusive() {
    assert_eq!(affected_axis(5, 2, 2, 3).unwrap(), (0, 2));
    assert_eq!(affected_axis(5, 2, 4, 5).unwrap(), (1, 2));
    assert_eq!(affected_axis(4, 2, 1, 2).unwrap(), (0, 1));
    assert_eq!(affected_axis(4, 2, 2, 3).unwrap(), (1, 2));
    assert_eq!(affected_axis(3, 1, 2, 3).unwrap(), (0, 1));
    assert_eq!(affected_axis(1, 1, 0, 1).unwrap(), (0, 1));
}

#[test]
fn huge_axis_endpoints_and_neighbors_bound_every_affected_interval() {
    for source in [u32::MAX, u32::MAX - 1, (1 << 31) + 1, 1 << 31] {
        let destination = source / 2;
        let ratio = f64::from(source) / f64::from(destination);
        for (start, end) in [
            (0, 1),
            (1, 2),
            (source / 2, source / 2 + 1),
            (source - 2, source - 1),
            (source - 1, source),
            (0, source),
        ] {
            let (first, last) = affected_axis(source, destination, start, end).unwrap();
            let overlap = |index: u32| {
                let left = f64::from(index) * ratio;
                let right = (f64::from(index) + 1.0) * ratio;
                right.min(f64::from(end)) - left.max(f64::from(start)) > 0.0
            };
            assert!(first < last && last <= destination);
            assert!(overlap(first));
            assert!(overlap(last - 1));
            if first > 0 {
                assert!(!overlap(first - 1));
            }
            if last < destination {
                assert!(!overlap(last));
            }
        }
    }
}

#[test]
fn invalid_axes_regions_and_strided_spans_are_rejected() {
    assert_eq!(affected_axis(0, 1, 0, 1), Err(ImageError::ZeroDimension));
    assert_eq!(affected_axis(1, 0, 0, 1), Err(ImageError::ZeroDimension));
    for (start, end) in [(0, 0), (2, 1), (1, 4)] {
        assert_eq!(
            affected_axis(3, 1, start, end),
            Err(ImageError::UpdateRegionOutOfBounds)
        );
    }
    let region = MipRegion {
        x: 2,
        y: 1,
        width: 1,
        height: 1,
    };
    let level = MipLevel {
        width: 4,
        height: 2,
        offset: 0,
        byte_count: 32,
    };
    assert_eq!(validate_region(level, region), Ok(4));
    assert_eq!(
        validate_region(
            MipLevel {
                byte_count: 24,
                ..level
            },
            region
        ),
        Err(ImageError::DimensionsTooLarge)
    );
    assert_eq!(
        validate_region(
            MipLevel {
                offset: usize::MAX - 27,
                ..level
            },
            region
        ),
        Err(ImageError::DimensionsTooLarge)
    );
    assert_eq!(
        validate_region(
            level,
            MipRegion {
                x: u32::MAX,
                ..region
            }
        ),
        Err(ImageError::DimensionsTooLarge)
    );
    assert_eq!(
        validate_region(level, MipRegion { y: 2, ..region }),
        Err(ImageError::UpdateRegionOutOfBounds)
    );
}

#[test]
fn base_only_plans_keep_one_region_and_zero_lower_bytes() {
    for (width, height) in [(1, 1), (19, 13)] {
        let level = MipLevel {
            width,
            height,
            offset: 0,
            byte_count: width as usize * height as usize * 4,
        };
        let region = ImageTexelRect::new(width - 1, height - 1, 1, 1).unwrap();
        let (plan, allocations) =
            crate::test_allocations::count(|| MipRegionPlan::new(&[level], region).unwrap());
        assert_eq!(allocations, 0);
        assert_eq!(plan.size(), (width, height));
        assert_eq!(plan.regions().len(), 1);
        assert_eq!(plan.lower_upload_bytes(), 0);
        assert_eq!(
            plan.regions()[0],
            MipRegion {
                x: width - 1,
                y: height - 1,
                width: 1,
                height: 1
            }
        );
    }
    let region = ImageTexelRect::new(0, 0, 1, 1).unwrap();
    assert!(matches!(
        MipRegionPlan::new(&[], region),
        Err(ImageError::ZeroDimension)
    ));
    assert!(matches!(
        MipRegionPlan::new(&[MipLevel::default(); MAX_LEVELS + 1], region),
        Err(ImageError::DimensionsTooLarge)
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn maximum_u32_chain_fits_fixed_plan_without_heap_storage() {
    for vertical in [false, true] {
        let mut levels = [MipLevel::default(); MAX_LEVELS];
        let mut extent = u32::MAX;
        let mut offset = 0;
        for level in &mut levels {
            let (width, height) = if vertical { (1, extent) } else { (extent, 1) };
            let byte_count = extent as usize * 4;
            *level = MipLevel {
                width,
                height,
                offset,
                byte_count,
            };
            offset += byte_count;
            extent = (extent / 2).max(1);
        }
        let (x, y) = if vertical {
            (0, u32::MAX - 1)
        } else {
            (u32::MAX - 1, 0)
        };
        let region = ImageTexelRect::new(x, y, 1, 1).unwrap();
        let (plan, allocations) =
            crate::test_allocations::count(|| MipRegionPlan::new(&levels, region).unwrap());
        assert_eq!(allocations, 0);
        assert_eq!(plan.regions().len(), MAX_LEVELS);
        assert_eq!(plan.lower_upload_bytes(), (MAX_LEVELS - 1) * 4);
        for (level, region) in levels.iter().zip(plan.regions()) {
            assert_eq!((region.width, region.height), (1, 1));
            assert_eq!(region.x, level.width - 1);
            assert_eq!(region.y, level.height - 1);
        }
    }
}
