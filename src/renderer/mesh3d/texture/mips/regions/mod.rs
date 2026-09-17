//! Fixed-size dirty footprints matching the full filter's floating boundaries.

use super::super::ImageTexelRect;
use super::{ImageError, MAX_LEVELS, MipLevel};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::renderer::mesh3d::texture) struct MipRegion {
    pub(in crate::renderer::mesh3d::texture) x: u32,
    pub(in crate::renderer::mesh3d::texture) y: u32,
    pub(in crate::renderer::mesh3d::texture) width: u32,
    pub(in crate::renderer::mesh3d::texture) height: u32,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::renderer::mesh3d::texture) struct MipRegionPlan {
    regions: [MipRegion; MAX_LEVELS],
    count: usize,
    size: (u32, u32),
    lower_upload_bytes: usize,
}

impl MipRegionPlan {
    pub(super) fn new(levels: &[MipLevel], region: ImageTexelRect) -> Result<Self, ImageError> {
        let base = levels.first().ok_or(ImageError::ZeroDimension)?;
        if levels.len() > MAX_LEVELS {
            return Err(ImageError::DimensionsTooLarge);
        }
        let mut result = Self {
            regions: [MipRegion::default(); MAX_LEVELS],
            count: levels.len(),
            size: (base.width, base.height),
            lower_upload_bytes: 0,
        };
        result.regions[0] = MipRegion {
            x: region.x(),
            y: region.y(),
            width: region.width(),
            height: region.height(),
        };
        validate_region(*base, result.regions[0])?;
        for index in 1..levels.len() {
            let source = levels[index - 1];
            let destination = levels[index];
            let previous = result.regions[index - 1];
            let (x, end_x) = affected_axis(
                source.width,
                destination.width,
                previous.x,
                previous
                    .x
                    .checked_add(previous.width)
                    .ok_or(ImageError::DimensionsTooLarge)?,
            )?;
            let (y, end_y) = affected_axis(
                source.height,
                destination.height,
                previous.y,
                previous
                    .y
                    .checked_add(previous.height)
                    .ok_or(ImageError::DimensionsTooLarge)?,
            )?;
            let region = MipRegion {
                x,
                y,
                width: end_x - x,
                height: end_y - y,
            };
            result.lower_upload_bytes = result
                .lower_upload_bytes
                .checked_add(validate_region(destination, region)?)
                .ok_or(ImageError::DimensionsTooLarge)?;
            result.regions[index] = region;
        }
        Ok(result)
    }

    pub(in crate::renderer::mesh3d::texture) fn regions(&self) -> &[MipRegion] {
        &self.regions[..self.count]
    }

    pub(in crate::renderer::mesh3d::texture) fn lower_upload_bytes(&self) -> usize {
        self.lower_upload_bytes
    }

    pub(super) fn size(&self) -> (u32, u32) {
        self.size
    }
}

fn validate_region(level: MipLevel, region: MipRegion) -> Result<usize, ImageError> {
    let end_x = region
        .x
        .checked_add(region.width)
        .ok_or(ImageError::DimensionsTooLarge)?;
    let end_y = region
        .y
        .checked_add(region.height)
        .ok_or(ImageError::DimensionsTooLarge)?;
    if region.width == 0 || region.height == 0 || end_x > level.width || end_y > level.height {
        return Err(ImageError::UpdateRegionOutOfBounds);
    }
    // Validate the complete strided slice span up front, not only its compact
    // texel count. Queue uploads will borrow rows of the existing full level.
    let span_end = (end_y as usize - 1)
        .checked_mul(level.width as usize)
        .and_then(|offset| offset.checked_add(end_x as usize))
        .and_then(|texels| texels.checked_mul(4))
        .ok_or(ImageError::DimensionsTooLarge)?;
    level
        .offset
        .checked_add(span_end)
        .ok_or(ImageError::DimensionsTooLarge)?;
    if span_end > level.byte_count {
        return Err(ImageError::DimensionsTooLarge);
    }
    (region.width as usize)
        .checked_mul(region.height as usize)
        .and_then(|texels| texels.checked_mul(4))
        .ok_or(ImageError::DimensionsTooLarge)
}

fn affected_axis(
    source: u32,
    destination: u32,
    start: u32,
    end: u32,
) -> Result<(u32, u32), ImageError> {
    if source == 0 || destination == 0 {
        return Err(ImageError::ZeroDimension);
    }
    if start >= end || end > source {
        return Err(ImageError::UpdateRegionOutOfBounds);
    }
    let ratio = f64::from(source) / f64::from(destination);
    // Preserve the full filter's arithmetic ordering. Rational integer scaling
    // alone can disagree where rounded odd-size footprint boundaries coincide.
    let first = first_matching(destination, |index| {
        (f64::from(index) + 1.0) * ratio > f64::from(start)
    });
    let last = first_matching(destination, |index| {
        f64::from(index) * ratio >= f64::from(end)
    });
    if first >= last {
        return Err(ImageError::UpdateRegionOutOfBounds);
    }
    Ok((first, last))
}

fn first_matching(length: u32, predicate: impl Fn(u32) -> bool) -> u32 {
    let mut lower = 0;
    let mut upper = length;
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if predicate(middle) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    lower
}

#[cfg(test)]
mod tests;
