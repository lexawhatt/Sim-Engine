//! Bounded mip chains for independently filtered textures.

use super::{ImageBudget, ImageError, image};

const MAX_LEVELS: usize = u32::BITS as usize;

pub(super) fn validate_layout(
    width: u32,
    height: u32,
    source_bytes: usize,
    budget: ImageBudget,
    limit: u32,
    mode: TextureMipmaps3d,
) -> Result<(), ImageError> {
    MipLayout::new(width, height, source_bytes, budget, limit, mode).map(|_| ())
}

/// Optional mip generation for a retained 3D texture.
///
/// This policy applies to a complete, independently filtered image, not to
/// individual cells of a packed atlas. Crop each tile into its own texture
/// before generating mipmaps when neighboring cells must remain isolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TextureMipmaps3d {
    /// Keep the supplied image unchanged and sample only its original level.
    #[default]
    None,
    /// Generate every level down to 1 x 1 with an area-weighted box filter.
    ///
    /// RGB is filtered in linear light with premultiplied alpha, then converted
    /// back to straight sRGB RGBA8. Source pixels remain byte-exact. Generated
    /// levels with alpha rounded to zero have zero RGB. This does not preserve
    /// alpha-test coverage: thin masked features can disappear in distant mips.
    Generate,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MipLevel {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) offset: usize,
    pub(super) byte_count: usize,
}

#[derive(Debug)]
pub(in crate::renderer::mesh3d) struct CpuMipChain {
    pixels: Vec<u8>,
    layout: MipLayout,
    budget: ImageBudget,
    mode: TextureMipmaps3d,
}

#[derive(Debug, Clone, Copy)]
struct MipLayout {
    levels: [MipLevel; MAX_LEVELS],
    count: usize,
    byte_count: usize,
}

impl CpuMipChain {
    pub(super) fn new(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        budget: ImageBudget,
        maximum_texture_dimension: u32,
        mode: TextureMipmaps3d,
    ) -> Result<Self, ImageError> {
        let layout = MipLayout::new(
            width,
            height,
            pixels.len(),
            budget,
            maximum_texture_dimension,
            mode,
        )?;
        // Preflight includes every generated level before reserving storage.
        // An over-capacity caller Vec must not bypass the retained byte limit.
        let mut pixels = if pixels.capacity() > budget.max_bytes() {
            let mut compact = allocate_pixels(layout.byte_count, budget)?;
            compact.extend_from_slice(&pixels);
            compact
        } else {
            let mut pixels = pixels;
            let additional = layout.byte_count - pixels.len();
            pixels
                .try_reserve_exact(additional)
                .map_err(|_| ImageError::AllocationFailed {
                    requested_bytes: layout.byte_count,
                })?;
            validate_capacity(pixels.capacity(), budget)?;
            pixels
        };
        pixels.resize(layout.byte_count, 0);
        if layout.count > 1 {
            let linear_channels = std::array::from_fn::<_, 256, _>(|channel| {
                let encoded = channel as f64 / 255.0;
                if encoded <= 0.04045 {
                    encoded / 12.92
                } else {
                    ((encoded + 0.055) / 1.055).powf(2.4)
                }
            });
            for level in 1..layout.count {
                let source = layout.levels[level - 1];
                let destination = layout.levels[level];
                let (earlier, later) = pixels.split_at_mut(destination.offset);
                downsample(
                    &earlier[source.offset..source.offset + source.byte_count],
                    source.width,
                    source.height,
                    &mut later[..destination.byte_count],
                    destination.width,
                    destination.height,
                    &linear_channels,
                );
            }
        }
        Ok(Self {
            pixels,
            layout,
            budget,
            mode,
        })
    }

    pub(super) fn size(&self) -> (u32, u32) {
        (self.layout.levels[0].width, self.layout.levels[0].height)
    }

    pub(super) fn pixels(&self) -> &[u8] {
        &self.pixels[..self.layout.levels[0].byte_count]
    }

    pub(super) fn level_pixels(&self, level: MipLevel) -> &[u8] {
        &self.pixels[level.offset..level.offset + level.byte_count]
    }

    pub(super) fn levels(&self) -> &[MipLevel] {
        &self.layout.levels[..self.layout.count]
    }

    pub(super) fn budget(&self) -> ImageBudget {
        self.budget
    }

    pub(super) fn mode(&self) -> TextureMipmaps3d {
        self.mode
    }

    pub(super) fn recovery_memory_bytes(&self) -> usize {
        self.pixels.capacity()
    }

    pub(super) fn gpu_allocation_bytes(&self) -> usize {
        self.layout.byte_count
    }

    pub(super) fn validate_device_limit(
        &self,
        maximum_texture_dimension: u32,
    ) -> Result<(), ImageError> {
        MipLayout::new(
            self.size().0,
            self.size().1,
            self.pixels().len(),
            self.budget,
            maximum_texture_dimension,
            self.mode,
        )?;
        Ok(())
    }

    pub(super) fn try_clone(&self) -> Result<Self, ImageError> {
        let mut pixels = allocate_pixels(self.pixels.len(), self.budget)?;
        pixels.extend_from_slice(&self.pixels);
        Ok(Self {
            pixels,
            layout: self.layout,
            budget: self.budget,
            mode: self.mode,
        })
    }

    pub(super) fn allocate_gpu(&self, device: &wgpu::Device) -> wgpu::Texture {
        let (width, height) = self.size();
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine retained 3D texture mip chain"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: self.layout.count as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    pub(super) fn upload_levels(
        &self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        first_level: usize,
    ) {
        for (index, level) in self.levels().iter().copied().enumerate().skip(first_level) {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: index as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                self.level_pixels(level),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level.width * 4),
                    rows_per_image: Some(level.height),
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub(super) fn patch_base(&mut self, region: super::ImageTexelRect, packed: &[u8]) {
        let width = self.size().0 as usize;
        let row_bytes = region.width() as usize * 4;
        for row in 0..region.height() as usize {
            let offset = ((region.y() as usize + row) * width + region.x() as usize) * 4;
            self.pixels[offset..offset + row_bytes]
                .copy_from_slice(&packed[row * row_bytes..(row + 1) * row_bytes]);
        }
    }

    pub(super) fn regenerate(&mut self) {
        if self.layout.count <= 1 {
            return;
        }
        let linear_channels = linear_channels();
        for level in 1..self.layout.count {
            let source = self.layout.levels[level - 1];
            let destination = self.layout.levels[level];
            let (earlier, later) = self.pixels.split_at_mut(destination.offset);
            downsample(
                &earlier[source.offset..source.offset + source.byte_count],
                source.width,
                source.height,
                &mut later[..destination.byte_count],
                destination.width,
                destination.height,
                &linear_channels,
            );
        }
    }
}

impl MipLayout {
    fn new(
        width: u32,
        height: u32,
        source_byte_count: usize,
        budget: ImageBudget,
        maximum_texture_dimension: u32,
        mode: TextureMipmaps3d,
    ) -> Result<Self, ImageError> {
        image::validate_image_shape(
            width,
            height,
            source_byte_count,
            budget,
            maximum_texture_dimension,
        )?;
        let mut result = Self {
            levels: [MipLevel::default(); MAX_LEVELS],
            count: 0,
            byte_count: 0,
        };
        let mut width = width;
        let mut height = height;
        loop {
            let byte_count = (width as usize)
                .checked_mul(height as usize)
                .and_then(|texels| texels.checked_mul(4))
                .ok_or(ImageError::DimensionsTooLarge)?;
            let offset = result.byte_count;
            result.byte_count = offset
                .checked_add(byte_count)
                .ok_or(ImageError::DimensionsTooLarge)?;
            if result.byte_count > budget.max_bytes() {
                return Err(ImageError::BudgetExceeded {
                    limit: budget.max_bytes(),
                    actual: result.byte_count,
                });
            }
            result.levels[result.count] = MipLevel {
                width,
                height,
                offset,
                byte_count,
            };
            result.count += 1;
            if mode == TextureMipmaps3d::None || (width == 1 && height == 1) {
                return Ok(result);
            }
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
    }
}

fn allocate_pixels(byte_count: usize, budget: ImageBudget) -> Result<Vec<u8>, ImageError> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(byte_count)
        .map_err(|_| ImageError::AllocationFailed {
            requested_bytes: byte_count,
        })?;
    validate_capacity(pixels.capacity(), budget)?;
    Ok(pixels)
}

fn validate_capacity(capacity: usize, budget: ImageBudget) -> Result<(), ImageError> {
    if capacity > budget.max_bytes() {
        Err(ImageError::BudgetExceeded {
            limit: budget.max_bytes(),
            actual: capacity,
        })
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn downsample(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    destination: &mut [u8],
    width: u32,
    height: u32,
    linear_channels: &[f64; 256],
) {
    // Integer-halving dimensions still need the entire source extent. Using a
    // fractional footprint includes the last row/column of odd-sized images.
    let horizontal_ratio = f64::from(source_width) / f64::from(width);
    let vertical_ratio = f64::from(source_height) / f64::from(height);
    for y in 0..height {
        let top = f64::from(y) * vertical_ratio;
        let bottom = (f64::from(y) + 1.0) * vertical_ratio;
        for x in 0..width {
            let left = f64::from(x) * horizontal_ratio;
            let right = (f64::from(x) + 1.0) * horizontal_ratio;
            let mut channels = [0.0; 3];
            let mut alpha_weight = 0.0;
            let mut total_weight = 0.0;
            for source_y in (top.floor() as u32)..(bottom.ceil() as u32).min(source_height) {
                let vertical_weight =
                    bottom.min(f64::from(source_y) + 1.0) - top.max(f64::from(source_y));
                for source_x in (left.floor() as u32)..(right.ceil() as u32).min(source_width) {
                    let horizontal_weight =
                        right.min(f64::from(source_x) + 1.0) - left.max(f64::from(source_x));
                    let weight = vertical_weight * horizontal_weight;
                    let index = (source_y as usize * source_width as usize + source_x as usize) * 4;
                    let alpha = f64::from(source[index + 3]) / 255.0;
                    for channel in 0..3 {
                        channels[channel] +=
                            linear_channels[source[index + channel] as usize] * alpha * weight;
                    }
                    alpha_weight += alpha * weight;
                    total_weight += weight;
                }
            }
            let index = (y as usize * width as usize + x as usize) * 4;
            let alpha = (alpha_weight / total_weight * 255.0).round() as u8;
            if alpha == 0 {
                destination[index..index + 4].fill(0);
                continue;
            }
            for channel in 0..3 {
                destination[index + channel] = linear_to_srgb8(channels[channel] / alpha_weight);
            }
            destination[index + 3] = alpha;
        }
    }
}

fn linear_to_srgb8(linear: f64) -> u8 {
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn linear_channels() -> [f64; 256] {
    std::array::from_fn(|channel| {
        let encoded = channel as f64 / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    })
}

#[cfg(test)]
#[path = "mesh3d_texture_mips_tests.rs"]
mod tests;
