//! Complete-chain creation and physically isolated atlas snapshots.

use super::*;

impl WgpuRenderer {
    /// Creates an independently filtered image with explicit alpha/mipmap policy.
    /// ImageBudget covers the complete committed CPU/GPU chain, not only mip zero.
    pub fn create_texture3d_rgba8_with_options(
        &self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        budget: ImageBudget,
        options: Texture3dOptions,
    ) -> Result<Texture3d, Texture3dError> {
        create_texture_with_options(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &self.mesh3d_renderer.textures.layout,
            width,
            height,
            pixels,
            budget,
            options,
        )
    }

    /// Crops physical source mip-zero texels into an independent local texture.
    /// Mip generation sees only the selected region. Later parent updates never
    /// propagate. Use full-domain normalized UVs and material Repeat to repeat a
    /// tile without sampling neighboring atlas entries at any mip level.
    pub fn crop_texture3d_tile(
        &self,
        source: &Texture3d,
        region: ImageTexelRect,
        budget: ImageBudget,
        options: Texture3dOptions,
    ) -> Result<Texture3d, Texture3dError> {
        crop_texture_tile(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &self.mesh3d_renderer.textures.layout,
            source,
            region,
            budget,
            options,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer::mesh3d) fn create_texture_with_options(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    budget: ImageBudget,
    options: Texture3dOptions,
) -> Result<Texture3d, Texture3dError> {
    image::validate_image_shape(
        width,
        height,
        pixels.len(),
        budget,
        device.limits().max_texture_dimension_2d,
    )
    .map_err(Texture3dError::Image)?;
    if !options.preserves_alpha()
        && let Some(texel) = pixels.chunks_exact(4).position(|pixel| pixel[3] != 255)
    {
        return Err(Texture3dError::NonOpaquePixel { texel });
    }
    let chain = CpuMipChain::new(
        width,
        height,
        pixels,
        budget,
        device.limits().max_texture_dimension_2d,
        options.mipmaps(),
    )
    .map_err(Texture3dError::Image)?;
    Ok(upload_chain(
        device,
        queue,
        identity,
        layout,
        chain,
        options.preserves_alpha(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer::mesh3d) fn crop_texture_tile(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    source: &Texture3d,
    region: ImageTexelRect,
    budget: ImageBudget,
    options: Texture3dOptions,
) -> Result<Texture3d, Texture3dError> {
    if !source.belongs_to(identity) {
        return Err(Texture3dError::RendererMismatch);
    }
    if !region.fits(source.size().0, source.size().1) {
        return Err(Texture3dError::InvalidRegion);
    }
    let bytes = (region.width() as usize)
        .checked_mul(region.height() as usize)
        .and_then(|count| count.checked_mul(4))
        .ok_or(Texture3dError::Image(ImageError::DimensionsTooLarge))?;
    // Validate the whole chain before even allocating the cropped base image.
    mips::validate_layout(
        region.width(),
        region.height(),
        bytes,
        budget,
        device.limits().max_texture_dimension_2d,
        options.mipmaps(),
    )
    .map_err(Texture3dError::Image)?;
    if !options.preserves_alpha() {
        for y in 0..region.height() as usize {
            for x in 0..region.width() as usize {
                let source_texel =
                    (region.y() as usize + y) * source.size().0 as usize + region.x() as usize + x;
                if source.pixels()[source_texel * 4 + 3] != 255 {
                    return Err(Texture3dError::NonOpaquePixel {
                        texel: y * region.width() as usize + x,
                    });
                }
            }
        }
    }
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(bytes).map_err(|_| {
        Texture3dError::Image(ImageError::AllocationFailed {
            requested_bytes: bytes,
        })
    })?;
    if pixels.capacity() > budget.max_bytes() {
        return Err(Texture3dError::Image(ImageError::BudgetExceeded {
            limit: budget.max_bytes(),
            actual: pixels.capacity(),
        }));
    }
    let row_bytes = region.width() as usize * 4;
    for y in 0..region.height() as usize {
        let start =
            ((region.y() as usize + y) * source.size().0 as usize + region.x() as usize) * 4;
        pixels.extend_from_slice(&source.pixels()[start..start + row_bytes]);
    }
    create_texture_with_options(
        device,
        queue,
        identity,
        layout,
        region.width(),
        region.height(),
        pixels,
        budget,
        options,
    )
}
