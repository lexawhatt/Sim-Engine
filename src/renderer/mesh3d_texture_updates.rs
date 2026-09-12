//! Bounded transactional mip-consistent updates with immutable alias isolation.

use super::*;

#[path = "mesh3d_texture_update_types.rs"]
mod types;
pub use types::{
    Texture3dUpdateBudget, Texture3dUpdateBudgetResource, Texture3dUpdateError,
    Texture3dUpdateReport,
};

struct PreparedUpdate {
    chain: CpuMipChain,
    packed: Vec<u8>,
    first_nonopaque: Option<usize>,
    report: Texture3dUpdateReport,
}

impl WgpuRenderer {
    /// Updates this handle only, preserving all existing material/scene aliases.
    ///
    /// Input is straight sRGB RGBA8 starting at byte zero; exact required length
    /// is `(height - 1) * bytes_per_row + width * 4`, with stride at least one row.
    /// Unique handles reuse GPU storage. Shared handles detach by GPU-copying the
    /// old chain before uploading the base patch and regenerated lower levels.
    /// Old materials remain unchanged until explicitly rebound with `with_texture`.
    /// All lower mips are fully regenerated; only mip-zero upload is region-local.
    /// Synchronous errors preserve CPU/GPU drawable state; asynchronous device
    /// loss is separate and recovery restores the last committed complete chain.
    pub fn update_texture3d_region(
        &self,
        texture: &mut Texture3d,
        region: ImageTexelRect,
        pixels: &[u8],
        bytes_per_row: usize,
        budget: Texture3dUpdateBudget,
    ) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
        update_texture_region(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &self.mesh3d_renderer.textures.layout,
            texture,
            region,
            pixels,
            bytes_per_row,
            budget,
        )
    }

    /// Updates only the selected object's texture, preserving its ID/material and
    /// every external/other-object snapshot. A uniquely scene-owned texture reuses
    /// GPU storage; aliases detach. Mesh topology is never copied or uploaded.
    /// Texture, material alpha policy and final scene budgets are checked before
    /// writes. Private bounded scene bookkeeping may be prepared before failure.
    #[allow(clippy::too_many_arguments)]
    pub fn update_scene3d_texture_region(
        &self,
        scene: &mut Scene3d,
        object_id: Object3dId,
        region: ImageTexelRect,
        pixels: &[u8],
        bytes_per_row: usize,
        budget: Texture3dUpdateBudget,
    ) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
        update_scene_texture_region(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &self.mesh3d_renderer.textures.layout,
            scene,
            object_id,
            region,
            pixels,
            bytes_per_row,
            budget,
        )
    }
}

fn sum(left: usize, right: usize) -> Result<usize, Texture3dUpdateError> {
    left.checked_add(right)
        .ok_or(Texture3dUpdateError::CapacityTooLarge)
}
fn limit(
    resource: Texture3dUpdateBudgetResource,
    actual: usize,
    maximum: usize,
) -> Result<(), Texture3dUpdateError> {
    if actual > maximum {
        Err(Texture3dUpdateError::BudgetExceeded {
            resource,
            limit: maximum,
            actual,
        })
    } else {
        Ok(())
    }
}

fn validate_layout(
    region: ImageTexelRect,
    pixels: &[u8],
    stride: usize,
) -> Result<(usize, usize), Texture3dUpdateError> {
    let row = (region.width() as usize)
        .checked_mul(4)
        .ok_or(Texture3dUpdateError::CapacityTooLarge)?;
    let required = (region.height() as usize - 1)
        .checked_mul(stride)
        .and_then(|count| count.checked_add(row))
        .ok_or(Texture3dUpdateError::InvalidSourceLayout)?;
    if stride < row || pixels.len() != required {
        return Err(Texture3dUpdateError::InvalidSourceLayout);
    }
    let bytes = row
        .checked_mul(region.height() as usize)
        .ok_or(Texture3dUpdateError::CapacityTooLarge)?;
    Ok((row, bytes))
}

#[allow(clippy::too_many_arguments)]
fn prepare_update(
    device: &wgpu::Device,
    identity: &Arc<()>,
    source: &Texture3d,
    region: ImageTexelRect,
    pixels: &[u8],
    stride: usize,
    budget: Texture3dUpdateBudget,
    require_opaque: bool,
    detached: bool,
) -> Result<PreparedUpdate, Texture3dUpdateError> {
    let started = Instant::now();
    if !source.belongs_to(identity) {
        return Err(Texture3dError::RendererMismatch.into());
    }
    if !region.fits(source.size().0, source.size().1) {
        return Err(Texture3dError::InvalidRegion.into());
    }
    source
        .storage
        .chain
        .validate_device_limit(device.limits().max_texture_dimension_2d)
        .map_err(Texture3dError::Image)?;
    let (row_bytes, base) = validate_layout(region, pixels, stride)?;
    if require_opaque || !source.preserves_alpha() {
        for y in 0..region.height() as usize {
            if let Some(x) = pixels[y * stride..y * stride + row_bytes]
                .chunks_exact(4)
                .position(|pixel| pixel[3] != 255)
            {
                return Err(Texture3dError::NonOpaquePixel {
                    texel: (region.y() as usize + y) * source.size().0 as usize
                        + region.x() as usize
                        + x,
                }
                .into());
            }
        }
    }
    let gpu = source.gpu_allocation_bytes();
    let mips = gpu - source.pixels().len();
    let uploaded = sum(base, mips)?;
    let copy = if detached { gpu } else { 0 };
    let peak_gpu = sum(gpu, copy)?;
    let peak_recovery = sum(source.recovery_memory_bytes(), gpu)?;
    let staging = sum(base, uploaded)?;
    for (resource, actual, maximum) in [
        (
            Texture3dUpdateBudgetResource::UploadBytes,
            uploaded,
            budget.upload,
        ),
        (
            Texture3dUpdateBudgetResource::StagingBytes,
            staging,
            budget.staging,
        ),
        (
            Texture3dUpdateBudgetResource::PeakRecoveryBytes,
            peak_recovery,
            budget.recovery,
        ),
        (
            Texture3dUpdateBudgetResource::PeakGpuBytes,
            peak_gpu,
            budget.gpu,
        ),
        (
            Texture3dUpdateBudgetResource::GpuCopyBytes,
            copy,
            budget.copy,
        ),
    ] {
        limit(resource, actual, maximum)?;
    }
    let mut packed = Vec::new();
    packed.try_reserve_exact(base).map_err(|_| {
        Texture3dError::Image(ImageError::AllocationFailed {
            requested_bytes: base,
        })
    })?;
    let staging = sum(packed.capacity(), uploaded)?;
    limit(
        Texture3dUpdateBudgetResource::StagingBytes,
        staging,
        budget.staging,
    )?;
    let mut chain = source
        .storage
        .chain
        .try_clone()
        .map_err(Texture3dError::Image)?;
    let recovery = chain.recovery_memory_bytes();
    let peak_recovery = sum(source.recovery_memory_bytes(), recovery)?;
    limit(
        Texture3dUpdateBudgetResource::PeakRecoveryBytes,
        peak_recovery,
        budget.recovery,
    )?;
    for y in 0..region.height() as usize {
        packed.extend_from_slice(&pixels[y * stride..y * stride + row_bytes]);
    }
    chain.patch_base(region, &packed);
    chain.regenerate();
    let first_nonopaque = chain
        .pixels()
        .chunks_exact(4)
        .position(|pixel| pixel[3] != 255);
    let level_count = chain.levels().len();
    Ok(PreparedUpdate {
        chain,
        packed,
        first_nonopaque,
        report: Texture3dUpdateReport {
            base,
            mips,
            uploads: level_count,
            copy,
            copies: if detached { level_count } else { 0 },
            allocations: usize::from(detached),
            submissions: 1 + usize::from(detached),
            recovery,
            gpu,
            staging,
            peak_recovery,
            peak_gpu,
            preparation: started.elapsed(),
            encode_submit: Duration::ZERO,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer::mesh3d) fn update_texture_region(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    texture: &mut Texture3d,
    region: ImageTexelRect,
    pixels: &[u8],
    stride: usize,
    budget: Texture3dUpdateBudget,
) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
    let detached = Arc::strong_count(&texture.storage) != 1;
    let prepared = prepare_update(
        device, identity, texture, region, pixels, stride, budget, false, detached,
    )?;
    Ok(apply_update(
        device, queue, layout, texture, region, prepared,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer::mesh3d) fn update_scene_texture_region(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    scene: &mut Scene3d,
    object_id: Object3dId,
    region: ImageTexelRect,
    pixels: &[u8],
    stride: usize,
    budget: Texture3dUpdateBudget,
) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
    let material = scene
        .instance(object_id)?
        .mesh
        .material()
        .ok_or(Texture3dUpdateError::MissingTexture)?;
    let source = &material.texture;
    // No temporary source/material clones may influence the ownership decision.
    let detached = Arc::strong_count(&source.storage) != 1;
    let prepared = prepare_update(
        device,
        identity,
        source,
        region,
        pixels,
        stride,
        budget,
        material.require_opaque_texture,
        detached,
    )?;
    let change = scene.prepare_dynamic_texture_change(
        object_id,
        prepared.report.recovery,
        prepared.report.gpu,
        detached,
    )?;
    let material = scene.instances[change.object_index()]
        .mesh
        .material
        .as_mut()
        .ok_or(Texture3dUpdateError::MissingTexture)?;
    let report = apply_update(
        device,
        queue,
        layout,
        &mut material.texture,
        region,
        prepared,
    );
    scene.commit_dynamic_texture_change(change);
    Ok(report)
}

fn upload_patch(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    region: ImageTexelRect,
    prepared: &PreparedUpdate,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: region.x(),
                y: region.y(),
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &prepared.packed,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(region.width() * 4),
            rows_per_image: Some(region.height()),
        },
        wgpu::Extent3d {
            width: region.width(),
            height: region.height(),
            depth_or_array_layers: 1,
        },
    );
    prepared.chain.upload_levels(queue, texture, 1);
    submit_pending_uploads(queue);
}

fn apply_update(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    target: &mut Texture3d,
    region: ImageTexelRect,
    mut prepared: PreparedUpdate,
) -> Texture3dUpdateReport {
    let started = Instant::now();
    if let Some(storage) = Arc::get_mut(&mut target.storage) {
        upload_patch(queue, &storage.texture, region, &prepared);
        storage.chain = prepared.chain;
        storage.first_nonopaque_texel = prepared.first_nonopaque;
    } else {
        let texture = prepared.chain.allocate_gpu(device);
        let bindings = create_bindings(device, layout, &texture, prepared.chain.levels().len());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine 3D texture alias detachment"),
        });
        for (index, level) in prepared.chain.levels().iter().enumerate() {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &target.storage.texture,
                    mip_level: index as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: index as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        // Pending writes execute before submitted command buffers. Submit the
        // preservation copy first, then enqueue patches, never in reverse order.
        queue.submit([encoder.finish()]);
        upload_patch(queue, &texture, region, &prepared);
        target.storage = Arc::new(TextureStorage {
            renderer_identity: Arc::clone(&target.storage.renderer_identity),
            preserve_alpha: target.storage.preserve_alpha,
            first_nonopaque_texel: prepared.first_nonopaque,
            chain: prepared.chain,
            texture,
            bindings,
        });
    }
    prepared.report.encode_submit = started.elapsed();
    prepared.report
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn region_source_stride_is_checked_without_allocation() {
        let region = ImageTexelRect::new(1, 1, 2, 2).unwrap();
        assert_eq!(validate_layout(region, &[0; 20], 12), Ok((8, 16)));
        for (data, stride) in [
            (&[0; 19][..], 12),
            (&[0; 16][..], 7),
            (&[0; 16][..], usize::MAX),
        ] {
            assert!(validate_layout(region, data, stride).is_err());
        }
    }
}
