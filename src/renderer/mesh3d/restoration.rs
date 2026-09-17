//! Standalone retained restoration preserves the full immutable material policy.

use super::*;

pub(super) fn restore_retained_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: Arc<()>,
    texture_layout: &wgpu::BindGroupLayout,
    source: &RetainedMesh3d,
) -> Result<RetainedMesh3d, Mesh3dResourceError> {
    let prepared = upload::prepare_restoration(device, source)?;
    let material = source
        .material()
        .map(|material| {
            texture::restore_texture(device, queue, &identity, texture_layout, material.texture())
                .map(|texture| material.with_restored_texture(&texture))
                .map_err(Mesh3dResourceError::Texture)
        })
        .transpose()?;
    let mut restored = upload_prepared_retained_mesh(device, queue, identity, prepared);
    restored.material = material;
    Ok(restored)
}
