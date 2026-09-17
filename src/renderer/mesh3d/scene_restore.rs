//! Transactional scene resource restoration with alias preservation.

use super::{
    Arc, Mesh3dInstance, Mesh3dResourceError, PreparedRetainedMeshUpload, RetainedMesh3d, Scene3d,
    Scene3dRestoreReport, Texture3d, texture, upload, upload_prepared_retained_mesh,
};

pub(super) fn restore_scene3d_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    renderer_identity: Arc<()>,
    scene: &mut Scene3d,
) -> Result<Scene3dRestoreReport, Mesh3dResourceError> {
    let prepared_textures = texture::prepare_scene_restoration(device, &renderer_identity, scene)?;
    let mut restored_textures = Vec::new();
    restored_textures
        .try_reserve_exact(prepared_textures.len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: prepared_textures
                .len()
                .saturating_mul(std::mem::size_of::<(usize, Texture3d)>())
                as u64,
        })?;
    let pending_staging_bytes = scene
        .instances
        .len()
        .checked_mul(std::mem::size_of::<(usize, RetainedMesh3d)>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    let prepared_staging_bytes = scene
        .instances
        .len()
        .checked_mul(std::mem::size_of::<(
            usize,
            Arc<wgpu::Buffer>,
            PreparedRetainedMeshUpload,
        )>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    let restored_staging_bytes = scene
        .instances
        .len()
        .checked_mul(std::mem::size_of::<(usize, RetainedMesh3d)>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    let replacement_staging_bytes = scene
        .instances
        .len()
        .checked_mul(std::mem::size_of::<Mesh3dInstance>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    let mut pending = Vec::<(usize, RetainedMesh3d)>::new();
    pending
        .try_reserve_exact(scene.instances.len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: pending_staging_bytes,
        })?;
    let mut prepared = Vec::<(usize, Arc<wgpu::Buffer>, PreparedRetainedMeshUpload)>::new();
    prepared
        .try_reserve_exact(scene.instances.len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: prepared_staging_bytes,
        })?;
    let mut restored = Vec::<(usize, RetainedMesh3d)>::new();
    restored
        .try_reserve_exact(scene.instances.len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: restored_staging_bytes,
        })?;
    let mut replacement_instances = Vec::new();
    replacement_instances
        .try_reserve_exact(scene.instances.len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: replacement_staging_bytes,
        })?;
    replacement_instances.extend(scene.instances.iter().cloned());

    for instance in &scene.instances {
        if Arc::ptr_eq(&renderer_identity, &instance.mesh.renderer_identity) {
            continue;
        }
        let key = Arc::as_ptr(&instance.mesh.vertex_buffer) as usize;
        pending.push((key, instance.mesh.clone()));
    }
    pending.sort_unstable_by_key(|entry| entry.0);
    pending.dedup_by_key(|entry| entry.0);

    // Complete every device-limit check and caller-scale host allocation
    // before the first replacement GPU resource is created or written.
    for (key, mesh) in pending {
        let upload = upload::prepare_restoration(device, &mesh)?;
        prepared.push((key, Arc::clone(&mesh.vertex_buffer), upload));
    }
    for (key, _old_vertex_buffer, upload) in prepared {
        let replacement =
            upload_prepared_retained_mesh(device, queue, Arc::clone(&renderer_identity), upload);
        restored.push((key, replacement));
    }
    for prepared in prepared_textures {
        if let Some(chain) = prepared.chain {
            let restored = texture::upload_chain(
                device,
                queue,
                &renderer_identity,
                texture_layout,
                chain,
                prepared.source.preserves_alpha(),
            );
            restored_textures.push((prepared.key, restored));
        }
    }

    let mut migrated_object_count = 0;
    for instance in &mut replacement_instances {
        let key = Arc::as_ptr(&instance.mesh.vertex_buffer) as usize;
        let previous_material = instance.mesh.material.clone();
        let mut migrated = false;
        if let Ok(index) = restored.binary_search_by_key(&key, |entry| entry.0) {
            instance.mesh = restored[index].1.clone();
            migrated = true;
        }
        if let Some(material) = previous_material {
            if let Ok(index) = restored_textures
                .binary_search_by_key(&material.texture().identity_key(), |entry| entry.0)
            {
                instance.mesh.material =
                    Some(material.with_restored_texture(&restored_textures[index].1));
                migrated = true;
            } else {
                instance.mesh.material = Some(material);
            }
        }
        migrated_object_count += usize::from(migrated);
    }
    let restored_gpu_bytes = restored.iter().fold(0_usize, |total, (_, mesh)| {
        total.saturating_add(mesh.gpu_allocation_bytes())
    });
    scene.instances = replacement_instances;
    scene.renderer_identity = Some(renderer_identity);
    scene.rebuild_resource_keys();
    Ok(Scene3dRestoreReport {
        object_count: scene.object_count(),
        migrated_object_count,
        restored_mesh_count: restored.len(),
        restored_gpu_bytes,
        restored_texture_count: restored_textures.len(),
        restored_texture_bytes: restored_textures
            .iter()
            .map(|(_, texture)| texture.gpu_allocation_bytes())
            .sum(),
    })
}
