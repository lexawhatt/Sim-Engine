//! Scene-owned whole-bundle updates with immutable snapshot isolation.

use super::*;

#[path = "mesh3d_dynamic_types.rs"]
mod types;
pub use types::{
    DynamicMesh3dBudget, DynamicMesh3dBudgetResource, DynamicMesh3dError, DynamicMesh3dUpdateReport,
};

#[cfg(test)]
#[path = "mesh3d_dynamic_tests.rs"]
mod tests;
#[cfg(test)]
pub(super) use tests::{assert_gpu_dynamic_contract, assert_gpu_dynamic_recovery};

#[derive(Default)]
pub(super) struct DynamicMesh3dScratch {
    vertices: Vec<MeshVertexGpu>,
    edges: Vec<MeshEdgeGpu>,
    coordinates: Vec<MeshUvGpu>,
    colors: Vec<MeshColorGpu>,
    normals: Vec<MeshNormalGpu>,
}

impl WgpuRenderer {
    /// Renderer-owned dynamic mesh conversion capacity, separate from scene
    /// resources and `FrameCacheBudget`. Backend upload staging is excluded.
    pub fn mesh3d_update_scratch_bytes(&self) -> usize {
        self.mesh3d_renderer.dynamic_scratch.capacity_bytes()
    }

    /// Releases retained dynamic conversion scratch without changing scene/GPU
    /// resources, returning the nominal CPU capacity released. Clear before
    /// requesting a smaller retained-scratch limit than a previous update used.
    pub fn clear_mesh3d_update_scratch(&mut self) -> usize {
        let previous = std::mem::take(&mut self.mesh3d_renderer.dynamic_scratch);
        previous.capacity_bytes()
    }

    /// Updates only the selected scene object, reusing unique GPU buffer capacity.
    ///
    /// Other objects and exported immutable mesh/instance clones remain unchanged:
    /// aliases cause one whole-bundle copy-on-write. Once only this scene object
    /// owns the new bundle, subsequent fitting updates allocate no GPU buffers.
    /// Growth or an optional-attribute layout change replaces the whole bundle; all live vertices,
    /// indices, UVs, colors, normals and edges are uploaded even for an identical source.
    ///
    /// Limits include retained capacity and old/new overlap. Validation, scene
    /// accounting and every fallible host preparation complete before GPU writes.
    /// Returned errors preserve the object's previous geometry and scene state;
    /// private bounded conversion scratch may warm before a later device failure.
    /// Asynchronous device loss is separate; recovery restores committed source
    /// and reserved capacities. Queue ordering protects earlier submitted frames.
    ///
    /// Source topology is a validated nonempty `Mesh3d`. Hide/remove an object for
    /// an empty chunk. Its ID, transform, visibility and material are preserved.
    /// UV-bearing material cannot be retained after removing source UVs.
    pub fn update_scene3d_mesh(
        &mut self,
        scene: &mut Scene3d,
        object_id: Object3dId,
        source: Mesh3d,
        budget: DynamicMesh3dBudget,
    ) -> Result<DynamicMesh3dUpdateReport, DynamicMesh3dError> {
        update_scene_mesh(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &mut self.mesh3d_renderer.dynamic_scratch,
            scene,
            object_id,
            source,
            budget,
        )
    }
}

fn final_limit(
    resource: Mesh3dUploadBudgetResource,
    actual: usize,
    limit: usize,
) -> Result<(), DynamicMesh3dError> {
    if actual > limit {
        return Err(Mesh3dResourceError::BudgetExceeded {
            resource,
            actual,
            limit,
        }
        .into());
    }
    Ok(())
}

fn peak_limit(
    resource: DynamicMesh3dBudgetResource,
    actual: usize,
    limit: usize,
) -> Result<(), DynamicMesh3dError> {
    if actual > limit {
        return Err(DynamicMesh3dError::BudgetExceeded {
            resource,
            actual,
            limit,
        });
    }
    Ok(())
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, DynamicMesh3dError> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| Mesh3dResourceError::CapacityTooLarge.into())
    })
}

fn bundle_is_unique(mesh: &RetainedMesh3d) -> bool {
    Arc::strong_count(&mesh.vertex_buffer) == 1
        && [
            &mesh.index_buffer,
            &mesh.edge_buffer,
            &mesh.texture_coordinate_buffer,
            &mesh.color_buffer,
            &mesh.normal_buffer,
        ]
        .into_iter()
        .all(|buffer| {
            buffer
                .as_ref()
                .is_none_or(|buffer| Arc::strong_count(buffer) == 1)
        })
}

fn planned_allocation(
    old: Mesh3dUploadLayout,
    source: &Mesh3d,
    budget: DynamicMesh3dBudget,
    max_buffer_size: u64,
) -> Result<Mesh3dUploadLayout, Mesh3dResourceError> {
    let vertices = (old.vertex_bytes as usize / std::mem::size_of::<MeshVertexGpu>())
        .max(source.vertices().len())
        .max(budget.minimum_vertices);
    let indices = (old.index_bytes as usize / std::mem::size_of::<u32>())
        .max(source.triangle_indices().len())
        .max(budget.minimum_indices);
    let edges = (old.edge_bytes as usize / std::mem::size_of::<MeshEdgeGpu>())
        .max(source.display_edges().len())
        .max(budget.minimum_edges);
    let mut allocation = preflight_mesh3d_upload(vertices, indices, edges, max_buffer_size)?;
    if !source.texture_coordinates().is_empty() {
        allocation.texture_coordinate_bytes = u64::try_from(vertices)
            .ok()
            .and_then(|vertices| vertices.checked_mul(std::mem::size_of::<MeshUvGpu>() as u64))
            .filter(|bytes| *bytes <= max_buffer_size)
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
        allocation.total_bytes = allocation
            .total_bytes
            .checked_add(allocation.texture_coordinate_bytes)
            .filter(|bytes| usize::try_from(*bytes).is_ok())
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    }
    if !source.vertex_colors().is_empty() {
        allocation.color_bytes = u64::try_from(vertices)
            .ok()
            .and_then(|count| count.checked_mul(std::mem::size_of::<MeshColorGpu>() as u64))
            .filter(|bytes| *bytes <= max_buffer_size)
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
        allocation.total_bytes = allocation
            .total_bytes
            .checked_add(allocation.color_bytes)
            .filter(|bytes| usize::try_from(*bytes).is_ok())
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    }
    if !source.normals().is_empty() {
        allocation.normal_bytes = u64::try_from(vertices)
            .ok()
            .and_then(|count| count.checked_mul(std::mem::size_of::<MeshNormalGpu>() as u64))
            .filter(|bytes| *bytes <= max_buffer_size)
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
        allocation.total_bytes = allocation
            .total_bytes
            .checked_add(allocation.normal_bytes)
            .filter(|bytes| usize::try_from(*bytes).is_ok())
            .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    }
    Ok(allocation)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update_scene_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    scratch: &mut DynamicMesh3dScratch,
    scene: &mut Scene3d,
    object_id: Object3dId,
    source: Mesh3d,
    budget: DynamicMesh3dBudget,
) -> Result<DynamicMesh3dUpdateReport, DynamicMesh3dError> {
    let instance = scene.instance(object_id)?;
    let old = &instance.mesh;
    if !Arc::ptr_eq(identity, &old.renderer_identity)
        || old
            .material
            .as_ref()
            .is_some_and(|material| !material.texture().belongs_to(identity))
    {
        return Err(Mesh3dResourceError::RendererMismatch.into());
    }
    validate_mesh_source_style(&source, instance.style)?;
    if old.material.is_some()
        && (source.texture_coordinates().len() != source.vertices().len()
            || source.triangle_count() == 0)
    {
        return Err(Mesh3dResourceError::Texture(Texture3dError::MissingTextureCoordinates).into());
    }
    let live = preflight_mesh3d_source(&source, device.limits().max_buffer_size)?;
    if !mesh3d_source_is_portable(&source) {
        return Err(Mesh3dResourceError::NonPortableVertex.into());
    }
    if source.texture_coordinates().iter().any(|coordinate| {
        !is_portable_shader_source(coordinate.u()) || !is_portable_shader_source(coordinate.v())
    }) {
        return Err(Mesh3dResourceError::NonPortableTextureCoordinate.into());
    }
    let allocation = planned_allocation(
        old.allocation,
        &source,
        budget,
        device.limits().max_buffer_size,
    )?;
    upload::validate_allocation(device, allocation, budget.upload)?;
    final_limit(
        Mesh3dUploadBudgetResource::RecoveryBytes,
        source.recovery_memory_bytes(),
        budget.upload.max_recovery_bytes(),
    )?;
    let detached_aliases = !bundle_is_unique(old);
    let layout_changed = old.allocation != allocation;
    let grew_capacity = [
        (allocation.vertex_bytes, old.allocation.vertex_bytes),
        (allocation.index_bytes, old.allocation.index_bytes),
        (allocation.edge_bytes, old.allocation.edge_bytes),
        (allocation.color_bytes, old.allocation.color_bytes),
        (allocation.normal_bytes, old.allocation.normal_bytes),
        (
            allocation.texture_coordinate_bytes,
            old.allocation.texture_coordinate_bytes,
        ),
    ]
    .into_iter()
    .any(|(new, old)| new > old);
    let replace_buffers = detached_aliases || layout_changed;
    let peak_gpu_bytes = checked_sum([
        old.gpu_allocation_bytes(),
        if replace_buffers {
            allocation.total_bytes as usize
        } else {
            0
        },
    ])?;
    let same_source = old.source.vertices().as_ptr() == source.vertices().as_ptr();
    let peak_recovery_bytes = checked_sum([
        old.recovery_memory_bytes(),
        if same_source {
            0
        } else {
            source.recovery_memory_bytes()
        },
    ])?;
    peak_limit(
        DynamicMesh3dBudgetResource::PeakGpuBytes,
        peak_gpu_bytes,
        budget.peak_gpu,
    )?;
    peak_limit(
        DynamicMesh3dBudgetResource::PeakRecoveryBytes,
        peak_recovery_bytes,
        budget.peak_recovery,
    )?;
    // This clone is a private candidate, after testing external snapshot aliases.
    // No caller can acquire another scene alias while its exclusive borrow lasts.
    let mut candidate = old.clone();
    candidate.source = source;
    candidate.index_count = live.index_count;
    candidate.edge_count = live.edge_count;
    candidate.budget = budget.upload;
    let change = scene.prepare_dynamic_mesh_change(
        object_id,
        &candidate.source,
        allocation.total_bytes as usize,
        replace_buffers,
    )?;
    let (scratch_reallocations, peak_staging_bytes) = scratch.prepare(&candidate.source, budget)?;
    let gpu_allocations = if replace_buffers {
        let material = candidate.material.take();
        candidate = allocate_retained_mesh(
            device,
            Arc::clone(identity),
            candidate.source,
            live,
            allocation,
            budget.upload,
        );
        candidate.material = material;
        [
            allocation.vertex_bytes,
            allocation.index_bytes,
            allocation.edge_bytes,
            allocation.texture_coordinate_bytes,
            allocation.color_bytes,
            allocation.normal_bytes,
        ]
        .into_iter()
        .filter(|bytes| *bytes > 0)
        .count()
    } else {
        0
    };
    let upload_calls = write_retained_mesh_uploads(
        queue,
        &candidate,
        &scratch.vertices,
        &scratch.edges,
        &scratch.coordinates,
        &scratch.colors,
        &scratch.normals,
    );
    submit_pending_uploads(queue);
    let scene_statistics = scene.commit_dynamic_mesh_change(change, candidate);
    Ok(DynamicMesh3dUpdateReport {
        uploaded_bytes: live.total_bytes as usize,
        upload_calls,
        gpu_allocations,
        detached_aliases,
        grew_capacity,
        capacity_gpu_bytes: allocation.total_bytes as usize,
        scratch_capacity_bytes: scratch.capacity_bytes(),
        scratch_reallocations,
        peak_staging_bytes,
        peak_gpu_bytes,
        peak_recovery_bytes,
        scene_statistics,
    })
}

impl DynamicMesh3dScratch {
    fn capacity_bytes(&self) -> usize {
        self.vertices.capacity() * std::mem::size_of::<MeshVertexGpu>()
            + self.edges.capacity() * std::mem::size_of::<MeshEdgeGpu>()
            + self.coordinates.capacity() * std::mem::size_of::<MeshUvGpu>()
            + self.colors.capacity() * std::mem::size_of::<MeshColorGpu>()
            + self.normals.capacity() * std::mem::size_of::<MeshNormalGpu>()
    }

    fn prepare(
        &mut self,
        source: &Mesh3d,
        budget: DynamicMesh3dBudget,
    ) -> Result<(usize, usize), DynamicMesh3dError> {
        let counts = [
            source.vertices().len(),
            source.display_edges().len(),
            source.texture_coordinates().len(),
            source.vertex_colors().len(),
            source.normals().len(),
        ];
        let capacities = [
            self.vertices.capacity(),
            self.edges.capacity(),
            self.coordinates.capacity(),
            self.colors.capacity(),
            self.normals.capacity(),
        ];
        let strides = [
            std::mem::size_of::<MeshVertexGpu>(),
            std::mem::size_of::<MeshEdgeGpu>(),
            std::mem::size_of::<MeshUvGpu>(),
            std::mem::size_of::<MeshColorGpu>(),
            std::mem::size_of::<MeshNormalGpu>(),
        ];
        let planned = std::array::from_fn::<_, 5, _>(|index| capacities[index].max(counts[index]));
        let bytes = |values: [usize; 5]| -> Result<usize, DynamicMesh3dError> {
            let mut total = 0usize;
            for (count, stride) in values.into_iter().zip(strides) {
                total = count
                    .checked_mul(stride)
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
            }
            Ok(total)
        };
        let final_bytes = bytes(planned)?;
        let replacing = std::array::from_fn(|index| {
            if counts[index] > capacities[index] {
                counts[index]
            } else {
                0
            }
        });
        let old_bytes = bytes(capacities)?;
        let peak = checked_sum([old_bytes, bytes(replacing)?])?;
        final_limit(
            Mesh3dUploadBudgetResource::StagingBytes,
            final_bytes,
            budget.upload.max_staging_bytes(),
        )?;
        peak_limit(
            DynamicMesh3dBudgetResource::PeakStagingBytes,
            peak,
            budget.peak_staging,
        )?;
        let vertices = reserve_conversion::<MeshVertexGpu>(capacities[0], counts[0])?;
        let edges = reserve_conversion::<MeshEdgeGpu>(capacities[1], counts[1])?;
        let coordinates = reserve_conversion::<MeshUvGpu>(capacities[2], counts[2])?;
        let colors = reserve_conversion::<MeshColorGpu>(capacities[3], counts[3])?;
        let normals = reserve_conversion::<MeshNormalGpu>(capacities[4], counts[4])?;
        let actual = [
            vertices.as_ref().map_or(capacities[0], Vec::capacity),
            edges.as_ref().map_or(capacities[1], Vec::capacity),
            coordinates.as_ref().map_or(capacities[2], Vec::capacity),
            colors.as_ref().map_or(capacities[3], Vec::capacity),
            normals.as_ref().map_or(capacities[4], Vec::capacity),
        ];
        let actual_new = [
            vertices.as_ref().map_or(0, Vec::capacity),
            edges.as_ref().map_or(0, Vec::capacity),
            coordinates.as_ref().map_or(0, Vec::capacity),
            colors.as_ref().map_or(0, Vec::capacity),
            normals.as_ref().map_or(0, Vec::capacity),
        ];
        let peak = checked_sum([old_bytes, bytes(actual_new)?])?;
        final_limit(
            Mesh3dUploadBudgetResource::StagingBytes,
            bytes(actual)?,
            budget.upload.max_staging_bytes(),
        )?;
        peak_limit(
            DynamicMesh3dBudgetResource::PeakStagingBytes,
            peak,
            budget.peak_staging,
        )?;
        let reallocations = usize::from(vertices.is_some())
            + usize::from(edges.is_some())
            + usize::from(coordinates.is_some())
            + usize::from(colors.is_some())
            + usize::from(normals.is_some());
        if let Some(vertices) = vertices {
            self.vertices = vertices;
        }
        if let Some(edges) = edges {
            self.edges = edges;
        }
        if let Some(coordinates) = coordinates {
            self.coordinates = coordinates;
        }
        if let Some(colors) = colors {
            self.colors = colors;
        }
        if let Some(normals) = normals {
            self.normals = normals;
        }
        self.normals.clear();
        self.normals
            .extend(source.normals().iter().map(MeshNormalGpu::from_source));
        self.colors.clear();
        self.colors
            .extend(source.vertex_colors().iter().map(|color| MeshColorGpu {
                color: color.to_array(),
            }));
        self.vertices.clear();
        self.vertices
            .extend(source.vertices().iter().map(|vertex| MeshVertexGpu {
                position: [vertex.x(), vertex.y(), vertex.z()],
            }));
        self.edges.clear();
        self.edges.extend(source.display_edges().iter().map(|edge| {
            let start = source.vertices()[edge.start() as usize];
            let end = source.vertices()[edge.end() as usize];
            MeshEdgeGpu {
                start: [start.x(), start.y(), start.z()],
                end: [end.x(), end.y(), end.z()],
            }
        }));
        self.coordinates.clear();
        self.coordinates.extend(
            source
                .texture_coordinates()
                .iter()
                .map(|coordinate| MeshUvGpu {
                    coordinate: [coordinate.u(), coordinate.v()],
                }),
        );
        Ok((reallocations, peak))
    }
}

fn reserve_conversion<T>(
    capacity: usize,
    count: usize,
) -> Result<Option<Vec<T>>, Mesh3dResourceError> {
    if count <= capacity {
        return Ok(None);
    }
    let requested_bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed { requested_bytes })?;
    Ok(Some(values))
}
