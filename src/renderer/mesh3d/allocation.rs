//! Bounded retained topology validation, staging and GPU allocation.

use super::*;

pub(super) fn create_retained_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: Mesh3d,
) -> Result<RetainedMesh3d, Mesh3dResourceError> {
    let prepared = upload::prepare_with_budget(device, source, Mesh3dUploadBudget::default())?;
    Ok(upload_prepared_retained_mesh(
        device,
        queue,
        renderer_identity,
        prepared,
    ))
}

pub(super) struct PreparedRetainedMeshUpload {
    pub(super) source: Mesh3d,
    pub(super) layout: Mesh3dUploadLayout,
    pub(super) allocation: Mesh3dUploadLayout,
    pub(super) vertices: Vec<MeshVertexGpu>,
    pub(super) edges: Vec<MeshEdgeGpu>,
    pub(super) texture_coordinates: Vec<MeshUvGpu>,
    pub(super) colors: Vec<MeshColorGpu>,
    pub(super) normals: Vec<MeshNormalGpu>,
    pub(super) budget: Mesh3dUploadBudget,
}

impl PreparedRetainedMeshUpload {
    pub(super) fn staging_capacity_bytes(&self) -> usize {
        self.vertices
            .capacity()
            .saturating_mul(std::mem::size_of::<MeshVertexGpu>())
            .saturating_add(
                self.edges
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshEdgeGpu>()),
            )
            .saturating_add(
                self.texture_coordinates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshUvGpu>()),
            )
            .saturating_add(
                self.colors
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshColorGpu>()),
            )
            .saturating_add(
                self.normals
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshNormalGpu>()),
            )
    }
}

pub(super) fn prepare_retained_mesh_upload(
    device: &wgpu::Device,
    source: Mesh3d,
) -> Result<PreparedRetainedMeshUpload, Mesh3dResourceError> {
    let layout = preflight_mesh3d_source(&source, device.limits().max_buffer_size)?;
    if !mesh3d_source_is_portable(&source) {
        return Err(Mesh3dResourceError::NonPortableVertex);
    }
    if source.texture_coordinates().iter().any(|coordinate| {
        !is_portable_shader_source(coordinate.u()) || !is_portable_shader_source(coordinate.v())
    }) {
        return Err(Mesh3dResourceError::NonPortableTextureCoordinate);
    }
    let mut normals = Vec::new();
    normals
        .try_reserve_exact(source.normals().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.normal_bytes,
        })?;
    normals.extend(source.normals().iter().map(MeshNormalGpu::from_source));
    let mut colors = Vec::new();
    colors
        .try_reserve_exact(source.vertex_colors().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.color_bytes,
        })?;
    colors.extend(source.vertex_colors().iter().map(|color| MeshColorGpu {
        color: color.to_array(),
    }));
    let mut texture_coordinates = Vec::new();
    texture_coordinates
        .try_reserve_exact(source.texture_coordinates().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.texture_coordinate_bytes,
        })?;
    texture_coordinates.extend(
        source
            .texture_coordinates()
            .iter()
            .map(|coordinate| MeshUvGpu {
                coordinate: [coordinate.u(), coordinate.v()],
            }),
    );
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(source.vertices().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.vertex_bytes,
        })?;
    vertices.extend(source.vertices().iter().map(|vertex| MeshVertexGpu {
        position: [vertex.x(), vertex.y(), vertex.z()],
    }));
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(source.display_edges().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.edge_bytes,
        })?;
    edges.extend(source.display_edges().iter().map(|edge| {
        let start = source.vertices()[edge.start() as usize];
        let end = source.vertices()[edge.end() as usize];
        MeshEdgeGpu {
            start: [start.x(), start.y(), start.z()],
            end: [end.x(), end.y(), end.z()],
        }
    }));
    Ok(PreparedRetainedMeshUpload {
        source,
        layout,
        allocation: layout,
        vertices,
        edges,
        texture_coordinates,
        colors,
        normals,
        budget: Mesh3dUploadBudget::default(),
    })
}

pub(super) fn mesh3d_source_is_portable(source: &Mesh3d) -> bool {
    source.vertices().iter().all(|vertex| {
        [vertex.x(), vertex.y(), vertex.z()]
            .into_iter()
            .all(is_portable_shader_source)
    })
}

pub(super) fn upload_prepared_retained_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    prepared: PreparedRetainedMeshUpload,
) -> RetainedMesh3d {
    let PreparedRetainedMeshUpload {
        source,
        layout,
        allocation,
        vertices,
        edges,
        texture_coordinates,
        colors,
        normals,
        budget,
    } = prepared;
    let mesh = allocate_retained_mesh(
        device,
        renderer_identity,
        source,
        layout,
        allocation,
        budget,
    );
    write_retained_mesh_uploads(
        queue,
        &mesh,
        &vertices,
        &edges,
        &texture_coordinates,
        &colors,
        &normals,
    );
    submit_pending_uploads(queue);
    mesh
}

pub(super) fn allocate_retained_mesh(
    device: &wgpu::Device,
    renderer_identity: Arc<()>,
    source: Mesh3d,
    live: Mesh3dUploadLayout,
    allocation: Mesh3dUploadLayout,
    budget: Mesh3dUploadBudget,
) -> RetainedMesh3d {
    let vertex_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine retained 3D vertex buffer"),
        size: allocation.vertex_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let index_buffer = if allocation.index_bytes == 0 {
        None
    } else {
        Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine retained 3D index buffer"),
            size: allocation.index_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })))
    };
    let edge_buffer = if allocation.edge_bytes == 0 {
        None
    } else {
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine retained 3D edge buffer"),
            size: allocation.edge_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Some(buffer)
    };
    let texture_coordinate_buffer = if allocation.texture_coordinate_bytes == 0 {
        None
    } else {
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine 3D UV buffer"),
            size: allocation.texture_coordinate_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Some(buffer)
    };
    let color_buffer = (allocation.color_bytes > 0).then(|| {
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine 3D vertex color buffer"),
            size: allocation.color_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    });
    let normal_buffer = (allocation.normal_bytes > 0).then(|| {
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine 3D normal buffer"),
            size: allocation.normal_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    });
    RetainedMesh3d {
        renderer_identity,
        vertex_buffer,
        index_buffer,
        edge_buffer,
        texture_coordinate_buffer,
        color_buffer,
        normal_buffer,
        material: None,
        source,
        index_count: live.index_count,
        edge_count: live.edge_count,
        gpu_allocation_bytes: allocation.total_bytes as usize,
        budget,
        allocation,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_retained_mesh_uploads(
    queue: &wgpu::Queue,
    mesh: &RetainedMesh3d,
    vertices: &[MeshVertexGpu],
    edges: &[MeshEdgeGpu],
    coordinates: &[MeshUvGpu],
    colors: &[MeshColorGpu],
    normals: &[MeshNormalGpu],
) -> usize {
    let mut calls = 0;
    for (buffer, bytes) in [
        (Some(&mesh.vertex_buffer), bytemuck::cast_slice(vertices)),
        (
            mesh.index_buffer.as_ref(),
            bytemuck::cast_slice(mesh.source.triangle_indices()),
        ),
        (mesh.edge_buffer.as_ref(), bytemuck::cast_slice(edges)),
        (mesh.color_buffer.as_ref(), bytemuck::cast_slice(colors)),
        (mesh.normal_buffer.as_ref(), bytemuck::cast_slice(normals)),
        (
            mesh.texture_coordinate_buffer.as_ref(),
            bytemuck::cast_slice(coordinates),
        ),
    ] {
        if let Some(buffer) = buffer
            && !bytes.is_empty()
        {
            queue.write_buffer(buffer, 0, bytes);
            calls += 1;
        }
    }
    calls
}
