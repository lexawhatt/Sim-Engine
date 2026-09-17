//! GPU capacity preflight and stream byte layout.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Mesh3dUploadLayout {
    pub(super) vertex_bytes: u64,
    pub(super) index_bytes: u64,
    pub(super) edge_bytes: u64,
    pub(super) texture_coordinate_bytes: u64,
    pub(super) color_bytes: u64,
    pub(super) normal_bytes: u64,
    pub(super) total_bytes: u64,
    pub(super) index_count: u32,
    pub(super) edge_count: u32,
}

pub(super) fn preflight_mesh3d_source(
    source: &Mesh3d,
    max_buffer_size: u64,
) -> Result<Mesh3dUploadLayout, Mesh3dResourceError> {
    let mut layout = preflight_mesh3d_upload(
        source.vertices().len(),
        source.triangle_indices().len(),
        source.display_edges().len(),
        max_buffer_size,
    )?;
    layout.texture_coordinate_bytes = source
        .texture_coordinates()
        .len()
        .checked_mul(std::mem::size_of::<MeshUvGpu>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|bytes| *bytes <= max_buffer_size)
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    layout.color_bytes = source
        .vertex_colors()
        .len()
        .checked_mul(std::mem::size_of::<MeshColorGpu>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|bytes| *bytes <= max_buffer_size)
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    layout.normal_bytes = source
        .normals()
        .len()
        .checked_mul(std::mem::size_of::<MeshNormalGpu>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|bytes| *bytes <= max_buffer_size)
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    layout.total_bytes = layout
        .total_bytes
        .checked_add(layout.texture_coordinate_bytes)
        .and_then(|bytes| bytes.checked_add(layout.color_bytes))
        .and_then(|bytes| bytes.checked_add(layout.normal_bytes))
        .filter(|bytes| usize::try_from(*bytes).is_ok())
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    Ok(layout)
}

pub(super) fn preflight_mesh3d_upload(
    vertex_count: usize,
    index_count: usize,
    edge_count: usize,
    max_buffer_size: u64,
) -> Result<Mesh3dUploadLayout, Mesh3dResourceError> {
    let draw_index_count =
        u32::try_from(index_count).map_err(|_| Mesh3dResourceError::CapacityTooLarge)?;
    let draw_edge_count =
        u32::try_from(edge_count).map_err(|_| Mesh3dResourceError::CapacityTooLarge)?;
    let checked_buffer_bytes = |element_size: usize, element_count: usize| {
        u64::try_from(element_size)
            .ok()
            .and_then(|size| u64::try_from(element_count).ok()?.checked_mul(size))
            .filter(|bytes| *bytes <= max_buffer_size && usize::try_from(*bytes).is_ok())
            .ok_or(Mesh3dResourceError::CapacityTooLarge)
    };
    let vertex_bytes = checked_buffer_bytes(std::mem::size_of::<MeshVertexGpu>(), vertex_count)?;
    let index_bytes = checked_buffer_bytes(std::mem::size_of::<u32>(), index_count)?;
    let edge_bytes = checked_buffer_bytes(std::mem::size_of::<MeshEdgeGpu>(), edge_count)?;
    let total_bytes = vertex_bytes
        .checked_add(index_bytes)
        .and_then(|bytes| bytes.checked_add(edge_bytes))
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    usize::try_from(total_bytes).map_err(|_| Mesh3dResourceError::CapacityTooLarge)?;
    Ok(Mesh3dUploadLayout {
        vertex_bytes,
        index_bytes,
        edge_bytes,
        texture_coordinate_bytes: 0,
        color_bytes: 0,
        normal_bytes: 0,
        total_bytes,
        index_count: draw_index_count,
        edge_count: draw_edge_count,
    })
}
