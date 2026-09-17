//! Adjacent retained-instance batching without reordering or new frame storage.

use super::{Arc, Mesh3dInstance, SurfaceAlphaMode3d};

pub(super) fn compatible(left: &Mesh3dInstance, right: &Mesh3dInstance) -> bool {
    let (Some(left_style), Some(right_style)) =
        (left.style.surface_style(), right.style.surface_style())
    else {
        return false;
    };
    // Blend retains its existing per-object sorted draw contract. Uniform-only
    // properties (tint, UV transform, mask cutoff, fog) may vary per instance.
    if left_style.alpha_mode() == SurfaceAlphaMode3d::Blend
        || right_style.alpha_mode() == SurfaceAlphaMode3d::Blend
        || left_style.sidedness() != right_style.sidedness()
        || left_style.lighting() != right_style.lighting()
    {
        return false;
    }
    let (left, right) = (&left.mesh, &right.mesh);
    left.index_count == right.index_count
        && Arc::ptr_eq(&left.vertex_buffer, &right.vertex_buffer)
        && same_buffer(&left.index_buffer, &right.index_buffer)
        && same_buffer(
            &left.texture_coordinate_buffer,
            &right.texture_coordinate_buffer,
        )
        && same_buffer(&left.color_buffer, &right.color_buffer)
        && same_buffer(&left.normal_buffer, &right.normal_buffer)
        && match (left.material(), right.material()) {
            (Some(left), Some(right)) => std::ptr::eq(left.bind_group(), right.bind_group()),
            (None, None) => true,
            _ => false,
        }
}

fn same_buffer(left: &Option<Arc<wgpu::Buffer>>, right: &Option<Arc<wgpu::Buffer>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}
