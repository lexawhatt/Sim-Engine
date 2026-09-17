//! Success-only shortcut for an unchanged original triangle.

use super::{ClipVertex, MAX_POLYGON_VERTICES, validate_projected_triangle_orientation};

pub(super) fn accepts(
    vertices: &[ClipVertex; MAX_POLYGON_VERTICES],
    all_planes_inside: bool,
) -> bool {
    if !all_planes_inside {
        return false;
    }
    let fixed: [[f32; 4]; 3] =
        std::array::from_fn(|index| vertices[index].ranges.map(|value| value.fixed));
    // Source vertices have different provenance. Even equal fixed coordinates
    // with identical attributes cannot collapse in the authoritative clipper.
    if fixed[0] == fixed[1] || fixed[1] == fixed[2] || fixed[2] == fixed[0] {
        return false;
    }
    for (vertex, fixed) in vertices[..3].iter().zip(fixed) {
        if vertex.ranges[3].minimum < f64::from(f32::MIN_POSITIVE) {
            return false;
        }
        let [x, y, z, w] = fixed.map(f64::from);
        for (plane, distance) in [w + x, w - x, w + y, w - y, z, w - z]
            .into_iter()
            .enumerate()
        {
            // Exact boundary masks override the raw fixed equation, just as
            // ClipVertex::plane_value does after canonical plane enforcement.
            if vertex.planes & (1 << plane) == 0 && distance < 0.0 {
                return false;
            }
        }
    }
    // Never introduce a new early failure: inconclusive orientation, boundary
    // or identity checks resume the original ordered clipping/error path.
    validate_projected_triangle_orientation([
        vertices[0].ranges,
        vertices[1].ranges,
        vertices[2].ranges,
    ])
    .is_ok()
}
