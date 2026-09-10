//! Conservative shader bounds for the independently anchored two-point arrow path.

use super::*;

pub(super) fn vertex_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<([(f64, f64); 2], bool)> {
    let mut center = vertex;
    center.world_offset = [0.0; 2];
    let base = tessellated_vertex_base_screen_ranges(center, uniform)?;
    let direction = Vec2::new(vertex.next_direction[0], vertex.next_direction[1]);
    let projected = [
        shader_direction_dot_range(uniform.world_to_screen_x, direction, direction)?,
        shader_direction_dot_range(uniform.world_to_screen_y, direction, direction)?,
    ];
    let tangent = stroke_safe_unit_range(projected)?;
    let normal = [range_negate(tangent[1]), tangent[0]];
    let horizontal = range_abs(projected[0]);
    let vertical = range_abs(projected[1]);
    let maximum_inset = rounded_f32_product_range(
        (horizontal.0.max(vertical.0), horizontal.1.max(vertical.1)),
        (0.25, 0.25),
    )?;
    let requested = f64::from(vertex.tangent_distance);
    let inset = if requested < 0.0 {
        (
            -(-requested).min(maximum_inset.1),
            -(-requested).min(maximum_inset.0),
        )
    } else {
        (
            requested.min(maximum_inset.0),
            requested.min(maximum_inset.1),
        )
    };
    let normal_distance = if vertex.stroke_role == -2.0 {
        let offset = Vec2::new(vertex.world_offset[0], vertex.world_offset[1]);
        let projected_offset = [
            shader_direction_dot_range(uniform.world_to_screen_x, offset, offset)?,
            shader_direction_dot_range(uniform.world_to_screen_y, offset, offset)?,
        ];
        range_dot(projected_offset, normal)?
    } else {
        (
            f64::from(vertex.normal_distance),
            f64::from(vertex.normal_distance),
        )
    };
    let screen = range_vector_add(
        range_vector_add(base, range_vector_scale(normal, normal_distance)?)?,
        range_vector_scale(tangent, inset)?,
    )?;
    let offset = [
        (
            f64::from(vertex.screen_offset[0]),
            f64::from(vertex.screen_offset[0]),
        ),
        (
            f64::from(vertex.screen_offset[1]),
            f64::from(vertex.screen_offset[1]),
        ),
    ];
    Some((range_vector_add(screen, offset)?, false))
}

#[cfg(test)]
#[path = "exact_markers_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) use tests::assert_gpu_pixels;
