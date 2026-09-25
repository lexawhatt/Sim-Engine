use super::super::geometry::PositionProofs;
use super::super::{Vertex, clip_triangle_has_stable_signed_area, clip_triangle_is_wholly_outside};
use super::{CameraUniform, ScreenPrimitiveGpu};

pub(super) fn instances_are_safe_accelerated(
    instances: &[ScreenPrimitiveGpu],
    certificate: super::certificate::Certificate,
    uniform: CameraUniform,
) -> bool {
    if certificate.accepts(uniform) {
        return true;
    }
    if instances.is_empty() {
        return instances_are_safe(instances, uniform);
    }
    // A nearly axis-aligned edge must not force every unrelated quad through
    // the interval kernel. Keep the same independent fallback per uncertain
    // primitive without allocating a parallel exception list.
    instances.iter().all(|instance| {
        let one = std::slice::from_ref(instance);
        super::certificate::Certificate::new(one).accepts(uniform)
            || instances_are_safe(one, uniform)
    })
}

pub(super) fn instances_are_safe(instances: &[ScreenPrimitiveGpu], uniform: CameraUniform) -> bool {
    if !uniform.sources_are_portable()
        || uniform.world_to_screen_x[..3] != [1.0, 0.0, 0.0]
        || uniform.world_to_screen_y[..3] != [0.0, -1.0, 0.0]
    {
        return false;
    }
    let mut positions = PositionProofs::new(uniform);
    for instance in instances {
        let mut clip = [[(0.0, 0.0); 2]; 4];
        for (index, corner) in instance.corners.iter().enumerate() {
            let vertex = Vertex {
                world_position: instance.anchor,
                depth: 0.0,
                world_offset: *corner,
                screen_offset: [0.0; 2],
                previous_direction: [0.0; 2],
                next_direction: [0.0; 2],
                normal_distance: 0.0,
                tangent_distance: 0.0,
                miter_limit: 1.0,
                stroke_role: 0.0,
                stroke_parameter: 0.0,
                color: instance.color,
            };
            let Some(ranges) = positions.clip(vertex) else {
                return false;
            };
            clip[index] = ranges;
        }
        for indices in [[0, 1, 2], [0, 2, 3]] {
            let triangle = indices.map(|index| clip[index]);
            if !clip_triangle_is_wholly_outside(triangle)
                && !clip_triangle_has_stable_signed_area(triangle, f64::from(f32::MIN_POSITIVE))
            {
                return false;
            }
        }
    }
    true
}
