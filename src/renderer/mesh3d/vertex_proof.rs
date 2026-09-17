//! Per-vertex clip classification and projected triangle orientation.

use super::{
    Mesh3dRenderError, ShaderValueRange, Vec3, rounded_f32_add_range, rounded_f32_product_range,
    shader_dot_range, wgsl_division_range,
};

#[cfg(test)]
use super::{Mesh3d, surface};

#[cfg(test)]
pub(super) fn validate_shader_transform(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    validate_shader_points(mesh, model_rows, camera_rows)?;
    validate_surface_triangle_topology(mesh, model_rows, camera_rows)
}

#[cfg(test)]
pub(super) fn validate_shader_points(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    // Validate actual retained coordinates. An AABB-only proof cannot exclude
    // an interior source product entering the backend-dependent subnormal
    // domain, even when every synthetic corner has a stable clip side.
    for vertex in mesh.vertices() {
        validate_shader_point(*vertex, model_rows, camera_rows)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn validate_surface_triangle_topology(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    surface::classify_surface(mesh, model_rows, camera_rows, |_, _, _| Ok(()))
}

pub(super) fn validate_projected_triangle_orientation(
    clips: [[ShaderValueRange; 4]; 3],
) -> Result<(), Mesh3dRenderError> {
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    let ndc = clips.map(|clip| {
        let denominator = (clip[3].minimum, clip[3].maximum);
        [
            wgsl_division_range((clip[0].minimum, clip[0].maximum), denominator),
            wgsl_division_range((clip[1].minimum, clip[1].maximum), denominator),
        ]
    });
    let ndc = [
        [
            ndc[0][0].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
            ndc[0][1].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
        ],
        [
            ndc[1][0].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
            ndc[1][1].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
        ],
        [
            ndc[2][0].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
            ndc[2][1].ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?,
        ],
    ];
    let first_x = rounded_f32_add_range(ndc[1][0], ndc[0][0], true)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let first_y = rounded_f32_add_range(ndc[2][1], ndc[0][1], true)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let second_y = rounded_f32_add_range(ndc[1][1], ndc[0][1], true)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let second_x = rounded_f32_add_range(ndc[2][0], ndc[0][0], true)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let positive = rounded_f32_product_range(first_x, first_y)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let negative = rounded_f32_product_range(second_y, second_x)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let signed_area = rounded_f32_add_range(positive, negative, true)
        .ok_or(Mesh3dRenderError::UnportableSurfaceTopology)?;
    let stable_area = signed_area.0 >= minimum_normal || signed_area.1 <= -minimum_normal;
    if !stable_area {
        return Err(Mesh3dRenderError::UnportableSurfaceTopology);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn validate_shader_point(
    point: Vec3,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    validate_clip_classification(shader_clip_point_ranges(point, model_rows, camera_rows)?)
}

pub(super) fn shader_clip_point_ranges(
    point: Vec3,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<[ShaderValueRange; 4], Mesh3dRenderError> {
    let model_point = [point.x(), point.y(), point.z(), 1.0].map(ShaderValueRange::exact);
    let world = [
        shader_dot_range(model_rows[0], model_point)?,
        shader_dot_range(model_rows[1], model_point)?,
        shader_dot_range(model_rows[2], model_point)?,
        ShaderValueRange::exact(1.0),
    ];
    Ok([
        shader_dot_range(camera_rows[0], world)?,
        shader_dot_range(camera_rows[1], world)?,
        shader_dot_range(camera_rows[2], world)?,
        shader_dot_range(camera_rows[3], world)?,
    ])
}

pub(super) fn validate_clip_classification(
    clip: [ShaderValueRange; 4],
) -> Result<(), Mesh3dRenderError> {
    let plane_ranges = clip_plane_ranges(clip)?;
    // Rasterization is portable only when no critical plane classification
    // changes across legal dot associations and the vertex is either always
    // inside or has at least one common outside plane. A crossing range can
    // otherwise turn the same retained triangle from visible to clipped on
    // another conforming backend.
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    let stable_planes = plane_ranges
        .iter()
        .all(|range| range.0 >= 0.0 || range.1 <= -minimum_normal);
    let always_inside =
        plane_ranges.iter().all(|range| range.0 >= 0.0) && clip[3].minimum >= minimum_normal;
    let always_outside = plane_ranges.iter().any(|range| range.1 <= -minimum_normal);
    (stable_planes && (always_inside || always_outside))
        .then_some(())
        .ok_or(Mesh3dRenderError::InvalidGeometryTransform)
}

pub(super) fn clip_plane_ranges(
    clip: [ShaderValueRange; 4],
) -> Result<[(f64, f64); 6], Mesh3dRenderError> {
    let x = clip[0];
    let y = clip[1];
    let z = clip[2];
    let w = clip[3];
    // These are sign envelopes, not a mirror of a raw f32 add. The edge
    // shader first applies a common normal homogeneous scale and hardware
    // clipping is homogeneous as well; the plane-sign envelope is evaluated
    // separately from the bounded shader add used by edge normalization.
    let plane_ranges = [
        (w.minimum + x.minimum, w.maximum + x.maximum),
        (w.minimum - x.maximum, w.maximum - x.minimum),
        (w.minimum + y.minimum, w.maximum + y.maximum),
        (w.minimum - y.maximum, w.maximum - y.minimum),
        (z.minimum, z.maximum),
        (w.minimum - z.maximum, w.maximum - z.minimum),
    ];
    plane_ranges
        .iter()
        .all(|range| range.0.is_finite() && range.1.is_finite())
        .then_some(plane_ranges)
        .ok_or(Mesh3dRenderError::InvalidGeometryTransform)
}
