//! Surface rasterization policy and source-attributed numerical validation.

use super::*;

/// Selects the portability contract for filled 3D surfaces, not display edges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceRasterization3d {
    /// Proves clip classification and projected orientation across the portable
    /// shader arithmetic envelope, using canonical CPU clipping for crossings.
    /// Ambiguous and edge-on triangles return a source-attributed error.
    #[default]
    StrictPortable,
    /// Submits original indexed surfaces for native hardware clipping and
    /// rasterization. Finite shader arithmetic is still validated, but grazing,
    /// edge-on and camera-plane surfaces need no CPU topology proof.
    ///
    /// Coverage at numerical boundaries may differ across adapters. No uncertain
    /// source triangles are dropped by the CPU. Hardware-clipped/zero-area
    /// primitives may produce no fragments; preflight reports submissions, not
    /// visibility, and does not claim CPU classification counts in this mode.
    /// Mathematical display edges retain their independent strict validation.
    Native,
}

/// Specific cause of failure for an original source surface triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Mesh3dSurfaceError {
    /// An active Lambert source normal loses a provably usable transformed direction.
    NormalTransform,
    /// Active fog camera-forward world-distance arithmetic leaves the finite envelope.
    FogArithmetic,
    /// Model/camera products or sums leave the finite portable shader envelope.
    TransformArithmetic,
    /// A vertex's side of a clipping plane cannot be proved consistently.
    ClipPlaneClassification,
    /// A clip intersection or its interpolated attributes cannot be proved safe.
    ClipIntersection,
    /// Rounded coincident endpoints cannot be proved to be the same vertex.
    ClippedVertexIdentity,
    /// The fixed clipping polygon capacity would be exceeded.
    ClippedPolygonCapacity,
    /// A nonempty clipped polygon has fewer than three proven vertices.
    ClippedPolygonDegenerate,
    /// A surviving vertex cannot be safely divided by its homogeneous W.
    PerspectiveDivide,
    /// A generated vertex cannot be proved inside the complete clip volume.
    ClippedVertexOutside,
    /// The projected signed area cannot be proved to have a nonzero sign.
    ProjectedOrientation,
}

impl Mesh3dSurfaceError {
    #[cfg(test)]
    pub(super) fn legacy(self) -> Mesh3dRenderError {
        match self {
            Self::TransformArithmetic => Mesh3dRenderError::InvalidGeometryTransform,
            _ => Mesh3dRenderError::UnportableSurfaceTopology,
        }
    }

    pub(super) fn for_triangle(
        self,
        object_id: Object3dId,
        triangle_index: usize,
    ) -> Mesh3dRenderError {
        Mesh3dRenderError::ObjectFailure {
            object_id,
            reason: Mesh3dObjectError::SurfaceTriangle {
                triangle_index,
                reason: self,
            },
        }
    }
}

pub(super) fn validate_points_for_policy(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
    policy: SurfaceRasterization3d,
    object_id: Object3dId,
) -> Result<(), Mesh3dRenderError> {
    for (vertex_index, vertex) in mesh.vertices().iter().enumerate() {
        let result = validate_point_for_policy(*vertex, model_rows, camera_rows, policy);
        if let Err(reason) = result {
            // Only failures search for a source owner; the ordinary path does
            // not allocate an adjacency map or validate shared vertices twice.
            return Err(
                match mesh
                    .triangle_indices()
                    .iter()
                    .position(|index| *index as usize == vertex_index)
                {
                    Some(index) => reason.for_triangle(object_id, index / 3),
                    None => Mesh3dRenderError::ObjectFailure {
                        object_id,
                        reason: Mesh3dObjectError::Vertex {
                            vertex_index,
                            reason,
                        },
                    },
                },
            );
        }
    }
    Ok(())
}

pub(super) fn validate_point_for_policy(
    vertex: Vec3,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
    policy: SurfaceRasterization3d,
) -> Result<(), Mesh3dSurfaceError> {
    let clip = shader_clip_point_ranges(vertex, model_rows, camera_rows)
        .map_err(|_| Mesh3dSurfaceError::TransformArithmetic)?;
    if policy == SurfaceRasterization3d::StrictPortable {
        validate_clip_classification(clip)
            .map_err(|_| Mesh3dSurfaceError::ClipPlaneClassification)?;
    }
    Ok(())
}
