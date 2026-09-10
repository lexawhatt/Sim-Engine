//! Bounded homogeneous surface clipping and its authoritative frame preflight.

use super::*;

#[cfg(test)]
#[path = "mesh3d_surface_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "mesh3d_surface_red_tests.rs"]
mod red_tests;

#[cfg(test)]
pub(super) use gpu_tests::{
    assert_gpu_clipped_recovery, assert_gpu_surface_contract, read_pixels as test_read_pixels,
    target as test_target,
};

/// Limits additional clip-space surface topology generated during one 3D draw.
///
/// Retained, wholly inside objects need no generated vertices. A crossing
/// object's surfaces are transformed and clipped together so one draw preserves
/// their shared depth interpolation. Limits apply before GPU writes or clearing
/// the target, and may be zero to disallow generated topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mesh3dRenderBudget {
    max_generated_vertices: usize,
    max_generated_triangles: usize,
    max_generated_upload_bytes: usize,
}

impl Mesh3dRenderBudget {
    /// Sets exact generated surface vertex, triangle, and total upload-byte
    /// ceilings. Upload bytes include canonical display-edge endpoints.
    pub const fn new(vertices: usize, triangles: usize, upload_bytes: usize) -> Self {
        Self {
            max_generated_vertices: vertices,
            max_generated_triangles: triangles,
            max_generated_upload_bytes: upload_bytes,
        }
    }

    /// Returns the maximum generated clip-space vertices in one frame.
    pub const fn max_generated_vertices(self) -> usize {
        self.max_generated_vertices
    }

    /// Returns the maximum generated triangle-list primitives in one frame.
    pub const fn max_generated_triangles(self) -> usize {
        self.max_generated_triangles
    }

    /// Returns maximum bytes uploaded for generated surfaces and their edges.
    pub const fn max_generated_upload_bytes(self) -> usize {
        self.max_generated_upload_bytes
    }
}

impl Default for Mesh3dRenderBudget {
    fn default() -> Self {
        Self::new(1_048_575, 349_525, 16_777_200)
    }
}

/// Non-mutating preflight counts for the exact scene, camera, target and budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mesh3dPreflightReport {
    pub(super) generated_vertices: usize,
    pub(super) generated_triangles: usize,
    pub(super) generated_upload_bytes: usize,
    pub(super) generated_edges: usize,
    pub(super) clipped_source_triangles: usize,
    pub(super) discarded_source_triangles: usize,
}

impl Mesh3dPreflightReport {
    /// Returns uploaded clip-space vertices, including unchanged triangles in
    /// objects that also contain crossing triangles.
    pub const fn generated_vertex_count(self) -> usize {
        self.generated_vertices
    }

    /// Returns generated triangle-list primitives submitted to the GPU.
    pub const fn generated_triangle_count(self) -> usize {
        self.generated_triangles
    }

    /// Returns exact additional surface and display-edge upload bytes.
    pub const fn generated_upload_bytes(self) -> usize {
        self.generated_upload_bytes
    }

    /// Returns canonical clip-space display edges uploaded for crossing objects.
    /// Their two endpoints share the surface's CPU-defined transform so legal
    /// shader dot associations cannot shift edges away from their own faces.
    pub const fn generated_edge_count(self) -> usize {
        self.generated_edges
    }

    /// Returns original triangles that intersect the frustum boundary and
    /// produce a nonempty clipped polygon.
    pub const fn clipped_source_triangle_count(self) -> usize {
        self.clipped_source_triangles
    }

    /// Returns original triangles proven entirely outside the frustum.
    pub const fn discarded_source_triangle_count(self) -> usize {
        self.discarded_source_triangles
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SurfaceClipVertex {
    pub(super) clip: [f32; 4],
    pub(super) uv: [f32; 2],
}

impl SurfaceClipVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x4];
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
    const TEXTURED_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x4, 5 => Float32x2];
    pub(super) const TEXTURED_LAYOUT: wgpu::VertexBufferLayout<'static> =
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::TEXTURED_ATTRIBUTES,
        };
}

pub(super) struct SurfaceFrame {
    pub(super) report: Mesh3dPreflightReport,
    pub(super) vertices: Vec<SurfaceClipVertex>,
    pub(super) edges: Vec<SurfaceClipEdge>,
    pub(super) objects: Vec<SurfaceObject>,
}

pub(super) struct SurfaceObject {
    pub(super) generated: Option<std::ops::Range<u32>>,
    pub(super) model_rows: [[f32; 4]; 3],
    pub(super) generated_edges: Option<std::ops::Range<u32>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SurfaceClipEdge {
    pub(super) start: [f32; 4],
    pub(super) end: [f32; 4],
}

impl SurfaceClipEdge {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &Self::ATTRIBUTES,
    };
}

pub(super) fn preflight(
    renderer_identity: &Arc<()>,
    scene: &Scene3d,
    camera: Camera3dUniform,
    budget: Mesh3dRenderBudget,
    max_buffer_size: u64,
) -> Result<SurfaceFrame, Mesh3dRenderError> {
    let capacity_error = Mesh3dRenderError::GeneratedGeometryCapacityTooLarge;
    let visible_count = scene.visible_object_count();
    let object_bytes = visible_count
        .checked_mul(std::mem::size_of::<SurfaceObject>())
        .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
    if u64::try_from(object_bytes).map_or(true, |bytes| bytes > max_buffer_size) {
        return Err(Mesh3dRenderError::InstanceCapacityTooLarge);
    }
    let mut frame = SurfaceFrame {
        report: Mesh3dPreflightReport::default(),
        vertices: Vec::new(),
        edges: Vec::new(),
        objects: Vec::new(),
    };
    frame
        .objects
        .try_reserve_exact(visible_count)
        .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
    for instance in scene.instances().iter().filter(|instance| instance.visible) {
        validate_mesh_identity(renderer_identity, &instance.mesh)?;
        let object_result = (|| {
            let model = instance
                .transform
                .model_rows()
                .map_err(|_| Mesh3dRenderError::InvalidGeometryTransform)?;
            validate_shader_points(instance.mesh.source(), model, camera.rows())?;
            if let Some(style) = instance.wireframe() {
                validate_edge_projection(
                    instance.mesh.source(),
                    model,
                    camera.rows(),
                    style,
                    camera.viewport,
                )?;
            }
            let mut crossing = false;
            let mut vertex_count = 0usize;
            if instance.style.surface_style().is_some() {
                classify_surface(
                    instance.mesh.source(),
                    model,
                    camera.rows(),
                    |_, vertices, clipped| {
                        crossing |= clipped;
                        vertex_count = vertex_count
                            .checked_add(vertices.len())
                            .ok_or(capacity_error)?;
                        if vertices.is_empty() {
                            frame.report.discarded_source_triangles = frame
                                .report
                                .discarded_source_triangles
                                .checked_add(1)
                                .ok_or(capacity_error)?;
                        } else if clipped {
                            frame.report.clipped_source_triangles = frame
                                .report
                                .clipped_source_triangles
                                .checked_add(1)
                                .ok_or(capacity_error)?;
                        }
                        Ok(())
                    },
                )?;
            }
            if crossing {
                let start = frame.report.generated_vertices;
                let count = start.checked_add(vertex_count).ok_or(capacity_error)?;
                let vertex_bytes = count
                    .checked_mul(std::mem::size_of::<SurfaceClipVertex>())
                    .ok_or(capacity_error)?;
                let edge_start = frame.report.generated_edges;
                let edge_count = if instance.wireframe().is_some() {
                    instance.mesh.source().display_edges().len()
                } else {
                    0
                };
                let edge_end = edge_start.checked_add(edge_count).ok_or(capacity_error)?;
                let edge_bytes = edge_end
                    .checked_mul(std::mem::size_of::<SurfaceClipEdge>())
                    .ok_or(capacity_error)?;
                let bytes = vertex_bytes.checked_add(edge_bytes).ok_or(capacity_error)?;
                if count > budget.max_generated_vertices
                    || count / 3 > budget.max_generated_triangles
                    || bytes > budget.max_generated_upload_bytes
                    || u64::try_from(vertex_bytes.max(edge_bytes))
                        .map_or(true, |bytes| bytes > max_buffer_size)
                    || count > u32::MAX as usize
                    || edge_end > u32::MAX as usize
                {
                    return Err(capacity_error);
                }
                frame.report.generated_vertices = count;
                frame.report.generated_triangles = count / 3;
                frame.report.generated_upload_bytes = bytes;
                frame.report.generated_edges = edge_end;
                frame.objects.push(SurfaceObject {
                    generated: Some(start as u32..count as u32),
                    model_rows: model,
                    generated_edges: Some(edge_start as u32..edge_end as u32),
                });
            } else {
                frame.objects.push(SurfaceObject {
                    generated: None,
                    model_rows: model,
                    generated_edges: None,
                });
            }
            Ok(())
        })();
        object_result.map_err(|error: Mesh3dRenderError| error.for_object(instance.id))?;
    }
    // All source proofs and exact aggregate capacities succeeded. This private
    // scratch is discarded on failure and never publishes a partial target.
    frame
        .vertices
        .try_reserve_exact(frame.report.generated_vertices)
        .map_err(|_| capacity_error)?;
    frame
        .edges
        .try_reserve_exact(frame.report.generated_edges)
        .map_err(|_| capacity_error)?;
    for (instance, object) in scene
        .instances()
        .iter()
        .filter(|instance| instance.visible)
        .zip(&frame.objects)
    {
        if object.generated.is_none() {
            continue;
        }
        classify_surface(
            instance.mesh.source(),
            object.model_rows,
            camera.rows(),
            |_, vertices, _| {
                frame.vertices.extend_from_slice(vertices);
                Ok(())
            },
        )
        .map_err(|error| error.for_object(instance.id))?;
        if instance.wireframe().is_some() {
            for edge in instance.mesh.source().display_edges() {
                let endpoints = [edge.start(), edge.end()];
                let mut clips = [[0.0; 4]; 2];
                for (clip, vertex) in clips.iter_mut().zip(endpoints) {
                    *clip = shader_clip_point_ranges(
                        instance.mesh.source().vertices()[vertex as usize],
                        object.model_rows,
                        camera.rows(),
                    )
                    .map_err(|error| error.for_object(instance.id))?
                    .map(|component| component.fixed);
                }
                frame.edges.push(SurfaceClipEdge {
                    start: clips[0],
                    end: clips[1],
                });
            }
        }
    }
    Ok(frame)
}

// A triangle intersected with six convex half-spaces has at most nine vertices.
const MAX_POLYGON_VERTICES: usize = 9;

#[derive(Clone, Copy)]
struct ClipVertex {
    ranges: [ShaderValueRange; 4],
    planes: u8,
    provenance: u8,
    uv: [f32; 2],
}

impl ClipVertex {
    fn same_proven_vertex(self, other: Self) -> bool {
        self.provenance == other.provenance
            && self.planes == other.planes
            && self.uv == other.uv
            && self
                .ranges
                .into_iter()
                .zip(other.ranges)
                .all(|(left, right)| {
                    left.fixed == right.fixed
                        && left.minimum == right.minimum
                        && left.maximum == right.maximum
                })
    }
    fn plane_range(self, plane: usize) -> Result<(f64, f64), Mesh3dRenderError> {
        if self.planes & (1 << plane) != 0 {
            Ok((0.0, 0.0))
        } else {
            clip_plane_ranges(self.ranges).map(|ranges| ranges[plane])
        }
    }

    fn plane_value(self, plane: usize) -> f64 {
        if self.planes & (1 << plane) != 0 {
            return 0.0;
        }
        let [x, y, z, w] = self.ranges.map(|range| f64::from(range.fixed));
        [w + x, w - x, w + y, w - y, z, w - z][plane]
    }
}

#[derive(Clone, Copy)]
struct ClippedTriangle {
    vertices: [ClipVertex; MAX_POLYGON_VERTICES],
    count: usize,
    crossing: bool,
}

fn inside(vertex: ClipVertex, plane: usize) -> Result<bool, Mesh3dRenderError> {
    let range = vertex.plane_range(plane)?;
    if range.0 >= 0.0 {
        Ok(true)
    } else if range.1 <= -f64::from(f32::MIN_POSITIVE) {
        Ok(false)
    } else {
        Err(Mesh3dRenderError::UnportableSurfaceTopology)
    }
}

fn intersection(
    start: ClipVertex,
    end: ClipVertex,
    plane: usize,
    provenance: u8,
) -> Result<ClipVertex, Mesh3dRenderError> {
    let error = Mesh3dRenderError::UnportableSurfaceTopology;
    let start_distance = start.plane_range(plane)?;
    let end_distance = end.plane_range(plane)?;
    if start_distance == (0.0, 0.0) {
        return Ok(start);
    }
    if end_distance == (0.0, 0.0) {
        return Ok(end);
    }
    let denominator = rounded_f32_add_range(start_distance, end_distance, true).ok_or(error)?;
    let amount = wgsl_signed_division_range(start_distance, denominator).ok_or(error)?;
    if amount.0 < 0.0 || amount.1 > 1.0 {
        return Err(error);
    }
    // The emitted position has a single CPU-defined representation. Intervals
    // still prove topology for all legal transform associations, so baking the
    // position is not permission to accept an ambiguous source triangle.
    let distance = start.plane_value(plane);
    let fixed_amount = distance / (distance - end.plane_value(plane));
    let mut ranges = [ShaderValueRange::exact(0.0); 4];
    for (axis, range) in ranges.iter_mut().enumerate() {
        let left = start.ranges[axis];
        let right = end.ranges[axis];
        let (minimum, maximum) = interval_lerp_range(
            (left.minimum, left.maximum),
            (right.minimum, right.maximum),
            amount,
        )
        .ok_or(error)?;
        let fixed = (f64::from(left.fixed)
            + (f64::from(right.fixed) - f64::from(left.fixed)) * fixed_amount)
            as f32;
        if !is_portable_shader_source(fixed) {
            return Err(error);
        }
        *range = ShaderValueRange {
            fixed,
            minimum: minimum.min(f64::from(fixed)),
            maximum: maximum.max(f64::from(fixed)),
        };
    }
    let planes = (start.planes & end.planes) | (1 << plane);
    let uv = std::array::from_fn(|axis| {
        (f64::from(start.uv[axis])
            + (f64::from(end.uv[axis]) - f64::from(start.uv[axis])) * fixed_amount) as f32
    });
    if uv
        .into_iter()
        .any(|value| !is_portable_shader_source(value) || !(0.0..=1.0).contains(&value))
    {
        return Err(error);
    }
    // Preserve the exact plane relation, including corners shared by two
    // previously clipped planes. This prevents a second hardware clip caused
    // only by independently rounding x and w at the boundary.
    for boundary in 0..6 {
        if planes & (1 << boundary) == 0 {
            continue;
        }
        let w = ranges[3];
        let (axis, negative) = match boundary {
            0 => (0, true),
            1 => (0, false),
            2 => (1, true),
            3 => (1, false),
            4 => {
                ranges[2] = ShaderValueRange::exact(0.0);
                continue;
            }
            _ => (2, false),
        };
        ranges[axis] = if negative {
            ShaderValueRange {
                fixed: -w.fixed,
                minimum: -w.maximum,
                maximum: -w.minimum,
            }
        } else {
            w
        };
    }
    Ok(ClipVertex {
        ranges,
        planes,
        provenance,
        uv,
    })
}

#[cfg(test)]
fn clipped_triangle(
    clips: [[ShaderValueRange; 4]; 3],
) -> Result<ClippedTriangle, Mesh3dRenderError> {
    clipped_triangle_with_uv(clips, [[0.0; 2]; 3])
}

fn clipped_triangle_with_uv(
    clips: [[ShaderValueRange; 4]; 3],
    uv: [[f32; 2]; 3],
) -> Result<ClippedTriangle, Mesh3dRenderError> {
    let empty = ClipVertex {
        ranges: [ShaderValueRange::exact(0.0); 4],
        planes: 0,
        provenance: 0,
        uv: [0.0; 2],
    };
    let mut result = ClippedTriangle {
        vertices: [empty; MAX_POLYGON_VERTICES],
        count: 3,
        crossing: false,
    };
    for (provenance, (destination, ranges)) in result.vertices.iter_mut().zip(clips).enumerate() {
        let plane_ranges = clip_plane_ranges(ranges)?;
        let planes = plane_ranges
            .iter()
            .enumerate()
            .fold(0, |mask, (plane, range)| {
                mask | if *range == (0.0, 0.0) { 1 << plane } else { 0 }
            });
        *destination = ClipVertex {
            ranges,
            planes,
            provenance: provenance as u8,
            uv: uv[provenance],
        };
    }
    // Common-plane rejection does not need a projected orientation proof.
    let mut next_provenance = 3u8;
    for plane in 0..6 {
        if result.vertices[..3]
            .iter()
            .all(|vertex| vertex.plane_range(plane).is_ok_and(|range| range.1 < 0.0))
        {
            result.count = 0;
            return Ok(result);
        }
    }
    for plane in 0..6 {
        if result.count == 0 {
            break;
        }
        let input = result;
        result.count = 0;
        let mut previous = input.vertices[input.count - 1];
        let mut previous_inside = inside(previous, plane)?;
        for current in input.vertices[..input.count].iter().copied() {
            let current_inside = inside(current, plane)?;
            let intersection = if previous_inside != current_inside {
                let intersection = intersection(previous, current, plane, next_provenance)?;
                next_provenance += 1;
                Some(intersection)
            } else {
                None
            };
            result.crossing |= !current_inside;
            for vertex in intersection
                .into_iter()
                .chain(current_inside.then_some(current))
            {
                if result.count > 0
                    && result.vertices[result.count - 1]
                        .ranges
                        .map(|value| value.fixed)
                        == vertex.ranges.map(|value| value.fixed)
                {
                    // Only identical, proven boundary endpoints may collapse.
                    // An uncertainty envelope is never discarded as zero area.
                    if !result.vertices[result.count - 1].same_proven_vertex(vertex) {
                        return Err(Mesh3dRenderError::UnportableSurfaceTopology);
                    }
                    continue;
                }
                if result.count == MAX_POLYGON_VERTICES {
                    return Err(Mesh3dRenderError::UnportableSurfaceTopology);
                }
                result.vertices[result.count] = vertex;
                result.count += 1;
            }
            previous = current;
            previous_inside = current_inside;
        }
        if result.count > 1
            && result.vertices[0].ranges.map(|value| value.fixed)
                == result.vertices[result.count - 1]
                    .ranges
                    .map(|value| value.fixed)
        {
            if !result.vertices[0].same_proven_vertex(result.vertices[result.count - 1]) {
                return Err(Mesh3dRenderError::UnportableSurfaceTopology);
            }
            result.count -= 1;
        }
    }
    if result.count > 0 && result.count < 3 {
        return Err(Mesh3dRenderError::UnportableSurfaceTopology);
    }
    for index in 1..result.count.saturating_sub(1) {
        let vertices = [
            result.vertices[0],
            result.vertices[index],
            result.vertices[index + 1],
        ];
        for vertex in vertices {
            if vertex.ranges[3].minimum < f64::from(f32::MIN_POSITIVE) {
                return Err(Mesh3dRenderError::UnportableSurfaceTopology);
            }
            for plane in 0..6 {
                if !inside(vertex, plane)? || vertex.plane_value(plane) < 0.0 {
                    return Err(Mesh3dRenderError::UnportableSurfaceTopology);
                }
            }
        }
        validate_projected_triangle_orientation(vertices.map(|vertex| vertex.ranges))?;
    }
    Ok(result)
}

pub(super) fn classify_surface(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
    mut visit: impl FnMut(usize, &[SurfaceClipVertex], bool) -> Result<(), Mesh3dRenderError>,
) -> Result<(), Mesh3dRenderError> {
    for (index, triangle) in mesh.triangle_indices().chunks_exact(3).enumerate() {
        let mut clips = [[ShaderValueRange::exact(0.0); 4]; 3];
        for (destination, vertex) in clips.iter_mut().zip(triangle) {
            *destination = shader_clip_point_ranges(
                mesh.vertices()[*vertex as usize],
                model_rows,
                camera_rows,
            )?;
        }
        let uv = std::array::from_fn(|index| {
            mesh.texture_coordinates()
                .get(triangle[index] as usize)
                .map_or([0.0, 0.0], |coordinate| [coordinate.u(), coordinate.v()])
        });
        let polygon = clipped_triangle_with_uv(clips, uv)?;
        let mut emitted = [SurfaceClipVertex {
            clip: [0.0; 4],
            uv: [0.0; 2],
        }; 21];
        let mut count = 0;
        for fan in 1..polygon.count.saturating_sub(1) {
            for vertex in [
                polygon.vertices[0],
                polygon.vertices[fan],
                polygon.vertices[fan + 1],
            ] {
                emitted[count] = SurfaceClipVertex {
                    clip: vertex.ranges.map(|range| range.fixed),
                    uv: vertex.uv,
                };
                count += 1;
            }
        }
        visit(index, &emitted[..count], polygon.crossing)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(points: [[f32; 4]; 3]) -> [[ShaderValueRange; 4]; 3] {
        points.map(|point| point.map(ShaderValueRange::exact))
    }

    #[test]
    fn surface_clips_every_frustum_plane_in_both_windings() {
        let base = [
            [-0.4, -0.3, 0.4, 1.0],
            [0.5, -0.2, 0.6, 1.0],
            [0.1, 0.5, 0.5, 1.0],
        ];
        for plane in 0..6 {
            let mut points = base;
            match plane {
                0 => points[0][0] = -2.0,
                1 => points[0][0] = 2.0,
                2 => points[0][1] = -2.0,
                3 => points[0][1] = 2.0,
                4 => points[0][2] = -0.5,
                _ => points[0][2] = 1.5,
            }
            for reverse in [false, true] {
                let mut input = points;
                if reverse {
                    input.swap(1, 2);
                }
                let result = clipped_triangle(exact(input))
                    .unwrap_or_else(|error| panic!("plane={plane} reverse={reverse}: {error}"));
                assert!(result.crossing);
                assert_eq!(result.count, 4);
                for vertex in &result.vertices[..result.count] {
                    assert!(vertex.ranges[3].fixed > 0.0);
                    for boundary in 0..6 {
                        assert!(vertex.plane_value(boundary) >= 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn surface_clipping_handles_camera_plane_and_multi_plane_corners() {
        for points in [
            [
                [-0.1, -0.2, -0.1, 0.0],
                [0.5, -0.2, 0.5, 1.0],
                [0.1, 0.5, 0.5, 1.0],
            ],
            [
                [-0.1, -0.2, -0.1, f32::MIN_POSITIVE],
                [0.5, -0.2, 0.5, 1.0],
                [0.1, 0.5, 0.5, 1.0],
            ],
            [
                [-2.0, -1.5, 0.5, 1.0],
                [2.0, -0.7, 0.5, 1.0],
                [0.1, 2.0, 0.5, 1.0],
            ],
        ] {
            let result = clipped_triangle(exact(points)).unwrap();
            assert!(result.crossing);
            assert!(result.count >= 3 && result.count <= MAX_POLYGON_VERTICES);
        }
    }

    #[test]
    fn surface_clipping_keeps_ambiguous_and_edge_on_rejections() {
        let edge_on = exact([
            [-0.5, 0.0, 0.2, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.5, 0.0, 0.8, 1.0],
        ]);
        assert!(matches!(
            clipped_triangle(edge_on),
            Err(Mesh3dRenderError::UnportableSurfaceTopology)
        ));
        let mut grazing = exact([
            [-1.0, -0.3, 0.4, 1.0],
            [0.5, -0.2, 0.6, 1.0],
            [0.1, 0.5, 0.5, 1.0],
        ]);
        grazing[0][0].minimum = -1.000001;
        grazing[0][0].maximum = -0.999999;
        assert!(matches!(
            clipped_triangle(grazing),
            Err(Mesh3dRenderError::UnportableSurfaceTopology)
        ));
        let outside = exact([
            [-2.0, 0.0, 0.5, 1.0],
            [-2.5, 0.0, 0.5, 1.0],
            [-3.0, 0.0, 0.5, 1.0],
        ]);
        assert_eq!(clipped_triangle(outside).unwrap().count, 0);
    }
}
