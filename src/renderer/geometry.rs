//! Exact, bounded reuse of CPU position proofs within one validation call.

use super::*;

type ClipRanges = [(f64, f64); 2];

#[derive(Clone, Copy, PartialEq, Eq)]
struct PositionKey([u32; 7]);

impl PositionKey {
    fn new(vertex: Vertex) -> Self {
        // Only these fields participate in tessellated_vertex_clip_ranges.
        // Bit equality intentionally distinguishes even signed zero; no fuzzy
        // position matching or GPU approximation is used as a cache key.
        Self([
            vertex.world_position[0].to_bits(),
            vertex.world_position[1].to_bits(),
            vertex.world_offset[0].to_bits(),
            vertex.world_offset[1].to_bits(),
            vertex.screen_offset[0].to_bits(),
            vertex.screen_offset[1].to_bits(),
            vertex.depth.to_bits(),
        ])
    }
}

struct PositionProofs {
    uniform: CameraUniform,
    entries: [Option<(PositionKey, ClipRanges)>; 8],
    next: usize,
}

impl PositionProofs {
    fn new(uniform: CameraUniform) -> Self {
        Self {
            uniform,
            entries: [None; 8],
            next: 0,
        }
    }

    fn clip(&mut self, vertex: Vertex) -> Option<ClipRanges> {
        let key = PositionKey::new(vertex);
        if let Some((_, ranges)) = self
            .entries
            .iter()
            .flatten()
            .find(|(cached, _)| *cached == key)
        {
            return Some(*ranges);
        }
        let ranges = tessellated_vertex_clip_ranges(vertex, self.uniform)?;
        self.entries[self.next] = Some((key, ranges));
        self.next = (self.next + 1) % self.entries.len();
        Some(ranges)
    }
}

pub(super) fn tessellated_topology_is_portable(
    vertices: &[Vertex],
    uniform: CameraUniform,
) -> bool {
    if !vertices.len().is_multiple_of(3) {
        return false;
    }
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    let mut positions = PositionProofs::new(uniform);
    for triangle in vertices.chunks_exact(3) {
        let has_shader_extrusion = !triangle.iter().all(|vertex| {
            vertex.normal_distance == 0.0
                && vertex.tangent_distance == 0.0
                && vertex.stroke_role >= 0.0
        });
        if has_shader_extrusion {
            // Preserve the separate center/clip proof for extruded triangles,
            // including inactive candidates which can skip their area proof.
            // Plain triangles below already perform this identical projection.
            if triangle
                .iter()
                .any(|vertex| positions.clip(*vertex).is_none())
            {
                return false;
            }
            let Some(first) = logical_stroke_vertex_screen_ranges(triangle[0], uniform) else {
                return false;
            };
            let Some(second) = logical_stroke_vertex_screen_ranges(triangle[1], uniform) else {
                return false;
            };
            let Some(third) = logical_stroke_vertex_screen_ranges(triangle[2], uniform) else {
                return false;
            };
            if [first, second, third].iter().all(|output| output.1)
                && triangle_position_sources_equal(triangle)
            {
                continue;
            }
            let Some(first) = screen_ranges_to_clip(first.0, uniform) else {
                return false;
            };
            let Some(second) = screen_ranges_to_clip(second.0, uniform) else {
                return false;
            };
            let Some(third) = screen_ranges_to_clip(third.0, uniform) else {
                return false;
            };
            let clip = [first, second, third];
            if !clip_triangle_is_wholly_outside(clip)
                && !clip_triangle_has_stable_signed_area(clip, minimum_normal)
            {
                return false;
            }
            continue;
        }
        let Some(first) = positions.clip(triangle[0]) else {
            return false;
        };
        let Some(second) = positions.clip(triangle[1]) else {
            return false;
        };
        let Some(third) = positions.clip(triangle[2]) else {
            return false;
        };
        let clip = [first, second, third];
        if !clip_triangle_is_wholly_outside(clip)
            && !clip_triangle_has_stable_signed_area(clip, minimum_normal)
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
pub(super) fn legacy_is_safe_for(
    extents: GeometryExtents,
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    geometry_sources_are_portable(source)
        && match source {
            GeometryValidationSource::Tessellated(vertices) => {
                geometry_vertex_centers_are_portable(source, uniform)
                    && tessellated_triangle_topology_is_portable(vertices, uniform)
                    && vertices
                        .iter()
                        .all(|vertex| logical_stroke_branches_are_stable(*vertex, uniform))
            }
            GeometryValidationSource::Dynamic(vertices) => {
                dynamic_triangle_topology_is_portable(vertices, uniform)
            }
        }
        && uniform.sources_are_portable()
        && extents.is_safe_for(uniform)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
