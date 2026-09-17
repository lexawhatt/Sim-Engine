use super::*;

mod interval;
#[cfg(test)]
pub(in crate::renderer) use interval::shader_interval_sum_is_safe;
use interval::{
    interval_products, is_nonzero_subnormal_f64, next_f32_down, next_f32_up,
    rounded_f32_division_range, rounded_f32_range, shader_interval_difference,
    shader_interval_product,
};
pub(in crate::renderer) use interval::{
    interval_products_f64, is_nonzero_subnormal, range_abs, range_dot, range_negate,
    range_vector_add, range_vector_scale, rounded_f32_add_range, rounded_f32_product_range,
    shader_interval_sum_range,
};

pub(in crate::renderer) fn center_axis(
    center: ((f64, f64), (f64, f64)),
    axis: usize,
) -> (f64, f64) {
    if axis == 0 { center.0 } else { center.1 }
}

impl CameraUniform {
    pub(in crate::renderer) fn new(camera: Camera2d, viewport: LogicalViewport) -> Option<Self> {
        Self::new_in_region(camera, viewport, Vec2::ZERO, viewport)
    }

    pub(in crate::renderer) fn new_in_region(
        camera: Camera2d,
        viewport: LogicalViewport,
        target_origin: Vec2,
        target_viewport: LogicalViewport,
    ) -> Option<Self> {
        let projection_cosine = camera.projection().tilt().cos();
        let rotation_cosine = camera.rotation().cos();
        let rotation_sine = camera.rotation().sin();
        let zoom = camera.zoom();
        let center = camera.center();
        let projection_sine = camera.projection().tilt().sin();
        let depth_scale = camera.projection().depth_scale();

        let horizontal_x = zoom * rotation_cosine;
        let horizontal_y = -zoom * rotation_sine * projection_cosine;
        let vertical_x = -zoom * rotation_sine;
        let vertical_y = -zoom * rotation_cosine * projection_cosine;
        let horizontal_depth =
            zoom * depth_scale * projection_sine * (rotation_cosine * 0.5 - rotation_sine);
        let vertical_depth =
            -zoom * depth_scale * projection_sine * (rotation_sine * 0.5 + rotation_cosine);

        let uniform = Self {
            camera_center: [center.x, center.y, 0.0, 0.0],
            world_to_screen_x: [
                horizontal_x,
                horizontal_y,
                horizontal_depth,
                target_origin.x + viewport.width() * 0.5,
            ],
            world_to_screen_y: [
                vertical_x,
                vertical_y,
                vertical_depth,
                target_origin.y + viewport.height() * 0.5,
            ],
            screen_to_clip: [
                2.0 / target_viewport.width(),
                -2.0 / target_viewport.height(),
                -1.0,
                1.0,
            ],
        };
        (uniform.is_finite() && uniform.sources_are_portable()).then_some(uniform)
    }

    pub(in crate::renderer) fn sources_are_portable(self) -> bool {
        self.camera_center
            .into_iter()
            .chain(self.world_to_screen_x)
            .chain(self.world_to_screen_y)
            .chain(self.screen_to_clip)
            .all(is_portable_shader_source)
    }

    fn is_finite(self) -> bool {
        self.camera_center
            .iter()
            .chain(self.world_to_screen_x.iter())
            .chain(self.world_to_screen_y.iter())
            .chain(self.screen_to_clip.iter())
            .all(|value| value.is_finite())
    }

    pub(in crate::renderer) fn world_to_screen(self, world: Vec2, depth: f32) -> Vec2 {
        let relative = world - Vec2::new(self.camera_center[0], self.camera_center[1]);
        Vec2::new(
            self.world_to_screen_x[0] * relative.x
                + self.world_to_screen_x[1] * relative.y
                + self.world_to_screen_x[2] * depth
                + self.world_to_screen_x[3],
            self.world_to_screen_y[0] * relative.x
                + self.world_to_screen_y[1] * relative.y
                + self.world_to_screen_y[2] * depth
                + self.world_to_screen_y[3],
        )
    }

    #[cfg(test)]
    pub(in crate::renderer) fn direction_to_screen(self, direction: Vec2) -> Vec2 {
        Vec2::new(
            self.world_to_screen_x[0] * direction.x + self.world_to_screen_x[1] * direction.y,
            self.world_to_screen_y[0] * direction.x + self.world_to_screen_y[1] * direction.y,
        )
    }
}

impl GeometryExtents {
    fn empty(is_empty: bool) -> Self {
        Self {
            world_min: Vec2::splat(f32::INFINITY),
            world_max: Vec2::splat(f32::NEG_INFINITY),
            world_offset_min: Vec2::splat(f32::INFINITY),
            world_offset_max: Vec2::splat(f32::NEG_INFINITY),
            depth_min: f32::INFINITY,
            depth_max: f32::NEG_INFINITY,
            direction_min: Vec2::splat(f32::INFINITY),
            direction_max: Vec2::splat(f32::NEG_INFINITY),
            screen_offset_max_abs: Vec2::ZERO,
            normal_distance_max_abs: 0.0,
            tangent_distance_max_abs: 0.0,
            miter_limit_max: 1.0,
            vertex_count: 0,
            empty: is_empty,
        }
    }

    fn include(&mut self, vertex: Vertex) {
        self.vertex_count += 1;
        let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
        self.world_min.x = self.world_min.x.min(world.x);
        self.world_min.y = self.world_min.y.min(world.y);
        self.world_max.x = self.world_max.x.max(world.x);
        self.world_max.y = self.world_max.y.max(world.y);
        let world_offset = Vec2::new(vertex.world_offset[0], vertex.world_offset[1]);
        self.world_offset_min.x = self.world_offset_min.x.min(world_offset.x);
        self.world_offset_min.y = self.world_offset_min.y.min(world_offset.y);
        self.world_offset_max.x = self.world_offset_max.x.max(world_offset.x);
        self.world_offset_max.y = self.world_offset_max.y.max(world_offset.y);
        self.depth_min = self.depth_min.min(vertex.depth);
        self.depth_max = self.depth_max.max(vertex.depth);

        for direction in [vertex.previous_direction, vertex.next_direction] {
            self.direction_min.x = self.direction_min.x.min(direction[0]);
            self.direction_min.y = self.direction_min.y.min(direction[1]);
            self.direction_max.x = self.direction_max.x.max(direction[0]);
            self.direction_max.y = self.direction_max.y.max(direction[1]);
        }

        self.screen_offset_max_abs.x = self
            .screen_offset_max_abs
            .x
            .max(vertex.screen_offset[0].abs());
        self.screen_offset_max_abs.y = self
            .screen_offset_max_abs
            .y
            .max(vertex.screen_offset[1].abs());
        self.normal_distance_max_abs = self
            .normal_distance_max_abs
            .max(vertex.normal_distance.abs());
        self.tangent_distance_max_abs = self
            .tangent_distance_max_abs
            .max(vertex.tangent_distance.abs());
        self.miter_limit_max = self.miter_limit_max.max(vertex.miter_limit);
    }

    fn include_dynamic(&mut self, vertex: DynamicGpu) {
        self.vertex_count += 1;
        let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
        self.world_min.x = self.world_min.x.min(world.x);
        self.world_min.y = self.world_min.y.min(world.y);
        self.world_max.x = self.world_max.x.max(world.x);
        self.world_max.y = self.world_max.y.max(world.y);
        self.world_offset_min = Vec2::ZERO;
        self.world_offset_max = Vec2::ZERO;
        self.depth_min = self.depth_min.min(vertex.depth);
        self.depth_max = self.depth_max.max(vertex.depth);
        self.direction_min = Vec2::ZERO;
        self.direction_max = Vec2::ZERO;
    }

    pub(in crate::renderer) fn from_dynamic_vertices(vertices: &[DynamicGpu]) -> Self {
        let mut extents = Self::empty(vertices.is_empty());
        for vertex in vertices {
            extents.include_dynamic(*vertex);
        }
        extents
    }

    pub(in crate::renderer) fn from_vertices(vertices: &[Vertex]) -> Self {
        let mut extents = Self::empty(vertices.is_empty());
        for vertex in vertices {
            extents.include(*vertex);
        }

        extents
    }

    pub(in crate::renderer) fn is_safe_for(self, uniform: CameraUniform) -> bool {
        if self.empty {
            return true;
        }
        let center = Vec2::new(uniform.camera_center[0], uniform.camera_center[1]);
        // Match the vertex shader's split-anchor operation order. Generated
        // local offsets are projected separately so they cannot disappear
        // when their world anchor is much larger than the local geometry.
        let Some(relative_horizontal) = shader_relative_component_bounds(
            self.world_min.x,
            self.world_max.x,
            center.x,
            0.0,
            0.0,
        ) else {
            return false;
        };
        let Some(relative_vertical) = shader_relative_component_bounds(
            self.world_min.y,
            self.world_max.y,
            center.y,
            0.0,
            0.0,
        ) else {
            return false;
        };
        let relative_minimum = [relative_horizontal.0, relative_vertical.0];
        let relative_maximum = [relative_horizontal.1, relative_vertical.1];
        let Some(projected_world_horizontal) = shader_world_dot_range(
            uniform.world_to_screen_x,
            relative_minimum,
            relative_maximum,
            self.depth_min,
            self.depth_max,
        ) else {
            return false;
        };
        let Some(projected_world_vertical) = shader_world_dot_range(
            uniform.world_to_screen_y,
            relative_minimum,
            relative_maximum,
            self.depth_min,
            self.depth_max,
        ) else {
            return false;
        };
        let Some(projected_offset_horizontal) = shader_direction_dot_range(
            uniform.world_to_screen_x,
            self.world_offset_min,
            self.world_offset_max,
        ) else {
            return false;
        };
        let Some(projected_offset_vertical) = shader_direction_dot_range(
            uniform.world_to_screen_y,
            self.world_offset_min,
            self.world_offset_max,
        ) else {
            return false;
        };
        let Some(world_horizontal) =
            shader_interval_sum_range([projected_world_horizontal, projected_offset_horizontal])
        else {
            return false;
        };
        let Some(world_vertical) =
            shader_interval_sum_range([projected_world_vertical, projected_offset_vertical])
        else {
            return false;
        };
        let Some(direction_horizontal) = shader_direction_dot_range(
            uniform.world_to_screen_x,
            self.direction_min,
            self.direction_max,
        ) else {
            return false;
        };
        let Some(direction_vertical) = shader_direction_dot_range(
            uniform.world_to_screen_y,
            self.direction_min,
            self.direction_max,
        ) else {
            return false;
        };
        if [
            self.screen_offset_max_abs.x,
            self.screen_offset_max_abs.y,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
        ]
        .into_iter()
        .any(is_nonzero_subnormal)
        {
            return false;
        }
        let Some(horizontal_bounds) = shader_stroke_screen_bounds(
            world_horizontal,
            self.screen_offset_max_abs.x,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
            self.miter_limit_max,
        ) else {
            return false;
        };
        let Some(vertical_bounds) = shader_stroke_screen_bounds(
            world_vertical,
            self.screen_offset_max_abs.y,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
            self.miter_limit_max,
        ) else {
            return false;
        };

        if !shader_clip_interval_is_safe(
            horizontal_bounds.0,
            horizontal_bounds.1,
            uniform.screen_to_clip[0],
            uniform.screen_to_clip[2],
        ) || !shader_clip_interval_is_safe(
            vertical_bounds.0,
            vertical_bounds.1,
            uniform.screen_to_clip[1],
            uniform.screen_to_clip[3],
        ) {
            return false;
        }

        [
            world_horizontal.0,
            world_horizontal.1,
            world_vertical.0,
            world_vertical.1,
            direction_horizontal.0,
            direction_horizontal.1,
            direction_vertical.0,
            direction_vertical.1,
            horizontal_bounds.0,
            horizontal_bounds.1,
            vertical_bounds.0,
            vertical_bounds.1,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value.abs() <= f32::MAX as f64)
    }
}

pub(in crate::renderer) fn shader_stroke_screen_bounds(
    world_screen: (f64, f64),
    screen_offset_max_abs: f32,
    normal_distance_max_abs: f32,
    tangent_distance_max_abs: f32,
    miter_limit_max: f32,
) -> Option<(f64, f64)> {
    // `safe_unit` and the miter path are backend-selected inverse-square-root
    // arithmetic. A component of a mathematically unit vector is at most one;
    // two is a deliberately loose portable bound that also contains its
    // permitted approximation error. Propagate the shader's two additions
    // separately so a tiny later clip scale cannot hide screen-space overflow.
    const UNIT_COMPONENT_BOUND: f64 = 2.0;
    let normal = f64::from(normal_distance_max_abs);
    let tangent = f64::from(tangent_distance_max_abs);
    let miter = normal * f64::from(miter_limit_max.max(1.0));
    let normal_component = UNIT_COMPONENT_BOUND * normal.max(miter);
    let tangent_component = UNIT_COMPONENT_BOUND * tangent;
    let extrusion = shader_interval_sum_range([
        (-normal_component, normal_component),
        (-tangent_component, tangent_component),
    ])?;
    let offset = f64::from(screen_offset_max_abs);
    shader_interval_sum_range([world_screen, extrusion, (-offset, offset)])
}

#[derive(Clone, Copy)]
pub(in crate::renderer) enum GeometryValidationSource<'a> {
    Tessellated(&'a [Vertex]),
    Dynamic(&'a [DynamicGpu]),
}

pub(in crate::renderer) const GEOMETRY_VALIDATION_CACHE_CAPACITY: usize = 8;

#[derive(Clone, Copy)]
pub(in crate::renderer) struct GeometryValidationCacheEntry {
    uniform: CameraUniform,
    safe: bool,
}

pub(in crate::renderer) struct GeometryValidationCacheState {
    pub(in crate::renderer) entries:
        [Option<GeometryValidationCacheEntry>; GEOMETRY_VALIDATION_CACHE_CAPACITY],
    next: usize,
}

impl Default for GeometryValidationCacheState {
    fn default() -> Self {
        Self {
            entries: [None; GEOMETRY_VALIDATION_CACHE_CAPACITY],
            next: 0,
        }
    }
}

#[derive(Default)]
pub(in crate::renderer) struct GeometryValidationCache {
    pub(in crate::renderer) state: Mutex<GeometryValidationCacheState>,
}

impl GeometryValidationCache {
    pub(in crate::renderer) fn validate(
        &self,
        extents: GeometryExtents,
        source: GeometryValidationSource<'_>,
        uniform: CameraUniform,
    ) -> bool {
        {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(entry) = state
                .entries
                .iter()
                .flatten()
                .find(|entry| entry.uniform == uniform)
            {
                return entry.safe;
            }
        }

        let safe = geometry_is_safe_for(extents, source, uniform);
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(entry) = state
            .entries
            .iter()
            .flatten()
            .find(|entry| entry.uniform == uniform)
        {
            return entry.safe;
        }
        let next = state.next;
        state.entries[next] = Some(GeometryValidationCacheEntry { uniform, safe });
        state.next = (next + 1) % GEOMETRY_VALIDATION_CACHE_CAPACITY;
        safe
    }

    pub(in crate::renderer) fn clear(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        *state = GeometryValidationCacheState::default();
    }
}

pub(in crate::renderer) fn geometry_is_safe_for_cached(
    cache: Option<&GeometryValidationCache>,
    extents: GeometryExtents,
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    cache.map_or_else(
        || geometry_is_safe_for(extents, source, uniform),
        |cache| cache.validate(extents, source, uniform),
    )
}

pub(in crate::renderer) fn geometry_is_safe_for(
    extents: GeometryExtents,
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    geometry_sources_are_portable(source)
        && match source {
            GeometryValidationSource::Tessellated(vertices) => {
                geometry::tessellated_topology_is_portable(vertices, uniform)
                    && vertices
                        .iter()
                        .all(|vertex| logical_stroke_branches_are_stable(*vertex, uniform))
            }
            GeometryValidationSource::Dynamic(vertices) => {
                // The triangle proof projects every vertex itself, so do not
                // duplicate the same per-vertex dot envelopes first.
                dynamic_triangle_topology_is_portable(vertices, uniform)
            }
        }
        && uniform.sources_are_portable()
        && extents.is_safe_for(uniform)
}

pub(in crate::renderer) fn dynamic_triangle_topology_is_portable(
    vertices: &[DynamicGpu],
    uniform: CameraUniform,
) -> bool {
    if !vertices.len().is_multiple_of(3) {
        return false;
    }
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    for triangle in vertices.chunks_exact(3) {
        let Some(first) = dynamic_vertex_clip_ranges(triangle[0], uniform) else {
            return false;
        };
        let Some(second) = dynamic_vertex_clip_ranges(triangle[1], uniform) else {
            return false;
        };
        let Some(third) = dynamic_vertex_clip_ranges(triangle[2], uniform) else {
            return false;
        };
        let clip = [first, second, third];
        if clip_triangle_is_wholly_outside(clip) {
            continue;
        }
        if !clip_triangle_is_wholly_inside(clip) {
            // v0.2 does not yet carry conservative interval polygons through
            // hardware clipping. Reject partially clipped dynamic triangles
            // rather than permit backend-selected topology.
            return false;
        }
        if !clip_triangle_has_stable_signed_area(clip, minimum_normal) {
            return false;
        }
    }
    true
}

// Frozen pre-optimization topology reference for differential regression tests.
#[cfg(test)]
pub(in crate::renderer) fn tessellated_triangle_topology_is_portable(
    vertices: &[Vertex],
    uniform: CameraUniform,
) -> bool {
    if !vertices.len().is_multiple_of(3) {
        return false;
    }
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    for triangle in vertices.chunks_exact(3) {
        let has_shader_extrusion = !triangle.iter().all(|vertex| {
            vertex.normal_distance == 0.0
                && vertex.tangent_distance == 0.0
                && vertex.stroke_role >= 0.0
        });
        if has_shader_extrusion {
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
        let Some(first) = tessellated_vertex_clip_ranges(triangle[0], uniform) else {
            return false;
        };
        let Some(second) = tessellated_vertex_clip_ranges(triangle[1], uniform) else {
            return false;
        };
        let Some(third) = tessellated_vertex_clip_ranges(triangle[2], uniform) else {
            return false;
        };
        let clip = [first, second, third];
        if clip_triangle_is_wholly_outside(clip) {
            continue;
        }
        if !clip_triangle_has_stable_signed_area(clip, minimum_normal) {
            return false;
        }
    }
    true
}

pub(in crate::renderer) fn clip_triangle_is_wholly_outside(clip: [[(f64, f64); 2]; 3]) -> bool {
    (0..2).any(|axis| {
        clip.iter().all(|point| point[axis].1 < -1.0)
            || clip.iter().all(|point| point[axis].0 > 1.0)
    })
}

fn clip_triangle_is_wholly_inside(clip: [[(f64, f64); 2]; 3]) -> bool {
    clip.iter()
        .all(|point| point.iter().all(|axis| axis.0 >= -1.0 && axis.1 <= 1.0))
}

pub(in crate::renderer) fn clip_triangle_has_stable_signed_area(
    clip: [[(f64, f64); 2]; 3],
    minimum_magnitude: f64,
) -> bool {
    let Some(first_x) = shader_interval_difference(clip[1][0], clip[0][0]) else {
        return false;
    };
    let Some(first_y) = shader_interval_difference(clip[2][1], clip[0][1]) else {
        return false;
    };
    let Some(second_y) = shader_interval_difference(clip[1][1], clip[0][1]) else {
        return false;
    };
    let Some(second_x) = shader_interval_difference(clip[2][0], clip[0][0]) else {
        return false;
    };
    let Some(positive) = shader_interval_product(first_x, first_y) else {
        return false;
    };
    let Some(negative) = shader_interval_product(second_y, second_x) else {
        return false;
    };
    let Some(area) = shader_interval_difference(positive, negative) else {
        return false;
    };
    area.0 >= minimum_magnitude || area.1 <= -minimum_magnitude
}

fn dynamic_vertex_clip_ranges(
    vertex: DynamicGpu,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let relative_x = shader_relative_component_bounds(
        vertex.world_position[0],
        vertex.world_position[0],
        uniform.camera_center[0],
        0.0,
        0.0,
    )?;
    let relative_y = shader_relative_component_bounds(
        vertex.world_position[1],
        vertex.world_position[1],
        uniform.camera_center[1],
        0.0,
        0.0,
    )?;
    let minimum = [relative_x.0, relative_y.0];
    let maximum = [relative_x.1, relative_y.1];
    let screen = [
        shader_world_dot_range(
            uniform.world_to_screen_x,
            minimum,
            maximum,
            vertex.depth,
            vertex.depth,
        )?,
        shader_world_dot_range(
            uniform.world_to_screen_y,
            minimum,
            maximum,
            vertex.depth,
            vertex.depth,
        )?,
    ];
    screen_ranges_to_clip(screen, uniform)
}

#[cfg(test)]
pub(in crate::renderer) fn tessellated_vertex_clip_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    screen_ranges_to_clip(tessellated_vertex_screen_ranges(vertex, uniform)?, uniform)
}

#[cfg(test)]
fn tessellated_vertex_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let base = tessellated_vertex_base_screen_ranges(vertex, uniform)?;
    Some([
        rounded_f32_add_range(
            base[0],
            (
                f64::from(vertex.screen_offset[0]),
                f64::from(vertex.screen_offset[0]),
            ),
            false,
        )?,
        rounded_f32_add_range(
            base[1],
            (
                f64::from(vertex.screen_offset[1]),
                f64::from(vertex.screen_offset[1]),
            ),
            false,
        )?,
    ])
}

pub(in crate::renderer) fn tessellated_vertex_base_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let relative_x = shader_relative_component_bounds(
        vertex.world_position[0],
        vertex.world_position[0],
        uniform.camera_center[0],
        0.0,
        0.0,
    )?;
    let relative_y = shader_relative_component_bounds(
        vertex.world_position[1],
        vertex.world_position[1],
        uniform.camera_center[1],
        0.0,
        0.0,
    )?;
    let minimum = [relative_x.0, relative_y.0];
    let maximum = [relative_x.1, relative_y.1];
    let world_offset = Vec2::new(vertex.world_offset[0], vertex.world_offset[1]);
    Some([
        shader_interval_sum_range([
            shader_world_dot_range(
                uniform.world_to_screen_x,
                minimum,
                maximum,
                vertex.depth,
                vertex.depth,
            )?,
            shader_direction_dot_range(uniform.world_to_screen_x, world_offset, world_offset)?,
        ])?,
        shader_interval_sum_range([
            shader_world_dot_range(
                uniform.world_to_screen_y,
                minimum,
                maximum,
                vertex.depth,
                vertex.depth,
            )?,
            shader_direction_dot_range(uniform.world_to_screen_y, world_offset, world_offset)?,
        ])?,
    ])
}

pub(in crate::renderer) fn triangle_position_sources_equal(triangle: &[Vertex]) -> bool {
    triangle.windows(2).all(|pair| {
        pair[0].world_position == pair[1].world_position
            && pair[0].depth == pair[1].depth
            && pair[0].world_offset == pair[1].world_offset
            && pair[0].screen_offset == pair[1].screen_offset
    })
}

pub(in crate::renderer) fn logical_stroke_vertex_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<([(f64, f64); 2], bool)> {
    if vertex.stroke_role < 0.0 {
        return exact_markers::vertex_screen_ranges(vertex, uniform);
    }
    let base = tessellated_vertex_base_screen_ranges(vertex, uniform)?;
    let previous_source = Vec2::new(vertex.previous_direction[0], vertex.previous_direction[1]);
    let next_source = Vec2::new(vertex.next_direction[0], vertex.next_direction[1]);
    let previous_projected = [
        shader_direction_dot_range(uniform.world_to_screen_x, previous_source, previous_source)?,
        shader_direction_dot_range(uniform.world_to_screen_y, previous_source, previous_source)?,
    ];
    let next_projected = if vertex.previous_direction == vertex.next_direction {
        previous_projected
    } else {
        [
            shader_direction_dot_range(uniform.world_to_screen_x, next_source, next_source)?,
            shader_direction_dot_range(uniform.world_to_screen_y, next_source, next_source)?,
        ]
    };
    let previous_tangent = stroke_safe_unit_range(previous_projected)?;
    let next_tangent = if vertex.previous_direction == vertex.next_direction {
        previous_tangent
    } else {
        stroke_safe_unit_range(next_projected)?
    };
    let previous_normal = [range_negate(previous_tangent[1]), previous_tangent[0]];
    let next_normal = [range_negate(next_tangent[1]), next_tangent[0]];
    let turn = if vertex.previous_direction == vertex.next_direction {
        (0.0, 0.0)
    } else {
        rounded_f32_add_range(
            rounded_f32_product_range(previous_tangent[0], next_tangent[1])?,
            rounded_f32_product_range(previous_tangent[1], next_tangent[0])?,
            true,
        )?
    };
    let turn_sign = if turn.0 > 0.0 {
        1.0
    } else if turn.1 < 0.0 {
        -1.0
    } else if vertex.previous_direction == vertex.next_direction {
        0.0
    } else {
        return None;
    };
    let combined_normal = [
        rounded_f32_add_range(previous_normal[0], next_normal[0], false)?,
        rounded_f32_add_range(previous_normal[1], next_normal[1], false)?,
    ];
    let miter = stroke_safe_unit_range(combined_normal);
    let miter_state = miter.and_then(|miter| {
        let denominator = range_dot(miter, next_normal)?;
        if denominator.0 <= 0.001 && denominator.1 >= -0.001 {
            return None;
        }
        let reciprocal = rounded_f32_division_range((1.0, 1.0), denominator)?;
        let multiple = range_abs(reciprocal);
        let limit = f64::from(vertex.miter_limit);
        let within_limit = if multiple.1 <= limit {
            true
        } else if multiple.0 > limit {
            false
        } else {
            return None;
        };
        let scalar = rounded_f32_product_range(
            reciprocal,
            (
                f64::from(vertex.normal_distance),
                f64::from(vertex.normal_distance),
            ),
        )?;
        Some((range_vector_scale(miter, scalar)?, within_limit))
    });
    let normal_scalar = (
        f64::from(vertex.normal_distance),
        f64::from(vertex.normal_distance),
    );
    let tangent_scalar = (
        f64::from(vertex.tangent_distance),
        f64::from(vertex.tangent_distance),
    );
    let next_normal_offset = range_vector_scale(next_normal, normal_scalar)?;
    let previous_normal_offset = range_vector_scale(previous_normal, normal_scalar)?;
    let tangent_offset = range_vector_scale(next_tangent, tangent_scalar)?;
    let mut inactive_candidate = false;
    let mut extrusion = range_vector_add(next_normal_offset, tangent_offset)?;
    if (1.0..=3.0).contains(&vertex.stroke_role) {
        if turn_sign == 0.0 {
            extrusion = next_normal_offset;
        } else {
            let side = vertex.normal_distance.signum();
            let outer_side = -turn_sign;
            if side * outer_side <= 0.0 {
                extrusion = miter_state
                    .filter(|state| state.1)
                    .map_or([(0.0, 0.0); 2], |state| state.0);
            } else if vertex.stroke_role == 2.0 && miter_state.is_some_and(|state| state.1) {
                extrusion = miter_state?.0;
            } else if vertex.stroke_parameter < 0.0 {
                extrusion = previous_normal_offset;
            } else {
                extrusion = next_normal_offset;
            }
        }
        extrusion = range_vector_add(extrusion, tangent_offset)?;
    } else if vertex.stroke_role >= 4.0 {
        let inner = matches!(vertex.stroke_role as i32, 5 | 7 | 9);
        let side = vertex.normal_distance.signum();
        let candidate_side = if inner { -side } else { side };
        let mut active = turn_sign != 0.0 && candidate_side * -turn_sign > 0.0;
        if matches!(vertex.stroke_role as i32, 6 | 7) {
            active &= miter_state.is_none_or(|state| !state.1);
        }
        if !active {
            inactive_candidate = true;
            extrusion = [(0.0, 0.0); 2];
        } else if inner {
            extrusion = miter_state
                .filter(|state| state.1)
                .map_or([(0.0, 0.0); 2], |state| state.0);
        } else if vertex.stroke_role == 8.0 {
            let side_range = (f64::from(candidate_side), f64::from(candidate_side));
            let start = range_vector_scale(previous_normal, side_range)?;
            let finish = range_vector_scale(next_normal, side_range)?;
            let amount = f64::from(vertex.stroke_parameter);
            let mixed = range_vector_add(
                range_vector_scale(start, (1.0 - amount, 1.0 - amount))?,
                range_vector_scale(finish, (amount, amount))?,
            )?;
            extrusion = range_vector_scale(
                stroke_safe_unit_range(mixed)?,
                (
                    f64::from(vertex.normal_distance.abs()),
                    f64::from(vertex.normal_distance.abs()),
                ),
            )?;
        } else if vertex.stroke_parameter < 0.0 {
            extrusion = previous_normal_offset;
        } else {
            extrusion = next_normal_offset;
        }
    } else if miter_state.is_some_and(|state| state.1) {
        extrusion = range_vector_add(miter_state?.0, tangent_offset)?;
    }
    let screen = range_vector_add(base, extrusion)?;
    let screen_offset = [
        (
            f64::from(vertex.screen_offset[0]),
            f64::from(vertex.screen_offset[0]),
        ),
        (
            f64::from(vertex.screen_offset[1]),
            f64::from(vertex.screen_offset[1]),
        ),
    ];
    Some((range_vector_add(screen, screen_offset)?, inactive_candidate))
}

pub(in crate::renderer) fn stroke_safe_unit_range(
    direction: [(f64, f64); 2],
) -> Option<[(f64, f64); 2]> {
    let horizontal_abs = range_abs(direction[0]);
    let vertical_abs = range_abs(direction[1]);
    let scale = (
        horizontal_abs.0.max(vertical_abs.0),
        horizontal_abs.1.max(vertical_abs.1),
    );
    if scale.0 < f64::from(f32::MIN_POSITIVE) {
        return None;
    }
    let scaled = [
        rounded_f32_division_range(direction[0], scale)?,
        rounded_f32_division_range(direction[1], scale)?,
    ];
    let length_squared = range_dot(scaled, scaled)?;
    if length_squared.0 < f64::from(f32::MIN_POSITIVE) {
        return None;
    }
    let mut inverse_length =
        rounded_f32_range(1.0 / length_squared.1.sqrt(), 1.0 / length_squared.0.sqrt())?;
    // WGSL permits two ULP error for inverseSqrt. Add two outward neighbours;
    // an inexact endpoint may already carry one additional rounding neighbour.
    for _ in 0..2 {
        inverse_length = (
            f64::from(next_f32_down(inverse_length.0 as f32)?),
            f64::from(next_f32_up(inverse_length.1 as f32)?),
        );
    }
    range_vector_scale(scaled, inverse_length)
}

pub(in crate::renderer) fn screen_ranges_to_clip(
    screen: [(f64, f64); 2],
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    Some([
        shader_interval_sum_range([
            interval_products_f64(uniform.screen_to_clip[0], screen[0].0, screen[0].1),
            (
                f64::from(uniform.screen_to_clip[2]),
                f64::from(uniform.screen_to_clip[2]),
            ),
        ])?,
        shader_interval_sum_range([
            interval_products_f64(uniform.screen_to_clip[1], screen[1].0, screen[1].1),
            (
                f64::from(uniform.screen_to_clip[3]),
                f64::from(uniform.screen_to_clip[3]),
            ),
        ])?,
    ])
}

#[cfg(test)]
pub(in crate::renderer) fn geometry_vertex_centers_are_portable(
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    let dynamic_center_is_portable = |world: [f32; 2], depth: f32| {
        let Some(relative_x) = shader_relative_component_bounds(
            world[0],
            world[0],
            uniform.camera_center[0],
            0.0,
            0.0,
        ) else {
            return false;
        };
        let Some(relative_y) = shader_relative_component_bounds(
            world[1],
            world[1],
            uniform.camera_center[1],
            0.0,
            0.0,
        ) else {
            return false;
        };
        let minimum = [relative_x.0, relative_y.0];
        let maximum = [relative_x.1, relative_y.1];
        shader_world_dot_range(uniform.world_to_screen_x, minimum, maximum, depth, depth).is_some()
            && shader_world_dot_range(uniform.world_to_screen_y, minimum, maximum, depth, depth)
                .is_some()
    };
    match source {
        GeometryValidationSource::Tessellated(vertices) => vertices
            .iter()
            .all(|vertex| tessellated_vertex_clip_ranges(*vertex, uniform).is_some()),
        GeometryValidationSource::Dynamic(vertices) => vertices
            .iter()
            .all(|vertex| dynamic_center_is_portable(vertex.world_position, vertex.depth)),
    }
}

pub(in crate::renderer) fn geometry_sources_are_portable(
    source: GeometryValidationSource<'_>,
) -> bool {
    match source {
        GeometryValidationSource::Tessellated(vertices) => vertices.iter().all(|vertex| {
            vertex
                .world_position
                .into_iter()
                .chain(vertex.world_offset)
                .chain(vertex.screen_offset)
                .chain(vertex.previous_direction)
                .chain(vertex.next_direction)
                .chain([
                    vertex.depth,
                    vertex.normal_distance,
                    vertex.tangent_distance,
                    vertex.miter_limit,
                ])
                .all(is_portable_shader_source)
        }),
        GeometryValidationSource::Dynamic(vertices) => vertices.iter().all(|vertex| {
            vertex
                .world_position
                .into_iter()
                .chain([vertex.depth])
                .all(is_portable_shader_source)
        }),
    }
}

pub(in crate::renderer) fn logical_stroke_branches_are_stable(
    vertex: Vertex,
    uniform: CameraUniform,
) -> bool {
    if vertex.normal_distance == 0.0 && vertex.tangent_distance == 0.0 {
        return true;
    }
    let project = |direction: [f32; 2]| {
        [
            f64::from(uniform.world_to_screen_x[0]) * f64::from(direction[0])
                + f64::from(uniform.world_to_screen_x[1]) * f64::from(direction[1]),
            f64::from(uniform.world_to_screen_y[0]) * f64::from(direction[0])
                + f64::from(uniform.world_to_screen_y[1]) * f64::from(direction[1]),
        ]
    };
    let stable_projected_direction = |direction: [f32; 2]| {
        let direction = Vec2::new(direction[0], direction[1]);
        let horizontal =
            shader_direction_dot_range(uniform.world_to_screen_x, direction, direction)?;
        let vertical = shader_direction_dot_range(uniform.world_to_screen_y, direction, direction)?;
        let fixed = project([direction.x, direction.y]);
        let component_minimum_magnitude = |range: (f64, f64)| {
            if range.0 > 0.0 {
                range.0
            } else if range.1 < 0.0 {
                -range.1
            } else {
                0.0
            }
        };
        let lower_length =
            component_minimum_magnitude(horizontal).hypot(component_minimum_magnitude(vertical));
        let fixed_length = fixed[0].hypot(fixed[1]);
        let uncertainty = (horizontal.0 - fixed[0])
            .abs()
            .max((horizontal.1 - fixed[0]).abs())
            .hypot(
                (vertical.0 - fixed[1])
                    .abs()
                    .max((vertical.1 - fixed[1]).abs()),
            );
        let minimum_normal = f64::from(f32::MIN_POSITIVE);
        (lower_length >= minimum_normal
            && fixed_length.is_finite()
            && uncertainty.is_finite()
            && uncertainty <= fixed_length * 1.0e-6)
            .then_some(fixed)
    };
    let Some(previous_projected) = stable_projected_direction(vertex.previous_direction) else {
        return false;
    };
    let Some(next_projected) = stable_projected_direction(vertex.next_direction) else {
        return false;
    };
    if vertex.previous_direction == vertex.next_direction {
        // The shader detects source equality and reuses one projection and
        // normalization result, so identical directions cannot acquire a
        // backend-dependent artificial turn.
        return true;
    }

    let normalize = |value: [f64; 2]| {
        let scale = value[0].abs().max(value[1].abs());
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let scaled = [value[0] / scale, value[1] / scale];
        let length = scaled[0].hypot(scaled[1]);
        (length.is_finite() && length > 0.0).then_some([scaled[0] / length, scaled[1] / length])
    };
    let Some(previous) = normalize(previous_projected) else {
        return false;
    };
    let Some(next) = normalize(next_projected) else {
        return false;
    };
    let turn = previous[0] * next[1] - previous[1] * next[0];
    let tangent_dot = previous[0] * next[0] + previous[1] * next[1];
    if !turn.is_finite() || !tangent_dot.is_finite() {
        return false;
    }
    // Keep every topology-changing comparison far outside WGSL's permitted
    // normal-operation error. Exact repeated directions took the fast path.
    if turn.abs() <= 1.0e-4 || tangent_dot <= -0.999_9 {
        return false;
    }

    let previous_normal = [-previous[1], previous[0]];
    let next_normal = [-next[1], next[0]];
    let combined = [
        previous_normal[0] + next_normal[0],
        previous_normal[1] + next_normal[1],
    ];
    let Some(miter) = normalize(combined) else {
        return false;
    };
    let denominator = miter[0] * next_normal[0] + miter[1] * next_normal[1];
    if !denominator.is_finite() || denominator.abs() <= 0.002 {
        return false;
    }
    let miter_multiple = 1.0 / denominator.abs();
    let limit = f64::from(vertex.miter_limit);
    let comparison_margin = 1.0e-4 * miter_multiple.abs().max(limit.abs()).max(1.0);
    (miter_multiple - limit).abs() > comparison_margin
}

pub(in crate::renderer) fn shader_clip_interval_is_safe(
    screen_minimum: f64,
    screen_maximum: f64,
    scale: f32,
    offset: f32,
) -> bool {
    if ((is_nonzero_subnormal_f64(screen_minimum) || is_nonzero_subnormal_f64(screen_maximum))
        && scale != 0.0)
        || (is_nonzero_subnormal(scale) && (screen_minimum != 0.0 || screen_maximum != 0.0))
    {
        return false;
    }
    shader_interval_sum_range([
        interval_products_f64(scale, screen_minimum, screen_maximum),
        (f64::from(offset), f64::from(offset)),
    ])
    .is_some()
}

pub(in crate::renderer) fn shader_relative_component_bounds(
    world_minimum: f32,
    world_maximum: f32,
    camera_center: f32,
    offset_minimum: f32,
    offset_maximum: f32,
) -> Option<(f64, f64)> {
    // Exact equal f32 operands subtract to exact zero on every backend. Keep
    // that important camera-relative case tight so separately projected local
    // tessellation offsets remain usable at large world anchors.
    let subtraction = if world_minimum == camera_center && world_maximum == camera_center {
        (0.0, 0.0)
    } else {
        shader_interval_sum_range([
            (f64::from(world_minimum), f64::from(world_maximum)),
            (-f64::from(camera_center), -f64::from(camera_center)),
        ])?
    };
    if offset_minimum == 0.0 && offset_maximum == 0.0 {
        Some(subtraction)
    } else {
        shader_interval_sum_range([
            subtraction,
            (f64::from(offset_minimum), f64::from(offset_maximum)),
        ])
    }
}

pub(in crate::renderer) fn shader_world_dot_range(
    row: [f32; 4],
    minimum: [f64; 2],
    maximum: [f64; 2],
    depth_minimum: f32,
    depth_maximum: f32,
) -> Option<(f64, f64)> {
    if (is_nonzero_subnormal(row[0]) && (minimum[0] != 0.0 || maximum[0] != 0.0))
        || (is_nonzero_subnormal(row[1]) && (minimum[1] != 0.0 || maximum[1] != 0.0))
        || (is_nonzero_subnormal(row[2]) && (depth_minimum != 0.0 || depth_maximum != 0.0))
        || ((is_nonzero_subnormal_f64(minimum[0]) || is_nonzero_subnormal_f64(maximum[0]))
            && row[0] != 0.0)
        || ((is_nonzero_subnormal_f64(minimum[1]) || is_nonzero_subnormal_f64(maximum[1]))
            && row[1] != 0.0)
        || ((is_nonzero_subnormal(depth_minimum) || is_nonzero_subnormal(depth_maximum))
            && row[2] != 0.0)
    {
        return None;
    }
    shader_interval_sum_range([
        interval_products_f64(row[0], minimum[0], maximum[0]),
        interval_products_f64(row[1], minimum[1], maximum[1]),
        interval_products(row[2], depth_minimum, depth_maximum),
        (f64::from(row[3]), f64::from(row[3])),
    ])
}

pub(in crate::renderer) fn shader_direction_dot_range(
    row: [f32; 4],
    minimum: Vec2,
    maximum: Vec2,
) -> Option<(f64, f64)> {
    if (is_nonzero_subnormal(row[0]) && (minimum.x != 0.0 || maximum.x != 0.0))
        || (is_nonzero_subnormal(row[1]) && (minimum.y != 0.0 || maximum.y != 0.0))
        || ((is_nonzero_subnormal(minimum.x) || is_nonzero_subnormal(maximum.x)) && row[0] != 0.0)
        || ((is_nonzero_subnormal(minimum.y) || is_nonzero_subnormal(maximum.y)) && row[1] != 0.0)
    {
        return None;
    }
    shader_interval_sum_range([
        interval_products(row[0], minimum.x, maximum.x),
        interval_products(row[1], minimum.y, maximum.y),
    ])
}
