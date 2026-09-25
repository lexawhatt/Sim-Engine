//! Bounded exact reuse of stroke shader proofs within one immutable uniform.

use super::{
    CameraUniform, Vec2, Vertex, exact_markers, logical_stroke_branches_are_stable, range_abs,
    range_dot, range_negate, range_vector_add, range_vector_scale, rounded_f32_add_range,
    rounded_f32_division_range, rounded_f32_product_range, shader_direction_dot_range,
    stroke_safe_unit_range, tessellated_vertex_base_screen_ranges,
};

type Ranges = [(f64, f64); 2];
type ScreenProof = (Ranges, bool);

#[derive(Clone, Copy, PartialEq, Eq)]
struct VertexKey([u32; 16]);

impl VertexKey {
    fn new(vertex: Vertex) -> Self {
        Self(
            [
                vertex.world_position[0],
                vertex.world_position[1],
                vertex.world_offset[0],
                vertex.world_offset[1],
                vertex.screen_offset[0],
                vertex.screen_offset[1],
                vertex.depth,
                vertex.previous_direction[0],
                vertex.previous_direction[1],
                vertex.next_direction[0],
                vertex.next_direction[1],
                vertex.normal_distance,
                vertex.tangent_distance,
                vertex.miter_limit,
                vertex.stroke_role,
                vertex.stroke_parameter,
            ]
            .map(f32::to_bits),
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DirectionKey([u32; 4]);

impl DirectionKey {
    fn new(vertex: Vertex) -> Self {
        Self(
            [
                vertex.previous_direction[0],
                vertex.previous_direction[1],
                vertex.next_direction[0],
                vertex.next_direction[1],
            ]
            .map(f32::to_bits),
        )
    }
}

#[derive(Clone, Copy)]
struct DirectionProof {
    next_tangent: Ranges,
    previous_normal: Ranges,
    next_normal: Ranges,
    turn_sign: f32,
    miter: Option<(Ranges, (f64, f64))>,
}

impl DirectionProof {
    fn new(vertex: Vertex, uniform: CameraUniform) -> Option<Self> {
        let previous_source = Vec2::new(vertex.previous_direction[0], vertex.previous_direction[1]);
        let next_source = Vec2::new(vertex.next_direction[0], vertex.next_direction[1]);
        let previous_projected = [
            shader_direction_dot_range(
                uniform.world_to_screen_x,
                previous_source,
                previous_source,
            )?,
            shader_direction_dot_range(
                uniform.world_to_screen_y,
                previous_source,
                previous_source,
            )?,
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

        let miter = miter.and_then(|miter| {
            let denominator = range_dot(miter, next_normal)?;
            if denominator.0 <= 0.001 && denominator.1 >= -0.001 {
                return None;
            }
            let reciprocal = rounded_f32_division_range((1.0, 1.0), denominator)?;
            Some((miter, reciprocal))
        });
        Some(Self {
            next_tangent,
            previous_normal,
            next_normal,
            turn_sign,
            miter,
        })
    }

    fn screen(self, vertex: Vertex, base: Ranges) -> Option<ScreenProof> {
        let Self {
            next_tangent,
            previous_normal,
            next_normal,
            turn_sign,
            miter,
        } = self;
        let miter_state = miter.and_then(|(miter, reciprocal)| {
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
}

// Keys contain exact input bits rather than addresses. The uniform is immutable
// and owned by this short-lived proof set, so every viewport/camera/device pass
// begins with an empty memo. Failed proofs can be reused without becoming passes.
pub(in crate::renderer) struct StrokeProofs {
    uniform: CameraUniform,
    outputs: [Option<(VertexKey, Option<ScreenProof>)>; 8],
    output_next: usize,
    directions: [Option<(DirectionKey, Option<DirectionProof>)>; 8],
    direction_next: usize,
    bases: [Option<([u32; 5], Option<Ranges>)>; 8],
    base_next: usize,
    branches: [Option<((DirectionKey, u32), bool)>; 8],
    branch_next: usize,
}

impl StrokeProofs {
    pub(in crate::renderer) fn new(uniform: CameraUniform) -> Self {
        Self {
            uniform,
            outputs: [None; 8],
            output_next: 0,
            directions: [None; 8],
            direction_next: 0,
            bases: [None; 8],
            base_next: 0,
            branches: [None; 8],
            branch_next: 0,
        }
    }

    pub(in crate::renderer) fn screen(&mut self, vertex: Vertex) -> Option<ScreenProof> {
        let key = VertexKey::new(vertex);
        if let Some((_, proof)) = self
            .outputs
            .iter()
            .flatten()
            .find(|(cached, _)| *cached == key)
        {
            return *proof;
        }
        let proof = self.screen_uncached(vertex);
        self.outputs[self.output_next] = Some((key, proof));
        self.output_next = (self.output_next + 1) % self.outputs.len();
        proof
    }

    fn screen_uncached(&mut self, vertex: Vertex) -> Option<ScreenProof> {
        if vertex.stroke_role < 0.0 {
            return exact_markers::vertex_screen_ranges(vertex, self.uniform);
        }
        let base_key = [
            vertex.world_position[0],
            vertex.world_position[1],
            vertex.world_offset[0],
            vertex.world_offset[1],
            vertex.depth,
        ]
        .map(f32::to_bits);
        let base = match self
            .bases
            .iter()
            .flatten()
            .find(|(key, _)| *key == base_key)
        {
            Some((_, proof)) => *proof,
            None => {
                let proof = tessellated_vertex_base_screen_ranges(vertex, self.uniform);
                self.bases[self.base_next] = Some((base_key, proof));
                self.base_next = (self.base_next + 1) % self.bases.len();
                proof
            }
        }?;
        let key = DirectionKey::new(vertex);
        let proof = match self
            .directions
            .iter()
            .flatten()
            .find(|(cached, _)| *cached == key)
        {
            Some((_, proof)) => *proof,
            None => {
                let proof = DirectionProof::new(vertex, self.uniform);
                self.directions[self.direction_next] = Some((key, proof));
                self.direction_next = (self.direction_next + 1) % self.directions.len();
                proof
            }
        };
        proof?.screen(vertex, base)
    }

    pub(in crate::renderer) fn branches_are_stable(&mut self, vertex: Vertex) -> bool {
        if vertex.normal_distance == 0.0 && vertex.tangent_distance == 0.0 {
            return true;
        }
        let key = (DirectionKey::new(vertex), vertex.miter_limit.to_bits());
        if let Some((_, proof)) = self
            .branches
            .iter()
            .flatten()
            .find(|(cached, _)| *cached == key)
        {
            return *proof;
        }
        let proof = logical_stroke_branches_are_stable(vertex, self.uniform);
        self.branches[self.branch_next] = Some((key, proof));
        self.branch_next = (self.branch_next + 1) % self.branches.len();
        proof
    }
}

#[cfg(test)]
mod tests;
