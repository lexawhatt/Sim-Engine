use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

use super::config::recovery_quarantine_has_capacity;
use super::*;
use crate::LogicalScreenVector;

#[path = "cpu_validation_diagnostics.rs"]
mod cpu_validation_diagnostics;

mod clipping;
mod configuration;
mod gpu_oracle;
mod gpu_pixels;
mod particles;
mod shapes;
mod strokes;
mod validation;

fn tessellate_for_test(scene: &Scene) -> (Vec<Vertex>, Vec<PreparedDrawBatch>) {
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    tessellate_scene(scene, &mut vertices, &mut draw_batches)
        .expect("validated test scene should tessellate within its budget");
    (vertices, draw_batches)
}

fn vertex_screen_position(vertex: Vertex, camera: Camera2d, viewport: LogicalViewport) -> Vec2 {
    let Some(uniform) = CameraUniform::new(camera, viewport) else {
        panic!("test camera uniform should be finite");
    };
    let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
    let mut screen = uniform.world_to_screen(world, vertex.depth)
        + uniform.direction_to_screen(Vec2::new(vertex.world_offset[0], vertex.world_offset[1]))
        + Vec2::new(vertex.screen_offset[0], vertex.screen_offset[1]);
    if vertex.normal_distance.abs() > 0.0 || vertex.tangent_distance.abs() > 0.0 {
        let previous = uniform.direction_to_screen(Vec2::new(
            vertex.previous_direction[0],
            vertex.previous_direction[1],
        ));
        let next = uniform.direction_to_screen(Vec2::new(
            vertex.next_direction[0],
            vertex.next_direction[1],
        ));
        let previous_tangent = previous.normalized();
        let next_tangent = next.normalized();
        let previous_normal = previous_tangent.perp();
        let next_normal = next_tangent.perp();
        let combined_normal = previous_normal + next_normal;
        let turn = previous_tangent
            .x
            .mul_add(next_tangent.y, -previous_tangent.y * next_tangent.x);
        let reverses = turn.abs() <= 0.000001 && previous_tangent.dot(next_tangent) < 0.0;
        let side = vertex.normal_distance.signum();
        let outer_side = -turn.signum();
        let mut extrusion =
            next_normal * vertex.normal_distance + next_tangent * vertex.tangent_distance;
        let mut miter_offset = Vec2::ZERO;
        let mut miter_multiple = f32::INFINITY;
        let mut miter_valid = false;
        if combined_normal.length_squared() > 0.000001 {
            let miter = combined_normal.normalized();
            let denominator = miter.dot(next_normal);
            if denominator.abs() > 0.001 {
                miter_multiple = (1.0 / denominator).abs();
                miter_offset = miter * (vertex.normal_distance / denominator);
                miter_valid = true;
            }
        }
        if (1.0..=3.0).contains(&vertex.stroke_role) {
            if reverses {
                extrusion = Vec2::ZERO;
            } else if turn.abs() <= 0.000001 {
                extrusion = next_normal * vertex.normal_distance;
            } else if side * outer_side <= 0.0 {
                extrusion = if miter_valid && miter_multiple <= vertex.miter_limit {
                    miter_offset
                } else {
                    Vec2::ZERO
                };
            } else if vertex.stroke_role == 2.0
                && miter_valid
                && miter_multiple <= vertex.miter_limit
            {
                extrusion = miter_offset;
            } else if vertex.stroke_parameter < 0.0 {
                extrusion = previous_normal * vertex.normal_distance;
            } else {
                extrusion = next_normal * vertex.normal_distance;
            }
            extrusion += next_tangent * vertex.tangent_distance;
        } else if vertex.stroke_role >= 4.0 {
            let inner = matches!(vertex.stroke_role as i32, 5 | 7 | 9);
            let candidate_side = if inner { -side } else { side };
            let mut join_active = turn.abs() > 0.000001 && candidate_side * outer_side > 0.0;
            if matches!(vertex.stroke_role as i32, 6 | 7) {
                join_active &= !miter_valid || miter_multiple > vertex.miter_limit;
            }
            extrusion = if !join_active {
                Vec2::ZERO
            } else if inner {
                if miter_valid && miter_multiple <= vertex.miter_limit {
                    miter_offset
                } else {
                    Vec2::ZERO
                }
            } else if vertex.stroke_role == 8.0 {
                let start = previous_normal * candidate_side;
                let finish = next_normal * candidate_side;
                let angle = start
                    .x
                    .mul_add(finish.y, -start.y * finish.x)
                    .atan2(start.dot(finish))
                    * vertex.stroke_parameter;
                Vec2::new(
                    start.x.mul_add(angle.cos(), -start.y * angle.sin()),
                    start.x.mul_add(angle.sin(), start.y * angle.cos()),
                ) * vertex.normal_distance.abs()
            } else if vertex.stroke_parameter < 0.0 {
                previous_normal * vertex.normal_distance
            } else {
                next_normal * vertex.normal_distance
            };
        } else if miter_valid && miter_multiple <= vertex.miter_limit {
            extrusion = miter_offset + next_tangent * vertex.tangent_distance;
        }
        screen += extrusion;
    }
    screen
}

#[test]
fn offscreen_gpu_readback_verifies_camera_depth_and_clip_contract() {
    gpu_oracle::run();
}
