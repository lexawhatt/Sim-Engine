use super::*;
use crate::renderer::validation2d::logical_stroke_vertex_screen_ranges;
use crate::renderer::{Camera2d, Color, LogicalViewport, Scene, tessellate_scene};

fn example_vertices() -> Vec<Vertex> {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_line(
            Vec2::new(-10.0, 3.0),
            Vec2::new(27.0, 9.0),
            3.0,
            Color::WHITE,
        )
        .unwrap();
    let mut vertices = Vec::new();
    tessellate_scene(&scene, &mut vertices, &mut Vec::new()).unwrap();
    vertices
}

fn proof_bits(value: Option<ScreenProof>) -> Option<([(u64, u64); 2], bool)> {
    value.map(|(ranges, inactive)| {
        (
            ranges.map(|(minimum, maximum)| (minimum.to_bits(), maximum.to_bits())),
            inactive,
        )
    })
}

fn assert_same(proofs: &mut StrokeProofs, vertex: Vertex, uniform: CameraUniform) {
    let expected = proof_bits(logical_stroke_vertex_screen_ranges(vertex, uniform));
    for _ in 0..2 {
        assert_eq!(proof_bits(proofs.screen(vertex)), expected);
        assert_eq!(
            proofs.branches_are_stable(vertex),
            logical_stroke_branches_are_stable(vertex, uniform)
        );
    }
}

#[test]
fn stroke_memo_matches_frozen_intervals_for_every_input_and_uniform() {
    let vertices = example_vertices();
    let viewport = LogicalViewport::new(1280.0, 800.0).unwrap();
    let values = [
        0.0,
        -0.0,
        0.00001,
        -1.0,
        3.5,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        1e30,
        f32::MAX,
        f32::NAN,
        f32::INFINITY,
    ];
    for zoom in [0.0001, 1.0, 17.0, f32::MAX] {
        let mut camera = Camera2d::new(Vec2::new(13.0, -4.0), zoom).unwrap();
        for rotation in [0.0, 0.17, -0.59] {
            camera.set_rotation(rotation).unwrap();
            let Some(uniform) = CameraUniform::new(camera, viewport) else {
                continue;
            };
            let mut proofs = StrokeProofs::new(uniform);
            for vertex in &vertices {
                assert_same(&mut proofs, *vertex, uniform);
            }
            for field in 0..16 {
                for value in values {
                    let mut vertex = vertices[1];
                    match field {
                        0..=1 => vertex.world_position[field] = value,
                        2..=3 => vertex.world_offset[field - 2] = value,
                        4..=5 => vertex.screen_offset[field - 4] = value,
                        6 => vertex.depth = value,
                        7..=8 => vertex.previous_direction[field - 7] = value,
                        9..=10 => vertex.next_direction[field - 9] = value,
                        11 => vertex.normal_distance = value,
                        12 => vertex.tangent_distance = value,
                        13 => vertex.miter_limit = value,
                        14 => vertex.stroke_role = value,
                        _ => vertex.stroke_parameter = value,
                    }
                    assert_same(&mut proofs, vertex, uniform);
                    assert!(proofs.outputs.iter().flatten().count() <= 8);
                    assert!(proofs.directions.iter().flatten().count() <= 8);
                    assert!(proofs.bases.iter().flatten().count() <= 8);
                    assert!(proofs.branches.iter().flatten().count() <= 8);
                }
            }
        }
    }
}

#[test]
fn randomized_stroke_memo_preserves_rejections_and_candidate_branches() {
    let mut state = 0x2917_cca5_u32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let original = example_vertices()[1];
    let mut accepted = 0;
    let mut rejected = 0;
    for index in 0..4096 {
        let mut camera = Camera2d::new(Vec2::new(8.0, -16.0), 1.0).unwrap();
        camera.set_rotation(index as f32 * 0.007).unwrap();
        let uniform =
            CameraUniform::new(camera, LogicalViewport::new(913.0, 727.0).unwrap()).unwrap();
        let mut proofs = StrokeProofs::new(uniform);
        let mut vertex = original;
        vertex.previous_direction = [
            (random() % 200) as f32 - 100.0,
            (random() % 200) as f32 - 100.0,
        ];
        vertex.next_direction = if index % 3 == 0 {
            vertex.previous_direction
        } else {
            [
                (random() % 200) as f32 - 100.0,
                (random() % 200) as f32 - 100.0,
            ]
        };
        vertex.stroke_role = (index % 10) as f32;
        vertex.stroke_parameter = (random() % 100) as f32 / 50.0 - 1.0;
        vertex.normal_distance = (random() % 100) as f32 / 10.0 - 5.0;
        vertex.miter_limit = (random() % 80) as f32 / 10.0 + 1.0;
        if index % 7 == 0 {
            vertex.next_direction[0] = f32::from_bits(random());
        }
        assert_same(&mut proofs, vertex, uniform);
        if proofs.screen(vertex).is_some() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    assert!(accepted > 0 && rejected > 0);
}
