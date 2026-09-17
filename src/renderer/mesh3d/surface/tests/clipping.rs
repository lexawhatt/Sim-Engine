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
