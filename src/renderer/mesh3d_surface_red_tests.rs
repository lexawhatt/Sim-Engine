use super::*;

#[test]
fn rounded_coincident_clip_intersections_do_not_discard_uncertain_sliver() {
    // The vertex lies 128 representable steps outside x = -w. Its two incident
    // intersections both round to (-1, .5, .5, 1), but have nonzero uncertainty
    // and represent distinct points. Dropping one silently removes topology.
    let points = [
        [-f32::from_bits(1.0_f32.to_bits() + 128), 0.5, 0.5, 1.0],
        [0.75, 0.4999, 0.5, 1.0],
        [0.75, 0.5001, 0.5, 1.0],
    ];
    let vertices = std::array::from_fn::<_, 3, _>(|index| ClipVertex {
        uv: [0.0; 2],
        color: [1.0; 4],
        lighting: [0.0; 4],
        ranges: points[index].map(ShaderValueRange::exact),
        planes: 0,
        provenance: index as u8,
    });
    let first = intersection(vertices[2], vertices[0], 0, 3).unwrap();
    let second = intersection(vertices[0], vertices[1], 0, 4).unwrap();
    assert_eq!(
        first.ranges.map(|value| value.fixed),
        second.ranges.map(|value| value.fixed)
    );
    assert_eq!(first.planes, second.planes);
    assert!(!first.same_proven_vertex(second));
    assert!(
        first
            .ranges
            .iter()
            .any(|range| range.minimum < range.maximum)
    );
    assert!(
        second
            .ranges
            .iter()
            .any(|range| range.minimum < range.maximum)
    );
    // The old duplicate rule dropped the second intersection and validated
    // this large remaining triangle, concealing the uncertain fourth vertex.
    validate_projected_triangle_orientation([first.ranges, vertices[1].ranges, vertices[2].ranges])
        .unwrap();
    for reverse in [false, true] {
        let mut points = points;
        if reverse {
            points.swap(1, 2);
        }
        assert!(
            matches!(
                clipped_triangle(points.map(|point| point.map(ShaderValueRange::exact))),
                Err(Mesh3dRenderError::UnportableSurfaceTopology)
            ),
            "distinct intersections with one rounded position need rejection, winding={reverse}"
        );
    }
}

#[test]
fn repeated_exact_boundary_endpoint_preserves_a_valid_clipped_triangle() {
    let points = [
        [-2.0, 0.4, 0.5, 1.0],
        [-1.0, 0.5, 0.5, 1.0],
        [0.75, -0.2, 0.5, 1.0],
    ];
    for reverse in [false, true] {
        let mut points = points;
        if reverse {
            points.swap(1, 2);
        }
        let result =
            clipped_triangle(points.map(|point| point.map(ShaderValueRange::exact))).unwrap();
        assert!(result.crossing);
        assert_eq!(result.count, 3);
        assert_eq!(
            result.vertices[..result.count]
                .iter()
                .filter(|vertex| vertex.ranges.map(|range| range.fixed) == [-1.0, 0.5, 0.5, 1.0])
                .count(),
            1
        );
    }
}

#[test]
fn boundary_identity_requires_the_entire_uncertainty_envelope() {
    let original = ClipVertex {
        uv: [0.0; 2],
        color: [1.0; 4],
        lighting: [0.0; 4],
        ranges: [-1.0, 0.5, 0.5, 1.0].map(ShaderValueRange::exact),
        planes: 1,
        provenance: 4,
    };
    assert!(original.same_proven_vertex(original));
    let mut uncertain = original;
    uncertain.ranges[1].minimum = 0.49;
    assert!(!original.same_proven_vertex(uncertain));
    uncertain = original;
    uncertain.ranges[1].maximum = 0.51;
    assert!(!original.same_proven_vertex(uncertain));
    let mut independent = original;
    independent.provenance += 1;
    assert!(!original.same_proven_vertex(independent));
}
