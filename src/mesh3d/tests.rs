//! Core topology and presentation invariants, without renderer dependencies.

use super::{
    Mesh3d, Mesh3dError, Mesh3dStyleError, MeshEdge3d, MeshStyle3d, SurfaceStyle3d,
    WireframeStyle3d,
};
use crate::{Color, LogicalPixels, UnitError, Vec3};

fn vector(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).unwrap()
}

fn logical(value: f32) -> LogicalPixels {
    LogicalPixels::new(value).unwrap()
}

#[test]
fn texture_coordinates_are_finite_normalized_and_match_vertices() {
    use super::TextureCoordinate2d;
    for value in [f32::NAN, f32::INFINITY, -f32::INFINITY, -0.01, 1.01] {
        assert_eq!(
            TextureCoordinate2d::new(value, 0.0),
            Err(Mesh3dError::InvalidTextureCoordinate)
        );
        assert_eq!(
            TextureCoordinate2d::new(0.0, value),
            Err(Mesh3dError::InvalidTextureCoordinate)
        );
    }
    let uv = TextureCoordinate2d::new(0.0, 1.0).unwrap();
    assert_eq!((uv.u(), uv.v()), (0.0, 1.0));
    let vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    for count in [0, 2, 4] {
        assert_eq!(
            Mesh3d::textured(vertices.clone(), vec![uv; count], vec![0, 1, 2], Vec::new()),
            Err(Mesh3dError::TextureCoordinateCountMismatch {
                vertex_count: 3,
                coordinate_count: count
            })
        );
    }
    let bare = Mesh3d::new(vertices.clone(), vec![0, 1, 2]).unwrap();
    let mesh = Mesh3d::textured(vertices, vec![uv; 3], vec![0, 1, 2], Vec::new()).unwrap();
    assert_eq!(mesh.texture_coordinates(), &[uv; 3]);
    assert_eq!(
        mesh.recovery_memory_bytes(),
        bare.recovery_memory_bytes() + 3 * std::mem::size_of::<TextureCoordinate2d>()
    );
    assert_eq!(
        mesh.clone().texture_coordinates().as_ptr(),
        mesh.texture_coordinates().as_ptr()
    );
}

#[test]
fn mesh_validates_topology_and_preserves_explicit_edges() {
    let mesh = Mesh3d::with_display_edges(
        vec![
            vector(-1.0, -1.0, 0.0),
            vector(1.0, -1.0, 0.0),
            vector(0.0, 1.0, 0.0),
        ],
        vec![0, 1, 2],
        vec![
            MeshEdge3d::new(0, 1).unwrap(),
            MeshEdge3d::new(1, 2).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(mesh.triangle_count(), 1);
    assert_eq!(mesh.display_edges().len(), 2);
    assert_eq!(mesh.bounds_min(), vector(-1.0, -1.0, 0.0));
    assert_eq!(mesh.bounds_max(), vector(1.0, 1.0, 0.0));
}

#[test]
fn mesh_rejects_invalid_and_degenerate_topology() {
    assert_eq!(
        Mesh3d::new(Vec::new(), vec![0, 1, 2]),
        Err(Mesh3dError::EmptyVertices)
    );
    let vertices = vec![
        vector(0.0, 0.0, 0.0),
        vector(1.0, 0.0, 0.0),
        vector(2.0, 0.0, 0.0),
    ];
    assert_eq!(
        Mesh3d::new(vertices.clone(), vec![0, 1]),
        Err(Mesh3dError::InvalidTriangleIndexCount { index_count: 2 })
    );
    assert_eq!(
        Mesh3d::new(vertices.clone(), vec![0, 1, 3]),
        Err(Mesh3dError::IndexOutOfBounds {
            index: 3,
            vertex_count: 3,
        })
    );
    assert_eq!(
        Mesh3d::new(vertices, vec![0, 1, 2]),
        Err(Mesh3dError::DegenerateTriangle { triangle: 0 })
    );
}

#[test]
fn mesh_rejects_duplicate_undirected_display_edges() {
    let vertices = vec![
        vector(0.0, 0.0, 0.0),
        vector(1.0, 0.0, 0.0),
        vector(0.0, 1.0, 0.0),
    ];
    assert_eq!(
        Mesh3d::with_display_edges(
            vertices,
            vec![0, 1, 2],
            vec![
                MeshEdge3d::new(0, 1).unwrap(),
                MeshEdge3d::new(1, 0).unwrap(),
            ],
        ),
        Err(Mesh3dError::DuplicateDisplayEdge { start: 1, end: 0 })
    );
    assert_eq!(
        MeshEdge3d::new(2, 2),
        Err(Mesh3dError::DegenerateDisplayEdge { start: 2, end: 2 })
    );
}

#[test]
fn mesh_rejects_geometrically_zero_length_display_edges() {
    assert_eq!(
        Mesh3d::with_display_edges(
            vec![vector(1.0, 2.0, 3.0), vector(1.0, 2.0, 3.0)],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        ),
        Err(Mesh3dError::DegenerateDisplayEdge { start: 0, end: 1 })
    );
}

#[test]
fn mesh_accepts_edge_only_construction_geometry() {
    let mesh = Mesh3d::with_display_edges(
        vec![vector(0.0, 0.0, 0.0), vector(1.0, 0.0, 0.0)],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    assert_eq!(mesh.triangle_count(), 0);
    assert_eq!(mesh.display_edges().len(), 1);
    assert_eq!(
        Mesh3d::new(vec![vector(0.0, 0.0, 0.0)], Vec::new()),
        Err(Mesh3dError::EmptyGeometry)
    );
}

#[test]
fn wireframe_style_validates_logical_pixel_contract() {
    let style = WireframeStyle3d::visible(Color::WHITE, logical(2.0))
        .unwrap()
        .with_hidden(
            Color::rgba(0.5, 0.5, 0.5, 1.0),
            logical(1.5),
            logical(6.0),
            logical(4.0),
        )
        .unwrap();
    assert_eq!(style.visible_width(), logical(2.0));
    assert_eq!(style.hidden_pattern(), Some((logical(6.0), logical(4.0))));
    assert_eq!(
        LogicalPixels::new(0.0),
        Err(UnitError::InvalidLogicalPixels { value: 0.0 })
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::WHITE, logical(f32::MAX)),
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::WHITE, logical(f32::MIN_POSITIVE / 2.0)),
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::rgb(-0.01, 0.0, 0.0), logical(1.0)),
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::WHITE, logical(1.0))
            .unwrap()
            .with_hidden(Color::WHITE, logical(1.0), logical(f32::MAX), logical(2.0),),
        Err(Mesh3dStyleError::InvalidDashPattern)
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::WHITE, logical(1.0))
            .unwrap()
            .with_hidden(
                Color::WHITE,
                logical(1.0),
                logical(f32::MIN_POSITIVE / 2.0),
                logical(2.0),
            ),
        Err(Mesh3dStyleError::InvalidDashPattern)
    );
    assert_eq!(
        WireframeStyle3d::visible(Color::WHITE, logical(1.0))
            .unwrap()
            .with_hidden(
                Color::WHITE,
                logical(1.0),
                logical(f32::MAX),
                logical(f32::MAX),
            ),
        Err(Mesh3dStyleError::InvalidDashPattern)
    );
}

#[test]
fn mesh_style_keeps_surface_and_wireframe_extensible() {
    let surface = SurfaceStyle3d::opaque(Color::rgb(0.2, 0.3, 0.4)).unwrap();
    let wireframe = WireframeStyle3d::visible(Color::WHITE, logical(2.0)).unwrap();
    let style = MeshStyle3d::surface(surface).with_wireframe(wireframe);
    assert_eq!(style.surface_style(), Some(surface));
    assert_eq!(style.wireframe_style(), Some(wireframe));
    assert_eq!(
        SurfaceStyle3d::opaque(Color::rgba(1.0, 1.0, 1.0, 0.5)),
        Err(Mesh3dStyleError::InvalidSurfaceColor)
    );
    assert_eq!(
        SurfaceStyle3d::opaque(Color::rgb(1.01, 0.0, 0.0)),
        Err(Mesh3dStyleError::InvalidSurfaceColor)
    );
    assert_eq!(MeshStyle3d::wireframe(wireframe).surface_style(), None);
}
