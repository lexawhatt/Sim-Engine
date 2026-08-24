use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

use crate::{Color, Vec3};

const MAX_LOGICAL_EDGE_METRIC: f32 = 1_048_576.0;

/// Surface material for one retained 3D instance.
///
/// The representation is private so translucent and hatched section modes can
/// be added without changing [`crate::Mesh3dInstance`] construction. The first
/// renderer slice supports only opaque color/depth-writing surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceStyle3d {
    color: Color,
}

impl SurfaceStyle3d {
    /// Creates an opaque linear-RGBA surface.
    pub fn opaque(color: Color) -> Result<Self, Mesh3dStyleError> {
        if !color.is_finite() || color.alpha() != 1.0 {
            return Err(Mesh3dStyleError::InvalidSurfaceColor);
        }
        Ok(Self { color })
    }

    /// Returns the opaque linear-RGBA color used by the current surface pass.
    pub const fn color(self) -> Color {
        self.color
    }
}

/// Extensible visual material bundle for a retained 3D object.
///
/// Surface and wireframe presentation are independent. Edge-only construction
/// geometry uses `surface = None`; future section materials can extend
/// [`SurfaceStyle3d`] without changing scene insertion or object addressing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStyle3d {
    surface: Option<SurfaceStyle3d>,
    wireframe: Option<WireframeStyle3d>,
}

impl MeshStyle3d {
    /// Creates an object with an opaque surface and no display edges.
    pub const fn surface(surface: SurfaceStyle3d) -> Self {
        Self {
            surface: Some(surface),
            wireframe: None,
        }
    }

    /// Creates display edges without a surface/depth-writing face pass.
    pub const fn wireframe(wireframe: WireframeStyle3d) -> Self {
        Self {
            surface: None,
            wireframe: Some(wireframe),
        }
    }

    /// Adds or replaces display-edge presentation.
    pub const fn with_wireframe(mut self, wireframe: WireframeStyle3d) -> Self {
        self.wireframe = Some(wireframe);
        self
    }

    /// Returns the optional surface material.
    pub const fn surface_style(self) -> Option<SurfaceStyle3d> {
        self.surface
    }

    /// Returns the optional edge presentation.
    pub const fn wireframe_style(self) -> Option<WireframeStyle3d> {
        self.wireframe
    }
}

/// Logical-screen presentation for explicit mathematical mesh edges.
///
/// Visible fragments are solid. Hidden fragments may be disabled or drawn with
/// a logical-pixel dash/gap pattern after depth classification against opaque
/// surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireframeStyle3d {
    visible_color: Color,
    visible_width: f32,
    hidden: Option<HiddenEdgeStyle3d>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HiddenEdgeStyle3d {
    color: Color,
    width: f32,
    dash_length: f32,
    gap_length: f32,
}

impl WireframeStyle3d {
    /// Creates solid visible edges with hidden fragments disabled.
    pub fn visible(color: Color, width: f32) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        Ok(Self {
            visible_color: color,
            visible_width: width,
            hidden: None,
        })
    }

    /// Enables dashed hidden fragments in logical screen pixels.
    pub fn with_hidden(
        mut self,
        color: Color,
        width: f32,
        dash_length: f32,
        gap_length: f32,
    ) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        if !dash_length.is_finite()
            || dash_length <= 0.0
            || dash_length > MAX_LOGICAL_EDGE_METRIC
            || !gap_length.is_finite()
            || gap_length <= 0.0
            || gap_length > MAX_LOGICAL_EDGE_METRIC
            || !(dash_length + gap_length).is_finite()
        {
            return Err(Mesh3dStyleError::InvalidDashPattern);
        }
        self.hidden = Some(HiddenEdgeStyle3d {
            color,
            width,
            dash_length,
            gap_length,
        });
        Ok(self)
    }

    /// Returns the solid visible-edge color in linear RGBA.
    pub const fn visible_color(self) -> Color {
        self.visible_color
    }

    /// Returns visible-edge width in logical screen pixels.
    pub const fn visible_width(self) -> f32 {
        self.visible_width
    }

    /// Returns whether depth-occluded edge fragments should be drawn.
    pub const fn hidden_enabled(self) -> bool {
        self.hidden.is_some()
    }

    /// Returns hidden-edge color when enabled.
    pub const fn hidden_color(self) -> Option<Color> {
        match self.hidden {
            Some(hidden) => Some(hidden.color),
            None => None,
        }
    }

    /// Returns hidden-edge width in logical screen pixels when enabled.
    pub const fn hidden_width(self) -> Option<f32> {
        match self.hidden {
            Some(hidden) => Some(hidden.width),
            None => None,
        }
    }

    /// Returns hidden dash and gap lengths in logical screen pixels.
    pub const fn hidden_pattern(self) -> Option<(f32, f32)> {
        match self.hidden {
            Some(hidden) => Some((hidden.dash_length, hidden.gap_length)),
            None => None,
        }
    }
}

/// Rejection reason for logical-screen 3D edge presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dStyleError {
    /// Surface color must be finite and opaque in the initial material mode.
    InvalidSurfaceColor,
    /// Edge color must be finite and opaque, and width must be finite and positive.
    InvalidColorOrWidth,
    /// Hidden dash and gap lengths must both be finite and positive.
    InvalidDashPattern,
}

impl fmt::Display for Mesh3dStyleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSurfaceColor => write!(formatter, "3D surface color must be opaque"),
            Self::InvalidColorOrWidth => {
                write!(formatter, "3D edge color must be opaque and width positive")
            }
            Self::InvalidDashPattern => {
                write!(formatter, "3D hidden-edge dash and gap must be positive")
            }
        }
    }
}

impl Error for Mesh3dStyleError {}

fn validate_edge_color_width(color: Color, width: f32) -> Result<(), Mesh3dStyleError> {
    if color.is_finite()
        && color.alpha() == 1.0
        && width.is_finite()
        && width > 0.0
        && width <= MAX_LOGICAL_EDGE_METRIC
    {
        Ok(())
    } else {
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    }
}

/// One explicit display edge in a retained 3D mesh.
///
/// Endpoints index the mesh vertex array. A mesh validates the indices and
/// rejects repeated or zero-length edges when it is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MeshEdge3d {
    start: u32,
    end: u32,
}

impl MeshEdge3d {
    /// Describes an edge between two distinct vertex indices.
    pub fn new(start: u32, end: u32) -> Result<Self, Mesh3dError> {
        if start == end {
            return Err(Mesh3dError::DegenerateDisplayEdge { vertex: start });
        }
        Ok(Self { start, end })
    }

    /// Returns the first mesh vertex index.
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Returns the second mesh vertex index.
    pub const fn end(self) -> u32 {
        self.end
    }

    fn canonical(self) -> (u32, u32) {
        if self.start < self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }
}

/// Immutable validated topology for retained stereometry rendering.
///
/// Vertices use caller-defined model-space world units. Triangle indices are a
/// flat triangle list. Winding is preserved but deliberately not interpreted:
/// the current surface pass disables face culling, and a global
/// "counter-clockwise" rule is meaningless without a declared outward side.
/// Optional display edges identify the mathematical edges that a later
/// visible/hidden-line pass should draw; they are not inferred from triangle
/// adjacency because triangulation diagonals are not necessarily meaningful
/// construction edges. A mesh may contain surfaces, display edges, or both.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh3d {
    vertices: Arc<[Vec3]>,
    triangle_indices: Arc<[u32]>,
    display_edges: Arc<[MeshEdge3d]>,
    bounds_min: Vec3,
    bounds_max: Vec3,
}

impl Mesh3d {
    /// Builds a surface mesh without explicit display edges.
    pub fn new(vertices: Vec<Vec3>, triangle_indices: Vec<u32>) -> Result<Self, Mesh3dError> {
        Self::with_display_edges(vertices, triangle_indices, Vec::new())
    }

    /// Builds a mesh with explicit mathematical display edges.
    ///
    /// Construction rejects an empty vertex set, geometry with neither
    /// triangles nor display edges, incomplete triangle lists, out-of-range
    /// indices, repeated/collinear triangle vertices, duplicate display edges,
    /// and display edges outside the vertex array.
    pub fn with_display_edges(
        vertices: Vec<Vec3>,
        triangle_indices: Vec<u32>,
        display_edges: Vec<MeshEdge3d>,
    ) -> Result<Self, Mesh3dError> {
        if vertices.is_empty() {
            return Err(Mesh3dError::EmptyVertices);
        }
        if triangle_indices.is_empty() && display_edges.is_empty() {
            return Err(Mesh3dError::EmptyGeometry);
        }
        if !triangle_indices.len().is_multiple_of(3) {
            return Err(Mesh3dError::InvalidTriangleIndexCount {
                index_count: triangle_indices.len(),
            });
        }

        for (triangle, indices) in triangle_indices.chunks_exact(3).enumerate() {
            let [first, second, third] = [indices[0], indices[1], indices[2]];
            validate_index(first, vertices.len())?;
            validate_index(second, vertices.len())?;
            validate_index(third, vertices.len())?;
            if first == second
                || second == third
                || first == third
                || triangle_is_degenerate(
                    vertices[first as usize],
                    vertices[second as usize],
                    vertices[third as usize],
                )
            {
                return Err(Mesh3dError::DegenerateTriangle { triangle });
            }
        }

        let mut unique_edges = BTreeSet::new();
        for edge in &display_edges {
            validate_index(edge.start, vertices.len())?;
            validate_index(edge.end, vertices.len())?;
            if !unique_edges.insert(edge.canonical()) {
                return Err(Mesh3dError::DuplicateDisplayEdge {
                    start: edge.start,
                    end: edge.end,
                });
            }
        }

        let (bounds_min, bounds_max) = mesh_bounds(&vertices);
        Ok(Self {
            vertices: vertices.into(),
            triangle_indices: triangle_indices.into(),
            display_edges: display_edges.into(),
            bounds_min,
            bounds_max,
        })
    }

    /// Returns immutable model-space vertices.
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Returns the flat host-wound triangle index list.
    pub fn triangle_indices(&self) -> &[u32] {
        &self.triangle_indices
    }

    /// Returns explicit mathematical edges, excluding triangulation diagonals.
    pub fn display_edges(&self) -> &[MeshEdge3d] {
        &self.display_edges
    }

    /// Returns the number of retained triangles.
    pub fn triangle_count(&self) -> usize {
        self.triangle_indices.len() / 3
    }

    /// Returns the inclusive model-space lower bound.
    pub const fn bounds_min(&self) -> Vec3 {
        self.bounds_min
    }

    /// Returns the inclusive model-space upper bound.
    pub const fn bounds_max(&self) -> Vec3 {
        self.bounds_max
    }

    /// Returns retained CPU bytes used by vertices, indices, and display edges.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.vertices
            .len()
            .saturating_mul(std::mem::size_of::<Vec3>())
            .saturating_add(
                self.triangle_indices
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.display_edges
                    .len()
                    .saturating_mul(std::mem::size_of::<MeshEdge3d>()),
            )
    }
}

/// Rejection reason for retained 3D topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dError {
    /// A retained mesh requires at least one model-space vertex.
    EmptyVertices,
    /// A retained mesh requires at least one triangle or display edge.
    EmptyGeometry,
    /// The triangle index list was empty or not divisible by three.
    InvalidTriangleIndexCount {
        /// Rejected number of indices.
        index_count: usize,
    },
    /// A triangle or display edge referenced a missing vertex.
    IndexOutOfBounds {
        /// Rejected vertex index.
        index: u32,
        /// Number of vertices available in the mesh.
        vertex_count: usize,
    },
    /// A triangle repeated a vertex or had zero model-space area.
    DegenerateTriangle {
        /// Zero-based triangle number in the index list.
        triangle: usize,
    },
    /// A display edge used the same endpoint twice.
    DegenerateDisplayEdge {
        /// Repeated vertex index.
        vertex: u32,
    },
    /// The same undirected display edge appeared more than once.
    DuplicateDisplayEdge {
        /// First endpoint in the rejected declaration.
        start: u32,
        /// Second endpoint in the rejected declaration.
        end: u32,
    },
}

impl fmt::Display for Mesh3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVertices => write!(formatter, "3D mesh requires at least one vertex"),
            Self::EmptyGeometry => {
                write!(formatter, "3D mesh requires triangles or display edges")
            }
            Self::InvalidTriangleIndexCount { index_count } => write!(
                formatter,
                "3D mesh triangle index count must be non-zero and divisible by three, got {index_count}"
            ),
            Self::IndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                formatter,
                "3D mesh index {index} is outside {vertex_count} vertices"
            ),
            Self::DegenerateTriangle { triangle } => {
                write!(formatter, "3D mesh triangle {triangle} has zero area")
            }
            Self::DegenerateDisplayEdge { vertex } => {
                write!(formatter, "3D display edge repeats vertex {vertex}")
            }
            Self::DuplicateDisplayEdge { start, end } => {
                write!(formatter, "3D display edge {start}-{end} is duplicated")
            }
        }
    }
}

impl Error for Mesh3dError {}

fn validate_index(index: u32, vertex_count: usize) -> Result<(), Mesh3dError> {
    ((index as u64) < vertex_count as u64)
        .then_some(())
        .ok_or(Mesh3dError::IndexOutOfBounds {
            index,
            vertex_count,
        })
}

fn triangle_is_degenerate(first: Vec3, second: Vec3, third: Vec3) -> bool {
    let ab = (
        second.x() as f64 - first.x() as f64,
        second.y() as f64 - first.y() as f64,
        second.z() as f64 - first.z() as f64,
    );
    let ac = (
        third.x() as f64 - first.x() as f64,
        third.y() as f64 - first.y() as f64,
        third.z() as f64 - first.z() as f64,
    );
    let cross = (
        ab.1 * ac.2 - ab.2 * ac.1,
        ab.2 * ac.0 - ab.0 * ac.2,
        ab.0 * ac.1 - ab.1 * ac.0,
    );
    cross.0 == 0.0 && cross.1 == 0.0 && cross.2 == 0.0
}

fn mesh_bounds(vertices: &[Vec3]) -> (Vec3, Vec3) {
    let first = vertices[0];
    let mut minimum = [first.x(), first.y(), first.z()];
    let mut maximum = minimum;
    for vertex in &vertices[1..] {
        let components = [vertex.x(), vertex.y(), vertex.z()];
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(components[axis]);
            maximum[axis] = maximum[axis].max(components[axis]);
        }
    }
    (
        Vec3::new_unchecked(minimum[0], minimum[1], minimum[2]),
        Vec3::new_unchecked(maximum[0], maximum[1], maximum[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        Mesh3d, Mesh3dError, Mesh3dStyleError, MeshEdge3d, MeshStyle3d, SurfaceStyle3d,
        WireframeStyle3d,
    };
    use crate::{Color, Vec3};

    fn vector(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).unwrap()
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
            Err(Mesh3dError::DegenerateDisplayEdge { vertex: 2 })
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
        let style = WireframeStyle3d::visible(Color::WHITE, 2.0)
            .unwrap()
            .with_hidden(Color::rgba(0.5, 0.5, 0.5, 1.0), 1.5, 6.0, 4.0)
            .unwrap();
        assert_eq!(style.visible_width(), 2.0);
        assert_eq!(style.hidden_pattern(), Some((6.0, 4.0)));
        assert_eq!(
            WireframeStyle3d::visible(Color::WHITE, 0.0),
            Err(Mesh3dStyleError::InvalidColorOrWidth)
        );
        assert_eq!(
            WireframeStyle3d::visible(Color::WHITE, f32::MAX),
            Err(Mesh3dStyleError::InvalidColorOrWidth)
        );
        assert_eq!(
            WireframeStyle3d::visible(Color::WHITE, 1.0)
                .unwrap()
                .with_hidden(Color::WHITE, 1.0, 0.0, 2.0),
            Err(Mesh3dStyleError::InvalidDashPattern)
        );
        assert_eq!(
            WireframeStyle3d::visible(Color::WHITE, 1.0)
                .unwrap()
                .with_hidden(Color::WHITE, 1.0, f32::MAX, f32::MAX),
            Err(Mesh3dStyleError::InvalidDashPattern)
        );
    }

    #[test]
    fn mesh_style_keeps_surface_and_wireframe_extensible() {
        let surface = SurfaceStyle3d::opaque(Color::rgb(0.2, 0.3, 0.4)).unwrap();
        let wireframe = WireframeStyle3d::visible(Color::WHITE, 2.0).unwrap();
        let style = MeshStyle3d::surface(surface).with_wireframe(wireframe);
        assert_eq!(style.surface_style(), Some(surface));
        assert_eq!(style.wireframe_style(), Some(wireframe));
        assert_eq!(
            SurfaceStyle3d::opaque(Color::rgba(1.0, 1.0, 1.0, 0.5)),
            Err(Mesh3dStyleError::InvalidSurfaceColor)
        );
        assert_eq!(MeshStyle3d::wireframe(wireframe).surface_style(), None);
    }
}
