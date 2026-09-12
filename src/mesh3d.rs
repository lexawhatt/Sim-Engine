use std::{error::Error, fmt, sync::Arc};

use crate::{Color, LogicalPixels, Vec3};

const MAX_LOGICAL_EDGE_METRIC: f32 = 1_048_576.0;

/// Surface material for one retained 3D instance.
///
/// The representation is private so translucent and hatched section modes can
/// be added without changing `Mesh3dInstance` construction. The first renderer
/// slice supports only opaque color/depth-writing surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceStyle3d {
    color: Color,
}

impl SurfaceStyle3d {
    /// Creates an opaque linear-RGBA surface.
    pub fn opaque(color: Color) -> Result<Self, Mesh3dStyleError> {
        if !color.is_normalized() || color.alpha() != 1.0 {
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
/// Visible fragments are solid. Fragments occluded beyond the depth buffer's
/// conservative coplanar tolerance may be drawn with a logical-pixel dash/gap
/// pattern. Classification uses one implementation depth unit away from the
/// camera for hidden fragments and one unit toward it for visible fragments,
/// with visible fragments rendered last. Therefore coplanar and
/// sub-depth-resolution separations intentionally resolve as visible; this is
/// raster visibility, not exact analytic solid geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireframeStyle3d {
    visible_color: Color,
    visible_width: LogicalPixels,
    hidden: Option<HiddenEdgeStyle3d>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HiddenEdgeStyle3d {
    color: Color,
    width: LogicalPixels,
    dash_length: LogicalPixels,
    gap_length: LogicalPixels,
}

impl WireframeStyle3d {
    /// Largest accepted width, dash, or gap in logical pixels.
    ///
    /// The bound keeps every style-derived shader operation inside the
    /// cross-backend arithmetic envelope.
    pub const MAX_LOGICAL_PIXELS: f32 = MAX_LOGICAL_EDGE_METRIC;

    /// Creates solid visible edges with hidden fragments disabled.
    pub fn visible(color: Color, width: LogicalPixels) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        Ok(Self {
            visible_color: color,
            visible_width: width,
            hidden: None,
        })
    }

    /// Enables dashed depth-occluded fragments in logical screen pixels.
    ///
    /// Occlusion must exceed the conservative two-unit coplanar tolerance
    /// documented on [`WireframeStyle3d`] before a fragment resolves hidden.
    pub fn with_hidden(
        mut self,
        color: Color,
        width: LogicalPixels,
        dash_length: LogicalPixels,
        gap_length: LogicalPixels,
    ) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        if !is_stable_edge_metric(dash_length.get())
            || !is_stable_edge_metric(gap_length.get())
            || !(dash_length.get() + gap_length.get()).is_finite()
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
    pub const fn visible_width(self) -> LogicalPixels {
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
    pub const fn hidden_width(self) -> Option<LogicalPixels> {
        match self.hidden {
            Some(hidden) => Some(hidden.width),
            None => None,
        }
    }

    /// Returns hidden dash and gap lengths in logical screen pixels.
    pub const fn hidden_pattern(self) -> Option<(LogicalPixels, LogicalPixels)> {
        match self.hidden {
            Some(hidden) => Some((hidden.dash_length, hidden.gap_length)),
            None => None,
        }
    }
}

/// Rejection reason for logical-screen 3D edge presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dStyleError {
    /// Surface color must be normalized and opaque in the initial material mode.
    InvalidSurfaceColor,
    /// Edge color must be normalized and opaque, and width must be a normal
    /// positive value no greater than [`WireframeStyle3d::MAX_LOGICAL_PIXELS`].
    InvalidColorOrWidth,
    /// Hidden dash and gap must be normal positive values, each no greater than
    /// [`WireframeStyle3d::MAX_LOGICAL_PIXELS`], with a finite sum.
    InvalidDashPattern,
}

impl fmt::Display for Mesh3dStyleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSurfaceColor => write!(formatter, "3D surface color must be opaque"),
            Self::InvalidColorOrWidth => {
                write!(
                    formatter,
                    "3D edge color must be opaque and width must be normal and in 0..={MAX_LOGICAL_EDGE_METRIC} logical pixels"
                )
            }
            Self::InvalidDashPattern => {
                write!(
                    formatter,
                    "3D hidden-edge dash and gap must each be normal and in 0..={MAX_LOGICAL_EDGE_METRIC} logical pixels with a finite sum"
                )
            }
        }
    }
}

impl Error for Mesh3dStyleError {}

fn validate_edge_color_width(color: Color, width: LogicalPixels) -> Result<(), Mesh3dStyleError> {
    if color.is_normalized() && color.alpha() == 1.0 && is_stable_edge_metric(width.get()) {
        Ok(())
    } else {
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    }
}

fn is_stable_edge_metric(value: f32) -> bool {
    value.is_normal() && value <= MAX_LOGICAL_EDGE_METRIC
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
            return Err(Mesh3dError::DegenerateDisplayEdge { start, end });
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

#[derive(Debug, PartialEq)]
struct Mesh3dStorage {
    vertices: Vec<Vec3>,
    triangle_indices: Vec<u32>,
    display_edges: Vec<MeshEdge3d>,
    texture_coordinates: Vec<TextureCoordinate2d>,
    vertex_colors: Vec<Color>,
}

#[path = "mesh3d_attributes.rs"]
mod attributes;
pub use attributes::Mesh3dAttributes;

/// Normalized texture coordinate with a top-left origin and downward V axis.
///
/// Both components are finite in `0..=1`. The textured 3D path uses mip level
/// zero and clamp-to-edge addressing. Atlas cells should map geometry to their
/// edge texel centers for nearest sampling. Strict linear-filter isolation also
/// requires host-owned extruded gutters to tolerate interpolation rounding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureCoordinate2d {
    u: f32,
    v: f32,
}

impl TextureCoordinate2d {
    /// Validates normalized U/V without clamping invalid caller input.
    pub fn new(u: f32, v: f32) -> Result<Self, Mesh3dError> {
        if !u.is_finite()
            || !v.is_finite()
            || !(0.0..=1.0).contains(&u)
            || !(0.0..=1.0).contains(&v)
        {
            return Err(Mesh3dError::InvalidTextureCoordinate);
        }
        Ok(Self { u, v })
    }
    /// Horizontal normalized texture coordinate, increasing toward the right.
    pub const fn u(self) -> f32 {
        self.u
    }
    /// Vertical normalized texture coordinate, increasing toward the bottom.
    pub const fn v(self) -> f32 {
        self.v
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
    storage: Arc<Mesh3dStorage>,
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
    /// display edges with coincident endpoints, and display edges outside the
    /// vertex array.
    pub fn with_display_edges(
        vertices: Vec<Vec3>,
        triangle_indices: Vec<u32>,
        display_edges: Vec<MeshEdge3d>,
    ) -> Result<Self, Mesh3dError> {
        Self::with_attributes(
            vertices,
            triangle_indices,
            display_edges,
            Mesh3dAttributes::new(),
        )
    }

    /// Builds surfaces with one validated texture coordinate per model vertex.
    ///
    /// Duplicate model positions with different UVs are valid for atlas seams.
    /// Winding and explicit display edges keep the untextured mesh contract.
    /// A texture/material is attached only when uploading or binding to a
    /// renderer; constructing this data requires no GPU dependencies.
    pub fn textured(
        vertices: Vec<Vec3>,
        texture_coordinates: Vec<TextureCoordinate2d>,
        triangle_indices: Vec<u32>,
        display_edges: Vec<MeshEdge3d>,
    ) -> Result<Self, Mesh3dError> {
        if texture_coordinates.len() != vertices.len() {
            return Err(Mesh3dError::TextureCoordinateCountMismatch {
                vertex_count: vertices.len(),
                coordinate_count: texture_coordinates.len(),
            });
        }
        Self::with_attributes(
            vertices,
            triangle_indices,
            display_edges,
            Mesh3dAttributes::new().with_texture_coordinates(texture_coordinates),
        )
    }

    /// Builds topology with optional UVs and normalized straight-linear colors.
    /// Supplied attributes must match every vertex, including unused vertices.
    /// Missing colors multiply by white; the opaque renderer ignores their
    /// alpha. All buffers are moved into immutable shared storage without copies.
    pub fn with_attributes(
        vertices: Vec<Vec3>,
        triangle_indices: Vec<u32>,
        display_edges: Vec<MeshEdge3d>,
        attributes: Mesh3dAttributes,
    ) -> Result<Self, Mesh3dError> {
        if let Some(coordinates) = &attributes.texture_coordinates
            && coordinates.len() != vertices.len()
        {
            return Err(Mesh3dError::TextureCoordinateCountMismatch {
                vertex_count: vertices.len(),
                coordinate_count: coordinates.len(),
            });
        }
        if let Some(colors) = &attributes.vertex_colors
            && colors.len() != vertices.len()
        {
            return Err(Mesh3dError::VertexColorCountMismatch {
                vertex_count: vertices.len(),
                color_count: colors.len(),
            });
        }
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

        for edge in &display_edges {
            validate_index(edge.start, vertices.len())?;
            validate_index(edge.end, vertices.len())?;
            if vertices[edge.start as usize] == vertices[edge.end as usize] {
                return Err(Mesh3dError::DegenerateDisplayEdge {
                    start: edge.start,
                    end: edge.end,
                });
            }
        }

        let canonical_bytes = display_edges
            .len()
            .checked_mul(std::mem::size_of::<((u32, u32), usize)>())
            .ok_or(Mesh3dError::AllocationFailed {
                requested_bytes: usize::MAX,
            })?;
        let mut canonical_edges = Vec::new();
        canonical_edges
            .try_reserve_exact(display_edges.len())
            .map_err(|_| Mesh3dError::AllocationFailed {
                requested_bytes: canonical_bytes,
            })?;
        for (index, edge) in display_edges.iter().enumerate() {
            canonical_edges.push((edge.canonical(), index));
        }
        canonical_edges.sort_unstable_by_key(|entry| entry.0);
        if let Some(pair) = canonical_edges
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
        {
            let duplicate = display_edges[pair[0].1.max(pair[1].1)];
            return Err(Mesh3dError::DuplicateDisplayEdge {
                start: duplicate.start,
                end: duplicate.end,
            });
        }

        let (bounds_min, bounds_max) = mesh_bounds(&vertices);
        Ok(Self {
            // One fixed-size Arc control block makes Mesh3d cloning O(1)
            // without copying any caller-scale topology buffers.
            storage: Arc::new(Mesh3dStorage {
                vertices,
                triangle_indices,
                display_edges,
                texture_coordinates: attributes.texture_coordinates.unwrap_or_default(),
                vertex_colors: attributes.vertex_colors.unwrap_or_default(),
            }),
            bounds_min,
            bounds_max,
        })
    }

    /// Returns immutable model-space vertices.
    pub fn vertices(&self) -> &[Vec3] {
        &self.storage.vertices
    }

    /// Returns the flat host-wound triangle index list.
    pub fn triangle_indices(&self) -> &[u32] {
        &self.storage.triangle_indices
    }

    /// Returns explicit mathematical edges, excluding triangulation diagonals.
    pub fn display_edges(&self) -> &[MeshEdge3d] {
        &self.storage.display_edges
    }

    /// Returns per-vertex normalized UVs, or an empty slice for untextured data.
    pub fn texture_coordinates(&self) -> &[TextureCoordinate2d] {
        &self.storage.texture_coordinates
    }

    /// Returns normalized straight-linear vertex RGBA, or empty for white.
    /// The opaque surface pass ignores alpha; mathematical edges are unaffected.
    pub fn vertex_colors(&self) -> &[Color] {
        &self.storage.vertex_colors
    }

    /// Returns the number of retained triangles.
    pub fn triangle_count(&self) -> usize {
        self.storage.triangle_indices.len() / 3
    }

    /// Returns the inclusive model-space lower bound.
    pub const fn bounds_min(&self) -> Vec3 {
        self.bounds_min
    }

    /// Returns the inclusive model-space upper bound.
    pub const fn bounds_max(&self) -> Vec3 {
        self.bounds_max
    }

    /// Returns retained CPU capacity bytes for topology and optional attributes.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.storage
            .vertices
            .capacity()
            .saturating_mul(std::mem::size_of::<Vec3>())
            .saturating_add(
                self.storage
                    .triangle_indices
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.storage
                    .display_edges
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshEdge3d>()),
            )
            .saturating_add(
                self.storage
                    .texture_coordinates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<TextureCoordinate2d>()),
            )
            .saturating_add(
                self.storage
                    .vertex_colors
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Color>()),
            )
    }
}

/// Rejection reason for retained 3D topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dError {
    /// A vertex color contains a non-finite or non-normalized channel.
    InvalidVertexColor {
        /// Zero-based source vertex/attribute index, including unused vertices.
        vertex_index: usize,
    },
    /// An explicitly supplied color array must cover every model vertex.
    VertexColorCountMismatch {
        /// Number of model vertices.
        vertex_count: usize,
        /// Number of supplied straight-linear RGBA values.
        color_count: usize,
    },
    /// A U/V component was non-finite or outside normalized `0..=1` bounds.
    InvalidTextureCoordinate,
    /// Textured topology must supply exactly one UV for each model vertex.
    TextureCoordinateCountMismatch {
        /// Number of model-space vertices.
        vertex_count: usize,
        /// Number of normalized texture coordinates provided.
        coordinate_count: usize,
    },
    /// A retained mesh requires at least one model-space vertex.
    EmptyVertices,
    /// A retained mesh requires at least one triangle or display edge.
    EmptyGeometry,
    /// The triangle index list was not divisible by three.
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
    /// A display edge used one index twice or referenced coincident vertices.
    DegenerateDisplayEdge {
        /// First endpoint in the rejected declaration.
        start: u32,
        /// Second endpoint in the rejected declaration.
        end: u32,
    },
    /// The same undirected display edge appeared more than once.
    DuplicateDisplayEdge {
        /// First endpoint in the rejected declaration.
        start: u32,
        /// Second endpoint in the rejected declaration.
        end: u32,
    },
    /// Temporary validation storage could not be reserved.
    AllocationFailed {
        /// Bytes requested for duplicate-edge validation.
        requested_bytes: usize,
    },
}

impl fmt::Display for Mesh3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVertexColor { vertex_index } => write!(
                formatter,
                "3D vertex {vertex_index} color must be finite straight-linear RGBA in 0..=1"
            ),
            Self::VertexColorCountMismatch {
                vertex_count,
                color_count,
            } => write!(
                formatter,
                "{vertex_count} mesh vertices require matching colors, got {color_count}"
            ),
            Self::InvalidTextureCoordinate => {
                write!(formatter, "texture coordinates must be finite in 0..=1")
            }
            Self::TextureCoordinateCountMismatch {
                vertex_count,
                coordinate_count,
            } => write!(
                formatter,
                "{vertex_count} mesh vertices require matching UVs, got {coordinate_count}"
            ),
            Self::EmptyVertices => write!(formatter, "3D mesh requires at least one vertex"),
            Self::EmptyGeometry => {
                write!(formatter, "3D mesh requires triangles or display edges")
            }
            Self::InvalidTriangleIndexCount { index_count } => write!(
                formatter,
                "3D mesh triangle index count, when present, must be divisible by three, got {index_count}"
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
            Self::DegenerateDisplayEdge { start, end } => {
                write!(formatter, "3D display edge {start}-{end} has zero length")
            }
            Self::DuplicateDisplayEdge { start, end } => {
                write!(formatter, "3D display edge {start}-{end} is duplicated")
            }
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for 3D mesh validation"
            ),
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
}
