//! Validated retained model topology, vertex attributes and visual materials.

use std::{error::Error, fmt, sync::Arc};

use crate::{Color, Vec3};

mod attributes;
mod environment;
mod material;
mod style;
#[cfg(test)]
mod tests;
mod uv;

pub use attributes::Mesh3dAttributes;
pub use environment::{
    AmbientLight3d, DirectionalLight3d, Fog3d, Fog3dError, Lighting3d, Lighting3dError,
    SurfaceLighting3d,
};
pub use material::{SurfaceAlphaMode3d, SurfaceSidedness3d, SurfaceStyle3d};
pub use style::{Mesh3dStyleError, MeshStyle3d, WireframeStyle3d};
pub use uv::{TextureAddressMode3d, TextureUvTransform3d, TextureUvTransformError};

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
    normals: Vec<Vec3>,
}

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
    minimum_nonzero_components: [f32; 3],
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
    /// Missing colors multiply by white. Opaque ignores alpha; Mask and Blend
    /// use combined vertex/material/texture alpha. All buffers are moved into immutable shared storage without copies.
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
        if let Some(normals) = &attributes.normals
            && normals.len() != vertices.len()
        {
            return Err(Mesh3dError::NormalCountMismatch {
                vertex_count: vertices.len(),
                normal_count: normals.len(),
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

        let (bounds_min, bounds_max, minimum_nonzero_components) = mesh_bounds(&vertices);
        Ok(Self {
            // One fixed-size Arc control block makes Mesh3d cloning O(1)
            // without copying any caller-scale topology buffers.
            storage: Arc::new(Mesh3dStorage {
                vertices,
                triangle_indices,
                display_edges,
                texture_coordinates: attributes.texture_coordinates.unwrap_or_default(),
                vertex_colors: attributes.vertex_colors.unwrap_or_default(),
                normals: attributes.normals.unwrap_or_default(),
            }),
            bounds_min,
            bounds_max,
            minimum_nonzero_components,
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

    /// Returns normalized application-supplied model normals, or an empty slice.
    /// Lambert surfaces require normals; Unlit surfaces never evaluate them.
    pub fn normals(&self) -> &[Vec3] {
        &self.storage.normals
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

    // Bounds alone cannot detect a subnormal source hidden between ordinary
    // extrema. Immutable source metadata also covers unreferenced vertices.
    #[cfg(feature = "wgpu")]
    pub(crate) const fn minimum_nonzero_components(&self) -> [f32; 3] {
        self.minimum_nonzero_components
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
            .saturating_add(
                self.storage
                    .normals
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec3>()),
            )
    }
}

/// Rejection reason for retained 3D topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dError {
    /// A supplied normal is zero or cannot be normalized to a finite direction.
    InvalidVertexNormal {
        /// Source attribute index, including unused vertices.
        vertex_index: usize,
    },
    /// Explicit normals must match every model vertex, including unused ones.
    NormalCountMismatch {
        /// Number of model vertices.
        vertex_count: usize,
        /// Number of supplied normals.
        normal_count: usize,
    },
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
            Self::InvalidVertexNormal { vertex_index } => write!(
                formatter,
                "3D vertex {vertex_index} normal must be a nonzero finite direction"
            ),
            Self::NormalCountMismatch {
                vertex_count,
                normal_count,
            } => write!(
                formatter,
                "{vertex_count} mesh vertices require matching normals, got {normal_count}"
            ),
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

fn mesh_bounds(vertices: &[Vec3]) -> (Vec3, Vec3, [f32; 3]) {
    let first = vertices[0];
    let mut minimum = [first.x(), first.y(), first.z()];
    let mut maximum = minimum;
    let mut minimum_nonzero = [f32::INFINITY; 3];
    for vertex in vertices {
        let components = [vertex.x(), vertex.y(), vertex.z()];
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(components[axis]);
            maximum[axis] = maximum[axis].max(components[axis]);
            if components[axis] != 0.0 {
                minimum_nonzero[axis] = minimum_nonzero[axis].min(components[axis].abs());
            }
        }
    }
    (
        Vec3::new_unchecked(minimum[0], minimum[1], minimum[2]),
        Vec3::new_unchecked(maximum[0], maximum[1], maximum[2]),
        minimum_nonzero,
    )
}
