//! Optional, validated application-owned model-vertex attributes.

use super::{Color, Mesh3dError, TextureCoordinate2d, Vec3};

/// Extensible model-vertex data for [`super::Mesh3d::with_attributes`].
///
/// Omitted colors multiply by white. Supplied arrays must match the model
/// vertex count exactly, including unused vertices; an explicit empty array
/// is not omission. No normals or simulation quantities are generated here.
///
/// ```
/// use sim_engine::{Color, Mesh3d, Mesh3dAttributes, Vec3};
/// let attributes = Mesh3dAttributes::new().with_vertex_colors(vec![
///     Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0), Color::rgb(0.0, 0.0, 1.0),
/// ])?;
/// let mesh = Mesh3d::with_attributes(
///     vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![0, 1, 2], Vec::new(), attributes,
/// )?;
/// assert_eq!(mesh.vertex_colors().len(), 3);
/// # Ok::<(), sim_engine::Mesh3dError>(())
/// ```
#[derive(Debug, Default, PartialEq)]
pub struct Mesh3dAttributes {
    pub(super) texture_coordinates: Option<Vec<TextureCoordinate2d>>,
    pub(super) vertex_colors: Option<Vec<Color>>,
    pub(super) normals: Option<Vec<Vec3>>,
}

impl Mesh3dAttributes {
    /// Omits every optional attribute without allocating backing arrays.
    pub const fn new() -> Self {
        Self {
            texture_coordinates: None,
            vertex_colors: None,
            normals: None,
        }
    }

    /// Supplies normalized model UVs; mesh construction validates the count.
    pub fn with_texture_coordinates(mut self, coordinates: Vec<TextureCoordinate2d>) -> Self {
        self.texture_coordinates = Some(coordinates);
        self
    }

    /// Supplies normalized, straight-linear RGBA colors without clamping.
    ///
    /// Reports the source vertex index of the first non-finite or out-of-range
    /// channel. Mesh construction separately checks the exact vertex count.
    /// Opaque surfaces ignore vertex alpha; Mask and Blend use the combined
    /// alpha. RGBA is interpolated perspectively and multiplies surface/texture tint.
    /// Mathematical display edges keep their independent wireframe colors.
    pub fn with_vertex_colors(mut self, colors: Vec<Color>) -> Result<Self, Mesh3dError> {
        if let Some(vertex_index) = colors.iter().position(|color| !color.is_normalized()) {
            return Err(Mesh3dError::InvalidVertexColor { vertex_index });
        }
        self.vertex_colors = Some(colors);
        Ok(self)
    }

    /// Supplies finite nonzero model normals, normalized robustly in place.
    /// No face normals are inferred. Every value, including unused vertices, is
    /// validated; mesh construction separately requires an exact vertex count.
    /// Nonuniform transforms use inverse-transpose direction transport, followed
    /// by perspective interpolation and fragment normalization, not vertex normalization.
    pub fn with_normals(mut self, mut normals: Vec<Vec3>) -> Result<Self, Mesh3dError> {
        for (vertex_index, normal) in normals.iter_mut().enumerate() {
            *normal = normal
                .normalized()
                .map_err(|_| Mesh3dError::InvalidVertexNormal { vertex_index })?;
        }
        self.normals = Some(normals);
        Ok(self)
    }

    /// Supplied normalized model normals, distinguishing omission from empty.
    pub fn normals(&self) -> Option<&[Vec3]> {
        self.normals.as_deref()
    }

    /// Supplied UVs, distinguishing omission from an explicit empty array.
    pub fn texture_coordinates(&self) -> Option<&[TextureCoordinate2d]> {
        self.texture_coordinates.as_deref()
    }

    /// Supplied straight-linear colors, distinguishing omission from empty.
    pub fn vertex_colors(&self) -> Option<&[Color]> {
        self.vertex_colors.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mesh3d, Vec3};

    #[test]
    fn normals_validate_exact_count_and_normalize_extreme_directions() {
        let normal = Vec3::new(f32::MAX, f32::MAX, f32::MAX).unwrap();
        let attributes = Mesh3dAttributes::new()
            .with_normals(vec![normal; 3])
            .unwrap();
        let normalized = attributes.normals().unwrap()[0];
        assert!((normalized.x() - 1.0 / 3.0_f32.sqrt()).abs() < 1e-6);
        let tiny = Vec3::new(f32::from_bits(1), 0.0, 0.0).unwrap();
        assert_eq!(
            Mesh3dAttributes::new()
                .with_normals(vec![tiny])
                .unwrap()
                .normals()
                .unwrap(),
            &[Vec3::X]
        );
        assert_eq!(
            Mesh3dAttributes::new().with_normals(vec![normal, Vec3::ZERO]),
            Err(Mesh3dError::InvalidVertexNormal { vertex_index: 1 })
        );
        for count in [0, 1, 2, 4] {
            assert_eq!(
                Mesh3d::with_attributes(
                    vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                    vec![0, 1, 2],
                    vec![],
                    Mesh3dAttributes::new()
                        .with_normals(vec![Vec3::Z; count])
                        .unwrap()
                ),
                Err(Mesh3dError::NormalCountMismatch {
                    vertex_count: 3,
                    normal_count: count
                })
            );
        }
        let mesh = Mesh3d::with_attributes(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            vec![0, 1, 2],
            vec![],
            attributes,
        )
        .unwrap();
        assert_eq!(mesh.normals().as_ptr(), mesh.clone().normals().as_ptr());
        let plain = Mesh3d::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![0, 1, 2]).unwrap();
        assert_eq!(
            mesh.recovery_memory_bytes(),
            plain.recovery_memory_bytes() + 3 * std::mem::size_of::<Vec3>()
        );
        assert!(plain.normals().is_empty());
    }

    #[test]
    fn vertex_colors_validate_all_channels_and_attribute_indices() {
        for channel in 0..4 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
                let mut components = [0.5; 4];
                components[channel] = value;
                let invalid =
                    Color::rgba(components[0], components[1], components[2], components[3]);
                assert_eq!(
                    Mesh3dAttributes::new().with_vertex_colors(vec![Color::WHITE, invalid]),
                    Err(Mesh3dError::InvalidVertexColor { vertex_index: 1 })
                );
            }
        }
        assert!(
            Mesh3dAttributes::new()
                .with_vertex_colors(vec![Color::rgba(0.0, f32::from_bits(1), 1.0, 0.0)])
                .is_ok()
        );
    }

    #[test]
    fn optional_colors_require_exact_count_preserve_alpha_and_share_storage() {
        let vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        for count in [0, 1, 3, 5] {
            let attributes = Mesh3dAttributes::new()
                .with_vertex_colors(vec![Color::WHITE; count])
                .unwrap();
            assert_eq!(
                Mesh3d::with_attributes(vertices.clone(), vec![0, 1, 2], vec![], attributes),
                Err(Mesh3dError::VertexColorCountMismatch {
                    vertex_count: 4,
                    color_count: count
                })
            );
        }
        let color = Color::rgba(0.25, 0.5, 1.0, 0.0);
        let mut colors = Vec::with_capacity(9);
        colors.resize(4, color);
        let capacity_bytes = colors.capacity() * std::mem::size_of::<Color>();
        let colored = Mesh3d::with_attributes(
            vertices.clone(),
            vec![0, 1, 2],
            vec![],
            Mesh3dAttributes::new().with_vertex_colors(colors).unwrap(),
        )
        .unwrap();
        let plain = Mesh3d::new(vertices, vec![0, 1, 2]).unwrap();
        assert!(plain.vertex_colors().is_empty());
        assert_eq!(
            colored.recovery_memory_bytes(),
            plain.recovery_memory_bytes() + capacity_bytes
        );
        assert_eq!(colored.vertex_colors()[3], color);
        let alias = colored.clone();
        assert_eq!(
            colored.vertex_colors().as_ptr(),
            alias.vertex_colors().as_ptr()
        );
    }
}
