//! Host-owned fixtures; simulation and chunk models remain outside the library.

use sim_engine::{
    Camera3d, Color, DynamicMesh3dBudget, ImageBudget, ImageSampling, ImageTexelRect, Mesh3d,
    Mesh3dAttributes, Mesh3dRenderBudget, Mesh3dUploadBudget, MeshStyle3d, Object3dId,
    Projection3d, Rotation3d, Scene3d, SurfaceRasterization3d, SurfaceStyle3d, Texture3dOptions,
    Texture3dUpdateBudget, TextureAddressMode3d, TextureCoordinate2d, TextureMaterial3d,
    TextureMipmaps3d, TextureUvTransform3d, Transform3d, Vec2, Vec3, WgpuRenderer, WorldLength,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Repeated,
    Outside,
    HostHidden,
    Immutable,
    Dynamic,
    Growth,
    Distinct,
    Crossing,
    Textured,
    Mask,
    Blend,
    TextureUpdate,
    PreparedText,
}

impl Case {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "repeated" => Ok(Self::Repeated),
            "outside" => Ok(Self::Outside),
            "host_hidden" => Ok(Self::HostHidden),
            "immutable" => Ok(Self::Immutable),
            "dynamic" => Ok(Self::Dynamic),
            "growth" => Ok(Self::Growth),
            "distinct" => Ok(Self::Distinct),
            "crossing" => Ok(Self::Crossing),
            "textured" => Ok(Self::Textured),
            "mask" => Ok(Self::Mask),
            "blend" => Ok(Self::Blend),
            "texture_update" => Ok(Self::TextureUpdate),
            "prepared_text" => Ok(Self::PreparedText),
            _ => Err(format!("unknown fixture: {value}").into()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Repeated => "repeated",
            Self::Outside => "outside",
            Self::HostHidden => "host_hidden",
            Self::Immutable => "immutable",
            Self::Dynamic => "dynamic",
            Self::Growth => "growth",
            Self::Distinct => "distinct",
            Self::Crossing => "crossing",
            Self::Textured => "textured",
            Self::Mask => "mask",
            Self::Blend => "blend",
            Self::TextureUpdate => "texture_update",
            Self::PreparedText => "prepared_text",
        }
    }

    pub fn changes_mesh(self) -> bool {
        matches!(self, Self::Immutable | Self::Dynamic | Self::Growth)
    }

    pub fn textured(self) -> bool {
        matches!(
            self,
            Self::Textured | Self::Mask | Self::Blend | Self::TextureUpdate
        )
    }

    pub fn revision(self, frame: usize) -> usize {
        if self == Self::Growth {
            1
        } else if self.changes_mesh() {
            frame % 2
        } else {
            0
        }
    }
}

#[derive(Default)]
pub struct UpdateCounters {
    pub bytes: usize,
    pub calls: usize,
    pub gpu_allocations: usize,
    pub reused: bool,
    pub detached: bool,
    pub capacity: usize,
    pub scratch: usize,
    pub scratch_reallocations: usize,
    pub texture_uploaded_bytes: usize,
    pub texture_upload_calls: usize,
    pub texture_copy_bytes: usize,
    pub texture_gpu_allocations: usize,
    pub texture_staging_bytes: usize,
    pub peak_mesh_staging_bytes: usize,
    pub peak_mesh_gpu_bytes: usize,
    pub peak_mesh_recovery_bytes: usize,
    pub peak_texture_gpu_bytes: usize,
    pub peak_texture_recovery_bytes: usize,
}

pub struct Workload {
    pub scene: Scene3d,
    pub camera: Camera3d,
    pub budget: Mesh3dRenderBudget,
    pub expected_objects: usize,
    pub expected_triangles: usize,
    pub host_snapshot_bytes: usize,
    pub source_triangles: usize,
    columns: usize,
    sources: [Mesh3d; 2],
    object: Object3dId,
    case: Case,
}

impl Workload {
    pub fn new(
        renderer: &WgpuRenderer,
        case: Case,
        objects: usize,
        side: usize,
        policy: SurfaceRasterization3d,
    ) -> Result<Self> {
        let sources = [
            grid(side, 0.0, case.textured())?,
            grid(
                if case == Case::Growth { side * 2 } else { side },
                0.25,
                case.textured(),
            )?,
        ];
        let texture = if case.textured() {
            // Crop the central tile from a deliberately hostile red/blue atlas.
            // Every mip is generated after cropping; hardware repeat sees only the tile.
            let mut pixels = vec![0; 64 * 32 * 4];
            for y in 0..32 {
                for x in 0..64 {
                    let rgba = if x < 16 {
                        [255, 0, 0, 255]
                    } else if x >= 48 {
                        [0, 0, 255, 255]
                    } else if (x / 4 + y / 4) % 2 == 0 {
                        [210, 220, 80, 255]
                    } else {
                        [24, 80, 24, if case == Case::Mask { 0 } else { 255 }]
                    };
                    pixels[(y * 64 + x) * 4..(y * 64 + x + 1) * 4].copy_from_slice(&rgba);
                }
            }
            let options = Texture3dOptions::new().with_alpha_preservation(true);
            let atlas = renderer.create_texture3d_rgba8_with_options(
                64,
                32,
                pixels,
                ImageBudget::default(),
                options,
            )?;
            Some(renderer.crop_texture3d_tile(
                &atlas,
                ImageTexelRect::new(16, 0, 32, 32)?,
                ImageBudget::default(),
                options.with_mipmaps(TextureMipmaps3d::Generate),
            )?)
        } else {
            None
        };
        let material = texture
            .as_ref()
            .map(|texture| -> Result<_> {
                Ok(
                    TextureMaterial3d::with_alpha(texture, ImageSampling::Linear, Color::WHITE)?
                        .with_address_mode(TextureAddressMode3d::Repeat)
                        .with_uv_transform(TextureUvTransform3d::new(
                            Vec2::new(8.0, 8.0),
                            Vec2::new(-0.25, 0.125),
                        )?),
                )
            })
            .transpose()?;
        let upload = |source: Mesh3d| -> Result<_> {
            let mesh = renderer.create_mesh3d(source)?;
            Ok(if let Some(material) = material.as_ref() {
                renderer.with_mesh3d_material(&mesh, material)?
            } else {
                mesh
            })
        };
        let mesh = upload(sources[0].clone())?;
        let color = Color::rgb8(82, 176, 233);
        let style = MeshStyle3d::surface(match case {
            Case::Mask => SurfaceStyle3d::mask(Color::WHITE, 0.5)?,
            Case::Blend => SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5))?,
            _ => SurfaceStyle3d::opaque(if case.textured() { Color::WHITE } else { color })?,
        });
        let mut scene = Scene3d::new(Color::rgb8(8, 12, 20))?;
        let columns = (objects as f64).sqrt().ceil() as usize;
        let mut first = None;
        let mut expected_objects = 0;
        for index in 0..objects {
            let outside = matches!(case, Case::Outside | Case::HostHidden) && index % 10 != 0;
            let x = (index % columns) as f32 - columns as f32 * 0.5;
            let y = (index / columns) as f32 - columns as f32 * 0.5;
            let transform = Transform3d::new(
                Vec3::new(
                    x + if outside { 10_000.0 } else { 0.0 },
                    y,
                    if case == Case::Blend {
                        (index % 4) as f32 * 0.2
                    } else {
                        0.0
                    },
                )?,
                Rotation3d::IDENTITY,
                Vec3::new(0.9, 0.9, 0.9)?,
            )?;
            // Distinct control owns distinct CPU snapshots as well as GPU buffers.
            let unique = if case == Case::Distinct {
                Some(upload(grid(side, 0.0, false)?)?)
            } else {
                None
            };
            let id = scene.try_push(unique.as_ref().unwrap_or(&mesh), transform, style)?;
            first.get_or_insert(id);
            if case == Case::HostHidden && outside {
                scene.set_visible(id, false)?;
            } else {
                expected_objects += 1;
            }
        }
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, columns as f32 * 2.0)?,
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(
                WorldLength::new(columns as f32 * 1.1)?,
                1280.0 / 720.0,
                WorldLength::new(0.1)?,
                WorldLength::new(columns as f32 * 4.0)?,
            )?,
        )?;
        let expected_triangles = expected_objects * mesh.triangle_count();
        Ok(Self {
            scene,
            camera,
            budget: Mesh3dRenderBudget::default()
                .with_surface_policy(policy)
                .with_max_surface_triangles(expected_triangles.saturating_mul(4)),
            expected_objects,
            expected_triangles,
            host_snapshot_bytes: sources.iter().map(Mesh3d::recovery_memory_bytes).sum(),
            source_triangles: objects * mesh.triangle_count(),
            columns,
            sources,
            object: first.ok_or("at least one object is required")?,
            case,
        })
    }

    pub fn update(&mut self, renderer: &mut WgpuRenderer, frame: usize) -> Result<UpdateCounters> {
        let mut result = UpdateCounters::default();
        if self.case == Case::Crossing {
            // Orthographic side-plane crossings, deterministic and repeatable.
            let x = (frame as f32 * 0.07).sin() * self.columns as f32 * 0.8;
            self.camera = Camera3d::look_at(
                Vec3::new(x, 0.0, self.columns as f32 * 2.0)?,
                Vec3::new(x, 0.0, 0.0)?,
                Vec3::Y,
                self.camera.projection(),
            )?;
        }
        if self.case == Case::TextureUpdate {
            let color = if frame.is_multiple_of(2) {
                [255, 80, 20, 255]
            } else {
                [20, 255, 80, 255]
            };
            let mut pixels = [0u8; 4 * 4 * 4];
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.copy_from_slice(&color);
            }
            let report = renderer.update_scene3d_texture_region(
                &mut self.scene,
                self.object,
                ImageTexelRect::new((frame % 28) as u32, 4, 4, 4)?,
                &pixels,
                16,
                Texture3dUpdateBudget::default(),
            )?;
            result.texture_uploaded_bytes = report.uploaded_bytes();
            result.texture_upload_calls = report.upload_calls();
            result.texture_copy_bytes = report.gpu_copy_bytes();
            result.texture_gpu_allocations = report.gpu_allocation_count();
            result.texture_staging_bytes = report.staging_bytes();
            result.peak_texture_gpu_bytes = report.peak_gpu_bytes();
            result.peak_texture_recovery_bytes = report.peak_recovery_bytes();
        }
        if !self.case.changes_mesh() {
            return Ok(result);
        }
        // Growth includes a fresh small allocation followed by a larger update.
        // This deliberate reset is counted, not hidden by warmup or reused capacity.
        if self.case == Case::Growth {
            let small = renderer.create_mesh3d(self.sources[0].clone())?;
            result.bytes += small.gpu_allocation_bytes();
            result.calls += 2;
            result.gpu_allocations += 2;
            self.scene.set_mesh(self.object, &small)?;
        }
        let source = self.sources[self.case.revision(frame)].clone();
        self.expected_triangles = (self.expected_objects - 1) * self.sources[0].triangle_count()
            + source.triangle_count();
        self.source_triangles = (self.scene.statistics().object_count() - 1)
            * self.sources[0].triangle_count()
            + source.triangle_count();
        if self.case == Case::Immutable {
            let mesh = renderer.create_mesh3d_with_budget(source, Mesh3dUploadBudget::default())?;
            result.bytes = mesh.gpu_allocation_bytes();
            result.capacity = mesh.gpu_allocation_bytes();
            // This fixture has positions and triangle indices, with no UVs/edges.
            result.calls = 2;
            result.gpu_allocations = 2;
            self.scene.set_mesh(self.object, &mesh)?;
        } else {
            let report = renderer.update_scene3d_mesh(
                &mut self.scene,
                self.object,
                source,
                DynamicMesh3dBudget::default(),
            )?;
            result.bytes += report.uploaded_bytes();
            result.calls += report.upload_calls();
            result.gpu_allocations += report.gpu_allocation_count();
            result.reused = report.reused_buffers();
            result.detached = report.detached_aliases();
            result.capacity = report.capacity_gpu_bytes();
            result.scratch = report.scratch_capacity_bytes();
            result.scratch_reallocations = report.scratch_reallocations();
            result.peak_mesh_staging_bytes = report.peak_staging_bytes();
            result.peak_mesh_gpu_bytes = report.peak_gpu_bytes();
            result.peak_mesh_recovery_bytes = report.peak_recovery_bytes();
        }
        Ok(result)
    }
}

fn grid(side: usize, center_height: f32, textured: bool) -> Result<Mesh3d> {
    let mut vertices = Vec::with_capacity((side + 1) * (side + 1));
    for row in 0..=side {
        for column in 0..=side {
            let x = column as f32 / side as f32;
            let y = row as f32 / side as f32;
            let z =
                center_height * (x * std::f32::consts::PI).sin() * (y * std::f32::consts::PI).sin();
            vertices.push(Vec3::new(x, y, z)?);
        }
    }
    let mut indices = Vec::with_capacity(side * side * 6);
    for row in 0..side {
        for column in 0..side {
            let first = (row * (side + 1) + column) as u32;
            let next = first + side as u32 + 1;
            indices.extend([first, first + 1, next + 1, first, next + 1, next]);
        }
    }
    let attributes = if textured {
        Mesh3dAttributes::new().with_texture_coordinates(
            vertices
                .iter()
                .map(|p| TextureCoordinate2d::new(p.x(), p.y()))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    } else {
        Mesh3dAttributes::new()
    };
    Ok(Mesh3d::with_attributes(
        vertices,
        indices,
        vec![],
        attributes,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_revisions_have_equal_capacity_and_distinct_geometry() {
        for side in [1, 4, 32] {
            let first = grid(side, 0.0, false).unwrap();
            let second = grid(side, 0.25, false).unwrap();
            assert_eq!(first.triangle_count(), side * side * 2);
            assert_eq!(
                first.recovery_memory_bytes(),
                second.recovery_memory_bytes()
            );
            assert_eq!(first.triangle_indices(), second.triangle_indices());
            if side > 1 {
                assert_ne!(first.vertices(), second.vertices());
            }
        }
    }

    #[test]
    fn reported_revision_matches_the_update_source_even_for_odd_trials() {
        for frame in 0..5 {
            assert_eq!(Case::Growth.revision(frame), 1);
            assert_eq!(Case::Dynamic.revision(frame), frame % 2);
            assert_eq!(Case::Immutable.revision(frame), frame % 2);
            assert_eq!(Case::Repeated.revision(frame), 0);
        }
    }
}
