//! Host-owned fixtures; simulation and chunk models remain outside the library.

use sim_engine::{
    AmbientLight3d, Camera3d, Color, DirectionalLight3d, DynamicMesh3dBudget, Fog3d, ImageBudget,
    ImageSampling, ImageTexelRect, Lighting3d, Mesh3d, Mesh3dAttributes, Mesh3dRenderBudget,
    Mesh3dUploadBudget, MeshStyle3d, Object3dId, Projection3d, Rotation3d, Scene3d,
    SurfaceLighting3d, SurfaceRasterization3d, SurfaceStyle3d, Texture3dOptions,
    Texture3dUpdateBudget, TextureAddressMode3d, TextureCoordinate2d, TextureMaterial3d,
    TextureMipmaps3d, TextureUvTransform3d, Transform3d, Vec2, Vec3, WgpuRenderer, WorldLength,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Repeated,
    Outside,
    OutsideAll,
    OutsideDistinct,
    OutsideAlternating,
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
    Lit,
    Fog,
    LitFog,
    LitSmooth,
}

impl Case {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "repeated" => Ok(Self::Repeated),
            "outside" => Ok(Self::Outside),
            "outside_all" => Ok(Self::OutsideAll),
            "outside_distinct" => Ok(Self::OutsideDistinct),
            "outside_alternating" => Ok(Self::OutsideAlternating),
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
            "lit" => Ok(Self::Lit),
            "fog" => Ok(Self::Fog),
            "lit_fog" => Ok(Self::LitFog),
            "lit_smooth" => Ok(Self::LitSmooth),
            _ => Err(format!("unknown fixture: {value}").into()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Repeated => "repeated",
            Self::Outside => "outside",
            Self::OutsideAll => "outside_all",
            Self::OutsideDistinct => "outside_distinct",
            Self::OutsideAlternating => "outside_alternating",
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
            Self::Lit => "lit",
            Self::Fog => "fog",
            Self::LitFog => "lit_fog",
            Self::LitSmooth => "lit_smooth",
        }
    }

    pub fn changes_mesh(self) -> bool {
        matches!(self, Self::Immutable | Self::Dynamic | Self::Growth)
    }

    pub fn offscreen(self, index: usize) -> bool {
        match self {
            Self::OutsideAll => true,
            Self::OutsideAlternating => !index.is_multiple_of(2),
            Self::Outside | Self::OutsideDistinct | Self::HostHidden => !index.is_multiple_of(10),
            _ => false,
        }
    }

    pub fn culling_control(self) -> bool {
        matches!(
            self,
            Self::Repeated
                | Self::Distinct
                | Self::Outside
                | Self::OutsideAll
                | Self::OutsideDistinct
                | Self::OutsideAlternating
                | Self::HostHidden
        )
    }

    pub fn expected_culled_objects(self, objects: usize) -> usize {
        if self == Self::HostHidden {
            return 0;
        }
        let columns = (objects as f64).sqrt().ceil() as usize;
        (0..objects)
            .filter(|&index| {
                // Odd grids have one row at y=-0.5 whose transformed [0, 0.9]
                // source interval spans zero. The conservative model proof cannot
                // infer a nonzero gap there, even when X is far outside. Keep this
                // intentional fallback control; do not ask the renderer for expected
                // membership or move geometry to make the proof succeed.
                self.offscreen(index)
                    && (columns.is_multiple_of(2) || index / columns != columns / 2)
            })
            .count()
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

    fn lit(self) -> bool {
        matches!(self, Self::Lit | Self::LitFog | Self::LitSmooth)
    }

    fn fogged(self) -> bool {
        matches!(self, Self::Fog | Self::LitFog)
    }

    fn initial_source(self, side: usize) -> Result<Mesh3d> {
        grid(
            side,
            if self == Self::LitSmooth { 0.25 } else { 0.0 },
            self.textured(),
            self.lit(),
        )
    }

    fn surface_style(self) -> Result<SurfaceStyle3d> {
        let color = if self.textured() {
            Color::WHITE
        } else {
            Color::rgb8(82, 176, 233)
        };
        Ok(match self {
            Self::Mask => SurfaceStyle3d::mask(Color::WHITE, 0.5)?,
            Self::Blend => SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5))?,
            _ => SurfaceStyle3d::opaque(color)?,
        }
        .with_lighting(if self.lit() {
            SurfaceLighting3d::Lambert
        } else {
            SurfaceLighting3d::Unlit
        })
        .with_fog(self.fogged()))
    }

    fn configure_environment(self, scene: &mut Scene3d, columns: usize) -> Result<()> {
        if self.lit() {
            scene.set_lighting(
                Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.2)?).with_directional(Some(
                    DirectionalLight3d::new(
                        Vec3::new(0.35, 0.45, 1.0)?,
                        Color::rgb(1.0, 0.95, 0.85),
                        0.75,
                    )?,
                )),
            );
        }
        if self.fogged() {
            // Camera distance is 2 * columns. Scale the fog so both small and
            // dense workloads exercise a visible, nonsaturated contribution.
            scene.set_fog(Some(Fog3d::new(
                Color::rgb(0.08, 0.12, 0.2),
                columns as f32 * 0.5,
                0.35 / columns as f32,
            )?));
        }
        Ok(())
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
            case.initial_source(side)?,
            grid(
                if case == Case::Growth { side * 2 } else { side },
                0.25,
                case.textured(),
                case.lit(),
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
        let style = MeshStyle3d::surface(case.surface_style()?);
        let mut scene = Scene3d::new(Color::rgb8(8, 12, 20))?;
        let columns = (objects as f64).sqrt().ceil() as usize;
        case.configure_environment(&mut scene, columns)?;
        let mut first = None;
        let mut expected_objects = 0;
        for index in 0..objects {
            let outside = case.offscreen(index);
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
            let unique = if matches!(case, Case::Distinct | Case::OutsideDistinct) {
                Some(upload(grid(side, 0.0, false, false)?)?)
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

fn grid(side: usize, center_height: f32, textured: bool, lit: bool) -> Result<Mesh3d> {
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
    let mut attributes = if textured {
        Mesh3dAttributes::new().with_texture_coordinates(
            vertices
                .iter()
                .map(|p| TextureCoordinate2d::new(p.x(), p.y()))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
    } else {
        Mesh3dAttributes::new()
    };
    if lit {
        let normals = vertices
            .iter()
            .map(|point| {
                // Host-supplied analytic normals of this height field, not a
                // renderer-generated lighting model. Flat grids use exact +Z.
                if center_height == 0.0 {
                    return Ok(Vec3::Z);
                }
                let x = point.x() * std::f32::consts::PI;
                let y = point.y() * std::f32::consts::PI;
                let gradient = center_height * std::f32::consts::PI;
                Vec3::new(
                    -gradient * x.cos() * y.sin(),
                    -gradient * x.sin() * y.cos(),
                    1.0,
                )?
                .normalized()
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        attributes = attributes.with_normals(normals)?;
    }
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
            let first = grid(side, 0.0, false, false).unwrap();
            let second = grid(side, 0.25, false, false).unwrap();
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

    #[test]
    fn environment_cases_keep_repeated_geometry_and_enable_real_validation_work() {
        for (case, lit, fogged) in [
            (Case::Repeated, false, false),
            (Case::Lit, true, false),
            (Case::Fog, false, true),
            (Case::LitFog, true, true),
        ] {
            assert_eq!(Case::parse(case.name()).unwrap(), case);
            assert!(!case.changes_mesh());
            assert!(!case.textured());
            for frame in [0, 1, 119] {
                assert_eq!(case.revision(frame), 0);
            }
            let style = case.surface_style().unwrap();
            assert_eq!(style.lighting() == SurfaceLighting3d::Lambert, lit);
            assert_eq!(style.fog_enabled(), fogged);
            for (side, columns) in [(1, 23), (32, 8)] {
                let control = grid(side, 0.0, false, false).unwrap();
                let source = case.initial_source(side).unwrap();
                assert_eq!(source.vertices(), control.vertices());
                assert_eq!(source.triangle_indices(), control.triangle_indices());
                assert_eq!(
                    source.normals().len(),
                    if lit { (side + 1).pow(2) } else { 0 }
                );
                assert!(source.normals().iter().all(|normal| *normal == Vec3::Z));
                let mut scene = Scene3d::new(Color::BLACK).unwrap();
                case.configure_environment(&mut scene, columns).unwrap();
                let lighting = scene.lighting();
                assert_eq!(lighting.directional().is_some(), lit);
                if let Some(sun) = lighting.directional() {
                    assert_eq!(sun.intensity(), 0.75);
                    assert_eq!(lighting.ambient().intensity(), 0.2);
                    assert!(sun.direction().z() > 0.5);
                    assert!(sun.direction().x() > 0.0 && sun.direction().y() > 0.0);
                }
                assert_eq!(scene.fog().is_some(), fogged);
                if let Some(fog) = scene.fog() {
                    let distance = columns as f32 * 2.0;
                    assert!(fog.start() < distance);
                    assert!(fog.density() > 0.0);
                    let transmission = (-(distance - fog.start()) * fog.density()).exp();
                    assert!((0.5..0.7).contains(&transmission));
                }
            }
        }
    }

    #[test]
    fn lit_grid_normals_follow_host_height_field_and_are_normalized() {
        let source = grid(32, 0.25, false, true).unwrap();
        assert_eq!(source.normals().len(), source.vertices().len());
        assert!(source.normals().iter().any(|normal| *normal != Vec3::Z));
        for (point, normal) in source.vertices().iter().zip(source.normals()) {
            let length_squared =
                normal.x() * normal.x() + normal.y() * normal.y() + normal.z() * normal.z();
            assert!((length_squared - 1.0).abs() < 1e-6);
            let x = point.x() * std::f32::consts::PI;
            let y = point.y() * std::f32::consts::PI;
            let gradient = 0.25 * std::f32::consts::PI;
            assert!((normal.x() + normal.z() * gradient * x.cos() * y.sin()).abs() < 1e-6);
            assert!((normal.y() + normal.z() * gradient * x.sin() * y.cos()).abs() < 1e-6);
            assert!(normal.z() > 0.0);
        }
    }

    #[test]
    fn smooth_lit_fixture_renders_static_curvature_with_many_distinct_normals() {
        let case = Case::parse("lit_smooth").unwrap();
        assert_eq!(case, Case::LitSmooth);
        assert_eq!(case.name(), "lit_smooth");
        assert!(!case.changes_mesh());
        assert!(!case.textured());
        for frame in [0, 1, 59, 119] {
            assert_eq!(case.revision(frame), 0);
        }
        let source = case.initial_source(32).unwrap();
        let flat = Case::Lit.initial_source(32).unwrap();
        assert_eq!(source.triangle_indices(), flat.triangle_indices());
        assert_eq!(source.vertices().len(), 33 * 33);
        assert_eq!(source.normals().len(), source.vertices().len());
        assert_ne!(source.vertices(), flat.vertices());
        assert_eq!(source.bounds_max().z(), 0.25);
        assert_eq!(source, case.initial_source(32).unwrap());
        // More than eight successive distinct normals force the bounded
        // validation memo to saturate instead of measuring its flat-grid hit path.
        let leading_normals: std::collections::HashSet<_> = source
            .normals()
            .iter()
            .take(9)
            .map(|normal| [normal.x(), normal.y(), normal.z()].map(f32::to_bits))
            .collect();
        assert_eq!(leading_normals.len(), 9);
        let style = case.surface_style().unwrap();
        assert_eq!(style.lighting(), SurfaceLighting3d::Lambert);
        assert!(!style.fog_enabled());
        let mut smooth_scene = Scene3d::new(Color::BLACK).unwrap();
        let mut flat_scene = Scene3d::new(Color::BLACK).unwrap();
        case.configure_environment(&mut smooth_scene, 8).unwrap();
        Case::Lit.configure_environment(&mut flat_scene, 8).unwrap();
        assert_eq!(smooth_scene.lighting(), flat_scene.lighting());
        assert_eq!(
            smooth_scene.lighting().directional().unwrap().intensity(),
            0.75
        );
        assert_eq!(smooth_scene.fog(), None);
    }
}
