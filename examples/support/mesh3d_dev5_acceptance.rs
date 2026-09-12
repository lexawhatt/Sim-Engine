//! Public restoration/mutation checks across the gallery's real device recovery.

use sim_engine::{
    AmbientLight3d, Camera3d, Color, DynamicMesh3dBudget, GpuTimingId, ImageBudget, ImageSampling,
    Lighting3d, Mesh3d, Mesh3dAttributes, Mesh3dUploadBudget, MeshEdge3d, MeshStyle3d,
    RenderTarget3d, RetainedMesh3d, Scene3d, SurfaceLighting3d, SurfaceStyle3d, Texture3dOptions,
    TextureAddressMode3d, TextureCoordinate2d, TextureMaterial3d, TextureMipmaps3d,
    TextureUvTransform3d, Transform3d, Vec2, Vec3, WgpuRenderer,
};

use super::ExampleResult;

/// Retains stale standalone aliases until the existing gallery recovery occurs.
pub struct StandaloneRecovery {
    cases: Vec<Case>,
    submitted_cases: usize,
    mutations_checked: bool,
}

struct Case {
    name: &'static str,
    original: RetainedMesh3d,
    expected: Snapshot,
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    edges: Vec<MeshEdge3d>,
    coordinates: Vec<TextureCoordinate2d>,
    colors: Vec<Color>,
    normals: Vec<Vec3>,
    budget: Mesh3dUploadBudget,
    gpu_bytes: usize,
    cpu_bytes: usize,
    material: Option<MaterialSnapshot>,
}

#[derive(Debug, PartialEq)]
struct MaterialSnapshot {
    options: Texture3dOptions,
    budget: ImageBudget,
    dimensions: Vec<(u32, u32)>,
    levels: Vec<Vec<u8>>,
    sampling: ImageSampling,
    tint: Color,
    mapping: TextureUvTransform3d,
    addressing: TextureAddressMode3d,
}

impl Snapshot {
    fn read(mesh: &RetainedMesh3d) -> ExampleResult<Self> {
        let material = mesh.material().map(MaterialSnapshot::read).transpose()?;
        let source = mesh.source();
        Ok(Self {
            vertices: source.vertices().to_vec(),
            indices: source.triangle_indices().to_vec(),
            edges: source.display_edges().to_vec(),
            coordinates: source.texture_coordinates().to_vec(),
            colors: source.vertex_colors().to_vec(),
            normals: source.normals().to_vec(),
            budget: mesh.budget(),
            gpu_bytes: mesh.gpu_allocation_bytes(),
            cpu_bytes: mesh.recovery_memory_bytes(),
            material,
        })
    }
}

impl MaterialSnapshot {
    fn read(material: &TextureMaterial3d) -> ExampleResult<Self> {
        let texture = material.texture();
        let mut dimensions = Vec::new();
        let mut levels = Vec::new();
        for level in 0..texture.mip_level_count() {
            dimensions.push(
                texture
                    .mip_level_size(level)
                    .ok_or("missing mip dimensions")?,
            );
            levels.push(
                texture
                    .mip_level_pixels(level)
                    .ok_or("missing mip bytes")?
                    .to_vec(),
            );
        }
        Ok(Self {
            options: texture.options(),
            budget: texture.budget(),
            dimensions,
            levels,
            sampling: material.sampling(),
            tint: material.tint(),
            mapping: material.uv_transform(),
            addressing: material.address_mode(),
        })
    }
}

impl StandaloneRecovery {
    /// Creates all four D401 controls with live geometry smaller than its reserve.
    pub fn new(renderer: &mut WgpuRenderer) -> ExampleResult<Self> {
        let source = panel()?;
        let initial = renderer.create_mesh3d(source.clone())?;
        let mut staging_scene = Scene3d::new(Color::BLACK)?;
        let object = staging_scene.try_push(&initial, Transform3d::IDENTITY, style()?)?;
        renderer.update_scene3d_mesh(
            &mut staging_scene,
            object,
            source,
            DynamicMesh3dBudget::default().with_minimum_capacity(12, 18, 8),
        )?;
        let reserved = staging_scene.remove(object)?.mesh().clone();
        if reserved.gpu_allocation_bytes() <= initial.gpu_allocation_bytes() {
            return Err(
                "standalone recovery control did not reserve extra geometry capacity".into(),
            );
        }
        let options = Texture3dOptions::new()
            .with_alpha_preservation(true)
            .with_mipmaps(TextureMipmaps3d::Generate);
        let mut cases = Vec::new();
        for (index, name) in [
            "transparent_texel",
            "alpha_tint",
            "signed_uv_repeat",
            "alpha_policy_with_opaque_pixels",
        ]
        .into_iter()
        .enumerate()
        {
            let mut pixels = [90, 170, 230, 255].repeat(6);
            if index == 0 {
                pixels[3] = 0;
            }
            let texture = renderer.create_texture3d_rgba8_with_options(
                3,
                2,
                pixels,
                ImageBudget::default(),
                // Material rebinding policy is independent of whether the old
                // texture itself accepted nonopaque source pixels at creation.
                options.with_alpha_preservation(index != 3),
            )?;
            let tint = if index == 1 {
                Color::rgba(0.25, 0.75, 0.5, 0.4)
            } else {
                Color::WHITE
            };
            let mut material = TextureMaterial3d::with_alpha(
                &texture,
                if index.is_multiple_of(2) {
                    ImageSampling::Nearest
                } else {
                    ImageSampling::Linear
                },
                tint,
            )?;
            if index == 2 {
                material = material
                    .with_address_mode(TextureAddressMode3d::Repeat)
                    .with_uv_transform(TextureUvTransform3d::new(
                        Vec2::new(-2.5, 3.0),
                        Vec2::new(-0.25, 0.125),
                    )?);
            }
            let original = renderer.with_mesh3d_material(&reserved, &material)?;
            let expected = Snapshot::read(&original)?;
            cases.push(Case {
                name,
                original,
                expected,
            });
        }
        Ok(Self {
            cases,
            submitted_cases: 0,
            mutations_checked: false,
        })
    }

    /// Exercises the actual standalone public restore route after device replacement.
    /// Returned IDs join the gallery's bounded timing correlation, not a second ring.
    pub fn restore_and_submit(
        &mut self,
        renderer: &mut WgpuRenderer,
        target: &RenderTarget3d,
        camera: Camera3d,
    ) -> ExampleResult<Vec<GpuTimingId>> {
        let mut timing_ids = Vec::with_capacity(7);
        for (index, case) in self.cases.iter().enumerate() {
            let restored = renderer
                .restore_mesh3d(&case.original)
                .map_err(|error| format!("public restore_mesh3d {} failed: {error}", case.name))?;
            if Snapshot::read(&restored)? != case.expected
                || Snapshot::read(&case.original)? != case.expected
            {
                return Err(format!(
                    "public restore_mesh3d {} changed material/attributes/reserve or old alias",
                    case.name
                )
                .into());
            }
            let mut scene = Scene3d::new(Color::BLACK)?;
            scene.set_lighting(Lighting3d::new(AmbientLight3d::new(Color::WHITE, 1.0)?));
            let object = scene.try_push(&restored, Transform3d::IDENTITY, style()?)?;
            submit(renderer, target, &scene, camera, &mut timing_ids)?;
            self.submitted_cases += 1;
            if index == 3 {
                // The material is alpha-capable even though its original pixels
                // happen to be opaque. Rebuilding it through legacy `new` loses
                // that contract and this rebind must then fail.
                let alpha_texture = renderer.create_texture3d_rgba8_with_options(
                    3,
                    2,
                    [230, 80, 40, 90].repeat(6),
                    ImageBudget::default(),
                    Texture3dOptions::new()
                        .with_alpha_preservation(true)
                        .with_mipmaps(TextureMipmaps3d::Generate),
                )?;
                let rebound = restored
                    .material()
                    .ok_or("restored material missing")?
                    .with_texture(&alpha_texture)?;
                let alias = scene.try_push(&restored, Transform3d::IDENTITY, style()?)?;
                scene.set_visible(alias, false)?;
                let initial_statistics = scene.statistics();
                let old_background = scene.background();
                let lighting = scene.lighting();
                let fog = scene.fog();
                if scene
                    .set_background(Color::rgba(f32::NAN, 0.0, 0.0, 1.0))
                    .is_ok()
                    || scene.background() != old_background
                {
                    return Err(
                        "invalid public background edit was accepted or changed state".into(),
                    );
                }
                let background = Color::rgba(0.1, 0.2, 0.3, 0.5);
                scene.set_background(background)?;
                if scene.background() != background
                    || scene.statistics() != initial_statistics
                    || scene.lighting() != lighting
                    || scene.fog() != fog
                {
                    return Err(
                        "background-only update changed resources/environment or ignored color"
                            .into(),
                    );
                }
                let mut rebound_expected = Snapshot::read(&restored)?;
                rebound_expected.material = Some(MaterialSnapshot::read(&rebound)?);
                let update = scene.set_texture_material(object, Some(&rebound))?;
                if update.statistics().mesh_count() != 1
                    || update.statistics().texture_count() != 2
                    || scene.statistics().mesh_gpu_bytes() != initial_statistics.mesh_gpu_bytes()
                    || scene.statistics().mesh_cpu_bytes() != initial_statistics.mesh_cpu_bytes()
                    || scene.instance(object)?.mesh().source().vertices().as_ptr()
                        != restored.source().vertices().as_ptr()
                    || scene.instance(object)?.id() != object
                    || Snapshot::read(scene.instance(object)?.mesh())? != rebound_expected
                    || Snapshot::read(scene.instance(alias)?.mesh())? != case.expected
                {
                    return Err(
                        "material-only public update replaced geometry, IDs or another object"
                            .into(),
                    );
                }
                submit(renderer, target, &scene, camera, &mut timing_ids)?;
                scene.set_texture_material(object, None)?;
                let detached = restored.without_material();
                if detached.material().is_some()
                    || detached.source().vertices().as_ptr()
                        != restored.source().vertices().as_ptr()
                    || detached.gpu_allocation_bytes() != restored.gpu_allocation_bytes()
                    || scene.instance(object)?.mesh().material().is_some()
                    || scene.statistics().mesh_count() != 1
                    || scene.statistics().texture_count() != 1
                    || scene.statistics().mesh_gpu_bytes() != initial_statistics.mesh_gpu_bytes()
                    || Snapshot::read(scene.instance(alias)?.mesh())? != case.expected
                {
                    return Err(
                        "public texture detach changed topology or an immutable alias".into(),
                    );
                }
                submit(renderer, target, &scene, camera, &mut timing_ids)?;
                if Snapshot::read(&case.original)? != case.expected {
                    return Err("subsequent alpha rebinding mutated old standalone snapshot".into());
                }
                self.mutations_checked = true;
            }
            println!(
                "dev5_public_standalone_restore case={} submitted=true attributes_and_reserve=preserved source_alias=unchanged",
                case.name
            );
        }
        self.validate()?;
        Ok(timing_ids)
    }

    /// Requires every actual public restore/render and material-mutation control.
    pub fn validate(&self) -> ExampleResult<()> {
        if self.submitted_cases != 4 || !self.mutations_checked {
            return Err(
                "incomplete public standalone recovery/background/material acceptance".into(),
            );
        }
        Ok(())
    }
}

fn submit(
    renderer: &mut WgpuRenderer,
    target: &RenderTarget3d,
    scene: &Scene3d,
    camera: Camera3d,
    timing_ids: &mut Vec<GpuTimingId>,
) -> ExampleResult<()> {
    let report = renderer.render_scene3d_to_target(target, scene, camera)?;
    if report.object_count() != 1 || report.triangle_count() != 2 || report.draw_call_count() != 1 {
        return Err("restored standalone object did not submit its actual source triangles".into());
    }
    if let Some(id) = report.gpu_timing_id() {
        timing_ids.push(id);
    }
    Ok(())
}

fn style() -> ExampleResult<MeshStyle3d> {
    Ok(MeshStyle3d::surface(
        SurfaceStyle3d::blend(Color::WHITE)?.with_lighting(SurfaceLighting3d::Lambert),
    ))
}

fn panel() -> ExampleResult<Mesh3d> {
    let attributes = Mesh3dAttributes::new()
        .with_texture_coordinates(vec![
            TextureCoordinate2d::new(0.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 0.0)?,
            TextureCoordinate2d::new(0.0, 0.0)?,
        ])
        .with_vertex_colors(vec![
            Color::WHITE,
            Color::rgb(0.5, 1.0, 0.75),
            Color::rgba(1.0, 0.7, 0.5, 0.8),
            Color::WHITE,
        ])?
        .with_normals(vec![Vec3::Z; 4])?;
    Ok(Mesh3d::with_attributes(
        vec![
            Vec3::new(-1.0, -1.0, 0.0)?,
            Vec3::new(1.0, -1.0, 0.0)?,
            Vec3::new(1.0, 1.0, 0.0)?,
            Vec3::new(-1.0, 1.0, 0.0)?,
        ],
        vec![0, 1, 2, 0, 2, 3],
        vec![
            MeshEdge3d::new(0, 1)?,
            MeshEdge3d::new(1, 2)?,
            MeshEdge3d::new(2, 3)?,
            MeshEdge3d::new(3, 0)?,
        ],
        attributes,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standalone_control_has_every_optional_geometry_attribute() {
        let source = panel().unwrap();
        assert_eq!(source.vertices().len(), 4);
        assert_eq!(source.triangle_count(), 2);
        assert_eq!(source.display_edges().len(), 4);
        assert_eq!(source.texture_coordinates().len(), 4);
        assert_eq!(source.vertex_colors().len(), 4);
        assert_eq!(source.normals().len(), 4);
    }
}
