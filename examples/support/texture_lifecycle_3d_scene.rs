//! Host-authored texture panels, edit actions and bounded gallery state.

use sim_engine::{
    AmbientLight3d, Color, DirectionalLight3d, Fog3d, ImageBudget, ImageSampling, ImageTexelRect,
    Lighting3d, Mesh3d, Mesh3dAttributes, MeshStyle3d, Object3dId, Rotation3d, Scene3d,
    SurfaceLighting3d, SurfaceSidedness3d, SurfaceStyle3d, Texture3dOptions, Texture3dUpdateBudget,
    Texture3dUpdateReport, TextureAddressMode3d, TextureCoordinate2d, TextureMaterial3d,
    TextureMipmaps3d, TextureUvTransform3d, Transform3d, Vec2, Vec3, WgpuRenderer,
};

use super::ExampleResult;

const TILE: u32 = 32;
pub const BACKGROUND: Color = Color::rgb(0.0044, 0.0075, 0.0144);

pub struct Gallery {
    pub scene: Scene3d,
    pub repeat: bool,
    pub mirrored: bool,
    pub lit: bool,
    pub fog: bool,
    pub front_only: bool,
    pub alpha_mode: usize,
    pub phase: f32,
    pub edits: usize,
    pub detached_edits: usize,
    pub reused_edits: usize,
    pub uploaded_bytes: usize,
    pub copied_bytes: usize,
    ids: [Object3dId; 6],
    original: Vec<u8>,
}

impl Gallery {
    pub fn new(renderer: &WgpuRenderer) -> ExampleResult<Self> {
        let mut scene = Scene3d::new(BACKGROUND)?;
        let source = renderer.create_mesh3d(panel()?)?;
        let mut ids = Vec::new();
        // The top two panels share identical base pixels and continuous UVs.
        // Only mip policy differs, so zoom/orbit exposes aliasing directly.
        for mips in [false, true] {
            let texture = renderer.create_texture3d_rgba8_with_options(
                64,
                64,
                checker(),
                ImageBudget::default(),
                options(mips),
            )?;
            let material =
                TextureMaterial3d::with_alpha(&texture, ImageSampling::Linear, Color::WHITE)?
                    .with_address_mode(TextureAddressMode3d::Repeat)
                    .with_uv_transform(TextureUvTransform3d::new(
                        Vec2::new(16.0, 16.0),
                        Vec2::ZERO,
                    )?);
            let mesh = renderer.with_mesh3d_material(&source, &material)?;
            ids.push(scene.try_push(
                &mesh,
                transform(ids.len())?,
                style(0, false, false, false)?,
            )?);
        }

        // The green tile sits next to a bright magenta atlas cell. Cropping
        // creates an independent mip chain; repeated/minified panels never read
        // that neighbor. The tile itself has asymmetry to expose UV mirroring.
        let atlas = renderer.create_texture3d_rgba8_with_options(
            TILE * 2,
            TILE,
            atlas(),
            ImageBudget::default(),
            options(false),
        )?;
        let tile = renderer.crop_texture3d_tile(
            &atlas,
            ImageTexelRect::new(TILE, 0, TILE, TILE)?,
            ImageBudget::default(),
            options(true),
        )?;
        let material = TextureMaterial3d::with_alpha(&tile, ImageSampling::Linear, Color::WHITE)?;
        let mesh = renderer.with_mesh3d_material(&source, &material)?;
        for _ in 0..2 {
            ids.push(scene.try_push(
                &mesh,
                transform(ids.len())?,
                style(0, false, false, false)?,
            )?);
        }

        // Both bottom objects initially reference one immutable texture. The
        // first selected-object edit detaches; later edits can reuse its storage.
        let original = editable();
        let texture = renderer.create_texture3d_rgba8_with_options(
            TILE,
            TILE,
            original.clone(),
            ImageBudget::default(),
            options(true),
        )?;
        let material =
            TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)?;
        let mesh = renderer.with_mesh3d_material(&source, &material)?;
        for _ in 0..2 {
            ids.push(scene.try_push(
                &mesh,
                transform(ids.len())?,
                style(0, false, false, false)?,
            )?);
        }
        let ids = ids.try_into().map_err(|_| "gallery panel count mismatch")?;
        let mut result = Self {
            scene,
            repeat: true,
            mirrored: true,
            lit: false,
            fog: false,
            front_only: false,
            alpha_mode: 0,
            phase: 0.0,
            edits: 0,
            detached_edits: 0,
            reused_edits: 0,
            uploaded_bytes: 0,
            copied_bytes: 0,
            ids,
            original,
        };
        result.apply(renderer)?;
        Ok(result)
    }

    pub fn apply(&mut self, renderer: &WgpuRenderer) -> ExampleResult<()> {
        for id in self.ids {
            self.scene.set_style(
                id,
                style(self.alpha_mode, self.front_only, self.lit, self.fog)?,
            )?;
        }
        for (index, id) in self.ids[2..4].iter().copied().enumerate() {
            let sign = if self.mirrored && index == 1 {
                -1.0
            } else {
                1.0
            };
            let scale = if self.repeat { 3.0 } else { 1.4 };
            let instance = self.scene.instance(id)?;
            let material = instance
                .mesh()
                .material()
                .ok_or("tile lost its material")?
                .clone()
                .with_address_mode(if self.repeat {
                    TextureAddressMode3d::Repeat
                } else {
                    TextureAddressMode3d::Clamp
                })
                .with_uv_transform(TextureUvTransform3d::new(
                    Vec2::new(sign * scale, scale),
                    Vec2::new(self.phase, -self.phase * 0.5),
                )?);
            let mesh = renderer.with_mesh3d_material(instance.mesh(), &material)?;
            self.scene.set_mesh(id, &mesh)?;
        }
        self.scene.set_lighting(
            Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.2)?).with_directional(Some(
                DirectionalLight3d::new(Vec3::new(0.6, 0.5, 1.0)?, Color::WHITE, 0.8)?,
            )),
        );
        self.scene.set_fog(
            self.fog
                .then(|| Fog3d::new(BACKGROUND, 8.0, 0.12))
                .transpose()?,
        );
        Ok(())
    }

    pub fn edit(&mut self, renderer: &WgpuRenderer) -> ExampleResult<Texture3dUpdateReport> {
        let x = ((self.edits * 5) % 24) as u32;
        let y = ((self.edits * 7) % 24) as u32;
        let pixel = if self.edits.is_multiple_of(2) {
            [255, 198, 48, 255]
        } else {
            [42, 210, 255, 120]
        };
        // Explicit row padding proves the public strided input route. The final
        // row omits padding, as required by its exact source-layout contract.
        let stride = 8 * 4 + 8;
        let mut patch = vec![0; 7 * stride + 8 * 4];
        for row in 0..8 {
            for column in 0..8 {
                patch[row * stride + column * 4..row * stride + column * 4 + 4]
                    .copy_from_slice(&pixel);
            }
        }
        let report = renderer.update_scene3d_texture_region(
            &mut self.scene,
            self.ids[4],
            ImageTexelRect::new(x, y, 8, 8)?,
            &patch,
            stride,
            Texture3dUpdateBudget::default(),
        )?;
        self.edits += 1;
        self.detached_edits += usize::from(report.detached_aliases());
        self.reused_edits += usize::from(report.reused_allocation());
        self.uploaded_bytes += report.uploaded_bytes();
        self.copied_bytes += report.gpu_copy_bytes();
        self.validate_snapshot()?;
        println!(
            "texture_edit={} reused={} allocations={} base_upload={} mip_upload={} copy={} submissions={} staging={} CPU_prepare_ms={:.3}",
            self.edits,
            report.reused_allocation(),
            report.gpu_allocation_count(),
            report.base_upload_bytes(),
            report.mip_upload_bytes(),
            report.gpu_copy_bytes(),
            report.submission_count(),
            report.staging_bytes(),
            report.preparation_cpu().as_secs_f64() * 1000.0
        );
        Ok(report)
    }

    pub fn validate_snapshot(&self) -> ExampleResult<()> {
        let snapshot = self
            .scene
            .instance(self.ids[5])?
            .mesh()
            .material()
            .ok_or("snapshot lost its material")?
            .texture();
        if snapshot.pixels() != self.original {
            return Err("selected-object edit changed the original alias".into());
        }
        if self
            .scene
            .instance(self.ids[0])?
            .mesh()
            .material()
            .ok_or("mip-zero material missing")?
            .texture()
            .mip_level_count()
            != 1
            || self
                .scene
                .instance(self.ids[1])?
                .mesh()
                .material()
                .ok_or("mip-chain material missing")?
                .texture()
                .mip_level_count()
                != 7
        {
            return Err("mip comparison panels lost their distinct policies".into());
        }
        Ok(())
    }

    pub fn alpha_name(&self) -> &'static str {
        ["Opaque", "Mask", "Blend"][self.alpha_mode]
    }
}

fn options(mips: bool) -> Texture3dOptions {
    Texture3dOptions::new()
        .with_alpha_preservation(true)
        .with_mipmaps(if mips {
            TextureMipmaps3d::Generate
        } else {
            TextureMipmaps3d::None
        })
}

fn style(mode: usize, front_only: bool, lit: bool, fog: bool) -> ExampleResult<MeshStyle3d> {
    let surface = match mode {
        1 => SurfaceStyle3d::mask(Color::WHITE, 0.5)?,
        2 => SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.65))?,
        _ => SurfaceStyle3d::opaque(Color::WHITE)?,
    };
    Ok(MeshStyle3d::surface(
        surface
            .with_sidedness(if front_only {
                SurfaceSidedness3d::FrontOnly
            } else {
                SurfaceSidedness3d::TwoSided
            })
            .with_lighting(if lit {
                SurfaceLighting3d::Lambert
            } else {
                SurfaceLighting3d::Unlit
            })
            .with_fog(fog),
    ))
}

fn transform(index: usize) -> ExampleResult<Transform3d> {
    Ok(Transform3d::new(
        Vec3::new(
            if index.is_multiple_of(2) { -2.45 } else { 2.45 },
            2.2 - (index / 2) as f32 * 2.2,
            0.0,
        )?,
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0)?,
    )?)
}

fn panel() -> ExampleResult<Mesh3d> {
    let attributes = Mesh3dAttributes::new()
        .with_normals(vec![Vec3::Z; 4])?
        .with_texture_coordinates(vec![
            TextureCoordinate2d::new(0.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 0.0)?,
            TextureCoordinate2d::new(0.0, 0.0)?,
        ]);
    Ok(Mesh3d::with_attributes(
        vec![
            Vec3::new(-2.05, -0.85, 0.0)?,
            Vec3::new(2.05, -0.85, 0.0)?,
            Vec3::new(2.05, 0.85, 0.0)?,
            Vec3::new(-2.05, 0.85, 0.0)?,
        ],
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
        attributes,
    )?)
}

fn checker() -> Vec<u8> {
    (0..64)
        .flat_map(|y| {
            (0..64).flat_map(move |x| {
                if (x / 2 + y / 2) & 1 == 0 {
                    [255, 240, 210, 255]
                } else {
                    [5, 15, 30, 255]
                }
            })
        })
        .collect()
}

fn atlas() -> Vec<u8> {
    (0..TILE)
        .flat_map(|y| {
            (0..TILE * 2).flat_map(move |x| {
                if x < TILE {
                    [255, 0, 220, 255]
                } else if x - TILE < 6 && y < 14 {
                    [240, 255, 70, 255]
                } else if x - TILE > 23 && y > 22 {
                    [0, 160, 80, 0]
                } else {
                    [24, 180, 120, 255]
                }
            })
        })
        .collect()
}

fn editable() -> Vec<u8> {
    (0..TILE)
        .flat_map(|y| {
            (0..TILE).flat_map(move |x| {
                if (x / 4 + y / 4) & 1 == 0 {
                    [170, 74, 230, 255]
                } else {
                    [35, 28, 70, 80]
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gallery_sources_have_exact_rows_and_host_normals() {
        assert_eq!(checker().len(), 64 * 64 * 4);
        assert_eq!(atlas().len(), (TILE * TILE * 8) as usize);
        assert_eq!(editable().len(), (TILE * TILE * 4) as usize);
        let panel = panel().unwrap();
        assert_eq!(panel.triangle_count(), 2);
        assert_eq!(panel.normals(), &[Vec3::Z; 4]);
        assert_eq!(panel.texture_coordinates().len(), 4);
    }
}
