//! Host-owned visual fixture: no chunk generation or world rules live in Engine.

use sim_engine::{
    Camera3d, Color, ImageBudget, ImageSampling, ImageTexelRect, LogicalPixels, Mesh3d,
    Mesh3dUploadBudget, MeshEdge3d, MeshStyle3d, Object3dId, Projection3d, Rotation3d, Scene3d,
    Scene3dBudget, SurfaceStyle3d, TextureCoordinate2d, TextureMaterial3d, Transform3d, Vec3,
    WgpuRenderer, WireframeStyle3d, WorldLength,
};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Pool {
    pub scene: Scene3d,
    pub distance: f32,
    pub orbit: f32,
    pub phase: f32,
    pub playing: bool,
    slots: [Option<Object3dId>; 16],
    regions: [[TextureCoordinate2d; 4]; 6],
    sampling: ImageSampling,
    expanded: bool,
    wireframe: bool,
}

impl Pool {
    pub fn new(renderer: &WgpuRenderer) -> ExampleResult<Self> {
        // Six independent 8x8 cells, with one extruded texel of gutter each.
        // The atlas is one shared immutable allocation for every object/face.
        let palette: [[u8; 3]; 6] = [
            [235, 88, 77],
            [76, 182, 245],
            [107, 209, 134],
            [242, 192, 83],
            [187, 126, 236],
            [83, 218, 211],
        ];
        let pixels = (0..8)
            .flat_map(|y| {
                (0..48).flat_map(move |x| {
                    let face = x / 8;
                    let local_x = (x % 8).clamp(1, 6);
                    let local_y = y.clamp(1, 6);
                    let multiplier = if (local_x + local_y + face) % 3 == 0 {
                        0.45
                    } else {
                        1.0
                    };
                    let color = palette[face];
                    [
                        (f32::from(color[0]) * multiplier) as u8,
                        (f32::from(color[1]) * multiplier) as u8,
                        (f32::from(color[2]) * multiplier) as u8,
                        255,
                    ]
                })
            })
            .collect();
        let texture = renderer.create_texture3d_rgba8(48, 8, pixels, ImageBudget::default())?;
        let mut regions = [[TextureCoordinate2d::new(0.0, 0.0)?; 4]; 6];
        for (index, region) in regions.iter_mut().enumerate() {
            *region =
                texture.region_coordinates(ImageTexelRect::new(index as u32 * 8 + 1, 1, 6, 6)?)?;
        }
        let mesh = renderer.create_mesh3d(cube(&regions, 0.38)?)?;
        let material = TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE)?;
        let mesh = renderer.with_mesh3d_material(&mesh, &material)?;
        let budget = Scene3dBudget::new(16, 128 * 1024, 1024 * 1024, 1024 * 1024)?
            .with_texture_limits(48 * 8 * 4, 48 * 8 * 4);
        let mut scene = Scene3d::with_budget(Color::rgb8(15, 20, 30), budget)?;
        let mut slots = [None; 16];
        for (index, slot) in slots.iter_mut().enumerate() {
            *slot = Some(scene.try_push(&mesh, transform(index, 0.0)?, style()?)?);
        }
        Ok(Self {
            scene,
            distance: 7.0,
            orbit: 0.25,
            phase: 0.0,
            playing: true,
            slots,
            regions,
            sampling: ImageSampling::Nearest,
            expanded: false,
            wireframe: true,
        })
    }

    pub fn update(&mut self, delta: f32) -> ExampleResult<()> {
        if self.playing {
            self.phase += delta.min(0.1);
        }
        for (index, id) in self.slots.iter().enumerate() {
            if let Some(id) = id {
                self.scene
                    .set_transform(*id, transform(index, self.phase)?)?;
            }
        }
        Ok(())
    }

    pub fn camera(&self, width: u32, height: u32) -> ExampleResult<Camera3d> {
        let projection = Projection3d::perspective(
            55.0_f32.to_radians(),
            width as f32 / height as f32,
            WorldLength::new(0.2)?,
            WorldLength::new(60.0)?,
        )?;
        Ok(Camera3d::look_at(
            Vec3::new(
                self.orbit.sin() * self.distance,
                self.distance * 0.12,
                self.orbit.cos() * self.distance,
            )?,
            Vec3::ZERO,
            Vec3::Y,
            projection,
        )?)
    }

    pub fn toggle_filter(&mut self, renderer: &WgpuRenderer) -> ExampleResult<()> {
        self.sampling = if self.sampling == ImageSampling::Nearest {
            ImageSampling::Linear
        } else {
            ImageSampling::Nearest
        };
        for id in self.slots.iter().flatten() {
            let mesh = self.scene.instance(*id)?.mesh();
            let texture = mesh.material().ok_or("fixture texture missing")?.texture();
            let material = TextureMaterial3d::new(texture, self.sampling, Color::WHITE)?;
            let replacement = renderer.with_mesh3d_material(mesh, &material)?;
            self.scene.set_mesh(*id, &replacement)?;
        }
        println!(
            "filter={:?}; shared textures={}",
            self.sampling,
            self.scene.statistics().texture_count()
        );
        Ok(())
    }

    pub fn edit(&mut self, renderer: &WgpuRenderer) -> ExampleResult<()> {
        let Some(id) = self.slots[0] else {
            return Ok(());
        };
        let mut replacement = self.scene.instance(id)?.mesh().clone();
        self.expanded = !self.expanded;
        let report = renderer.replace_mesh3d(
            &mut replacement,
            cube(&self.regions, if self.expanded { 0.51 } else { 0.29 })?,
            Mesh3dUploadBudget::new(64 * 1024, 64 * 1024, 64 * 1024)?,
        )?;
        self.scene.set_mesh(id, &replacement)?;
        println!(
            "immutable edit: uploaded={} bytes, peak mesh buffers={} bytes; ID remains {id:?}",
            report.uploaded_bytes(),
            report.peak_gpu_bytes()
        );
        Ok(())
    }

    pub fn toggle_first(&mut self) -> ExampleResult<()> {
        if let Some(id) = self.slots[0].take() {
            self.scene.remove(id)?;
            assert!(
                self.scene.instance(id).is_err(),
                "removed ID aliased another object"
            );
        } else {
            let mesh = self.scene.instances()[0].mesh().clone();
            self.slots[0] = Some(self.scene.try_push(
                &mesh,
                transform(0, self.phase)?,
                if self.wireframe {
                    style()?
                } else {
                    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?)
                },
            )?);
        }
        println!(
            "live objects={}; slot capacity={}",
            self.scene.object_count(),
            self.scene.statistics().slot_capacity()
        );
        Ok(())
    }

    /// The solid free-camera fixture and mathematical-edge overlay deliberately
    /// remain separate: conservative edge-projection validation is not disabled.
    pub fn set_wireframe(&mut self, visible: bool) -> ExampleResult<()> {
        self.wireframe = visible;
        let style = if visible {
            style()?
        } else {
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?)
        };
        for id in self.slots.iter().flatten() {
            self.scene.set_style(*id, style)?;
        }
        println!(
            "mathematical-edge overlay={visible}; conservative edge-projection guard remains active"
        );
        Ok(())
    }

    pub fn toggle_wireframe(&mut self) -> ExampleResult<()> {
        self.set_wireframe(!self.wireframe)
    }
}

fn transform(index: usize, phase: f32) -> ExampleResult<Transform3d> {
    Ok(Transform3d::new(
        Vec3::new(
            (index % 4) as f32 * 1.15 - 1.725,
            (index / 4) as f32 * 1.15 - 1.725,
            0.0,
        )?,
        Rotation3d::from_euler_xyz(
            0.18 + phase * 0.07,
            0.25 + phase * 0.11,
            0.09 + index as f32 * 0.015,
        )?,
        Vec3::new(1.0, 1.0, 1.0)?,
    )?)
}

fn style() -> ExampleResult<MeshStyle3d> {
    Ok(
        MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?).with_wireframe(
            WireframeStyle3d::visible(Color::rgb8(230, 239, 255), LogicalPixels::new(1.0)?)?
                .with_hidden(
                    Color::rgb8(96, 117, 143),
                    LogicalPixels::new(0.7)?,
                    LogicalPixels::new(5.0)?,
                    LogicalPixels::new(4.0)?,
                )?,
        ),
    )
}

fn cube(regions: &[[TextureCoordinate2d; 4]; 6], half: f32) -> ExampleResult<Mesh3d> {
    let faces = [
        [[-1., 1., 1.], [1., 1., 1.], [1., -1., 1.], [-1., -1., 1.]],
        [
            [1., 1., -1.],
            [-1., 1., -1.],
            [-1., -1., -1.],
            [1., -1., -1.],
        ],
        [
            [-1., 1., -1.],
            [-1., 1., 1.],
            [-1., -1., 1.],
            [-1., -1., -1.],
        ],
        [[1., 1., 1.], [1., 1., -1.], [1., -1., -1.], [1., -1., 1.]],
        [[-1., 1., -1.], [1., 1., -1.], [1., 1., 1.], [-1., 1., 1.]],
        [
            [-1., -1., 1.],
            [1., -1., 1.],
            [1., -1., -1.],
            [-1., -1., -1.],
        ],
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut uv = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face, region) in faces.iter().zip(regions) {
        let offset = vertices.len() as u32;
        for position in face {
            vertices.push(Vec3::new(
                position[0] * half,
                position[1] * half,
                position[2] * half,
            )?);
        }
        uv.extend_from_slice(region);
        indices.extend_from_slice(&[
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ]);
    }
    let edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 5],
        [1, 4],
        [2, 7],
        [3, 6],
    ]
    .into_iter()
    .map(|[start, end]| MeshEdge3d::new(start, end))
    .collect::<Result<Vec<_>, _>>()?;
    Ok(Mesh3d::textured(vertices, uv, indices, edges)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_faces_keep_separate_uvs_and_twelve_mathematical_edges() {
        let uv = [[TextureCoordinate2d::new(0.25, 0.75).unwrap(); 4]; 6];
        let mesh = cube(&uv, 0.38).unwrap();
        assert_eq!(mesh.vertices().len(), 24);
        assert_eq!(mesh.texture_coordinates().len(), 24);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.display_edges().len(), 12);
    }
}
