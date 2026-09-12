//! Host-owned geometry and animation for the material gallery, not a simulation.

use sim_engine::{
    Color, ImageBudget, ImageSampling, Mesh3d, Mesh3dAttributes, MeshStyle3d, Object3dId,
    Rotation3d, Scene3d, SurfaceSidedness3d, SurfaceStyle3d, TextureCoordinate2d,
    TextureMaterial3d, Transform3d, Vec3, WgpuRenderer,
};

use super::ExampleResult;

const CENTERS: [f32; 3] = [-3.5, 0.0, 3.5];
const MODE_NAMES: [&str; 3] = ["Opaque", "Mask", "Blend"];

pub struct Gallery {
    pub scene: Scene3d,
    cards: [Object3dId; 3],
    overlays: [Object3dId; 2],
    preset: usize,
    front_only: bool,
    phase: f32,
}

impl Gallery {
    pub fn new(renderer: &WgpuRenderer) -> ExampleResult<Self> {
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT)?;
        let background = renderer.create_mesh3d(checker_background()?)?;
        let frame = renderer.create_mesh3d(quad(3.18, 4.58, None, false)?)?;
        let accents = [
            Color::rgb8(36, 113, 155),
            Color::rgb8(44, 136, 102),
            Color::rgb8(146, 75, 152),
        ];
        for (center, accent) in CENTERS.into_iter().zip(accents) {
            scene.try_push(
                &frame,
                transform(center, 0.0, -1.75, 0.0, 0.0)?,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(accent)?),
            )?;
            scene.try_push(
                &background,
                transform(center, 0.0, -1.6, 0.0, 0.0)?,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?),
            )?;
        }

        let texture = renderer.create_texture3d_rgba8_with_alpha(
            32,
            32,
            alpha_checker(),
            ImageBudget::new(32, 32, 32 * 32 * 4)?,
        )?;
        let material =
            TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)?;
        let colored = renderer.create_mesh3d(quad(2.28, 2.28, Some(corner_colors()), true)?)?;
        let card_mesh = renderer.with_mesh3d_material(&colored, &material)?;
        let mut insert = |index: usize| -> ExampleResult<Object3dId> {
            Ok(scene.try_push(
                &card_mesh,
                transform(CENTERS[index], 0.65, 0.25, 0.0, 0.0)?,
                card_style(index, false)?,
            )?)
        };
        let cards = [insert(0)?, insert(1)?, insert(2)?];

        // The lower-left swatch isolates vertex RGB interpolation from textures.
        let gradient = renderer.create_mesh3d(quad(2.3, 0.62, Some(corner_colors()), false)?)?;
        scene.try_push(
            &gradient,
            transform(CENTERS[0], -1.28, 0.2, 0.0, 0.0)?,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?),
        )?;
        // Alpha varies continuously across this untextured mask control.
        let mask_ramp = renderer.create_mesh3d(quad(
            2.3,
            0.62,
            Some([
                Color::rgba(0.3, 1.0, 0.6, 0.0),
                Color::rgba(0.3, 1.0, 0.6, 1.0),
                Color::rgba(0.3, 1.0, 0.6, 1.0),
                Color::rgba(0.3, 1.0, 0.6, 0.0),
            ]),
            false,
        )?)?;
        scene.try_push(
            &mask_ramp,
            transform(CENTERS[1], -1.28, 0.2, 0.0, 0.0)?,
            MeshStyle3d::surface(SurfaceStyle3d::mask(Color::WHITE, 0.5)?),
        )?;

        let overlay = renderer.create_mesh3d(quad(1.3, 0.86, None, false)?)?;
        // Deliberately insert the nearer face first. Back-to-front material
        // sorting, not this insertion order, must produce the visible overlap.
        let near = scene.try_push(
            &overlay,
            transform(CENTERS[2] + 0.31, -1.28, 0.9, 0.0, 0.0)?,
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::rgba(1.0, 0.1, 0.3, 0.5))?),
        )?;
        let far = scene.try_push(
            &overlay,
            transform(CENTERS[2] - 0.31, -1.28, 0.5, 0.0, 0.0)?,
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::rgba(0.05, 0.7, 1.0, 0.5))?),
        )?;
        Ok(Self {
            scene,
            cards,
            overlays: [near, far],
            preset: 0,
            front_only: false,
            phase: 0.0,
        })
    }

    pub fn preset(&self) -> usize {
        self.preset
    }

    pub fn front_only(&self) -> bool {
        self.front_only
    }

    pub fn names(&self) -> [&'static str; 3] {
        std::array::from_fn(|index| MODE_NAMES[(index + self.preset) % 3])
    }

    pub fn facing_counts(&self, eye: Vec3, perspective: bool) -> ExampleResult<(usize, usize)> {
        let mut front = 0;
        let mut back = 0;
        for object in self.cards {
            let transform = self.scene.instance(object)?.transform();
            let normal = transform.rotation().rotate(Vec3::Z)?;
            let translation = if perspective {
                transform.translation()
            } else {
                Vec3::ZERO
            };
            // All source cards are CCW when viewed along +Z. The orthographic
            // view has parallel rays, unlike perspective's per-object eye ray.
            let facing = f64::from(normal.x()) * (f64::from(eye.x()) - f64::from(translation.x()))
                + f64::from(normal.y()) * (f64::from(eye.y()) - f64::from(translation.y()))
                + f64::from(normal.z()) * (f64::from(eye.z()) - f64::from(translation.z()));
            front += usize::from(facing > 1e-6);
            back += usize::from(facing < -1e-6);
        }
        Ok((front, back))
    }

    pub fn set_preset(&mut self, preset: usize) -> ExampleResult<()> {
        self.preset = preset % 3;
        self.apply_styles()
    }

    pub fn toggle_sidedness(&mut self) -> ExampleResult<()> {
        self.front_only = !self.front_only;
        self.apply_styles()
    }

    fn apply_styles(&mut self) -> ExampleResult<()> {
        for (index, object) in self.cards.into_iter().enumerate() {
            self.scene.set_style(
                object,
                card_style((index + self.preset) % 3, self.front_only)?,
            )?;
        }
        for (object, color) in self.overlays.into_iter().zip([
            Color::rgba(1.0, 0.1, 0.3, 0.5),
            Color::rgba(0.05, 0.7, 1.0, 0.5),
        ]) {
            self.scene.set_style(
                object,
                MeshStyle3d::surface(
                    SurfaceStyle3d::blend(color)?.with_sidedness(sidedness(self.front_only)),
                ),
            )?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> ExampleResult<()> {
        self.preset = 0;
        self.front_only = false;
        self.phase = 0.0;
        self.apply_styles()?;
        self.animate(0.0)
    }

    pub fn animate(&mut self, delta_seconds: f32) -> ExampleResult<()> {
        // Every animation frequency is an integer multiple of 1/20. Wrapping
        // only the common period keeps independently rotating cards continuous.
        self.phase = (self.phase + delta_seconds * 0.8).rem_euclid(std::f32::consts::TAU * 20.0);
        for (index, object) in self.cards.into_iter().enumerate() {
            let rotation = match index {
                0 => self.phase,
                1 => -self.phase * 1.1 + 0.18,
                _ => self.phase * 0.85 - 0.18,
            };
            self.scene.set_transform(
                object,
                transform(
                    CENTERS[index],
                    0.65,
                    0.25,
                    rotation,
                    (self.phase + index as f32).sin() * 0.07,
                )?,
            )?;
        }
        for (index, object) in self.overlays.into_iter().enumerate() {
            let near = index == 0;
            self.scene.set_transform(
                object,
                transform(
                    CENTERS[2] + if near { 0.31 } else { -0.31 },
                    -1.28,
                    if near { 0.9 } else { 0.5 },
                    (self.phase + index as f32).sin() * 0.12,
                    (self.phase * 0.7 + index as f32).cos() * 0.1,
                )?,
            )?;
        }
        Ok(())
    }
}

fn sidedness(front_only: bool) -> SurfaceSidedness3d {
    if front_only {
        SurfaceSidedness3d::FrontOnly
    } else {
        SurfaceSidedness3d::TwoSided
    }
}

fn card_style(mode: usize, front_only: bool) -> ExampleResult<MeshStyle3d> {
    let surface = match mode {
        0 => SurfaceStyle3d::opaque(Color::WHITE)?,
        1 => SurfaceStyle3d::mask(Color::WHITE, 0.5)?,
        _ => SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.75))?,
    };
    Ok(MeshStyle3d::surface(
        surface.with_sidedness(sidedness(front_only)),
    ))
}

fn transform(
    x: f32,
    y: f32,
    z: f32,
    rotation_y: f32,
    rotation_z: f32,
) -> ExampleResult<Transform3d> {
    Ok(Transform3d::new(
        Vec3::new(x, y, z)?,
        Rotation3d::from_euler_xyz(0.0, rotation_y, rotation_z)?,
        Vec3::new(1.0, 1.0, 1.0)?,
    )?)
}

fn corner_colors() -> [Color; 4] {
    [
        Color::rgb8(255, 114, 107),
        Color::rgb8(255, 210, 105),
        Color::rgb8(95, 220, 193),
        Color::rgb8(124, 151, 255),
    ]
}

fn quad(
    width: f32,
    height: f32,
    colors: Option<[Color; 4]>,
    textured: bool,
) -> ExampleResult<Mesh3d> {
    let mut attributes = Mesh3dAttributes::new();
    if let Some(colors) = colors {
        attributes = attributes.with_vertex_colors(colors.to_vec())?;
    }
    if textured {
        attributes = attributes.with_texture_coordinates(vec![
            TextureCoordinate2d::new(0.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 1.0)?,
            TextureCoordinate2d::new(1.0, 0.0)?,
            TextureCoordinate2d::new(0.0, 0.0)?,
        ]);
    }
    Ok(Mesh3d::with_attributes(
        vec![
            Vec3::new(-width * 0.5, -height * 0.5, 0.0)?,
            Vec3::new(width * 0.5, -height * 0.5, 0.0)?,
            Vec3::new(width * 0.5, height * 0.5, 0.0)?,
            Vec3::new(-width * 0.5, height * 0.5, 0.0)?,
        ],
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
        attributes,
    )?)
}

fn checker_background() -> ExampleResult<Mesh3d> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    for y in 0..8 {
        for x in 0..6 {
            let left = -1.5 + x as f32 * 0.5;
            let bottom = -2.175 + y as f32 * 4.35 / 8.0;
            let start = vertices.len() as u32;
            vertices.extend([
                Vec3::new(left, bottom, 0.0)?,
                Vec3::new(left + 0.5, bottom, 0.0)?,
                Vec3::new(left + 0.5, bottom + 4.35 / 8.0, 0.0)?,
                Vec3::new(left, bottom + 4.35 / 8.0, 0.0)?,
            ]);
            indices.extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            let color = if (x + y) & 1 == 0 {
                Color::rgb8(50, 58, 74)
            } else {
                Color::rgb8(77, 87, 105)
            };
            colors.extend([color; 4]);
        }
    }
    Ok(Mesh3d::with_attributes(
        vertices,
        indices,
        Vec::new(),
        Mesh3dAttributes::new().with_vertex_colors(colors)?,
    )?)
}

fn alpha_checker() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(32 * 32 * 4);
    for y in 0..32 {
        for x in 0..32 {
            let alpha = match (x / 8 + y / 8) % 3 {
                0 => 0,
                1 => 96,
                _ => 255,
            };
            pixels.extend([255, 255, 255, alpha]);
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_engine::SurfaceAlphaMode3d;

    #[test]
    fn fixture_has_three_alpha_levels_and_distinct_material_policies() {
        let pixels = alpha_checker();
        assert_eq!(pixels.len(), 32 * 32 * 4);
        for alpha in [0, 96, 255] {
            assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] == alpha));
        }
        for (index, expected) in [
            SurfaceAlphaMode3d::Opaque,
            SurfaceAlphaMode3d::Mask,
            SurfaceAlphaMode3d::Blend,
        ]
        .into_iter()
        .enumerate()
        {
            let surface = card_style(index, true).unwrap().surface_style().unwrap();
            assert_eq!(surface.alpha_mode(), expected);
            assert_eq!(surface.sidedness(), SurfaceSidedness3d::FrontOnly);
        }
    }

    #[test]
    fn gallery_geometry_has_valid_host_supplied_colors_and_uvs() {
        let mesh = quad(2.28, 2.28, Some(corner_colors()), true).unwrap();
        assert_eq!(mesh.vertex_colors().len(), 4);
        assert_eq!(mesh.texture_coordinates().len(), 4);
        assert_eq!(mesh.triangle_count(), 2);
        let background = checker_background().unwrap();
        assert_eq!(background.triangle_count(), 96);
        assert_eq!(
            background.vertex_colors().len(),
            background.vertices().len()
        );
    }
}
