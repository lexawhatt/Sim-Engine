//! Host-authored normals, light animation and depth swatches for the gallery.

use sim_engine::{
    AmbientLight3d, Color, DirectionalLight3d, Fog3d, ImageBudget, ImageSampling, Lighting3d,
    Mesh3d, Mesh3dAttributes, MeshStyle3d, Object3dId, Rotation3d, Scene3d, SurfaceLighting3d,
    SurfaceSidedness3d, SurfaceStyle3d, TextureCoordinate2d, TextureMaterial3d, Transform3d, Vec3,
    WgpuRenderer,
};

use super::ExampleResult;

const SPHERE_RINGS: usize = 16;
const SPHERE_COLUMNS: usize = 32;
pub const FOG_COLOR: Color = Color::rgb(0.30, 0.40, 0.49);

pub struct Gallery {
    pub scene: Scene3d,
    pub sun_enabled: bool,
    pub fog_enabled: bool,
    pub nonuniform: bool,
    pub front_only: bool,
    pub alpha_mode: usize,
    shapes: [Object3dId; 2],
    marker: Object3dId,
    phase: f32,
}

impl Gallery {
    pub fn new(renderer: &WgpuRenderer) -> ExampleResult<Self> {
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT)?;
        let source = renderer.create_mesh3d(sphere()?)?;
        let left = scene.try_push(
            &source,
            shape_transform(-2.25, 0.0, true)?,
            shape_style(0, false, false)?,
        )?;
        let right = scene.try_push(
            &source,
            shape_transform(2.25, 0.0, true)?,
            shape_style(0, false, true)?,
        )?;

        let marker_mesh = renderer.create_mesh3d(marker_mesh()?)?;
        let marker = scene.try_push(
            &marker_mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb8(255, 222, 136))?),
        )?;

        let texture = renderer.create_texture3d_rgba8_with_alpha(
            16,
            16,
            alpha_texture(),
            ImageBudget::new(16, 16, 1024)?,
        )?;
        let material =
            TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)?;
        let swatch = renderer.create_mesh3d(swatch()?)?;
        let swatch = renderer.with_mesh3d_material(&swatch, &material)?;
        for index in 0..6 {
            let x = -3.75 + index as f32 * 1.5;
            let z = 1.5 - index as f32 * 1.25;
            let tint = if index & 1 == 0 {
                Color::rgb8(110, 220, 230)
            } else {
                Color::rgb8(255, 182, 100)
            };
            for (row, fog) in [(0, true), (1, false)] {
                let color = Color::rgba(
                    tint.red(),
                    tint.green(),
                    tint.blue(),
                    if index % 3 == 2 { 0.6 } else { 1.0 },
                );
                let style = surface(index % 3, color)?.with_fog(fog);
                scene.try_push(
                    &swatch,
                    Transform3d::new(
                        Vec3::new(x, -1.6 - row as f32 * 1.0, z)?,
                        Rotation3d::IDENTITY,
                        Vec3::new(1.0, 1.0, 1.0)?,
                    )?,
                    MeshStyle3d::surface(style),
                )?;
            }
        }
        let mut result = Self {
            scene,
            sun_enabled: true,
            fog_enabled: true,
            nonuniform: true,
            front_only: false,
            alpha_mode: 0,
            shapes: [left, right],
            marker,
            phase: 0.0,
        };
        result.update(0.0)?;
        Ok(result)
    }

    pub fn alpha_name(&self) -> &'static str {
        ["Opaque", "Mask", "Blend"][self.alpha_mode % 3]
    }

    pub fn light_horizontal(&self) -> f32 {
        self.phase.cos() * 0.85
    }

    pub fn update(&mut self, delta_seconds: f32) -> ExampleResult<()> {
        // All shape/light frequencies are integer multiples of one half.
        self.phase = (self.phase + delta_seconds * 0.8).rem_euclid(std::f32::consts::TAU * 2.0);
        for (object, x) in self.shapes.into_iter().zip([-2.25, 2.25]) {
            self.scene
                .set_transform(object, shape_transform(x, self.phase, self.nonuniform)?)?;
        }
        for (index, object) in self.shapes.into_iter().enumerate() {
            self.scene.set_style(
                object,
                shape_style(self.alpha_mode, self.front_only, index == 1)?,
            )?;
        }
        let direction = Vec3::new(self.light_horizontal(), 0.65, self.phase.sin() * 0.85)?;
        let sun = self
            .sun_enabled
            .then(|| DirectionalLight3d::new(direction, Color::rgb8(255, 244, 218), 0.82))
            .transpose()?;
        self.scene.set_lighting(
            Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.16)?).with_directional(sun),
        );
        self.scene.set_fog(
            self.fog_enabled
                .then(|| Fog3d::new(FOG_COLOR, 9.5, 0.40))
                .transpose()?,
        );
        self.scene.set_transform(
            self.marker,
            Transform3d::new(
                Vec3::new(direction.x() * 1.6, 3.05, direction.z() * 0.5)?,
                Rotation3d::from_euler_xyz(0.0, self.phase, 0.0)?,
                Vec3::new(0.13, 0.13, 0.13)?,
            )?,
        )?;
        self.scene.set_style(
            self.marker,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(if self.sun_enabled {
                Color::rgb8(255, 222, 136)
            } else {
                Color::rgb8(77, 86, 99)
            })?),
        )?;
        Ok(())
    }

    pub fn reset(&mut self) -> ExampleResult<()> {
        self.sun_enabled = true;
        self.fog_enabled = true;
        self.nonuniform = true;
        self.front_only = false;
        self.alpha_mode = 0;
        self.phase = 0.0;
        self.update(0.0)
    }
}

fn surface(mode: usize, color: Color) -> ExampleResult<SurfaceStyle3d> {
    Ok(match mode % 3 {
        0 => SurfaceStyle3d::opaque(color)?,
        1 => SurfaceStyle3d::mask(color, 0.5)?,
        _ => SurfaceStyle3d::blend(color)?,
    })
}

fn shape_style(mode: usize, front_only: bool, lit: bool) -> ExampleResult<MeshStyle3d> {
    let color = if mode % 3 == 2 {
        Color::rgba(1.0, 1.0, 1.0, 0.65)
    } else {
        Color::WHITE
    };
    Ok(MeshStyle3d::surface(
        surface(mode, color)?
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
            .with_fog(false),
    ))
}

fn shape_transform(x: f32, phase: f32, nonuniform: bool) -> ExampleResult<Transform3d> {
    let scale = if nonuniform {
        Vec3::new(1.45, 1.05, 0.6)?
    } else {
        Vec3::new(1.15, 1.15, 1.15)?
    };
    Ok(Transform3d::new(
        Vec3::new(x, 1.05, 0.0)?,
        Rotation3d::from_euler_xyz(0.2, phase * 0.5 + 0.35, -0.12)?,
        scale,
    )?)
}

fn sphere() -> ExampleResult<Mesh3d> {
    let mut vertices = vec![Vec3::Y];
    for ring in 1..SPHERE_RINGS {
        let latitude = std::f32::consts::PI * ring as f32 / SPHERE_RINGS as f32;
        for column in 0..SPHERE_COLUMNS {
            let longitude = std::f32::consts::TAU * column as f32 / SPHERE_COLUMNS as f32;
            vertices.push(Vec3::new(
                latitude.sin() * longitude.cos(),
                latitude.cos(),
                latitude.sin() * longitude.sin(),
            )?);
        }
    }
    let south = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, -1.0, 0.0)?);
    let mut indices = Vec::new();
    for column in 0..SPHERE_COLUMNS {
        outward_triangle(
            &vertices,
            &mut indices,
            [
                0,
                1 + column as u32,
                1 + ((column + 1) % SPHERE_COLUMNS) as u32,
            ],
        );
        for ring in 0..SPHERE_RINGS - 2 {
            let start = 1 + ring * SPHERE_COLUMNS;
            let next = 1 + (ring + 1) * SPHERE_COLUMNS;
            let following = (column + 1) % SPHERE_COLUMNS;
            outward_triangle(
                &vertices,
                &mut indices,
                [
                    (start + column) as u32,
                    (next + column) as u32,
                    (next + following) as u32,
                ],
            );
            outward_triangle(
                &vertices,
                &mut indices,
                [
                    (start + column) as u32,
                    (next + following) as u32,
                    (start + following) as u32,
                ],
            );
        }
        let last = 1 + (SPHERE_RINGS - 2) * SPHERE_COLUMNS;
        outward_triangle(
            &vertices,
            &mut indices,
            [
                south,
                (last + column) as u32,
                (last + (column + 1) % SPHERE_COLUMNS) as u32,
            ],
        );
    }
    let colors = vertices
        .iter()
        .map(|normal| {
            let amount = normal.y() * 0.5 + 0.5;
            Color::rgba(
                0.28 + 0.30 * amount,
                0.60 + 0.25 * amount,
                0.98,
                0.15 + 0.85 * amount,
            )
        })
        .collect();
    let attributes = Mesh3dAttributes::new()
        .with_normals(vertices.clone())?
        .with_vertex_colors(colors)?;
    Ok(Mesh3d::with_attributes(
        vertices,
        indices,
        Vec::new(),
        attributes,
    )?)
}

fn outward_triangle(vertices: &[Vec3], indices: &mut Vec<u32>, mut triangle: [u32; 3]) {
    let points = triangle.map(|index| vertices[index as usize]);
    let difference = |left: Vec3, right: Vec3| {
        [
            f64::from(left.x()) - f64::from(right.x()),
            f64::from(left.y()) - f64::from(right.y()),
            f64::from(left.z()) - f64::from(right.z()),
        ]
    };
    let left = difference(points[1], points[0]);
    let right = difference(points[2], points[0]);
    let cross = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
    let dot = cross[0] * f64::from(points[0].x())
        + cross[1] * f64::from(points[0].y())
        + cross[2] * f64::from(points[0].z());
    if dot < 0.0 {
        triangle.swap(1, 2);
    }
    indices.extend(triangle);
}

fn swatch() -> ExampleResult<Mesh3d> {
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
            Vec3::new(-0.56, -0.37, 0.0)?,
            Vec3::new(0.56, -0.37, 0.0)?,
            Vec3::new(0.56, 0.37, 0.0)?,
            Vec3::new(-0.56, 0.37, 0.0)?,
        ],
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
        attributes,
    )?)
}

fn alpha_texture() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(1024);
    for y in 0..16 {
        for x in 0..16 {
            let alpha = if (x / 4 + y / 4) & 1 == 0 { 255 } else { 0 };
            pixels.extend([255, 255, 255, alpha]);
        }
    }
    pixels
}

fn marker_mesh() -> ExampleResult<Mesh3d> {
    Ok(Mesh3d::new(
        vec![
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(-1.0, 0.0, 0.0)?,
            Vec3::new(0.0, -1.0, 0.0)?,
            Vec3::new(0.0, 0.0, -1.0)?,
        ],
        vec![
            0, 1, 2, 2, 1, 3, 3, 1, 5, 5, 1, 0, 2, 4, 0, 3, 4, 2, 5, 4, 3, 0, 4, 5,
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_supplies_smooth_normals_and_non_degenerate_outward_faces() {
        let mesh = sphere().unwrap();
        assert_eq!(mesh.vertices().len(), 482);
        assert_eq!(mesh.triangle_count(), 960);
        assert_eq!(mesh.normals().len(), mesh.vertices().len());
        assert_eq!(mesh.vertex_colors().len(), mesh.vertices().len());
        for (position, normal) in mesh.vertices().iter().zip(mesh.normals()) {
            let agreement =
                position.x() * normal.x() + position.y() * normal.y() + position.z() * normal.z();
            assert!((agreement - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn swatches_and_nonuniform_transforms_are_explicit_visual_state() {
        let mesh = swatch().unwrap();
        assert_eq!(mesh.normals(), &[Vec3::Z; 4]);
        assert_eq!(mesh.texture_coordinates().len(), 4);
        assert_eq!(alpha_texture().len(), 1024);
        let scaled = shape_transform(2.25, 0.4, true).unwrap();
        assert_ne!(scaled.scale().x(), scaled.scale().z());
        let uniform = shape_transform(2.25, 0.4, false).unwrap();
        assert_eq!(uniform.scale().x(), uniform.scale().z());
    }
}
