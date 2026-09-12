//! Host-owned fixtures; simulation and chunk models remain outside the library.

use sim_engine::{
    Camera3d, Color, DynamicMesh3dBudget, Mesh3d, Mesh3dRenderBudget, Mesh3dUploadBudget,
    MeshStyle3d, Object3dId, Projection3d, Rotation3d, Scene3d, SurfaceRasterization3d,
    SurfaceStyle3d, Transform3d, Vec3, WgpuRenderer, WorldLength,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Repeated,
    Outside,
    HostHidden,
    Immutable,
    Dynamic,
}

impl Case {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "repeated" => Ok(Self::Repeated),
            "outside" => Ok(Self::Outside),
            "host_hidden" => Ok(Self::HostHidden),
            "immutable" => Ok(Self::Immutable),
            "dynamic" => Ok(Self::Dynamic),
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
        }
    }

    pub fn changes_mesh(self) -> bool {
        matches!(self, Self::Immutable | Self::Dynamic)
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
}

pub struct Workload {
    pub scene: Scene3d,
    pub camera: Camera3d,
    pub budget: Mesh3dRenderBudget,
    pub expected_objects: usize,
    pub expected_triangles: usize,
    pub host_snapshot_bytes: usize,
    sources: [Mesh3d; 2],
    object: Object3dId,
    case: Case,
}

impl Workload {
    pub fn new(renderer: &WgpuRenderer, case: Case, objects: usize, side: usize) -> Result<Self> {
        let sources = [grid(side, 0.0)?, grid(side, 0.25)?];
        let mesh = renderer.create_mesh3d(sources[0].clone())?;
        let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb8(82, 176, 233))?);
        let mut scene = Scene3d::new(Color::rgb8(8, 12, 20))?;
        let columns = (objects as f64).sqrt().ceil() as usize;
        let mut first = None;
        let mut expected_objects = 0;
        for index in 0..objects {
            let outside = matches!(case, Case::Outside | Case::HostHidden) && index % 10 != 0;
            let x = (index % columns) as f32 - columns as f32 * 0.5;
            let y = (index / columns) as f32 - columns as f32 * 0.5;
            let transform = Transform3d::new(
                Vec3::new(x + if outside { 10_000.0 } else { 0.0 }, y, 0.0)?,
                Rotation3d::IDENTITY,
                Vec3::new(0.9, 0.9, 0.9)?,
            )?;
            let id = scene.try_push(&mesh, transform, style)?;
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
            budget: Mesh3dRenderBudget::new(0, 0, 0)
                .with_surface_policy(SurfaceRasterization3d::Native)
                .with_max_surface_triangles(expected_triangles),
            expected_objects,
            expected_triangles,
            host_snapshot_bytes: sources.iter().map(Mesh3d::recovery_memory_bytes).sum(),
            sources,
            object: first.ok_or("at least one object is required")?,
            case,
        })
    }

    pub fn update(&mut self, renderer: &mut WgpuRenderer, frame: usize) -> Result<UpdateCounters> {
        let mut result = UpdateCounters::default();
        if !self.case.changes_mesh() {
            return Ok(result);
        }
        let source = self.sources[frame % 2].clone();
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
            result.bytes = report.uploaded_bytes();
            result.calls = report.upload_calls();
            result.gpu_allocations = report.gpu_allocation_count();
            result.reused = report.reused_buffers();
            result.detached = report.detached_aliases();
            result.capacity = report.capacity_gpu_bytes();
            result.scratch = report.scratch_capacity_bytes();
            result.scratch_reallocations = report.scratch_reallocations();
        }
        Ok(result)
    }
}

fn grid(side: usize, center_height: f32) -> Result<Mesh3d> {
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
    Ok(Mesh3d::new(vertices, indices)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_revisions_have_equal_capacity_and_distinct_geometry() {
        for side in [1, 4, 32] {
            let first = grid(side, 0.0).unwrap();
            let second = grid(side, 0.25).unwrap();
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
}
