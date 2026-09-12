//! Shared surface-environment uniforms and source-attributed attribute proofs.

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SurfaceEnvironmentGpu {
    pub(super) depth_row: [f32; 4],
    pub(super) ambient: [f32; 4],
    pub(super) sunlight: [f32; 4],
    pub(super) direction: [f32; 4],
    pub(super) fog_color_start: [f32; 4],
    pub(super) fog_density: [f32; 4],
}

impl SurfaceEnvironmentGpu {
    pub(super) fn new(scene: &Scene3d, camera: Camera3d) -> Self {
        let mut result = Self::default();
        let ambient = scene.lighting().ambient();
        result.ambient = ambient
            .color()
            .to_array()
            .map(|channel| channel * ambient.intensity());
        if let Some(sun) = scene
            .lighting()
            .directional()
            .filter(|sun| sun.intensity() > 0.0)
        {
            result.sunlight = sun
                .color()
                .to_array()
                .map(|channel| channel * sun.intensity());
            let direction = sun.direction();
            result.direction = [direction.x(), direction.y(), direction.z(), 1.0];
        }
        if let Some(fog) = scene.fog().filter(|fog| fog.density() > 0.0) {
            result.fog_color_start = fog.color().to_array();
            result.fog_color_start[3] = fog.start();
            result.fog_density[0] = fog.density();
            let forward = camera.forward();
            let position = camera.position();
            let offset = -(f64::from(forward.x()) * f64::from(position.x())
                + f64::from(forward.y()) * f64::from(position.y())
                + f64::from(forward.z()) * f64::from(position.z()));
            // Invalid conversion is rejected with the first participating source
            // vertex, not hidden behind an unrelated camera-construction error.
            result.depth_row = [forward.x(), forward.y(), forward.z(), offset as f32];
        }
        result
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SurfaceTransport {
    pub(super) normal_rows: [[f32; 4]; 3],
    depth_row: Option<[f32; 4]>,
    lit: bool,
    pub(super) generated_attributes: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SurfaceLightingVertex {
    pub(super) normal_depth: [f32; 4],
}
impl SurfaceLightingVertex {
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![8 => Float32x4],
    };
}

impl SurfaceTransport {
    pub(super) fn new(
        instance: &Mesh3dInstance,
        environment: SurfaceEnvironmentGpu,
    ) -> Result<Self, Mesh3dRenderError> {
        let Some(style) = instance.style.surface_style() else {
            return Ok(Self::default());
        };
        if instance.mesh.source().triangle_count() == 0 {
            return Ok(Self::default());
        }
        let requested_lighting = style.lighting() == SurfaceLighting3d::Lambert;
        let lit = requested_lighting && environment.direction[3] > 0.0;
        let mut result = Self {
            lit,
            generated_attributes: requested_lighting || style.fog_enabled(),
            ..Self::default()
        };
        if lit {
            result.normal_rows = normal_rows(instance.transform)
                .map_err(|reason| vertex_error(instance.id, 0, reason))?;
        }
        if style.fog_enabled() && environment.fog_density[0] > 0.0 {
            result.depth_row = Some(environment.depth_row);
        }
        Ok(result)
    }

    pub(super) fn validate(
        self,
        instance: &Mesh3dInstance,
        model: [[f32; 4]; 3],
    ) -> Result<(), Mesh3dRenderError> {
        if !self.lit && self.depth_row.is_none() {
            return Ok(());
        }
        for vertex_index in 0..instance.mesh.source().vertices().len() {
            self.vertex(instance.mesh.source(), vertex_index, model)
                .map_err(|reason| vertex_error(instance.id, vertex_index, reason))?;
        }
        Ok(())
    }

    pub(super) fn vertex(
        self,
        mesh: &Mesh3d,
        index: usize,
        model: [[f32; 4]; 3],
    ) -> Result<[f32; 4], Mesh3dSurfaceError> {
        let mut output = [0.0; 4];
        if self.lit {
            let normal = mesh
                .normals()
                .get(index)
                .ok_or(Mesh3dSurfaceError::NormalTransform)?;
            let normal = [normal.x(), normal.y(), normal.z()];
            let mut guaranteed_nonzero = false;
            for (axis, output_component) in output.iter_mut().take(3).enumerate() {
                let row = self.normal_rows[axis];
                let mut value = 0.0_f32;
                let mut magnitude = 0.0_f64;
                for component in 0..3 {
                    let product = row[component] * normal[component];
                    value += product;
                    magnitude += (f64::from(row[component]) * f64::from(normal[component])).abs();
                }
                // Every operand is bounded by one. Include association/FMA and
                // denormal flushing in a conservative absolute cancellation bound.
                let uncertainty = magnitude * 32.0 * f64::from(f32::EPSILON)
                    + 16.0 * f64::from(f32::MIN_POSITIVE);
                guaranteed_nonzero |= f64::from(value).abs() > uncertainty;
                *output_component = value;
            }
            if !guaranteed_nonzero {
                return Err(Mesh3dSurfaceError::NormalTransform);
            }
        }
        if let Some(depth_row) = self.depth_row {
            let point = mesh.vertices()[index];
            let point = [point.x(), point.y(), point.z(), 1.0].map(ShaderValueRange::exact);
            let mut world = [ShaderValueRange::exact(1.0); 4];
            for axis in 0..3 {
                world[axis] = shader_dot_range(model[axis], point)
                    .map_err(|_| Mesh3dSurfaceError::FogArithmetic)?;
            }
            output[3] = shader_dot_range(depth_row, world)
                .map_err(|_| Mesh3dSurfaceError::FogArithmetic)?
                .fixed;
        }
        Ok(output)
    }
}

fn vertex_error(
    object_id: Object3dId,
    vertex_index: usize,
    reason: Mesh3dSurfaceError,
) -> Mesh3dRenderError {
    Mesh3dRenderError::ObjectFailure {
        object_id,
        reason: Mesh3dObjectError::Vertex {
            vertex_index,
            reason,
        },
    }
}

fn normal_rows(transform: Transform3d) -> Result<[[f32; 4]; 3], Mesh3dSurfaceError> {
    let scale = transform.scale();
    let scale = [scale.x(), scale.y(), scale.z()].map(f64::from);
    let minimum = scale.into_iter().fold(f64::INFINITY, f64::min);
    let mut rows = [[0.0; 4]; 3];
    for (column, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let rotated = transform
            .rotation()
            .rotate(axis)
            .map_err(|_| Mesh3dSurfaceError::NormalTransform)?;
        for (row, value) in [rotated.x(), rotated.y(), rotated.z()]
            .into_iter()
            .enumerate()
        {
            let value = (f64::from(value) * (minimum / scale[column])) as f32;
            // Canonicalize coefficients the backend may flush to zero. A lost
            // source direction is rejected by the per-vertex proof above.
            rows[row][column] = if value.is_normal() { value } else { 0.0 };
        }
    }
    Ok(rows)
}

pub(super) fn shader_source(body: &'static str) -> Cow<'static, str> {
    Cow::Owned(format!(
        "{}\n{body}",
        include_str!("mesh3d_environment.wgsl")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_transpose_preserves_uniform_tiny_scales_and_rejects_lost_source_directions() {
        let mesh = |normal| {
            Mesh3d::with_attributes(
                vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                vec![0, 1, 2],
                vec![],
                crate::Mesh3dAttributes::new()
                    .with_normals(vec![normal; 3])
                    .unwrap(),
            )
            .unwrap()
        };
        let tiny = Transform3d::from_rotation_scale(crate::Rotation3d::IDENTITY, 1e-30).unwrap();
        let transport = SurfaceTransport {
            normal_rows: normal_rows(tiny).unwrap(),
            lit: true,
            ..SurfaceTransport::default()
        };
        assert_eq!(
            transport
                .vertex(&mesh(Vec3::Z), 0, tiny.model_rows().unwrap())
                .unwrap(),
            [0.0, 0.0, 1.0, 0.0]
        );
        let extreme = Transform3d::new(
            Vec3::ZERO,
            crate::Rotation3d::IDENTITY,
            Vec3::new(1e-20, 1e20, 1.0).unwrap(),
        )
        .unwrap();
        let transport = SurfaceTransport {
            normal_rows: normal_rows(extreme).unwrap(),
            lit: true,
            ..SurfaceTransport::default()
        };
        assert_eq!(
            transport.vertex(&mesh(Vec3::Y), 0, extreme.model_rows().unwrap()),
            Err(Mesh3dSurfaceError::NormalTransform)
        );
        assert!(
            transport
                .vertex(&mesh(Vec3::X), 0, extreme.model_rows().unwrap())
                .is_ok()
        );
        assert!(
            SurfaceTransport::default()
                .vertex(&mesh(Vec3::Y), 0, extreme.model_rows().unwrap())
                .is_ok()
        );
        let anisotropic = Transform3d::new(
            Vec3::ZERO,
            crate::Rotation3d::IDENTITY,
            Vec3::new(2.0, 4.0, 8.0).unwrap(),
        )
        .unwrap();
        assert_eq!(
            normal_rows(anisotropic).unwrap(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.0]
            ]
        );
    }
}
