//! Optional model-normal storage, independent of the legacy position stream.

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MeshNormalGpu {
    pub(super) normal: [f32; 3],
}

impl MeshNormalGpu {
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![8 => Float32x3],
    };
    pub(super) fn from_source(normal: &Vec3) -> Self {
        Self {
            normal: [normal.x(), normal.y(), normal.z()],
        }
    }
}
