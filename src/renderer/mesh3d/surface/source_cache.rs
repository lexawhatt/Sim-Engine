//! Invocation-local source-index memoization, without caller-sized allocation.

use super::{
    Mesh3d, Mesh3dSurfaceError, ShaderValueRange, SurfaceTransport, shader_clip_point_ranges,
};

pub(super) const CAPACITY: usize = 64;

pub(super) fn worthwhile(mesh: &Mesh3d) -> bool {
    // Tiny and unshared meshes retain the zero-storage specialization. Indexed
    // grids amortize initialization across many repeated source references.
    mesh.triangle_count() >= 16 && mesh.triangle_indices().len() > mesh.vertices().len()
}

pub(super) struct SourceCache<'a, const SLOTS: usize> {
    mesh: &'a Mesh3d,
    model: [[f32; 4]; 3],
    camera: [[f32; 4]; 4],
    transport: SurfaceTransport,
    clips: [Option<(u32, [ShaderValueRange; 4])>; SLOTS],
    auxiliary: [Option<(u32, [f32; 4])>; SLOTS],
    #[cfg(test)]
    evaluations: [usize; 2],
}

impl<'a, const SLOTS: usize> SourceCache<'a, SLOTS> {
    pub(super) fn new(
        mesh: &'a Mesh3d,
        model: [[f32; 4]; 3],
        camera: [[f32; 4]; 4],
        transport: SurfaceTransport,
    ) -> Self {
        Self {
            mesh,
            model,
            camera,
            transport,
            clips: [None; SLOTS],
            auxiliary: [None; SLOTS],
            #[cfg(test)]
            evaluations: [0; 2],
        }
    }

    #[inline]
    pub(super) fn clip(&mut self, index: u32) -> Result<[ShaderValueRange; 4], Mesh3dSurfaceError> {
        if SLOTS > 0
            && let Some((source, value)) = &self.clips[index as usize % SLOTS]
            && *source == index
        {
            return Ok(*value);
        }
        #[cfg(test)]
        {
            self.evaluations[0] += 1;
        }
        let value = shader_clip_point_ranges(
            self.mesh.vertices()[index as usize],
            self.model,
            self.camera,
        )
        .map_err(|_| Mesh3dSurfaceError::TransformArithmetic)?;
        if SLOTS > 0 {
            self.clips[index as usize % SLOTS] = Some((index, value));
        }
        Ok(value)
    }

    #[inline]
    pub(super) fn auxiliary(&mut self, index: u32) -> Result<[f32; 4], Mesh3dSurfaceError> {
        if SLOTS > 0
            && let Some((source, value)) = &self.auxiliary[index as usize % SLOTS]
            && *source == index
        {
            return Ok(*value);
        }
        #[cfg(test)]
        {
            self.evaluations[1] += 1;
        }
        let value = self
            .transport
            .vertex(self.mesh, index as usize, self.model)?;
        if SLOTS > 0 {
            self.auxiliary[index as usize % SLOTS] = Some((index, value));
        }
        Ok(value)
    }

    #[cfg(test)]
    pub(super) fn evaluations(&self) -> [usize; 2] {
        self.evaluations
    }
}
