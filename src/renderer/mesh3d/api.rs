//! Public renderer entry points for retained 3D resources and frames.

use super::*;

impl WgpuRenderer {
    /// Uploads validated immutable topology into retained GPU buffers.
    ///
    /// Counts, byte arithmetic, draw-count representation, and active-device
    /// buffer limits are checked before conversion staging allocation. Host
    /// staging reservation failure is returned explicitly; GPU allocation
    /// itself follows wgpu's device-error model.
    pub fn create_mesh3d(&self, source: Mesh3d) -> Result<RetainedMesh3d, Mesh3dResourceError> {
        create_retained_mesh(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }

    /// Restores retained topology and its complete optional texture material
    /// onto this renderer after device replacement. Attribute/reserved buffer
    /// capacities, committed mip bytes/options, filtering, tint, UV transform,
    /// addressing and opaque/alpha-capable rebinding policy are preserved.
    /// Source handles remain unchanged. Separate calls restore separate GPU
    /// resources; use `restore_scene3d` to deduplicate shared scene resources.
    pub fn restore_mesh3d(
        &self,
        source: &RetainedMesh3d,
    ) -> Result<RetainedMesh3d, Mesh3dResourceError> {
        restore_retained_mesh(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            &self.mesh3d_renderer.textures.layout,
            source,
        )
    }

    /// Atomically restores every stale retained mesh referenced by a 3D scene.
    ///
    /// Distinct shared mesh and texture resources are uploaded once. Per-handle
    /// texture/filter/tint choices remain independent when topology is shared.
    /// Object IDs, insertion
    /// order, transforms, styles, visibility, scene provenance, and the next ID
    /// remain unchanged. If any capacity or host-staging allocation fails, the
    /// original scene is not modified. Targets remain separate resources and
    /// must be restored with [`WgpuRenderer::restore_render_target3d`].
    pub fn restore_scene3d(
        &self,
        scene: &mut Scene3d,
    ) -> Result<Scene3dRestoreReport, Mesh3dResourceError> {
        restore_scene3d_resources(
            &self.device,
            &self.queue,
            &self.mesh3d_renderer.textures.layout,
            Arc::clone(&self.renderer_identity),
            scene,
        )
    }

    /// Creates color/depth attachments for an explicit logical viewport.
    ///
    /// `width` and `height` are physical target texels. Their aspect must match
    /// `logical_viewport`; the resulting texels-per-logical-pixel ratio controls
    /// wireframe width independently of the window DPI scale.
    pub fn create_render_target3d(
        &self,
        width: u32,
        height: u32,
        logical_viewport: LogicalViewport,
    ) -> Result<RenderTarget3d, Mesh3dResourceError> {
        let pixels_per_logical = target_pixels_per_logical(width, height, logical_viewport)
            .ok_or(Mesh3dResourceError::InvalidViewportAspect)?;
        let color = self
            .create_render_target(width, height)
            .map_err(Mesh3dResourceError::Target)?;
        let depth_texture = create_depth_texture(&self.device, width, height);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(RenderTarget3d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            color,
            _depth_texture: depth_texture,
            depth_view,
            logical_viewport,
            pixels_per_logical,
        })
    }

    /// Restores empty color/depth attachments with the source dimensions.
    pub fn restore_render_target3d(
        &self,
        source: &RenderTarget3d,
    ) -> Result<RenderTarget3d, Mesh3dResourceError> {
        self.create_render_target3d(source.width(), source.height(), source.logical_viewport())
    }

    /// Draws retained opaque/masked surfaces, sorted blended surfaces, then edges
    /// into a reusable premultiplied color/depth target.
    ///
    /// Object insertion order does not determine visibility. Every surface
    /// writes and tests hardware depth. Model transforms are uploaded through a
    /// reusable instance buffer. Wholly inside objects retain their indexed
    /// topology; crossing surfaces use bounded homogeneous CPU clipping against
    /// all six frustum planes. Ambiguous classification/orientation produces an
    /// object-attributed error. Explicit display edges retain their complete
    /// interval-validated homogeneous shader clipper.
    pub fn render_scene3d_to_target(
        &mut self,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
    ) -> Result<Mesh3dRenderReport, Mesh3dRenderError> {
        self.render_scene3d_to_target_with_budget(
            target,
            scene,
            camera,
            Mesh3dRenderBudget::default(),
        )
    }

    /// Validates every visible object and exact submitted/generated work without
    /// changing target pixels, retained resources or submitting GPU work.
    ///
    /// This is the same authoritative preflight used by rendering. Private
    /// bounded scratch may be allocated; the report is not a reusable draw
    /// authorization after the scene, target or camera changes.
    /// [`Mesh3dRenderBudget::surface_policy`] controls filled surfaces only.
    /// Native mode reports no CPU clipped/discarded source counts.
    pub fn validate_scene3d_for_target(
        &self,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
    ) -> Result<Mesh3dPreflightReport, Mesh3dRenderError> {
        self.mesh3d_renderer
            .preflight_scene3d(
                &self.device,
                &self.renderer_identity,
                target,
                scene,
                camera,
                budget,
            )
            .map(|(_, frame)| frame.report)
    }

    /// Draws a scene with explicit surface policy and submitted/generated limits.
    ///
    /// All validation, generated counts and host/GPU capacity checks complete
    /// before the first GPU write or target mutation. Numerical ambiguity is
    /// attributed to the exact visible object; aggregate capacity remains a
    /// scene-level error. Wholly inside objects retain their indexed GPU path.
    /// With [`SurfaceRasterization3d::Native`], all filled surfaces keep their
    /// original indices and hardware clipping defines boundary coverage. Finite
    /// shader arithmetic and independent mathematical-edge proofs still apply.
    pub fn render_scene3d_to_target_with_budget(
        &mut self,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
    ) -> Result<Mesh3dRenderReport, Mesh3dRenderError> {
        self.mesh3d_renderer.render_scene3d_with_timing(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            target,
            scene,
            camera,
            budget,
            Some(&mut self.gpu_timing),
        )
    }
}
