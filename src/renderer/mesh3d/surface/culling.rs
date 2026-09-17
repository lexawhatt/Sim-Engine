//! Success-only Native surface omission; never a replacement for validation.

use super::*;

pub(super) fn omit_offscreen_surfaces(
    scene: &Scene3d,
    camera: Camera3dUniform,
    frame: &mut SurfaceFrame,
) {
    for (instance, object) in scene
        .instances()
        .iter()
        .filter(|instance| instance.visible)
        .zip(&mut frame.objects)
    {
        if instance.style.surface_style().is_none()
            || instance.wireframe().is_some()
            || instance.mesh.triangle_count() == 0
            || object.generated.is_some()
            || !validation::native_mesh_is_outside(
                instance.mesh.source(),
                object.model_rows,
                camera.rows(),
            )
        {
            continue;
        }
        // Reuse the encoder's empty surface range without changing object layout
        // or visible-index addressing. This is assigned AFTER generated emission;
        // it must never trigger a second Strict classifier or count as generated.
        object.generated = Some(0..0);
        frame.report.culled_objects += 1;
        frame.report.culled_triangles += instance.mesh.triangle_count();
        frame.report.submitted_triangles -= instance.mesh.triangle_count();
    }
}
