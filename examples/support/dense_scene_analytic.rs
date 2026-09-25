//! Alternate rendering of the exact same host Scene commands, not a new fixture.

use super::workload::Result;
use sim_engine::{
    Camera2d, DrawCommand, Fill, FrameComposer, FramePassOptions, LogicalPixels,
    LogicalScreenVector, LogicalViewport, Scene, ScreenPrimitive2d, ScreenPrimitiveBatch2d,
    ScreenPrimitiveBudget, Vec2, WgpuRenderer,
};
use std::time::{Duration, Instant};

pub struct AnalyticChunks {
    batches: Vec<ScreenPrimitiveBatch2d>,
    projected: Vec<Vec<ScreenPrimitive2d>>,
    camera: Camera2d,
    viewport: LogicalViewport,
    update_timing: (Duration, Duration),
}

impl AnalyticChunks {
    pub fn new(
        renderer: &WgpuRenderer,
        scenes: &[Scene],
        viewport: LogicalViewport,
    ) -> Result<Self> {
        let camera = Camera2d::new(Vec2::ZERO, 1.0)?;
        let mut projected = Vec::with_capacity(scenes.len());
        let mut batches = Vec::with_capacity(scenes.len());
        for scene in scenes {
            let mut primitives = Vec::with_capacity(scene.command_count());
            project(scene, camera, viewport, &mut primitives)?;
            batches.push(
                renderer
                    .create_screen_primitive_batch(&primitives, ScreenPrimitiveBudget::default())?,
            );
            projected.push(primitives);
        }
        Ok(Self {
            batches,
            projected,
            camera,
            viewport,
            update_timing: (Duration::ZERO, Duration::ZERO),
        })
    }

    pub fn update(
        &mut self,
        renderer: &WgpuRenderer,
        scenes: &[Scene],
        camera: Camera2d,
        changed_tile: Option<usize>,
    ) -> Result<usize> {
        let mut uploaded = 0;
        self.update_timing = (Duration::ZERO, Duration::ZERO);
        for (tile, scene) in scenes.iter().enumerate() {
            if camera != self.camera || changed_tile == Some(tile) {
                let projection_started = Instant::now();
                project(scene, camera, self.viewport, &mut self.projected[tile])?;
                self.update_timing.0 += projection_started.elapsed();
                let engine_started = Instant::now();
                uploaded += renderer
                    .update_screen_primitive_batch(&mut self.batches[tile], &self.projected[tile])?
                    .uploaded_bytes();
                self.update_timing.1 += engine_started.elapsed();
            }
        }
        self.camera = camera;
        Ok(uploaded)
    }

    pub fn update_timing(&self) -> (f64, f64) {
        (
            self.update_timing.0.as_secs_f64() * 1000.0,
            self.update_timing.1.as_secs_f64() * 1000.0,
        )
    }

    pub fn draw<'frame>(&'frame self, frame: &mut FrameComposer<'frame>) -> Result<()> {
        for (tile, batch) in self.batches.iter().enumerate() {
            frame.draw_screen_primitive_batch(batch, FramePassOptions::new(tile as i32))?;
        }
        Ok(())
    }

    pub fn restore(&mut self, renderer: &WgpuRenderer) -> Result<()> {
        self.batches = self
            .batches
            .iter()
            .map(|batch| Ok(renderer.restore_screen_primitive_batch(batch)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub fn projected_cpu_bytes(&self) -> usize {
        self.projected
            .iter()
            .map(|values| values.capacity() * std::mem::size_of::<ScreenPrimitive2d>())
            .sum()
    }
}

fn project(
    scene: &Scene,
    camera: Camera2d,
    viewport: LogicalViewport,
    output: &mut Vec<ScreenPrimitive2d>,
) -> Result<()> {
    output.clear();
    for command in scene.commands() {
        let mut primitive = match command.command() {
            DrawCommand::Circle(circle) => {
                let Some(Fill::Solid(color)) = circle.style().fill() else {
                    return Err("fixture circle is not solid".into());
                };
                ScreenPrimitive2d::circle(
                    camera.world_to_screen(circle.center(), viewport)?,
                    LogicalPixels::new(circle.radius() * camera.zoom())?,
                    color,
                )?
            }
            DrawCommand::Line(line) => ScreenPrimitive2d::round_line(
                camera.world_to_screen(line.from(), viewport)?,
                camera.world_to_screen(line.to(), viewport)?,
                LogicalPixels::new(line.stroke().width())?,
                line.stroke().color(),
            )?,
            DrawCommand::Rect(rectangle) => {
                let Some(Fill::Solid(color)) = rectangle.style().fill() else {
                    return Err("fixture rectangle is not solid".into());
                };
                let rectangle = rectangle.rect();
                let minimum = camera.world_to_screen(
                    Vec2::new(rectangle.min().x(), rectangle.max().y()),
                    viewport,
                )?;
                let maximum = camera.world_to_screen(
                    Vec2::new(rectangle.max().x(), rectangle.min().y()),
                    viewport,
                )?;
                ScreenPrimitive2d::rect(
                    minimum,
                    LogicalScreenVector::from_vec2(maximum.to_vec2() - minimum.to_vec2()),
                    color,
                )?
            }
            DrawCommand::Polyline(_) => {
                return Err("fixture unexpectedly contains a polyline".into());
            }
            _ => return Err("fixture contains an unsupported future scene command".into()),
        };
        if let Some(clip) = command.screen_clip() {
            primitive = primitive.with_clip(clip);
        }
        output.push(primitive);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_engine::{Color, LogicalScreenPosition, Rect, ScreenClipRect, ShapeStyle};

    #[test]
    fn camera_projection_keeps_all_source_commands_and_clips() {
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(0.25, 0.75),
            LogicalScreenVector::new(100.0, 100.0),
        )
        .unwrap();
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.set_screen_clip(Some(clip)).unwrap();
        scene
            .try_circle(
                Vec2::new(0.375, 1.625),
                4.0,
                ShapeStyle::filled(Color::WHITE),
            )
            .unwrap();
        scene
            .try_line(Vec2::ZERO, Vec2::new(12.0, 9.0), 1.25, Color::WHITE)
            .unwrap();
        scene
            .try_rect(
                Rect::from_center_size(Vec2::ZERO, Vec2::new(6.25, 3.75)),
                0.0,
                ShapeStyle::filled(Color::WHITE),
            )
            .unwrap();
        for zoom in [0.92, 1.0, 1.08] {
            let camera = Camera2d::new(Vec2::new(12.0, -9.0), zoom).unwrap();
            let mut projected = Vec::new();
            project(
                &scene,
                camera,
                LogicalViewport::new(1280.0, 800.0).unwrap(),
                &mut projected,
            )
            .unwrap();
            assert_eq!(projected.len(), scene.command_count());
            assert!(
                projected
                    .iter()
                    .all(|primitive| primitive.clip() == Some(clip))
            );
        }
    }
}
