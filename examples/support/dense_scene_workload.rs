//! Host-owned deterministic visual fixtures, without mathematical domain state.

use super::analytic::AnalyticChunks;

use sim_engine::{
    Camera2d, Color, FrameComposer, FramePassOptions, LogicalScreenPosition, LogicalScreenVector,
    PreparedScene, Rect, Scene, SceneBudget, ScreenClipRect, ShapeStyle, Vec2, WgpuRenderer,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const TILES: usize = 24;
pub const COMMANDS_PER_CONSTRUCTION: usize = 19;
pub const BACKGROUND: Color = Color::rgb(0.015, 0.021, 0.033);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Streaming,
    Prepared,
    Analytic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Pan,
    Zoom,
    Drag,
}

impl Phase {
    pub const ALL: [Self; 4] = [Self::Idle, Self::Pan, Self::Zoom, Self::Drag];

    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Pan => "pan",
            Self::Zoom => "zoom",
            Self::Drag => "drag",
        }
    }

    pub fn pose(self, frame: usize) -> Result<(Camera2d, f32)> {
        let angle = frame as f32 * 0.035;
        let center = if self == Self::Pan {
            Vec2::new(angle.sin() * 12.0, angle.cos() * 9.0)
        } else {
            Vec2::ZERO
        };
        let zoom = if self == Self::Zoom {
            1.0 + angle.sin() * 0.08
        } else {
            1.0
        };
        let displacement = if self == Self::Drag {
            angle.sin() * 7.0
        } else {
            0.0
        };
        Ok((Camera2d::new(center, zoom)?, displacement))
    }
}

pub struct Workload {
    pub scenes: Vec<Scene>,
    prepared: Vec<PreparedScene>,
    analytic: Option<AnalyticChunks>,
    pub constructions: usize,
    pub logical_size: Vec2,
    mode: Mode,
    displacement: f32,
}

impl Workload {
    pub fn new(
        renderer: &WgpuRenderer,
        constructions: usize,
        logical_size: Vec2,
        mode: Mode,
    ) -> Result<Self> {
        let scenes = (0..TILES)
            .map(|tile| tile_scene(constructions, tile, logical_size, 0.0))
            .collect::<Result<Vec<_>>>()?;
        let prepared = if mode == Mode::Prepared {
            scenes
                .iter()
                .map(|scene| Ok(renderer.prepare_scene(scene)?))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let analytic = if mode == Mode::Analytic {
            Some(AnalyticChunks::new(
                renderer,
                &scenes,
                sim_engine::LogicalViewport::new(logical_size.x(), logical_size.y())?,
            )?)
        } else {
            None
        };
        Ok(Self {
            scenes,
            prepared,
            analytic,
            constructions,
            logical_size,
            mode,
            displacement: 0.0,
        })
    }

    /// Only the final construction moves; its final tile is the bounded mutable
    /// chunk in both controls. The other 23 chunks keep identical painter order.
    pub fn update(
        &mut self,
        renderer: &WgpuRenderer,
        camera: Camera2d,
        displacement: f32,
    ) -> Result<usize> {
        let changed = displacement != self.displacement;
        let mut uploaded = 0;
        if displacement != self.displacement {
            let scene = tile_scene(
                self.constructions,
                TILES - 1,
                self.logical_size,
                displacement,
            )?;
            if self.mode == Mode::Prepared {
                self.prepared[TILES - 1] = renderer.prepare_scene(&scene)?;
                uploaded += self.prepared[TILES - 1].tessellation_stats().upload_bytes();
            }
            self.scenes[TILES - 1] = scene;
            self.displacement = displacement;
        }
        if let Some(analytic) = &mut self.analytic {
            uploaded +=
                analytic.update(renderer, &self.scenes, camera, changed.then_some(TILES - 1))?;
        }
        Ok(uploaded)
    }

    pub fn draw<'frame>(
        &'frame self,
        frame: &mut FrameComposer<'frame>,
        camera: Camera2d,
    ) -> Result<()> {
        if let Some(analytic) = &self.analytic {
            return analytic.draw(frame);
        }
        for (tile, scene) in self.scenes.iter().enumerate() {
            let options = FramePassOptions::new(tile as i32);
            match self.mode {
                Mode::Streaming => frame.draw_scene(scene, camera, options)?,
                Mode::Prepared => {
                    frame.draw_prepared_scene(&self.prepared[tile], camera, options)?;
                }
                Mode::Analytic => return Err("analytic fixture lost its retained batches".into()),
            }
        }
        Ok(())
    }

    pub fn restore(&mut self, renderer: &WgpuRenderer) -> Result<()> {
        if let Some(analytic) = &mut self.analytic {
            analytic.restore(renderer)?;
        }
        self.prepared = self
            .prepared
            .iter()
            .map(|source| Ok(renderer.restore_prepared_scene(source)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    pub fn host_source_bytes(&self) -> usize {
        self.scenes
            .iter()
            .map(Scene::allocation_bytes)
            .sum::<usize>()
            + self
                .analytic
                .as_ref()
                .map_or(0, AnalyticChunks::projected_cpu_bytes)
    }

    pub fn source_counts(&self) -> (usize, usize, usize) {
        (
            self.scenes.iter().map(Scene::command_count).sum(),
            self.scenes
                .iter()
                .map(|s| s.statistics().estimated_tessellated_vertices())
                .sum(),
            self.scenes
                .iter()
                .map(|s| s.statistics().estimated_upload_bytes())
                .sum(),
        )
    }

    pub fn analytic_update_timing(&self) -> Option<(f64, f64)> {
        self.analytic.as_ref().map(AnalyticChunks::update_timing)
    }
}

pub fn construction_tile(index: usize, constructions: usize) -> usize {
    index * TILES / constructions
}

pub fn tile_clip(tile: usize, logical_size: Vec2) -> Result<ScreenClipRect> {
    let width = logical_size.x() / 6.0;
    let height = logical_size.y() / 4.0;
    Ok(ScreenClipRect::from_min_size(
        LogicalScreenPosition::new((tile % 6) as f32 * width, (tile / 6) as f32 * height),
        LogicalScreenVector::new(width, height),
    )?)
}

pub fn construction_center(
    index: usize,
    constructions: usize,
    logical_size: Vec2,
    displacement: f32,
) -> Vec2 {
    let tile = construction_tile(index, constructions);
    let first = (tile * constructions).div_ceil(TILES);
    let end = ((tile + 1) * constructions).div_ceil(TILES);
    let columns = ((end - first) as f32).sqrt().ceil() as usize;
    let rows = (end - first).div_ceil(columns);
    let width = logical_size.x() / 6.0;
    let height = logical_size.y() / 4.0;
    let local = index - first;
    let x = (tile % 6) as f32 * width
        + ((local % columns) as f32 + 0.5) * width / columns as f32
        + 0.375;
    let y = (tile / 6) as f32 * height
        + ((local / columns) as f32 + 0.5) * height / rows as f32
        + 0.625;
    Vec2::new(
        x - logical_size.x() * 0.5
            + if index + 1 == constructions {
                displacement
            } else {
                0.0
            },
        logical_size.y() * 0.5 - y,
    )
}

fn tile_scene(
    constructions: usize,
    tile: usize,
    logical_size: Vec2,
    displacement: f32,
) -> Result<Scene> {
    let first = (tile * constructions).div_ceil(TILES);
    let end = ((tile + 1) * constructions).div_ceil(TILES);
    let budget = SceneBudget::new(
        100_000,
        100_000,
        8_000_000,
        64 << 20,
        128 << 20,
        640 << 20,
        100_000,
    );
    let mut scene = Scene::with_budget(BACKGROUND, budget)?;
    scene.set_screen_clip(Some(tile_clip(tile, logical_size)?))?;
    for index in first..end {
        let center = construction_center(index, constructions, logical_size, displacement);
        let tint = match index % 3 {
            0 => Color::rgba(0.1, 0.65, 0.95, 0.65),
            1 => Color::rgba(0.95, 0.46, 0.08, 0.7),
            _ => Color::rgba(0.1, 0.75, 0.4, 0.8),
        };
        for segment in 0..13 {
            let angle = segment as f32 * std::f32::consts::TAU / 13.0;
            let next = (segment + 1) as f32 * std::f32::consts::TAU / 13.0;
            scene.try_line(
                center + Vec2::new(angle.cos() * 17.0, angle.sin() * 13.0),
                center + Vec2::new(next.cos() * 17.0, next.sin() * 13.0),
                1.25,
                tint,
            )?;
        }
        for point in 0..3 {
            let offset = Vec2::new((point as f32 - 1.0) * 9.0, 3.5);
            scene.try_circle(center + offset, 4.0, ShapeStyle::filled(tint))?;
            scene.try_rect(
                Rect::from_center_size(
                    center + offset + Vec2::new(0.0, -11.0),
                    Vec2::new(6.25, 3.75),
                ),
                0.0,
                ShapeStyle::filled(tint),
            )?;
        }
    }
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_sizes_keep_every_construction_and_exact_clip_count() {
        for count in [100, 500, 2_000] {
            let scenes: Vec<_> = (0..TILES)
                .map(|tile| tile_scene(count, tile, Vec2::new(1280.0, 800.0), 0.0).unwrap())
                .collect();
            assert_eq!(
                scenes.iter().map(Scene::command_count).sum::<usize>(),
                count * COMMANDS_PER_CONSTRUCTION
            );
            let clips: Vec<_> = scenes
                .iter()
                .map(|scene| scene.commands()[0].screen_clip())
                .collect();
            assert_eq!(clips.len(), TILES);
            assert!(clips.windows(2).all(|pair| pair[0] != pair[1]));
            for (scene, clip) in scenes.iter().zip(clips) {
                assert!(
                    scene
                        .commands()
                        .iter()
                        .all(|command| command.screen_clip() == clip)
                );
            }
            assert!(
                scenes
                    .iter()
                    .all(|s| s.statistics().rejected_commands() == 0)
            );
        }
    }

    #[test]
    fn drag_changes_only_final_construction_and_remains_in_final_tile() {
        let size = Vec2::new(1280.0, 800.0);
        assert_eq!(
            construction_center(498, 500, size, 0.0),
            construction_center(498, 500, size, 7.0)
        );
        assert_eq!(
            construction_center(499, 500, size, 7.0) - construction_center(499, 500, size, 0.0),
            Vec2::new(7.0, 0.0)
        );
        assert_eq!(construction_tile(499, 500), TILES - 1);
    }
}
