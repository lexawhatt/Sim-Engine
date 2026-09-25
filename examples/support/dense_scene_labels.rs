//! Already shaped synthetic atlas labels isolate composition from font loading.

use super::workload::{Result, TILES, construction_center, construction_tile, tile_clip};
use sim_engine::{
    Camera2d, Color, FrameComposer, FramePassOptions, GlyphAtlas2d, GlyphAtlasBudget,
    GlyphAtlasEntry, GlyphBatch2d, GlyphBatchBudget, GlyphId, GlyphRun2d, GlyphRunBudget,
    GlyphRunPlacement2d, ImageBatchPlacement, ImageSampling, ImageTexelRect, LogicalScreenPosition,
    LogicalScreenVector, LogicalViewport, LogicalViewportRegion, PositionedGlyph2d, ScreenClipRect,
    Vec2, WgpuRenderer,
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelMode {
    Off,
    Runs,
    Batch,
}

pub struct Labels {
    atlas: GlyphAtlas2d,
    runs: Vec<GlyphRun2d>,
    placements: Vec<ImageBatchPlacement>,
    clips: Vec<ScreenClipRect>,
    // Spatial chunks preserve painter order while a dragged construction updates
    // only its tile. All text remains above every geometry chunk.
    batches: Option<Vec<GlyphBatch2d>>,
    constructions: usize,
    logical_size: Vec2,
    pose: Option<(Camera2d, f32)>,
    update_timing: (Duration, Duration),
}

impl Labels {
    pub fn new(
        renderer: &WgpuRenderer,
        count: usize,
        logical_size: Vec2,
        mode: LabelMode,
    ) -> Result<Option<Self>> {
        if mode == LabelMode::Off {
            return Ok(None);
        }
        // Tiny fixed source is deliberate: this measures existing prepared text,
        // not shaping/rasterization quality or font-cache initialization.
        let masks: [[u8; 7]; 4] = [
            [0, 0, 22, 25, 16, 16, 16],
            [0, 0, 17, 10, 4, 10, 17],
            [4, 12, 4, 4, 4, 4, 14],
            [14, 17, 1, 2, 4, 8, 31],
        ];
        let mut pixels = vec![0; 32 * 8 * 4];
        for (glyph, rows) in masks.iter().enumerate() {
            for (y, bits) in rows.iter().enumerate() {
                for x in 0..5 {
                    if bits & (1 << (4 - x)) != 0 {
                        let offset = (y * 32 + glyph * 8 + x + 1) * 4;
                        pixels[offset..offset + 4].copy_from_slice(&[255; 4]);
                    }
                }
            }
        }
        let entries = (0..4)
            .map(|glyph| {
                Ok(GlyphAtlasEntry::new(
                    GlyphId::new(glyph),
                    ImageTexelRect::new(glyph * 8, 0, 8, 8)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let atlas =
            renderer.create_glyph_atlas(32, 8, pixels, entries, GlyphAtlasBudget::default())?;
        let mut runs = Vec::with_capacity(count * 3);
        let mut clips = Vec::with_capacity(count * 3);
        for index in 0..count {
            for label in 0..3 {
                let glyphs = (0..4)
                    .map(|glyph| {
                        Ok(PositionedGlyph2d::new(
                            GlyphId::new((glyph + label) % 4),
                            LogicalViewportRegion::new(
                                LogicalScreenPosition::new(glyph as f32 * 5.5, 0.0),
                                LogicalViewport::new(6.0, 8.0)?,
                            )?,
                            Color::WHITE,
                        )?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                runs.push(renderer.create_glyph_run(&atlas, glyphs, GlyphRunBudget::default())?);
                clips.push(tile_clip(construction_tile(index, count), logical_size)?);
            }
        }
        let placements = vec![ImageBatchPlacement::default(); runs.len()];
        let batches =
            if mode == LabelMode::Batch {
                Some(
                    (0..TILES)
                        .map(|_| {
                            Ok(renderer.create_glyph_batch(
                                &atlas,
                                &[],
                                GlyphBatchBudget::default(),
                            )?)
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
        let mut result = Self {
            atlas,
            runs,
            placements,
            clips,
            batches,
            constructions: count,
            logical_size,
            pose: None,
            update_timing: (Duration::ZERO, Duration::ZERO),
        };
        result.update(renderer, Camera2d::new(Vec2::ZERO, 1.0)?, 0.0)?;
        Ok(Some(result))
    }

    pub fn update(
        &mut self,
        renderer: &WgpuRenderer,
        camera: Camera2d,
        displacement: f32,
    ) -> Result<usize> {
        self.update_timing = (Duration::ZERO, Duration::ZERO);
        if self.pose == Some((camera, displacement)) {
            return Ok(0);
        }
        let viewport = LogicalViewport::new(self.logical_size.x(), self.logical_size.y())?;
        let camera_changed = self.pose.is_none_or(|(previous, _)| previous != camera);
        let mut uploaded = 0;
        for tile in 0..TILES {
            if !tile_needs_update(tile, camera_changed) {
                continue;
            }
            let range = label_range(tile, self.constructions);
            let projection_started = Instant::now();
            for index in range.clone() {
                let center = construction_center(
                    index / 3,
                    self.constructions,
                    self.logical_size,
                    displacement,
                );
                let anchor = center
                    + Vec2::new(
                        -9.0 + (index % 3) as f32 * 7.0,
                        -18.0 - (index % 3) as f32 * 7.0,
                    );
                let screen = camera.world_to_screen(anchor, viewport)?.to_vec2();
                self.placements[index] = ImageBatchPlacement::new(
                    LogicalScreenVector::from_vec2(screen),
                    Color::rgba(0.85, 0.93, 1.0, 0.78 + (index % 2) as f32 * 0.1),
                )?;
            }
            self.update_timing.0 += projection_started.elapsed();
            if let Some(batches) = &mut self.batches {
                let placements = range
                    .map(|index| {
                        GlyphRunPlacement2d::new(&self.runs[index], self.placements[index])
                            .with_clip(self.clips[index])
                    })
                    .collect::<Vec<_>>();
                let engine_started = Instant::now();
                uploaded += renderer
                    .update_glyph_batch(&self.atlas, &mut batches[tile], &placements)?
                    .instances()
                    .uploaded_instance_bytes();
                self.update_timing.1 += engine_started.elapsed();
            }
        }
        self.pose = Some((camera, displacement));
        Ok(uploaded)
    }

    pub fn update_timing(&self) -> (f64, f64) {
        (
            self.update_timing.0.as_secs_f64() * 1000.0,
            self.update_timing.1.as_secs_f64() * 1000.0,
        )
    }

    pub fn draw<'frame>(&'frame self, frame: &mut FrameComposer<'frame>) -> Result<()> {
        if let Some(batches) = &self.batches {
            for batch in batches {
                frame.draw_glyph_batch(
                    &self.atlas,
                    batch,
                    ImageSampling::Linear,
                    FramePassOptions::new(TILES as i32),
                )?;
            }
        } else {
            for ((run, placement), clip) in self.runs.iter().zip(&self.placements).zip(&self.clips)
            {
                frame.draw_glyph_run_placed(
                    &self.atlas,
                    run,
                    *placement,
                    ImageSampling::Linear,
                    FramePassOptions::new(TILES as i32).with_clip(*clip),
                )?;
            }
        }
        Ok(())
    }

    pub fn restore(&mut self, renderer: &WgpuRenderer) -> Result<()> {
        let atlas = renderer.restore_glyph_atlas(&self.atlas)?;
        let runs = self
            .runs
            .iter()
            .map(|run| Ok(renderer.restore_glyph_run(&atlas, run)?))
            .collect::<Result<Vec<_>>>()?;
        let batches = self
            .batches
            .as_ref()
            .map(|batches| {
                batches
                    .iter()
                    .map(|batch| Ok(renderer.restore_glyph_batch(&atlas, batch)?))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        self.atlas = atlas;
        self.runs = runs;
        self.batches = batches;
        Ok(())
    }

    /// Total fixture-owned resource memory, including inactive source runs kept
    /// for the batch's changing placements. FrameStatistics counts drawn sources.
    pub fn owned_bytes(&self) -> (usize, usize) {
        let cpu = self.atlas.recovery_memory_bytes()
            + self
                .runs
                .iter()
                .map(GlyphRun2d::recovery_memory_bytes)
                .sum::<usize>()
            + self.runs.capacity() * std::mem::size_of::<GlyphRun2d>()
            + self.batches.as_ref().map_or(0, |batches| {
                batches
                    .iter()
                    .map(GlyphBatch2d::recovery_memory_bytes)
                    .sum::<usize>()
                    + batches.capacity() * std::mem::size_of::<GlyphBatch2d>()
            })
            + self.placements.capacity() * std::mem::size_of::<ImageBatchPlacement>()
            + self.clips.capacity() * std::mem::size_of::<ScreenClipRect>();
        let gpu = self.atlas.gpu_allocation_bytes()
            + self
                .runs
                .iter()
                .map(GlyphRun2d::gpu_allocation_bytes)
                .sum::<usize>()
            + self.batches.as_ref().map_or(0, |batches| {
                batches.iter().map(GlyphBatch2d::gpu_allocation_bytes).sum()
            });
        (cpu, gpu)
    }
}

fn label_range(tile: usize, constructions: usize) -> std::ops::Range<usize> {
    (tile * constructions).div_ceil(TILES) * 3..((tile + 1) * constructions).div_ceil(TILES) * 3
}

fn tile_needs_update(tile: usize, camera_changed: bool) -> bool {
    camera_changed || tile == TILES - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_label_tiles_preserve_complete_source_order_and_local_drag() {
        for count in [100, 500, 2000] {
            let indices = (0..TILES)
                .flat_map(|tile| label_range(tile, count))
                .collect::<Vec<_>>();
            assert_eq!(indices, (0..count * 3).collect::<Vec<_>>());
            for tile in 0..TILES {
                assert!(
                    label_range(tile, count)
                        .all(|index| construction_tile(index / 3, count) == tile)
                );
                assert!(tile_needs_update(tile, true));
                assert_eq!(tile_needs_update(tile, false), tile == TILES - 1);
            }
        }
    }
}
