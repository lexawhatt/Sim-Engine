//! Example-owned captions and retained raster resources, separate from widgets.

use sim_engine::{
    Color, FontFace, FrameComposer, FramePassOptions, ImageBatchPlacement, ImageSampling,
    LogicalPixels, LogicalScreenPosition, LogicalScreenVector, PhysicalPerLogical, ScreenClipRect,
    TextAtlas2d, TextAtlasBudget, TextLayoutBudget, TextRun2d, TextStyle, WgpuRenderer,
};

use super::Result;

const SMALL_SAMPLE: &str = "16 px | Latin: AVATAR, office. Cyrillic: Привет, мир!";
const MEDIUM_SAMPLE: &str = "24 px | Smooth Latin / Кириллица";
const LARGE_SAMPLE: &str = "48 px | Текст без пикселей";
const HEADING: &str = "Real TTF / OTF text - retained glyph atlases";
const MOTION_LABEL: &str = "Shared counter run / independent tint, translation and clip";
const INSTRUCTIONS: &str = "1 Fonts | 2 Unicode | 3 Motion | Space: pause | R: reset | Esc: exit";

pub struct TextContent {
    dpi: f32,
    small: TextAtlas2d,
    small_sample: TextRun2d,
    heading: TextRun2d,
    motion_label: TextRun2d,
    instructions: TextRun2d,
    medium: TextAtlas2d,
    medium_sample: TextRun2d,
    counter: TextRun2d,
    counter_value: u64,
    large: TextAtlas2d,
    large_sample: TextRun2d,
}

impl TextContent {
    pub fn new(renderer: &WgpuRenderer, font: FontFace, dpi: f32, counter: u64) -> Result<Self> {
        let make_atlas = |size| -> Result<TextAtlas2d> {
            let style = TextStyle::new(LogicalPixels::new(size)?, PhysicalPerLogical::new(dpi)?)?;
            Ok(TextAtlas2d::new(
                renderer,
                font.clone(),
                style,
                TextAtlasBudget::default(),
            )?)
        };
        let budget = TextLayoutBudget::default();
        let mut small = make_atlas(16.0)?;
        let small_sample = small.prepare(renderer, SMALL_SAMPLE, budget)?;
        let heading = small.prepare(renderer, HEADING, budget)?;
        let motion_label = small.prepare(renderer, MOTION_LABEL, budget)?;
        let instructions = small.prepare(renderer, INSTRUCTIONS, budget)?;
        let mut medium = make_atlas(24.0)?;
        let medium_sample = medium.prepare(renderer, MEDIUM_SAMPLE, budget)?;
        // Future counter digits are rasterized once, not during each UI tick.
        medium.prepare(renderer, "Счётчик / Counter: 0123456789", budget)?;
        let counter_run = medium.prepare(renderer, &counter_text(counter), budget)?;
        let mut large = make_atlas(48.0)?;
        let large_sample = large.prepare(renderer, LARGE_SAMPLE, budget)?;
        Ok(Self {
            dpi,
            small,
            small_sample,
            heading,
            motion_label,
            instructions,
            medium,
            medium_sample,
            counter: counter_run,
            counter_value: counter,
            large,
            large_sample,
        })
    }

    pub fn dpi(&self) -> f32 {
        self.dpi
    }

    pub fn update_counter(&mut self, renderer: &WgpuRenderer, value: u64) -> Result<bool> {
        if value == self.counter_value {
            return Ok(false);
        }
        self.medium.update(
            renderer,
            &mut self.counter,
            &counter_text(value),
            TextLayoutBudget::default(),
        )?;
        self.counter_value = value;
        Ok(true)
    }

    pub fn draw<'a>(
        &'a self,
        frame: &mut FrameComposer<'a>,
        seconds: f32,
        panel_width: f32,
    ) -> Result<()> {
        for (atlas, run, x, baseline, color) in [
            (&self.small, &self.heading, 32.0, 48.0, Color::WHITE),
            (
                &self.small,
                &self.small_sample,
                48.0,
                122.0,
                Color::rgb8(195, 218, 239),
            ),
            (
                &self.medium,
                &self.medium_sample,
                48.0,
                200.0,
                Color::rgb8(108, 203, 255),
            ),
            (&self.large, &self.large_sample, 48.0, 295.0, Color::WHITE),
            (
                &self.medium,
                &self.counter,
                48.0,
                389.0,
                Color::rgb8(255, 184, 75),
            ),
            (
                &self.small,
                &self.motion_label,
                32.0,
                440.0,
                Color::rgb8(150, 177, 191),
            ),
            (
                &self.small,
                &self.instructions,
                32.0,
                587.0,
                Color::rgb8(150, 177, 191),
            ),
        ] {
            frame.draw_glyph_run_placed(
                atlas.atlas(),
                run.glyph_run(),
                ImageBatchPlacement::new(LogicalScreenVector::new(x, baseline), color)?,
                ImageSampling::Linear,
                FramePassOptions::new(1),
            )?;
        }
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(32.0, 452.0),
            LogicalScreenVector::new(panel_width, 94.0),
        )?;
        let moving_x =
            32.0 + panel_width * (0.5 + seconds.sin() * 0.55) - self.counter.advance() * 0.5;
        for (x, baseline, tint) in [
            (48.0, 481.0, Color::rgb8(124, 216, 175).with_alpha(0.45)),
            (moving_x, 525.0, Color::rgb8(90, 240, 155)),
        ] {
            frame.draw_glyph_run_placed(
                self.medium.atlas(),
                self.counter.glyph_run(),
                ImageBatchPlacement::new(LogicalScreenVector::new(x, baseline), tint)?,
                ImageSampling::Linear,
                FramePassOptions::new(2).with_clip(clip),
            )?;
        }
        Ok(())
    }
}

fn counter_text(value: u64) -> String {
    format!("Счётчик / Counter: {value}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_keeps_real_cyrillic_and_full_integer_range() {
        assert_eq!(counter_text(9), "Счётчик / Counter: 9");
        assert_eq!(
            counter_text(u64::MAX),
            "Счётчик / Counter: 18446744073709551615"
        );
    }
}
