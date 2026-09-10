//! Explicit example font selection, not automatic fallback or paragraph layout.

use sim_engine::{
    Color, FontBudget, FontFace, FrameComposer, FramePassOptions, GlyphRunBudget,
    ImageBatchPlacement, ImageSampling, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    PhysicalPerLogical, ScreenClipRect, TextAtlas2d, TextAtlasBudget, TextDirection,
    TextLayoutBudget, TextRun2d, TextStyle, WgpuRenderer,
};

use super::Result;

pub const SCRIPT_SAMPLE: &str = "𝓣𝔂𝓹𝓮 𝓼𝓸𝓶𝓮𝓽𝓱𝓲𝓷𝓰 𝓽𝓸 𝓼𝓽𝓪𝓻𝓽";
pub const PAGE_COUNT: usize = 3;
pub const PRESENTS_PER_PAGE: usize = 40;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextPage {
    Fonts,
    #[default]
    Unicode,
    Motion,
}

impl TextPage {
    pub const fn index(self) -> usize {
        match self {
            Self::Fonts => 0,
            Self::Unicode => 1,
            Self::Motion => 2,
        }
    }
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fonts" | "1" => Ok(Self::Fonts),
            "unicode" | "2" => Ok(Self::Unicode),
            "motion" | "3" => Ok(Self::Motion),
            _ => Err("--page must be fonts, unicode or motion".into()),
        }
    }
    pub fn acceptance_page(drawn: usize) -> Self {
        match drawn / PRESENTS_PER_PAGE {
            0 => Self::Fonts,
            1 => Self::Unicode,
            _ => Self::Motion,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Face {
    Primary,
    Sans,
    Inter,
    Serif,
    Mono,
    Math,
    Japanese,
}

pub struct GalleryFonts {
    faces: [FontFace; 7],
}

impl GalleryFonts {
    pub fn load(primary: FontFace, custom: bool) -> Result<Self> {
        let load = |bytes: &[u8]| FontFace::from_bytes(bytes.to_vec(), FontBudget::default());
        let sans = if custom {
            load(include_bytes!("../assets/fonts/DejaVuSans.ttf"))?
        } else {
            primary.clone()
        };
        Ok(Self {
            faces: [
                primary,
                sans,
                load(include_bytes!("../assets/fonts/Inter-Regular.otf"))?,
                load(include_bytes!("../assets/fonts/DejaVuSerif.ttf"))?,
                load(include_bytes!("../assets/fonts/DejaVuSansMono.ttf"))?,
                load(include_bytes!("../assets/fonts/DejaVuMathTeXGyre.ttf"))?,
                load(include_bytes!(
                    "../assets/fonts/SimEngineJapaneseSubset.otf"
                ))?,
            ],
        })
    }
    fn face(&self, face: Face) -> FontFace {
        self.faces[match face {
            Face::Primary => 0,
            Face::Sans => 1,
            Face::Inter => 2,
            Face::Serif => 3,
            Face::Mono => 4,
            Face::Math => 5,
            Face::Japanese => 6,
        }]
        .clone()
    }
}

struct Sample {
    caption: &'static str,
    text: &'static str,
    face: Face,
    size: f32,
    direction: TextDirection,
    page: TextPage,
    baseline: f32,
}

fn samples() -> Vec<Sample> {
    let mut rows = Vec::new();
    for (index, (caption, face)) in [
        (
            "SELECTED FONT / DejaVu Sans by default; override with --font",
            Face::Primary,
        ),
        ("INTER / OpenType CFF", Face::Inter),
        ("DEJAVU SERIF / TrueType", Face::Serif),
        ("DEJAVU SANS MONO / monospaced", Face::Mono),
        ("DEJAVU MATH TEX GYRE / mathematical serif", Face::Math),
    ]
    .into_iter()
    .enumerate()
    {
        rows.push(Sample {
            caption,
            text: "Type something to start 0123456789",
            face,
            size: 32.0,
            direction: TextDirection::Ltr,
            page: TextPage::Fonts,
            baseline: 142.0 + index as f32 * 105.0,
        });
    }
    rows.push(Sample {
        caption: "CYRILLIC / DejaVu Sans",
        text: "Привет, мир! Текст без пиксельных букв.",
        face: Face::Sans,
        size: 32.0,
        direction: TextDirection::Ltr,
        page: TextPage::Fonts,
        baseline: 667.0,
    });
    for (caption, text, face, size, direction, baseline) in [
        (
            "YOUR EXACT UNICODE / mathematical script + fraktur",
            SCRIPT_SAMPLE,
            Face::Math,
            42.0,
            TextDirection::Ltr,
            142.0,
        ),
        (
            "GREEK + MATH SYMBOLS / DejaVu Math TeX Gyre",
            "∑ ∫ ∞ √ ∂ ∇ ≠ ≤ ≥ ± × π θ λ Ω",
            Face::Math,
            32.0,
            TextDirection::Ltr,
            242.0,
        ),
        (
            "HIRAGANA / example subset of Noto Sans CJK JP",
            "こんにちは、せかい！",
            Face::Japanese,
            32.0,
            TextDirection::Ltr,
            335.0,
        ),
        (
            "KATAKANA / example subset of Noto Sans CJK JP",
            "カタカナ・テキスト",
            Face::Japanese,
            32.0,
            TextDirection::Ltr,
            428.0,
        ),
        (
            "KANJI / example subset of Noto Sans CJK JP",
            "日本語 漢字 東京 世界",
            Face::Japanese,
            32.0,
            TextDirection::Ltr,
            521.0,
        ),
        (
            "COMBINING MARKS / precomposed and decomposed Unicode",
            "Café   Cafe\u{301}   Å   A\u{30a}   q\u{301}",
            Face::Sans,
            32.0,
            TextDirection::Ltr,
            614.0,
        ),
        (
            "ARABIC / explicit single right-to-left run, not paragraph bidi",
            "مرحبا بالعالم",
            Face::Sans,
            32.0,
            TextDirection::Rtl,
            707.0,
        ),
    ] {
        rows.push(Sample {
            caption,
            text,
            face,
            size,
            direction,
            page: TextPage::Unicode,
            baseline,
        });
    }
    rows
}

struct AtlasSlot {
    face: Face,
    size: f32,
    direction: TextDirection,
    atlas: TextAtlas2d,
}
struct Row {
    caption: TextRun2d,
    content: TextRun2d,
    slot: usize,
    page: TextPage,
    baseline: f32,
}

pub struct FontGallery {
    dpi: f32,
    captions: TextAtlas2d,
    heading: TextRun2d,
    instructions: TextRun2d,
    footer: TextRun2d,
    slots: Vec<AtlasSlot>,
    rows: Vec<Row>,
}

impl FontGallery {
    pub fn new(renderer: &WgpuRenderer, fonts: &GalleryFonts, dpi: f32) -> Result<Self> {
        // Scale fixed atlas capacity with the requested raster density; no per-frame
        // growth, eviction or font discovery. Typical DPI1/2 uses 1 MiB per atlas.
        let side = (256.0 * dpi).ceil().clamp(512.0, 2048.0) as u32;
        let atlas_budget = TextAtlasBudget::new(side, side, 256, GlyphRunBudget::default())?;
        let layout = TextLayoutBudget::default();
        let make_style = |size| -> Result<TextStyle> {
            Ok(TextStyle::new(
                LogicalPixels::new(size)?,
                PhysicalPerLogical::new(dpi)?,
            )?)
        };
        let mut captions = TextAtlas2d::new(
            renderer,
            fonts.face(Face::Sans),
            make_style(16.0)?,
            atlas_budget,
        )?;
        let heading = captions.prepare(renderer, "SIM;ENGINE / TYPE LAB", layout)?;
        let instructions = captions.prepare(
            renderer,
            "1 Fonts  |  2 Unicode + Japanese  |  3 Motion  |  Wheel Scroll  |  Esc Exit",
            layout,
        )?;
        let footer = captions.prepare(
            renderer,
            "Explicit fonts per row. Japanese asset is a demo subset; no automatic font fallback.",
            layout,
        )?;
        let mut slots: Vec<AtlasSlot> = Vec::new();
        let mut rows = Vec::new();
        for sample in samples() {
            let slot = match slots.iter().position(|slot| {
                slot.face == sample.face
                    && slot.size == sample.size
                    && slot.direction == sample.direction
            }) {
                Some(index) => index,
                None => {
                    let style = make_style(sample.size)?.with_direction(sample.direction);
                    slots.push(AtlasSlot {
                        face: sample.face,
                        size: sample.size,
                        direction: sample.direction,
                        atlas: TextAtlas2d::new(
                            renderer,
                            fonts.face(sample.face),
                            style,
                            atlas_budget,
                        )?,
                    });
                    slots.len() - 1
                }
            };
            let content = slots[slot]
                .atlas
                .prepare(renderer, sample.text, layout)
                .map_err(|error| format!("{}: {error}", sample.caption))?;
            rows.push(Row {
                caption: captions.prepare(renderer, sample.caption, layout)?,
                content,
                slot,
                page: sample.page,
                baseline: sample.baseline,
            });
        }
        Ok(Self {
            dpi,
            captions,
            heading,
            instructions,
            footer,
            slots,
            rows,
        })
    }

    pub fn dpi(&self) -> f32 {
        self.dpi
    }

    pub fn draw<'a>(
        &'a self,
        frame: &mut FrameComposer<'a>,
        page: TextPage,
        width: f32,
        scroll: f32,
    ) -> Result<()> {
        let draw = |frame: &mut FrameComposer<'a>,
                    atlas: &'a TextAtlas2d,
                    run: &'a TextRun2d,
                    x,
                    y,
                    color|
         -> Result<()> {
            frame.draw_glyph_run_placed(
                atlas.atlas(),
                run.glyph_run(),
                ImageBatchPlacement::new(LogicalScreenVector::new(x, y), color)?,
                ImageSampling::Linear,
                FramePassOptions::new(1),
            )?;
            Ok(())
        };
        draw(
            frame,
            &self.captions,
            &self.heading,
            32.0,
            35.0 - scroll,
            Color::rgb8(112, 213, 195),
        )?;
        draw(
            frame,
            &self.captions,
            &self.instructions,
            32.0,
            62.0 - scroll,
            Color::rgb8(190, 203, 220),
        )?;
        draw(
            frame,
            &self.captions,
            &self.footer,
            32.0,
            789.0 - scroll,
            Color::rgb8(155, 174, 193),
        )?;
        for row in self.rows.iter().filter(|row| row.page == page) {
            draw(
                frame,
                &self.captions,
                &row.caption,
                48.0,
                row.baseline - 48.0 - scroll,
                Color::rgb8(165, 182, 207),
            )?;
            let clip = ScreenClipRect::from_min_size(
                LogicalScreenPosition::new(40.0, row.baseline - 42.0 - scroll),
                LogicalScreenVector::new((width - 16.0).max(1.0), 64.0),
            )?;
            frame.draw_glyph_run_placed(
                self.slots[row.slot].atlas.atlas(),
                row.content.glyph_run(),
                ImageBatchPlacement::new(
                    LogicalScreenVector::new(48.0, row.baseline - scroll),
                    if row.baseline == 142.0 && page == TextPage::Unicode {
                        Color::rgb8(248, 200, 134)
                    } else {
                        Color::rgb8(222, 232, 244)
                    },
                )?,
                ImageSampling::Linear,
                FramePassOptions::new(2).with_clip(clip),
            )?;
        }
        Ok(())
    }

    pub fn retained_bytes(&self) -> (usize, usize) {
        let mut cpu = self.captions.recovery_memory_bytes();
        let mut gpu = self.captions.gpu_allocation_bytes();
        for slot in &self.slots {
            cpu += slot.atlas.recovery_memory_bytes();
            gpu += slot.atlas.gpu_allocation_bytes();
        }
        for run in [&self.heading, &self.instructions, &self.footer]
            .into_iter()
            .chain(
                self.rows
                    .iter()
                    .flat_map(|row| [&row.caption, &row.content]),
            )
        {
            cpu += run.recovery_memory_bytes();
            gpu += run.glyph_run().gpu_allocation_bytes();
        }
        (cpu, gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_preserves_exact_supplementary_unicode_and_page_contract() {
        assert_eq!(
            samples()
                .iter()
                .find(|row| row.caption.starts_with("YOUR EXACT"))
                .unwrap()
                .text,
            "𝓣𝔂𝓹𝓮 𝓼𝓸𝓶𝓮𝓽𝓱𝓲𝓷𝓰 𝓽𝓸 𝓼𝓽𝓪𝓻𝓽"
        );
        assert!(
            SCRIPT_SAMPLE
                .chars()
                .any(|character| character as u32 > 0xffff)
        );
        for page in [TextPage::Fonts, TextPage::Unicode] {
            assert!(samples().iter().filter(|row| row.page == page).count() >= 6);
        }
        assert_eq!(TextPage::acceptance_page(39), TextPage::Fonts);
        assert_eq!(TextPage::acceptance_page(40), TextPage::Unicode);
        assert_eq!(TextPage::acceptance_page(80), TextPage::Motion);
        assert!(TextPage::parse("unknown").is_err());
    }
}
