//! Explicit CPU enqueue diagnostic, not GPU execution or presentation throughput.

use super::*;
use crate::renderer::{
    GlyphAtlasBudget, GlyphAtlasEntry, GlyphId, GlyphRunBudget, ImageTexelRect, PositionedGlyph2d,
    glyph,
};
use std::{sync::Arc, time::Instant};

#[test]
#[ignore = "requires an idle host and --release; CPU construction/proof only"]
fn glyph_placement_construction_cpu_diagnostic() {
    assert!(!std::hint::black_box(cfg!(debug_assertions)));
    let sprite = ImageSprite2d::new(
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        LogicalViewportRegion::new(
            LogicalScreenPosition::new(7.0, -10.0),
            LogicalViewport::new(6.0, 12.0).unwrap(),
        )
        .unwrap(),
        Color::WHITE,
    )
    .unwrap();
    for proof in [false, true] {
        let started = Instant::now();
        for index in 0..240_000 {
            let placement = ImageBatchPlacement::new(
                LogicalScreenVector::new(
                    (index % 100) as f32 * 8.0 + 0.25,
                    (index / 100 % 60) as f32 * 12.0 + 20.0,
                ),
                Color::WHITE,
            )
            .unwrap();
            let transformed = if proof {
                placed_sprite(std::hint::black_box(sprite), placement)
            } else {
                translated_sprite(std::hint::black_box(sprite), placement)
            };
            std::hint::black_box(transformed.unwrap());
        }
        eprintln!(
            "glyph_construction_cpu portable_proof={proof} glyphs=24000 amortized_ms={:.6}",
            started.elapsed().as_secs_f64() * 100.0
        );
    }
}

#[test]
#[ignore = "requires an idle Vulkan-capable host and --release"]
fn glyph_batch_update_24000_glyphs_cpu_diagnostic() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "use --release for timing"
    );
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .expect("diagnostic requires a GPU adapter");
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("glyph batch update diagnostic"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();
        let identity = Arc::new(());
        let atlas = glyph::create_glyph_atlas_resources(
            &device,
            &queue,
            Arc::clone(&identity),
            1,
            1,
            vec![255; 4],
            vec![GlyphAtlasEntry::new(
                GlyphId::new(0),
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            )],
            GlyphAtlasBudget::default(),
        )
        .unwrap();
        let glyphs = (0..4)
            .map(|index| {
                PositionedGlyph2d::new(
                    GlyphId::new(0),
                    LogicalViewportRegion::new(
                        LogicalScreenPosition::new(index as f32 * 7.0, -10.0),
                        LogicalViewport::new(6.0, 12.0).unwrap(),
                    )
                    .unwrap(),
                    Color::WHITE,
                )
                .unwrap()
            })
            .collect();
        let run = glyph::create_glyph_run_resources(
            &device,
            &queue,
            Arc::clone(&identity),
            &atlas,
            glyphs,
            GlyphRunBudget::default(),
        )
        .unwrap();
        let placements = |shift| {
            (0..6000)
                .map(|index| {
                    GlyphRunPlacement2d::new(
                        &run,
                        ImageBatchPlacement::new(
                            LogicalScreenVector::new(
                                (index % 100) as f32 * 8.0 + shift,
                                (index / 100) as f32 * 12.0 + 20.0,
                            ),
                            Color::WHITE,
                        )
                        .unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = placements(0.0);
        let second = placements(0.25);
        let mut batch = create_resources(
            &device,
            &queue,
            identity,
            &atlas.image,
            &first,
            GlyphBatchBudget::default(),
        )
        .unwrap();
        for index in 0..6 {
            update_resources(
                &device,
                &queue,
                &atlas.image,
                &mut batch,
                if index % 2 == 0 { &second } else { &first },
            )
            .unwrap();
        }
        let mut times = Vec::new();
        let mut uploaded_bytes = 0;
        for index in 0..30 {
            let started = Instant::now();
            let report = update_resources(
                &device,
                &queue,
                &atlas.image,
                &mut batch,
                if index % 2 == 0 { &second } else { &first },
            )
            .unwrap();
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            assert!(!report.instances().replaced_instance_buffer());
            uploaded_bytes += report.instances().uploaded_instance_bytes();
        }
        times.sort_by(f64::total_cmp);
        assert_eq!(batch.glyph_count(), 24_000);
        assert_eq!(
            uploaded_bytes,
            30 * 24_000 * std::mem::size_of::<crate::renderer::image::ImageInstance>()
        );
        eprintln!(
            "glyph_update_cpu backend={:?} adapter={:?} driver={:?} pci={:?} placements=6000 glyphs=24000 samples=30 p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} upload_bytes_per_update={} retained_cpu_bytes={} gpu_buffer_bytes={}",
            info.backend,
            info.name,
            info.driver_info,
            info.device_pci_bus_id,
            times[14],
            times[28],
            times[29],
            uploaded_bytes / 30,
            batch.recovery_memory_bytes(),
            batch.gpu_allocation_bytes()
        );
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}
