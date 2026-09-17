//! Opt-in complete-update CPU measurements with a frozen full-chain byte oracle.
//!
//! Setup, polling, alias lifetime management and full CPU/GPU verification are
//! outside update timing and allocation counting. Durations describe CPU work
//! on the calling thread, not GPU completion. Use this identical fixture on baseline
//! and candidate and alternate executable order in the external trial runner.

use super::*;
use std::hint::black_box;

use full_mip_reference::Level;

const WARMUPS: usize = 2;
const SAMPLES: usize = 5;

#[derive(Clone, Copy)]
enum Patch {
    Center(u32),
    Full,
    BottomRight(u32, u32),
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    alpha: bool,
    patch: Patch,
}

impl Case {
    fn region(self) -> ImageTexelRect {
        let (x, y, width, height) = match self.patch {
            Patch::Center(side) => (
                (self.width - side) / 2,
                (self.height - side) / 2,
                side,
                side,
            ),
            Patch::Full => (0, 0, self.width, self.height),
            Patch::BottomRight(width, height) => {
                (self.width - width, self.height - height, width, height)
            }
        };
        ImageTexelRect::new(x, y, width, height).unwrap()
    }
}

const CASES: [Case; 12] = [
    Case {
        name: "opaque_512_center32",
        width: 512,
        height: 512,
        alpha: false,
        patch: Patch::Center(32),
    },
    Case {
        name: "opaque_512_full",
        width: 512,
        height: 512,
        alpha: false,
        patch: Patch::Full,
    },
    Case {
        name: "opaque_1024_center32",
        width: 1024,
        height: 1024,
        alpha: false,
        patch: Patch::Center(32),
    },
    Case {
        name: "opaque_1024_full",
        width: 1024,
        height: 1024,
        alpha: false,
        patch: Patch::Full,
    },
    Case {
        name: "opaque_2048_center32",
        width: 2048,
        height: 2048,
        alpha: false,
        patch: Patch::Center(32),
    },
    Case {
        name: "opaque_2048_full",
        width: 2048,
        height: 2048,
        alpha: false,
        patch: Patch::Full,
    },
    Case {
        name: "opaque_1024_corner1",
        width: 1024,
        height: 1024,
        alpha: false,
        patch: Patch::BottomRight(1, 1),
    },
    Case {
        name: "alpha_1024_center32",
        width: 1024,
        height: 1024,
        alpha: true,
        patch: Patch::Center(32),
    },
    Case {
        name: "alpha_1024_full",
        width: 1024,
        height: 1024,
        alpha: true,
        patch: Patch::Full,
    },
    Case {
        name: "alpha_511x257_corner8",
        width: 511,
        height: 257,
        alpha: true,
        patch: Patch::BottomRight(8, 8),
    },
    Case {
        name: "opaque_1023x769_center32",
        width: 1023,
        height: 769,
        alpha: false,
        patch: Patch::Center(32),
    },
    Case {
        name: "alpha_1x1023_last1x8",
        width: 1,
        height: 1023,
        alpha: true,
        patch: Patch::BottomRight(1, 8),
    },
];

fn pixels(width: u32, height: u32, alpha: bool, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let mixed = x.wrapping_mul(0x045d_9f3b)
                ^ y.wrapping_mul(0x119d_e1f3)
                ^ seed.wrapping_mul(0x9e37_79b9);
            let mixed = mixed ^ (mixed >> 16);
            let opacity = if alpha {
                [0, 1, 17, 127, 254, 255][((x + 3 * y + seed) % 6) as usize]
            } else {
                255
            };
            pixels.extend_from_slice(&[
                mixed as u8,
                (mixed >> 8) as u8,
                (mixed >> 16) as u8,
                opacity,
            ]);
        }
    }
    pixels
}

fn patched_reference(previous: &[Level], region: ImageTexelRect, patch: &[u8]) -> Vec<Level> {
    let mut base = previous[0].pixels.clone();
    let row_bytes = region.width() as usize * 4;
    for row in 0..region.height() as usize {
        let destination =
            ((region.y() as usize + row) * previous[0].width as usize + region.x() as usize) * 4;
        base[destination..destination + row_bytes]
            .copy_from_slice(&patch[row * row_bytes..(row + 1) * row_bytes]);
    }
    full_mip_reference::rebuild(previous[0].width, previous[0].height, base)
}

fn assert_bytes(label: &str, level: usize, row: usize, actual: &[u8], expected: &[u8]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: mip {level}, row {row} length"
    );
    if let Some(offset) = actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        panic!(
            "{label}: mip {level}, row {row}, byte {offset}: {} != {}",
            actual[offset], expected[offset]
        );
    }
}

fn assert_chain(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &Texture3d,
    expected: &[Level],
    label: &str,
) {
    assert_eq!(
        texture.mip_level_count() as usize,
        expected.len(),
        "{label}"
    );
    assert_eq!(
        texture.size(),
        (expected[0].width, expected[0].height),
        "{label}"
    );
    assert_eq!(
        texture.gpu_allocation_bytes(),
        expected
            .iter()
            .map(|level| level.pixels.len())
            .sum::<usize>(),
        "{label}"
    );
    let mut layouts = Vec::with_capacity(expected.len());
    let mut size = 0_u64;
    for (index, level) in expected.iter().enumerate() {
        assert_eq!(
            texture.mip_level_size(index as u32),
            Some((level.width, level.height)),
            "{label}"
        );
        assert_bytes(
            label,
            index,
            0,
            texture.mip_level_pixels(index as u32).unwrap(),
            &level.pixels,
        );
        let stride = (level.width * 4).div_ceil(256) * 256;
        layouts.push((size, stride));
        size += u64::from(stride) * u64::from(level.height);
    }
    assert!(size <= device.limits().max_buffer_size);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture update benchmark complete-chain verification"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (index, (level, &(offset, stride))) in expected.iter().zip(&layouts).enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.storage.texture,
                mip_level: index as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(level.height),
                },
            },
            wgpu::Extent3d {
                width: level.width,
                height: level.height,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    for (index, (level, &(offset, stride))) in expected.iter().zip(&layouts).enumerate() {
        let row_bytes = level.width as usize * 4;
        for row in 0..level.height as usize {
            let actual_start = offset as usize + row * stride as usize;
            let expected_start = row * row_bytes;
            assert_bytes(
                label,
                index,
                row,
                &mapped[actual_start..actual_start + row_bytes],
                &level.pixels[expected_start..expected_start + row_bytes],
            );
        }
    }
    drop(mapped);
    buffer.unmap();
}

fn assert_accounting(
    case: Case,
    aliased: bool,
    texture: &Texture3d,
    report: Texture3dUpdateReport,
    old_recovery: usize,
    expected: &[Level],
) {
    let region = case.region();
    let base_bytes = region.width() as usize * region.height() as usize * 4;
    let gpu_bytes = expected
        .iter()
        .map(|level| level.pixels.len())
        .sum::<usize>();
    let lower_bytes = gpu_bytes - expected[0].pixels.len();
    let copied_bytes = if aliased { gpu_bytes } else { 0 };
    assert_eq!(report.base_upload_bytes(), base_bytes);
    assert_eq!(
        report.uploaded_bytes(),
        base_bytes + report.mip_upload_bytes()
    );
    assert_eq!(
        report.regenerated_texel_count() * 4,
        report.mip_upload_bytes()
    );
    assert_eq!(report.upload_calls(), expected.len());
    assert!(report.mip_upload_bytes() <= lower_bytes);
    assert!(report.mip_upload_bytes() >= (expected.len() - 1) * 4);
    if matches!(case.patch, Patch::Full) {
        assert_eq!(report.mip_upload_bytes(), lower_bytes);
    }
    assert_eq!(report.gpu_copy_bytes(), copied_bytes);
    assert_eq!(
        report.gpu_copy_calls(),
        if aliased { expected.len() } else { 0 }
    );
    assert_eq!(report.gpu_allocation_count(), usize::from(aliased));
    assert_eq!(report.submission_count(), 1 + usize::from(aliased));
    assert_eq!(report.detached_aliases(), aliased);
    assert_eq!(report.reused_allocation(), !aliased);
    assert_eq!(report.gpu_bytes(), gpu_bytes);
    assert_eq!(report.recovery_bytes(), texture.recovery_memory_bytes());
    assert!(report.recovery_bytes() >= gpu_bytes);
    assert_eq!(
        report.peak_recovery_bytes(),
        old_recovery + report.recovery_bytes()
    );
    assert_eq!(report.peak_gpu_bytes(), gpu_bytes + copied_bytes);
    assert!(report.staging_bytes() >= base_bytes + report.uploaded_bytes());
    assert!(report.staging_bytes() <= Texture3dUpdateBudget::default().max_staging_bytes());
}

fn print_sample(
    case: Case,
    aliased: bool,
    iteration: usize,
    wall: Duration,
    allocations: usize,
    report: Texture3dUpdateReport,
) {
    let (phase, sample) = if iteration < WARMUPS {
        ("warmup", iteration)
    } else {
        ("sample", iteration - WARMUPS)
    };
    eprintln!(
        "texture_update_bench case={} ownership={} phase={} sample={} patch_variant={} wall_ns={} preparation_ns={} encode_submit_ns={} allocation_calls={} base_upload_bytes={} mip_upload_bytes={} uploaded_bytes={} upload_calls={} regenerated_texels={} gpu_copy_bytes={} gpu_copy_calls={} gpu_allocations={} submissions={} recovery_bytes={} gpu_bytes={} staging_bytes={} peak_recovery_bytes={} peak_gpu_bytes={}",
        case.name,
        if aliased { "aliased" } else { "unique" },
        phase,
        sample,
        iteration % 2,
        wall.as_nanos(),
        report.preparation_cpu().as_nanos(),
        report.encode_submit_cpu().as_nanos(),
        allocations,
        report.base_upload_bytes(),
        report.mip_upload_bytes(),
        report.uploaded_bytes(),
        report.upload_calls(),
        report.regenerated_texel_count(),
        report.gpu_copy_bytes(),
        report.gpu_copy_calls(),
        report.gpu_allocation_count(),
        report.submission_count(),
        report.recovery_bytes(),
        report.gpu_bytes(),
        report.staging_bytes(),
        report.peak_recovery_bytes(),
        report.peak_gpu_bytes(),
    );
}

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    case: Case,
    aliased: bool,
) {
    let base = pixels(case.width, case.height, case.alpha, 0);
    let mut expected = full_mip_reference::rebuild(case.width, case.height, base.clone());
    let mut texture = create_texture_with_options(
        device,
        queue,
        identity,
        layout,
        case.width,
        case.height,
        base,
        ImageBudget::default(),
        Texture3dOptions::new()
            .with_alpha_preservation(case.alpha)
            .with_mipmaps(TextureMipmaps3d::Generate),
    )
    .unwrap();
    assert_chain(device, queue, &texture, &expected, case.name);
    let region = case.region();
    let patches = [37, 113].map(|seed| pixels(region.width(), region.height(), case.alpha, seed));
    for iteration in 0..WARMUPS + SAMPLES {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(Arc::strong_count(&texture.storage), 1);
        let old_identity = texture.identity_key();
        let old_recovery = texture.recovery_memory_bytes();
        let patch = &patches[iteration % 2];
        let snapshot = aliased.then(|| texture.clone());
        assert_eq!(
            Arc::strong_count(&texture.storage),
            1 + usize::from(aliased)
        );
        let started = Instant::now();
        let (result, allocations) = crate::test_allocations::count(|| {
            update_texture_region(
                device,
                queue,
                identity,
                layout,
                black_box(&mut texture),
                black_box(region),
                black_box(patch),
                region.width() as usize * 4,
                Texture3dUpdateBudget::default(),
            )
        });
        let wall = started.elapsed();
        let report = black_box(result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let next = patched_reference(&expected, region, patch);
        assert_chain(device, queue, &texture, &next, case.name);
        assert_accounting(case, aliased, &texture, report, old_recovery, &next);
        if let Some(snapshot) = snapshot.as_ref() {
            assert_eq!(snapshot.identity_key(), old_identity);
            assert_ne!(texture.identity_key(), old_identity);
            assert_eq!(snapshot.recovery_memory_bytes(), old_recovery);
            assert_chain(device, queue, snapshot, &expected, "old alias");
        } else {
            assert_eq!(texture.identity_key(), old_identity);
        }
        drop(snapshot);
        assert_eq!(Arc::strong_count(&texture.storage), 1);
        expected = next;
        print_sample(case, aliased, iteration, wall, allocations, report);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

pub(in crate::renderer::mesh3d) fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    if std::env::var("SIM_ENGINE_BENCH_TEXTURE_UPDATES").as_deref() != Ok("1") {
        return;
    }
    if cfg!(debug_assertions) {
        panic!("texture update measurements require --release");
    }
    assert!(
        device.limits().max_texture_dimension_2d >= 2048,
        "texture update fixture requires its complete 2048-texel case matrix"
    );
    eprintln!(
        "texture_update_bench metadata package_version={} cases={} ownership_modes=2 warmups={} samples={} durations=cpu_calling_thread allocations=calling_thread_only gpu_timing=none oracle=frozen_dev6_full_filter",
        env!("CARGO_PKG_VERSION"),
        CASES.len(),
        WARMUPS,
        SAMPLES
    );
    eprintln!(
        "texture_update_bench provenance runtime={} fixture_sha256={} reference_sha256={} executable_sha256={}",
        std::env::var("SIM_ENGINE_RELEASE_SHA")
            .unwrap_or_else(|_| "unverified-working-tree".into()),
        std::env::var("SIM_ENGINE_TEXTURE_FIXTURE_SHA").unwrap_or_else(|_| "unverified".into()),
        std::env::var("SIM_ENGINE_TEXTURE_REFERENCE_SHA").unwrap_or_else(|_| "unverified".into()),
        std::env::var("SIM_ENGINE_BENCHMARK_BINARY_SHA256").unwrap_or_else(|_| "unverified".into())
    );
    let identity = Arc::new(());
    let renderer = Mesh3dRenderer::new(device, format);
    for case in CASES {
        for aliased in [false, true] {
            run_case(
                device,
                queue,
                &identity,
                &renderer.textures.layout,
                case,
                aliased,
            );
        }
    }
}
