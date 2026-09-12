//! Diagnostic large-scene/dynamic-mesh baseline, not an FPS/release gate.
//! Identical prevalidated host mesh snapshots drive immutable/dynamic controls.
//! Optional bounded GPU timestamps are distinct from CPU enqueue/surface wait.

#[path = "support/mesh3d_benchmark_cases.rs"]
mod cases;
#[path = "support/mesh3d_benchmark_metrics.rs"]
mod metrics;
#[cfg(feature = "text")]
#[path = "support/mesh3d_benchmark_text.rs"]
mod text_labels;

use cases::{Case, Result, Workload};
use metrics::{Samples, percentile};
use sim_engine::{
    BlendMode, Color, FrameBudget, FramePassOptions, GpuTimingStatus, LogicalViewport,
    RenderStatus, RenderTarget3d, RendererPresentMode, SurfaceRasterization3d, WgpuRenderer,
    WgpuRendererOptions,
};
use std::{sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    monitor::MonitorHandle,
    window::{Window, WindowId},
};

#[global_allocator]
static ALLOCATOR: metrics::Allocator = metrics::Allocator;

const WARMUP: usize = 20;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn source_frame(warmup_drawn: usize, measured_drawn: usize) -> usize {
    if warmup_drawn < WARMUP {
        warmup_drawn
    } else {
        measured_drawn
    }
}

#[derive(Clone, Copy)]
struct Configuration {
    case: Case,
    objects: usize,
    side: usize,
    frames: usize,
    trials: usize,
    policy: SurfaceRasterization3d,
}

impl Configuration {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut result = Self {
            case: Case::Repeated,
            objects: 1024,
            side: 1,
            frames: 120,
            trials: 3,
            policy: SurfaceRasterization3d::Native,
        };
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let value = arguments.next().ok_or("each argument needs a value")?;
            match argument.as_str() {
                "--case" => result.case = Case::parse(&value)?,
                "--objects" => result.objects = value.parse()?,
                "--side" => result.side = value.parse()?,
                "--frames" => result.frames = value.parse()?,
                "--trials" => result.trials = value.parse()?,
                "--policy" => {
                    result.policy = match value.as_str() {
                        "native" => SurfaceRasterization3d::Native,
                        "strict" => SurfaceRasterization3d::StrictPortable,
                        _ => return Err("policy must be native or strict".into()),
                    }
                }
                _ => return Err(format!("unknown option: {argument}").into()),
            }
        }
        if !(1..=8192).contains(&result.objects)
            || !(1..=128).contains(&result.side)
            || !(1..=10000).contains(&result.frames)
            || !(1..=20).contains(&result.trials)
            || result.objects * result.side * result.side * 2 > 4_000_000
        {
            return Err("limits: objects 1..8192, side 1..128, frames 1..10000, trials 1..20, total triangles <= 4M".into());
        }
        if result.case.changes_mesh() && result.side < 2 {
            return Err("changing geometry requires --side >= 2 (suggested: 32)".into());
        }
        if result.case == Case::Growth
            && (result.side > 64
                || (result.objects + 3) * result.side * result.side * 2 > 4_000_000)
        {
            return Err(
                "growth doubles side; side <= 64 and grown scene <= 4M triangles required".into(),
            );
        }
        if result.case == Case::PreparedText && !cfg!(feature = "text") {
            return Err("prepared_text requires --features text".into());
        }
        Ok(result)
    }
}

fn main() -> Result<()> {
    if std::env::args().any(|value| value == "--help" || value == "-h") {
        println!(
            "mesh3d_scene_benchmark --case repeated|distinct|outside|host_hidden|crossing|immutable|dynamic|growth|textured|mask|blend|texture_update|prepared_text --policy native|strict --objects 1024 --side 1 --frames 120 --trials 3\nFor chunk updates: --case immutable (or dynamic/growth) --objects 64 --side 32. prepared_text requires --features text. GPU queries are opt-in here, not a universal FPS gate."
        );
        return Ok(());
    }
    let configuration = Configuration::parse(std::env::args().skip(1))?;
    let mut application = Application {
        configuration,
        state: None,
        failure: None,
        completed: false,
    };
    EventLoop::new()?.run_app(&mut application)?;
    if let Some(failure) = application.failure {
        return Err(failure.into());
    }
    if !application.completed {
        return Err("benchmark closed before all trials completed".into());
    }
    Ok(())
}

struct Application {
    configuration: Configuration,
    state: Option<State>,
    failure: Option<String>,
    completed: bool,
}

struct State {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    target: RenderTarget3d,
    workload: Workload,
    configuration: Configuration,
    samples: Samples,
    trial: usize,
    warmup_drawn: usize,
    warmup_total_drawn: usize,
    attempts: usize,
    skipped: usize,
    trial_started: Option<Instant>,
    output: Option<OutputContext>,
    warmup_gpu_allocations: usize,
    warmup_detachments: usize,
    initial_timing_status: GpuTimingStatus,
    #[cfg(feature = "text")]
    labels: Option<text_labels::Labels>,
}

#[derive(PartialEq)]
struct OutputContext {
    size: PhysicalSize<u32>,
    scale: f64,
    monitor: Option<MonitorHandle>,
    refresh_millihertz: Option<u32>,
}

impl OutputContext {
    fn read(window: &Window) -> Self {
        let monitor = window.current_monitor();
        let refresh_millihertz = monitor
            .as_ref()
            .and_then(MonitorHandle::refresh_rate_millihertz);
        Self {
            size: window.inner_size(),
            scale: window.scale_factor(),
            monitor,
            refresh_millihertz,
        }
    }
}

impl State {
    fn new(events: &ActiveEventLoop, configuration: Configuration) -> Result<Self> {
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("Sim;Engine 3D scene diagnostics")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )?,
        );
        let size = window.inner_size();
        let options =
            WgpuRendererOptions::new(RendererPresentMode::NoVsync, window.scale_factor())?
                .with_gpu_timing(true);
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        if std::env::var_os("SIM_ENGINE_RELEASE_SHA").is_some() {
            if !renderer.adapter_backend().eq_ignore_ascii_case("vulkan") {
                return Err(
                    "revision-bound diagnostics require the selected Vulkan backend".into(),
                );
            }
            let required = std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")?;
            if !renderer
                .adapter_pci_bus_id()
                .eq_ignore_ascii_case(&required)
            {
                return Err("revision-bound diagnostics physical GPU identity mismatch".into());
            }
        }
        let target = renderer.create_render_target3d(
            WIDTH,
            HEIGHT,
            LogicalViewport::new(WIDTH as f32, HEIGHT as f32)?,
        )?;
        let started = Instant::now();
        let workload = Workload::new(
            &renderer,
            configuration.case,
            configuration.objects,
            configuration.side,
            configuration.policy,
        )?;
        #[cfg(feature = "text")]
        let labels = if configuration.case == Case::PreparedText {
            Some(text_labels::Labels::new(&renderer)?)
        } else {
            None
        };
        println!(
            "case={} adapter={:?} backend={} pci={:?} driver={:?} driver_info={:?} surface_format={:?} surface_msaa={} target=1280x720 target_msaa=1 present_mode={:?} construction_ms={:.3} host_source_build=excluded_from_frame_time release_gate=false",
            configuration.case.name(),
            renderer.adapter_name(),
            renderer.adapter_backend(),
            renderer.adapter_pci_bus_id(),
            renderer.adapter_driver(),
            renderer.adapter_driver_info(),
            renderer.surface_format(),
            renderer.surface_sample_count(),
            renderer.surface_present_mode(),
            started.elapsed().as_secs_f64() * 1000.0
        );
        println!(
            "policy={:?} gpu_timing={:?} query_period_ns={:?} diagnostic_ring_capacity={} diagnostic_query_count={} diagnostic_gpu_buffer_bytes={} source_sha={} executable_sha256={} growth_fixture=fresh_small_then_grown_update seed=0 camera_path=orthographic_sine_v1",
            configuration.policy,
            renderer.gpu_timing_statistics().status(),
            renderer
                .gpu_timing_statistics()
                .timestamp_period_nanoseconds(),
            renderer.gpu_timing_statistics().capacity(),
            renderer.gpu_timing_statistics().query_count(),
            renderer.gpu_timing_statistics().gpu_buffer_bytes(),
            std::env::var("SIM_ENGINE_RELEASE_SHA")
                .unwrap_or_else(|_| "unverified-working-tree".into()),
            std::env::var("SIM_ENGINE_BENCHMARK_BINARY_SHA256")
                .unwrap_or_else(|_| "unverified".into())
        );
        let initial_timing_status = renderer.gpu_timing_statistics().status();
        Ok(Self {
            window,
            renderer,
            target,
            workload,
            configuration,
            samples: Samples::with_capacity(configuration.frames),
            trial: 0,
            warmup_drawn: 0,
            warmup_total_drawn: 0,
            attempts: 0,
            skipped: 0,
            trial_started: None,
            output: None,
            warmup_gpu_allocations: 0,
            warmup_detachments: 0,
            initial_timing_status,
            #[cfg(feature = "text")]
            labels,
        })
    }

    fn draw(&mut self) -> Result<bool> {
        self.samples
            .gpu
            .collect(self.renderer.collect_gpu_timings())?;
        let output = OutputContext::read(&self.window);
        if self
            .output
            .as_ref()
            .is_some_and(|previous| *previous != output)
        {
            if self.trial_started.is_some() || self.trial > 0 {
                return Err(
                    "output changed during measurement; rerun under stable configuration".into(),
                );
            }
            self.warmup_drawn = 0;
        }
        self.output = Some(output);
        self.attempts += 1;
        let measured = self.warmup_drawn >= WARMUP;
        if measured && self.trial_started.is_none() {
            let output = self.output.as_ref().ok_or("missing output context")?;
            println!(
                "trial={} physical_surface={}x{} scale_factor={} monitor={:?} reported_refresh_millihertz={:?} warmup_drawn={} changed_mesh_objects_per_frame={} measured_initial_revision={}",
                self.trial + 1,
                output.size.width,
                output.size.height,
                output.scale,
                output.monitor.as_ref().and_then(MonitorHandle::name),
                output.refresh_millihertz,
                self.warmup_drawn,
                usize::from(self.configuration.case.changes_mesh()),
                self.configuration.case.revision(0)
            );
        }
        let source_frame = source_frame(self.warmup_drawn, self.samples.work_ms.len());
        let started = Instant::now();
        if measured {
            self.trial_started.get_or_insert(started);
        }
        let (rendered, allocations, allocated_bytes) = metrics::count(|| -> Result<_> {
            let update_started = Instant::now();
            let update = self.workload.update(&mut self.renderer, source_frame)?;
            #[cfg(feature = "text")]
            let text_uploaded_bytes = match &mut self.labels {
                Some(labels) => labels.update(&self.renderer, source_frame)?,
                None => 0,
            };
            #[cfg(not(feature = "text"))]
            let text_uploaded_bytes = 0;
            let update_ms = update_started.elapsed().as_secs_f64() * 1000.0;
            let render_started = Instant::now();
            let report = self.renderer.render_scene3d_to_target_with_budget(
                &self.target,
                &self.workload.scene,
                self.workload.camera,
                self.workload.budget,
            )?;
            let render_ms = render_started.elapsed().as_secs_f64() * 1000.0;
            if report.object_count() != self.workload.expected_objects
                || (self.configuration.policy == SurfaceRasterization3d::Native
                    && report.triangle_count() != self.workload.expected_triangles)
                || report.triangle_count() != report.preflight().submitted_triangle_count()
            {
                return Err("submission counts differ from the fixture workload".into());
            }
            let mut frame = self
                .renderer
                .begin_frame(Color::BLACK, FrameBudget::default())?;
            frame.draw_render_target(
                self.target.color_target(),
                BlendMode::Replace,
                1.0,
                FramePassOptions::new(0),
            )?;
            #[cfg(feature = "text")]
            if let Some(labels) = &self.labels {
                labels.draw(&mut frame)?;
            }
            let composed = frame.present()?;
            Ok((
                update,
                update_ms,
                report,
                render_ms,
                composed,
                text_uploaded_bytes,
            ))
        });
        let (update, update_ms, report, render_ms, composed, text_uploaded_bytes) = rendered?;
        if !measured {
            self.warmup_gpu_allocations += update.gpu_allocations;
            self.warmup_detachments += usize::from(update.detached);
        }
        if composed.status() != RenderStatus::Drawn {
            self.skipped += 1;
            if measured || self.skipped > 120 {
                return Err(format!(
                    "present skipped: {:?}; no successful measurement inferred",
                    composed.status()
                )
                .into());
            }
            return Ok(false);
        }
        if !measured {
            self.warmup_drawn += 1;
            self.warmup_total_drawn += 1;
            if self.warmup_drawn == WARMUP {
                self.renderer.wait_for_gpu_idle()?;
            }
            return Ok(false);
        }
        let acquire_ms = composed.metrics().surface_acquire().as_secs_f64() * 1000.0;
        self.samples
            .gpu
            .expect(report.gpu_timing_id(), composed.gpu_timing_id());
        self.samples
            .work_ms
            .push((started.elapsed().as_secs_f64() * 1000.0 - acquire_ms).max(0.0));
        self.samples.update_ms.push(update_ms);
        self.samples.render_3d_ms.push(render_ms);
        self.samples.acquire_ms.push(acquire_ms);
        self.samples.allocations += allocations;
        self.samples.allocated_bytes += allocated_bytes;
        self.samples.uploaded_mesh_bytes += update.bytes;
        self.samples.mesh_upload_calls += update.calls;
        self.samples.gpu_buffer_allocations += update.gpu_allocations;
        self.samples.reused_updates += usize::from(update.reused);
        self.samples.alias_detachments += usize::from(update.detached);
        self.samples.scratch_reallocations += update.scratch_reallocations;
        self.samples
            .preflight_ms
            .push(report.preflight_duration().as_secs_f64() * 1000.0);
        self.samples
            .staging_upload_ms
            .push(report.staging_upload_duration().as_secs_f64() * 1000.0);
        self.samples
            .encode_submit_ms
            .push(report.encode_submit().as_secs_f64() * 1000.0);
        self.samples.generated_upload_bytes += report.preflight().generated_upload_bytes();
        self.samples.submitted_triangles += report.triangle_count();
        self.samples.generated_triangles += report.preflight().generated_triangle_count();
        self.samples.discarded_triangles += report
            .preflight()
            .discarded_source_triangle_count()
            .unwrap_or(0);
        self.samples.clipped_triangles += report
            .preflight()
            .clipped_source_triangle_count()
            .unwrap_or(0);
        self.samples.draw_calls += report.draw_call_count();
        self.samples.texture_uploaded_bytes += update.texture_uploaded_bytes;
        self.samples.texture_upload_calls += update.texture_upload_calls;
        self.samples.texture_copy_bytes += update.texture_copy_bytes;
        self.samples.texture_gpu_allocations += update.texture_gpu_allocations;
        self.samples.texture_staging_peak = self
            .samples
            .texture_staging_peak
            .max(update.texture_staging_bytes);
        self.samples.prepared_text_uploaded_bytes += text_uploaded_bytes;
        self.samples.frame_upload_bytes += report.uploaded_bytes();
        self.samples.frame_upload_calls += report.upload_calls();
        self.samples.frame_buffer_allocations += report.buffer_allocation_count();
        self.samples.composition_upload_bytes += composed.statistics().upload_bytes();
        self.samples.composition_draw_calls += composed.statistics().draw_calls();
        self.samples.frame_staging_peak = self
            .samples
            .frame_staging_peak
            .max(report.staging_capacity_bytes());
        self.samples.frame_cpu_peak = self
            .samples
            .frame_cpu_peak
            .max(report.peak_frame_cpu_bytes());
        self.samples.frame_gpu_peak = self
            .samples
            .frame_gpu_peak
            .max(report.peak_frame_gpu_bytes());
        self.samples.mesh_staging_peak = self
            .samples
            .mesh_staging_peak
            .max(update.peak_mesh_staging_bytes);
        self.samples.mesh_update_gpu_peak = self
            .samples
            .mesh_update_gpu_peak
            .max(update.peak_mesh_gpu_bytes);
        self.samples.mesh_update_recovery_peak = self
            .samples
            .mesh_update_recovery_peak
            .max(update.peak_mesh_recovery_bytes);
        self.samples.texture_update_gpu_peak = self
            .samples
            .texture_update_gpu_peak
            .max(update.peak_texture_gpu_bytes);
        self.samples.texture_update_recovery_peak = self
            .samples
            .texture_update_recovery_peak
            .max(update.peak_texture_recovery_bytes);
        if self.samples.work_ms.len() < self.configuration.frames {
            return Ok(false);
        }
        let drain_started = Instant::now();
        self.renderer.wait_for_gpu_idle()?;
        for _ in 0..8 {
            self.samples
                .gpu
                .collect(self.renderer.collect_gpu_timings())?;
            if self.renderer.gpu_timing_statistics().pending() == 0 {
                break;
            }
        }
        let drain_ms = drain_started.elapsed().as_secs_f64() * 1000.0;
        let elapsed = self
            .trial_started
            .ok_or("missing measured start")?
            .elapsed()
            .as_secs_f64();
        if self.output.as_ref() != Some(&OutputContext::read(&self.window)) {
            return Err("output changed before trial completion".into());
        }
        let stats = self.workload.scene.statistics();
        let frames = self.samples.work_ms.len();
        println!(
            "case={} trial={} measured_drawn={} warmup_and_measurement_attempts={} startup_skipped={} completed_batch_wall_fps={:.3} renderer_work_ms[p50={:.3},p95={:.3},p99={:.3}] update_ms[p50={:.3},p95={:.3}] render3d_cpu_ms[p50={:.3},p95={:.3}] acquire_ms[p50={:.3},p95={:.3},p99={:.3}] objects={} submitted_objects={} triangles={} mesh_draw_calls={} unique_meshes={} scene_cpu_bytes={} scene_gpu_bytes={} scene_bookkeeping_bytes={} host_snapshot_bytes_including_shared={} target_bytes={} render_thread_alloc_realloc_calls={} render_thread_requested_allocation_bytes={} mesh_uploaded_bytes={} mesh_upload_calls={} mesh_gpu_allocation_count={} reused_updates={} detachments={} last_updated_capacity_bytes={} scratch_bytes={} scratch_reallocations={} composition_draw_calls_last={} composition_draw_calls_total={} end_of_trial_gpu_drain_ms={:.3}",
            self.configuration.case.name(),
            self.trial + 1,
            frames,
            self.attempts,
            self.skipped,
            frames as f64 / elapsed,
            percentile(&self.samples.work_ms, 50),
            percentile(&self.samples.work_ms, 95),
            percentile(&self.samples.work_ms, 99),
            percentile(&self.samples.update_ms, 50),
            percentile(&self.samples.update_ms, 95),
            percentile(&self.samples.render_3d_ms, 50),
            percentile(&self.samples.render_3d_ms, 95),
            percentile(&self.samples.acquire_ms, 50),
            percentile(&self.samples.acquire_ms, 95),
            percentile(&self.samples.acquire_ms, 99),
            stats.object_count(),
            report.object_count(),
            report.triangle_count(),
            report.draw_call_count(),
            stats.mesh_count(),
            stats.mesh_cpu_bytes(),
            stats.mesh_gpu_bytes(),
            stats.storage_bytes(),
            self.workload.host_snapshot_bytes,
            self.target.allocation_bytes(),
            self.samples.allocations,
            self.samples.allocated_bytes,
            self.samples.uploaded_mesh_bytes,
            self.samples.mesh_upload_calls,
            self.samples.gpu_buffer_allocations,
            self.samples.reused_updates,
            self.samples.alias_detachments,
            update.capacity,
            update.scratch,
            self.samples.scratch_reallocations,
            composed.statistics().draw_calls(),
            self.samples.composition_draw_calls,
            drain_ms
        );
        let timings = self.renderer.gpu_timing_statistics();
        println!(
            "trial={} dynamic_update_staging_peak={} dynamic_update_gpu_peak={} dynamic_update_recovery_peak={} texture_update_gpu_peak={} texture_update_recovery_peak={} preparation_frame_gpu_peak={} peak_scope=operation_local_not_process_RSS",
            self.trial + 1,
            self.samples.mesh_staging_peak,
            self.samples.mesh_update_gpu_peak,
            self.samples.mesh_update_recovery_peak,
            self.samples.texture_update_gpu_peak,
            self.samples.texture_update_recovery_peak,
            self.samples.frame_gpu_peak
        );
        if timings.status() != self.initial_timing_status {
            return Err("GPU diagnostics became unavailable during the trial".into());
        }
        println!(
            "trial={} frame_uploaded_bytes_total={} frame_upload_calls_total={} frame_gpu_buffer_allocations_total={} composition_upload_bytes_total={} retained_frame_cpu_bytes={} retained_frame_gpu_bytes={} frame_staging_capacity_peak={} preparation_frame_cpu_peak={}",
            self.trial + 1,
            self.samples.frame_upload_bytes,
            self.samples.frame_upload_calls,
            self.samples.frame_buffer_allocations,
            self.samples.composition_upload_bytes,
            report.retained_frame_cpu_bytes(),
            report.retained_frame_gpu_bytes(),
            self.samples.frame_staging_peak,
            self.samples.frame_cpu_peak
        );
        println!(
            "trial={} preflight_cpu_ms[p50={:.3},p95={:.3},p99={:.3}] staging_upload_cpu_ms[p50={:.3},p95={:.3},p99={:.3}] encode_submit_cpu_ms[p50={:.3},p95={:.3},p99={:.3}] source_triangles_last={} submitted_triangles_total={} generated_triangles_total={} discarded_source_triangles_total={:?} clipped_source_triangles_total={:?} mesh_draw_calls_total={} generated_upload_bytes_total={} texture_cpu_bytes={} texture_gpu_bytes={} texture_uploaded_bytes_total={} texture_upload_calls_total={} texture_copied_gpu_bytes_total={} texture_gpu_allocations_total={} texture_staging_peak={} prepared_text_instance_upload_bytes_total={}",
            self.trial + 1,
            percentile(&self.samples.preflight_ms, 50),
            percentile(&self.samples.preflight_ms, 95),
            percentile(&self.samples.preflight_ms, 99),
            percentile(&self.samples.staging_upload_ms, 50),
            percentile(&self.samples.staging_upload_ms, 95),
            percentile(&self.samples.staging_upload_ms, 99),
            percentile(&self.samples.encode_submit_ms, 50),
            percentile(&self.samples.encode_submit_ms, 95),
            percentile(&self.samples.encode_submit_ms, 99),
            self.workload.source_triangles,
            self.samples.submitted_triangles,
            self.samples.generated_triangles,
            (self.configuration.policy == SurfaceRasterization3d::StrictPortable)
                .then_some(self.samples.discarded_triangles),
            (self.configuration.policy == SurfaceRasterization3d::StrictPortable)
                .then_some(self.samples.clipped_triangles),
            self.samples.draw_calls,
            self.samples.generated_upload_bytes,
            stats.texture_cpu_bytes(),
            stats.texture_gpu_bytes(),
            self.samples.texture_uploaded_bytes,
            self.samples.texture_upload_calls,
            self.samples.texture_copy_bytes,
            self.samples.texture_gpu_allocations,
            self.samples.texture_staging_peak,
            self.samples.prepared_text_uploaded_bytes
        );
        println!(
            "trial={} gpu_timing={:?} matched_scene_samples={} matched_composition_samples={} requested_but_uncollected={} gpu_scene_ms={} gpu_composition_ms={} diagnostic_pending={} diagnostic_dropped_cumulative={} diagnostic_map_failures={} diagnostic_invalid_samples={} diagnostic_poll_failures={} diagnostic_collect_cpu_ms_cumulative={:.3} diagnostic_readback_bytes_cumulative={}",
            self.trial + 1,
            timings.status(),
            self.samples.gpu.scene_ms.len(),
            self.samples.gpu.composition_ms.len(),
            self.samples.gpu.missing(),
            metrics::gpu_distribution(&self.samples.gpu.scene_ms),
            metrics::gpu_distribution(&self.samples.gpu.composition_ms),
            timings.pending(),
            timings.dropped(),
            timings.map_failures(),
            timings.invalid_samples(),
            timings.poll_failures(),
            timings.collection_cpu().as_secs_f64() * 1000.0,
            timings.readback_bytes()
        );
        #[cfg(feature = "text")]
        if let Some(labels) = &self.labels {
            println!(
                "prepared_text_cpu_bytes={} prepared_text_gpu_bytes={} measured_shaper_calls=0",
                labels.cpu_bytes(),
                labels.gpu_bytes()
            );
        }
        if timings.status() == GpuTimingStatus::Enabled
            && (self.samples.gpu.scene_ms.len() != frames
                || self.samples.gpu.composition_ms.len() != frames)
        {
            return Err("incomplete GPU timing coverage; partial percentiles are diagnostic only, not a qualified baseline".into());
        }
        self.trial += 1;
        println!(
            "trial={} warmup_gpu_mesh_allocations={} warmup_alias_detachments={} measured_final_revision={} warmup_total_drawn={}",
            self.trial,
            self.warmup_gpu_allocations,
            self.warmup_detachments,
            self.configuration.case.revision(frames - 1),
            self.warmup_total_drawn
        );
        if self.trial == self.configuration.trials {
            return Ok(true);
        }
        self.samples = Samples::with_capacity(self.configuration.frames);
        self.warmup_drawn = 0;
        self.warmup_total_drawn = 0;
        self.attempts = 0;
        self.skipped = 0;
        self.trial_started = None;
        self.warmup_gpu_allocations = 0;
        self.warmup_detachments = 0;
        Ok(false)
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(events, self.configuration) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
                events.set_control_flow(ControlFlow::Poll);
            }
            Err(error) => {
                self.failure = Some(error.to_string());
                events.exit();
            }
        }
    }

    fn window_event(&mut self, events: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut().filter(|state| state.window.id() == id) else {
            return;
        };
        let result: Result<bool> = match event {
            WindowEvent::CloseRequested => {
                events.exit();
                return;
            }
            WindowEvent::Resized(size) => {
                if state.trial_started.is_some() || state.trial > 0 {
                    Err("resize interrupted measurement; rerun under stable configuration".into())
                } else {
                    state.warmup_drawn = 0;
                    state.window.request_redraw();
                    state
                        .renderer
                        .resize(size.width, size.height)
                        .map(|_| false)
                        .map_err(Into::into)
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if state.trial_started.is_some() || state.trial > 0 {
                    Err(
                        "scale change interrupted measurement; rerun under stable configuration"
                            .into(),
                    )
                } else {
                    state.warmup_drawn = 0;
                    (|| -> Result<bool> {
                        state.renderer.set_scale_factor(scale_factor)?;
                        #[cfg(feature = "text")]
                        if state.labels.is_some() {
                            state.labels = Some(text_labels::Labels::new(&state.renderer)?);
                        }
                        Ok(false)
                    })()
                }
            }
            WindowEvent::RedrawRequested => state.draw(),
            _ => return,
        };
        match result {
            Ok(true) => {
                self.completed = true;
                events.exit();
            }
            Ok(false) => state.window.request_redraw(),
            Err(error) => {
                self.failure = Some(error.to_string());
                events.exit();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_unbounded_or_empty_work() {
        for arguments in [
            vec!["--objects", "0"],
            vec!["--frames", "0"],
            vec!["--trials", "0"],
            vec!["--side", "1000000"],
            vec!["--objects", "8192", "--side", "128"],
            vec!["--case", "unknown"],
            vec!["--case", "dynamic", "--side", "1"],
            vec!["--case", "growth", "--side", "128"],
            vec!["--policy", "guess"],
        ] {
            assert!(Configuration::parse(arguments.into_iter().map(String::from)).is_err());
        }
    }

    #[test]
    fn diagnostics_cover_both_policies_and_all_new_workloads() {
        for case in [
            "repeated",
            "distinct",
            "outside",
            "host_hidden",
            "crossing",
            "immutable",
            "dynamic",
            "growth",
            "textured",
            "mask",
            "blend",
            "texture_update",
        ] {
            for policy in ["native", "strict"] {
                assert!(
                    Configuration::parse(
                        [
                            "--case",
                            case,
                            "--policy",
                            policy,
                            "--objects",
                            "4",
                            "--side",
                            "2"
                        ]
                        .map(String::from)
                    )
                    .is_ok()
                );
            }
        }
    }

    #[test]
    fn odd_trials_and_skipped_warmup_do_not_change_initial_text_update() {
        let mut line = 0;
        for trial_length in [1, 3, 60, 121] {
            for prior_skips in [0, 1, 20] {
                let mut drawn = 0;
                for attempt in 0..WARMUP + prior_skips {
                    line = source_frame(drawn, 0) % 2;
                    if attempt >= prior_skips {
                        drawn += 1;
                    }
                }
                assert_eq!(drawn, WARMUP);
                assert_eq!(line, 1);
                for measured in 0..trial_length {
                    let next = source_frame(drawn, measured) % 2;
                    assert_ne!(
                        line, next,
                        "trial {trial_length}, skips {prior_skips}, measured {measured}"
                    );
                    line = next;
                }
            }
        }
    }
}
