//! Diagnostic large-scene/dynamic-mesh baseline, not an FPS/release gate.
//! Identical prevalidated host mesh snapshots drive immutable/dynamic controls.
//! GPU timestamps are not implemented; CPU enqueue and surface wait are separate.

#[path = "support/mesh3d_benchmark_cases.rs"]
mod cases;
#[path = "support/mesh3d_benchmark_metrics.rs"]
mod metrics;

use cases::{Case, Result, Workload};
use metrics::{Samples, percentile};
use sim_engine::{
    BlendMode, Color, LogicalViewport, RenderStatus, RenderTarget3d, RendererPresentMode,
    WgpuRenderer, WgpuRendererOptions,
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

#[derive(Clone, Copy)]
struct Configuration {
    case: Case,
    objects: usize,
    side: usize,
    frames: usize,
    trials: usize,
}

impl Configuration {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut result = Self {
            case: Case::Repeated,
            objects: 1024,
            side: 1,
            frames: 120,
            trials: 3,
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
        Ok(result)
    }
}

fn main() -> Result<()> {
    if std::env::args().any(|value| value == "--help" || value == "-h") {
        println!(
            "mesh3d_scene_benchmark --case repeated|outside|host_hidden|immutable|dynamic --objects 1024 --side 1 --frames 120 --trials 3\nFor chunk updates: --case immutable (or dynamic) --objects 64 --side 32. No GPU-time or universal FPS claim."
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
    sequence: usize,
    trial_started: Option<Instant>,
    output: Option<OutputContext>,
    warmup_gpu_allocations: usize,
    warmup_detachments: usize,
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
            WgpuRendererOptions::new(RendererPresentMode::NoVsync, window.scale_factor())?;
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
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
        )?;
        println!(
            "case={} adapter={:?} backend={} pci={:?} driver={:?} driver_info={:?} surface_format={:?} surface_msaa={} target=1280x720 target_msaa=1 present_mode={:?} construction_ms={:.3} gpu_execution_ms=unavailable host_source_build=excluded_from_frame_time release_gate=false",
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
            sequence: 0,
            trial_started: None,
            output: None,
            warmup_gpu_allocations: 0,
            warmup_detachments: 0,
        })
    }

    fn draw(&mut self) -> Result<bool> {
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
                "trial={} physical_surface={}x{} scale_factor={} monitor={:?} reported_refresh_millihertz={:?} warmup_drawn={} changed_objects_per_frame={} measured_initial_revision=0",
                self.trial + 1,
                output.size.width,
                output.size.height,
                output.scale,
                output.monitor.as_ref().and_then(MonitorHandle::name),
                output.refresh_millihertz,
                self.warmup_drawn,
                usize::from(self.configuration.case.changes_mesh())
            );
        }
        let source_frame = if measured {
            self.samples.work_ms.len()
        } else {
            self.sequence
        };
        let started = Instant::now();
        if measured {
            self.trial_started.get_or_insert(started);
        }
        let (rendered, allocations, allocated_bytes) = metrics::count(|| -> Result<_> {
            let update_started = Instant::now();
            let update = self.workload.update(&mut self.renderer, source_frame)?;
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
                || report.triangle_count() != self.workload.expected_triangles
            {
                return Err("submission counts differ from the fixture workload".into());
            }
            let composed = self.renderer.compose_render_target(
                self.target.color_target(),
                BlendMode::Replace,
                1.0,
                Color::BLACK,
            )?;
            Ok((update, update_ms, report, render_ms, composed))
        });
        let (update, update_ms, report, render_ms, composed) = rendered?;
        self.sequence += 1;
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
        if self.samples.work_ms.len() < self.configuration.frames {
            return Ok(false);
        }
        let drain_started = Instant::now();
        self.renderer.wait_for_gpu_idle()?;
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
            "case={} trial={} measured_drawn={} warmup_and_measurement_attempts={} startup_skipped={} completed_batch_wall_fps={:.3} renderer_work_ms[p50={:.3},p95={:.3},p99={:.3}] update_ms[p50={:.3},p95={:.3}] render3d_cpu_ms[p50={:.3},p95={:.3}] acquire_ms[p50={:.3},p95={:.3},p99={:.3}] objects={} submitted_objects={} triangles={} mesh_draw_calls={} unique_meshes={} scene_cpu_bytes={} scene_gpu_bytes={} scene_bookkeeping_bytes={} host_snapshot_bytes_including_shared={} target_bytes={} render_thread_alloc_realloc_calls={} render_thread_requested_allocation_bytes={} mesh_uploaded_bytes={} mesh_upload_calls={} mesh_gpu_allocation_count={} reused_updates={} detachments={} last_updated_capacity_bytes={} scratch_bytes={} scratch_reallocations={} composition_draw_calls=1 end_of_trial_gpu_drain_ms={:.3}",
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
            drain_ms
        );
        self.trial += 1;
        println!(
            "trial={} warmup_gpu_mesh_allocations={} warmup_alias_detachments={} measured_final_revision={} warmup_total_drawn={}",
            self.trial,
            self.warmup_gpu_allocations,
            self.warmup_detachments,
            if self.configuration.case.changes_mesh() {
                (frames - 1) % 2
            } else {
                0
            },
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
                    state
                        .renderer
                        .set_scale_factor(scale_factor)
                        .map(|_| false)
                        .map_err(Into::into)
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
        ] {
            assert!(Configuration::parse(arguments.into_iter().map(String::from)).is_err());
        }
    }
}
