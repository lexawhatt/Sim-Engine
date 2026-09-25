//! One-redraw-per-frame measurement, surface generations and recovery.

use super::{
    configuration::Configuration,
    labels::{LabelMode, Labels},
    metrics::{Sample, Samples},
    workload::{BACKGROUND, COMMANDS_PER_CONSTRUCTION, Phase, Result, Workload},
};
use sim_engine::{
    FrameBudget, FrameReport, GpuTimingStatus, RenderStatus, RendererPresentMode, Vec2,
    WgpuRenderer, WgpuRendererOptions,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    dpi::PhysicalSize, event_loop::ActiveEventLoop, monitor::MonitorHandle, window::Window,
};

#[derive(PartialEq)]
struct OutputContext {
    size: PhysicalSize<u32>,
    scale: f64,
    monitor: Option<MonitorHandle>,
    refresh: Option<u32>,
}

impl OutputContext {
    fn read(window: &Window) -> Self {
        let monitor = window.current_monitor();
        let refresh = monitor
            .as_ref()
            .and_then(MonitorHandle::refresh_rate_millihertz);
        Self {
            size: window.inner_size(),
            scale: window.scale_factor(),
            monitor,
            refresh,
        }
    }
}

pub struct State {
    configuration: Configuration,
    pub window: Arc<Window>,
    renderer: WgpuRenderer,
    output: Option<OutputContext>,
    generation: usize,
    restarts: usize,
    workload: Option<Workload>,
    labels: Option<Labels>,
    samples: Samples,
    trial_reported: bool,
    phase: usize,
    trial: usize,
    cold_drawn: bool,
    warm_drawn: usize,
    measured_drawn: usize,
    total_drawn: usize,
    attempts: usize,
    skipped: usize,
    previous_present: Option<Instant>,
    draining: Option<Instant>,
    recovery_pending: bool,
    target_misses: usize,
}

impl State {
    pub fn new(events: &ActiveEventLoop, configuration: Configuration) -> Result<Self> {
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("Sim;Engine dense Math rendering diagnostic")
                    .with_decorations(false)
                    .with_inner_size(PhysicalSize::new(1280, 800)),
            )?,
        );
        let size = window.inner_size();
        let present = if configuration.vsync {
            RendererPresentMode::Vsync
        } else {
            RendererPresentMode::NoVsync
        };
        let options =
            WgpuRendererOptions::new(present, window.scale_factor())?.with_gpu_timing(true);
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        validate_renderer(&renderer, configuration)?;
        Ok(Self {
            configuration,
            window,
            renderer,
            output: None,
            generation: 0,
            restarts: 0,
            workload: None,
            labels: None,
            samples: Samples::new(configuration.frames),
            trial_reported: false,
            phase: 0,
            trial: 0,
            cold_drawn: false,
            warm_drawn: 0,
            measured_drawn: 0,
            total_drawn: 0,
            attempts: 0,
            skipped: 0,
            previous_present: None,
            draining: None,
            recovery_pending: false,
            target_misses: 0,
        })
    }

    pub fn context(&self) -> String {
        format!(
            "mode={:?} labels={:?} constructions={} phase={} trial={} generation={}",
            self.configuration.mode,
            self.configuration.labels,
            self.configuration.constructions,
            Phase::ALL[self.phase].name(),
            self.trial,
            self.generation
        )
    }

    fn reset_trial(&mut self) {
        self.workload = None;
        self.labels = None;
        self.samples = Samples::new(self.configuration.frames);
        self.trial_reported = false;
        self.cold_drawn = false;
        self.warm_drawn = 0;
        self.measured_drawn = 0;
        self.previous_present = None;
        self.draining = None;
        self.renderer.clear_frame_cache();
    }

    pub fn invalidate(&mut self) -> Result<()> {
        validate_surface_transition(self.recovery_pending, self.restarts)?;
        self.report_incomplete("surface_or_output_transition")?;
        self.restarts += 1;
        println!(
            "DENSE_RESTART {} discarded_records={} reason=surface_or_output_transition",
            self.context(),
            self.samples.records.len()
        );
        self.generation += 1;
        self.output = None;
        self.reset_trial();
        Ok(())
    }

    pub fn report_incomplete(&mut self, reason: &str) -> Result<()> {
        if !self.trial_reported {
            self.samples.write_incomplete(
                &mut std::io::stdout().lock(),
                &self.context(),
                reason,
            )?;
            self.trial_reported = true;
        }
        Ok(())
    }

    fn metadata(&self) {
        let d = self.renderer.diagnostics();
        println!(
            "DENSE_CONTEXT {} fixture_version=6 glyph_batch_chunks=24 vcs_sha={:?} adapter={:?} backend={} vendor={} device={} pci={:?} driver={:?} driver_info={:?} physical={:?} scale={} format={:?} samples={} present={:?} refresh_millihertz={:?} gpu_timing={:?}",
            self.context(),
            std::env::var("SIM_ENGINE_RELEASE_SHA").ok(),
            d.adapter_name(),
            d.backend(),
            d.vendor_id(),
            d.device_id(),
            d.pci_bus_id(),
            d.driver(),
            d.driver_info(),
            d.physical_size(),
            d.scale_factor(),
            d.surface_format(),
            d.sample_count(),
            d.present_mode(),
            self.output.as_ref().and_then(|o| o.refresh),
            self.renderer.gpu_timing_statistics().status()
        );
    }

    fn initialize_workload(&mut self) -> Result<()> {
        self.metadata();
        let logical = self.renderer.logical_size();
        let size = Vec2::new(logical.0, logical.1);
        let start = Instant::now();
        let workload = Workload::new(
            &self.renderer,
            self.configuration.constructions,
            size,
            self.configuration.mode,
        )?;
        let geometry_ms = start.elapsed().as_secs_f64() * 1000.0;
        let text_start = Instant::now();
        self.labels = Labels::new(
            &self.renderer,
            self.configuration.constructions,
            size,
            self.configuration.labels,
        )?;
        let text_ms = text_start.elapsed().as_secs_f64() * 1000.0;
        let (commands, estimated_vertices, estimated_bytes) = workload.source_counts();
        if commands != self.configuration.constructions * COMMANDS_PER_CONSTRUCTION {
            return Err("fixture lost source construction commands".into());
        }
        let (label_cpu, label_gpu) = self.labels.as_ref().map_or((0, 0), Labels::owned_bytes);
        println!(
            "DENSE_COLD_SETUP {} geometry_build_prepare_cpu_ms={geometry_ms:.6} glyph_resource_cpu_ms={text_ms:.6} source_commands={commands} estimated_vertices={estimated_vertices} estimated_upload_bytes={estimated_bytes} host_source_bytes={} owned_label_cpu_bytes={label_cpu} owned_label_gpu_bytes={label_gpu}",
            self.context(),
            workload.host_source_bytes()
        );
        self.workload = Some(workload);
        Ok(())
    }

    /// Exactly one submission is attempted in each window redraw. Collection is
    /// nonblocking; phase-end draining processes redraws without new GPU work.
    pub fn redraw(&mut self) -> Result<bool> {
        let current = OutputContext::read(&self.window);
        if current.size.width == 0 || current.size.height == 0 {
            return Ok(false);
        }
        if std::env::var("SIM_ENGINE_DENSE_ACCEPTANCE").as_deref() == Ok("1")
            && (current.size != PhysicalSize::new(1280, 800) || current.scale != 1.0)
        {
            return Err(format!(
                "dense acceptance requires actual 1280x800 physical pixels at scale 1; got {:?} at {}",
                current.size, current.scale
            ).into());
        }
        if self
            .output
            .as_ref()
            .is_some_and(|previous| previous != &current)
        {
            self.invalidate()?;
        }
        if self.output.is_none() {
            self.renderer.resize_with_scale_factor(
                current.size.width,
                current.size.height,
                current.scale,
            )?;
            let report = self
                .renderer
                .begin_frame(BACKGROUND, FrameBudget::default())?
                .present()?;
            if report.status() != RenderStatus::Drawn {
                return Err(
                    format!("output-confirmation frame skipped: {:?}", report.status()).into(),
                );
            }
            validate_preparation_sum(&report)?;
            self.output = Some(current);
            return Ok(false);
        }
        if self.draining.is_some() {
            return self.drain();
        }
        if self.workload.is_none() {
            self.initialize_workload()?;
        }
        let started = Instant::now();
        let collection = self.renderer.collect_gpu_timings();
        let collection_ms = collection.collection_cpu().as_secs_f64() * 1000.0;
        self.samples.collect(collection)?;
        let cold = !self.cold_drawn;
        let measured = !cold && self.warm_drawn >= self.configuration.warmup;
        let source_frame = if measured {
            self.measured_drawn
        } else {
            self.warm_drawn
        };
        let (camera, displacement) = Phase::ALL[self.phase].pose(source_frame)?;
        let update_started = Instant::now();
        let workload = self.workload.as_mut().ok_or("missing workload")?;
        let geometry_upload = workload.update(&self.renderer, camera, displacement)?;
        let geometry_update_ms = update_started.elapsed().as_secs_f64() * 1000.0;
        let analytic_update_timing = workload.analytic_update_timing();
        let glyph_started = Instant::now();
        let label_upload = self
            .labels
            .as_mut()
            .map(|labels| labels.update(&self.renderer, camera, displacement))
            .transpose()?
            .unwrap_or(0);
        let glyph_update_ms = glyph_started.elapsed().as_secs_f64() * 1000.0;
        let glyph_update_timing = self.labels.as_ref().map(Labels::update_timing);
        let update_ms = update_started.elapsed().as_secs_f64() * 1000.0;
        let budget = FrameBudget::new(16_384, 100_000, 8_000_000, 640 << 20, 64 << 20, 65_536);
        let mut frame = self.renderer.begin_frame(BACKGROUND, budget)?;
        workload.draw(&mut frame, camera)?;
        if let Some(labels) = &self.labels {
            labels.draw(&mut frame)?;
        }
        let report = frame.present()?;
        validate_preparation_sum(&report)?;
        let presented = Instant::now();
        self.attempts += 1;
        if report.status() != RenderStatus::Drawn {
            self.skipped += 1;
            return Err(format!(
                "attempt={} skipped={:?}; skipped attempts are not measured throughput",
                self.attempts,
                report.status()
            )
            .into());
        }
        self.total_drawn += 1;
        if report
            .metrics()
            .tessellation_stats()
            .dropped_command_count()
            != 0
        {
            return Err("accepted source geometry was dropped".into());
        }
        if report.metrics().tessellation_stats().command_count()
            != self.configuration.constructions * COMMANDS_PER_CONSTRUCTION
        {
            return Err("render report does not account for every source geometry command".into());
        }
        let expected_label_vertices = if self.configuration.labels == LabelMode::Off {
            0
        } else {
            self.configuration.constructions * 3 * 4 * 6
        };
        if report
            .statistics()
            .vertex_count()
            .checked_sub(report.metrics().tessellation_stats().vertex_count())
            != Some(expected_label_vertices)
        {
            return Err("render report does not account for every source glyph vertex".into());
        }
        if self.recovery_pending {
            println!(
                "DENSE_RECOVERY status=drawn source_commands={} vertices={} draws={} retained_cpu_bytes={} retained_gpu_bytes={}",
                workload.source_counts().0,
                report.statistics().vertex_count(),
                report.statistics().draw_calls(),
                report.statistics().retained_cpu_bytes(),
                report.statistics().retained_buffer_bytes()
            );
            return self.complete();
        }
        if cold || measured {
            let mut sample = Sample::from_report(source_frame, cold, report);
            sample.interval_ms = self
                .previous_present
                .map_or(presented.duration_since(started), |previous| {
                    presented.duration_since(previous)
                })
                .as_secs_f64()
                * 1000.0;
            sample.callback_ms = presented.duration_since(started).as_secs_f64() * 1000.0;
            sample.update_ms = update_ms;
            sample.geometry_update_ms = geometry_update_ms;
            sample.glyph_update_ms = glyph_update_ms;
            sample.geometry_projection_ms =
                analytic_update_timing.map(|(projection, _)| projection);
            sample.geometry_engine_update_ms = analytic_update_timing.map(|(_, engine)| engine);
            sample.glyph_projection_ms = glyph_update_timing.map(|(projection, _)| projection);
            sample.glyph_engine_update_ms = glyph_update_timing.map(|(_, engine)| engine);
            sample.collection_ms = collection_ms;
            sample.label_upload_bytes = label_upload;
            sample.geometry_update_upload_bytes = geometry_upload;
            let cache = self.renderer.frame_cache_statistics();
            sample.cache_cpu_bytes = cache.cpu_bytes();
            sample.cache_uniform_bytes = cache.uniform_bytes();
            sample.created_buffers = cache.created_buffers();
            if self.configuration.require_gpu && sample.gpu_id.is_none() {
                return Err("measured Drawn frame has no required GPU timestamp ID".into());
            }
            self.samples.records.push(sample);
        }
        self.previous_present = Some(presented);
        if cold {
            self.cold_drawn = true;
        } else if measured {
            self.measured_drawn += 1;
        } else {
            self.warm_drawn += 1;
        }
        if self.measured_drawn == self.configuration.frames {
            self.draining = Some(Instant::now());
        }
        Ok(false)
    }

    fn drain(&mut self) -> Result<bool> {
        self.samples.collect(self.renderer.collect_gpu_timings())?;
        let available = self.renderer.gpu_timing_statistics().status() == GpuTimingStatus::Enabled;
        let elapsed = self.draining.ok_or("missing drain state")?.elapsed();
        if available && self.samples.missing() > 0 && elapsed < Duration::from_secs(10) {
            return Ok(false);
        }
        if self.configuration.require_gpu && self.samples.missing() > 0 {
            return Err("GPU timing did not provide every requested cold/measured sample".into());
        }
        let context = self.context();
        self.samples.print(&context);
        self.trial_reported = true;
        let stats = self.renderer.gpu_timing_statistics();
        println!(
            "DENSE_TRIAL {context} status=complete measured_drawn={} cold_drawn={} warmup_drawn={} restarts={} gpu_dropped={} gpu_map_failures={} gpu_invalid={} drain_cpu_wall_ms={:.6} target_16_67ms_met={}",
            self.measured_drawn,
            usize::from(self.cold_drawn),
            self.warm_drawn,
            self.restarts,
            stats.dropped(),
            stats.map_failures(),
            stats.invalid_samples(),
            elapsed.as_secs_f64() * 1000.0,
            self.samples.target_met(16.67)
        );
        if self
            .configuration
            .maximum_ms
            .is_some_and(|limit| !self.samples.target_met(limit))
        {
            self.target_misses += 1;
        }
        self.trial += 1;
        if self.trial == self.configuration.trials {
            self.trial = 0;
            self.phase += 1;
        }
        if self.phase == Phase::ALL.len() {
            self.phase -= 1;
            if self.configuration.recovery {
                let started = Instant::now();
                pollster::block_on(self.renderer.recover_device_and_surface())?;
                validate_renderer(&self.renderer, self.configuration)?;
                self.workload
                    .as_mut()
                    .ok_or("missing recovery source")?
                    .restore(&self.renderer)?;
                if let Some(labels) = &mut self.labels {
                    labels.restore(&self.renderer)?;
                }
                println!(
                    "DENSE_RECOVERY restored_cpu_ms={:.6}",
                    started.elapsed().as_secs_f64() * 1000.0
                );
                self.draining = None;
                self.recovery_pending = true;
                self.metadata();
                return Ok(false);
            }
            return self.complete();
        }
        self.reset_trial();
        Ok(false)
    }

    fn complete(&self) -> Result<bool> {
        println!(
            "DENSE_COMPLETE attempts={} drawn={} skipped={} phases={} trials={} target_misses={}",
            self.attempts,
            self.total_drawn,
            self.skipped,
            Phase::ALL.len(),
            self.configuration.trials,
            self.target_misses
        );
        if self.target_misses != 0 {
            return Err(
                "one or more trials missed the explicitly requested end-to-end p95 target".into(),
            );
        }
        Ok(true)
    }
}

fn validate_surface_transition(recovery_pending: bool, restarts: usize) -> Result<()> {
    if recovery_pending {
        return Err(
            "surface/output transitioned after restoration; fresh resources cannot prove recovery"
                .into(),
        );
    }
    if restarts >= 8 {
        return Err("more than eight surface/output transitions; no stable measurement".into());
    }
    Ok(())
}

fn validate_preparation_sum(report: &FrameReport) -> Result<()> {
    if report
        .metrics()
        .preparation()
        .is_some_and(|stages| stages.total() != report.metrics().tessellation())
    {
        return Err("preparation substage total differs from coarse preparation interval".into());
    }
    Ok(())
}

fn validate_renderer(renderer: &WgpuRenderer, configuration: Configuration) -> Result<()> {
    if configuration.require_gpu
        && renderer.gpu_timing_statistics().status() != GpuTimingStatus::Enabled
    {
        return Err("hardware GPU timestamps are required but unavailable".into());
    }
    if let Ok(required) = std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")
        && !renderer
            .adapter_pci_bus_id()
            .eq_ignore_ascii_case(&required)
    {
        return Err("selected physical GPU does not match required PCI address".into());
    }
    if (std::env::var_os("SIM_ENGINE_RELEASE_SHA").is_some()
        || std::env::var("SIM_ENGINE_DENSE_ACCEPTANCE").as_deref() == Ok("1"))
        && !renderer.adapter_backend().eq_ignore_ascii_case("vulkan")
    {
        return Err("revision-bound benchmark requires Vulkan".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_transition_is_rejected_before_recreating_resources() {
        for restarts in 0..=8 {
            let error = validate_surface_transition(true, restarts).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("fresh resources cannot prove recovery")
            );
        }
        assert!(validate_surface_transition(false, 7).is_ok());
        assert!(validate_surface_transition(false, 8).is_err());
    }
}
