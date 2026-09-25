//! Dense small-primitive/label workload with real surface and independent phases.
//! This is a reproducible diagnostic, not a universal frame-rate promise.

#[path = "support/dense_scene_analytic.rs"]
mod analytic;
#[path = "support/dense_scene_labels.rs"]
mod labels;
#[path = "support/dense_scene_metrics.rs"]
mod metrics;
#[path = "support/dense_scene_workload.rs"]
mod workload;

#[path = "support/dense_scene_configuration.rs"]
mod configuration;
#[path = "support/dense_scene_runtime.rs"]
mod runtime;

use configuration::Configuration;
use runtime::State;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};
use workload::Result;

fn main() -> Result<()> {
    if std::env::args().any(|argument| argument == "--help" || argument == "-h") {
        println!(
            "dense_scene_benchmark --constructions 100|500|2000 --mode streaming|prepared|analytic --labels off|runs|batch --frames 120 --warmup 20 --trials 3 [--vsync] [--require-gpu] [--recovery] [--maximum-ms 16.67]"
        );
        println!(
            "Four phases: idle/pan/zoom/drag. Every trial reports cold setup/first draw, warm p50/p95/p99, exact Drawn counts, raw samples and asynchronously correlated GPU times. end_to_end_ms includes event scheduling/presentation backpressure, not hardware scanout. --maximum-ms is an explicit p95 diagnostic target, never an implicit refresh-independent FPS gate."
        );
        return Ok(());
    }
    env_logger::init();
    let configuration = Configuration::parse(std::env::args().skip(1))?;
    let mut application = Application {
        configuration,
        state: None,
        failure: None,
        completed: false,
    };
    EventLoop::new()?.run_app(&mut application)?;
    if let Some(error) = application.failure {
        return Err(error.into());
    }
    if !application.completed {
        return Err("benchmark exited before all requested trials and recovery completed".into());
    }
    Ok(())
}

struct Application {
    configuration: Configuration,
    state: Option<State>,
    failure: Option<String>,
    completed: bool,
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
            }
            Err(error) => {
                self.failure = Some(error.to_string());
                events.exit();
            }
        }
    }

    fn window_event(&mut self, events: &ActiveEventLoop, window: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if state.window.id() != window {
            return;
        }
        let result = match event {
            WindowEvent::CloseRequested => {
                Err("window closed before all requested trials and recovery completed".into())
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                state.invalidate().map(|()| false)
            }
            WindowEvent::RedrawRequested => state.redraw(),
            _ => Ok(false),
        };
        match result {
            Ok(true) => {
                self.completed = true;
                events.exit();
            }
            Ok(false) => state.window.request_redraw(),
            Err(error) => {
                if let Err(report_error) = state.report_incomplete("failure") {
                    eprintln!("could not preserve incomplete samples: {report_error}");
                }
                eprintln!("DENSE_FAILURE {} error={error:?}", state.context());
                self.failure = Some(error.to_string());
                events.exit();
            }
        }
    }

    fn about_to_wait(&mut self, events: &ActiveEventLoop) {
        events.set_control_flow(ControlFlow::Poll);
    }

    fn exiting(&mut self, _events: &ActiveEventLoop) {
        if !self.completed
            && let Some(state) = &mut self.state
            && let Err(error) = state.report_incomplete("event_loop_exit")
        {
            eprintln!("could not preserve incomplete samples: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_silent_workload_reduction_and_invalid_thresholds() {
        for arguments in [
            vec!["--constructions", "99"],
            vec!["--frames", "0"],
            vec!["--maximum-ms", "NaN"],
        ] {
            assert!(Configuration::parse(arguments.into_iter().map(str::to_owned)).is_err());
        }
        let configuration = Configuration::parse(
            ["--constructions", "2000", "--mode", "streaming"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(configuration.constructions, 2000);
    }
}
