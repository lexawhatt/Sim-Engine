//! Bounded workload and measurement controls.

use super::{
    labels::LabelMode,
    workload::{Mode, Result},
};

#[derive(Clone, Copy)]
pub struct Configuration {
    pub constructions: usize,
    pub frames: usize,
    pub warmup: usize,
    pub trials: usize,
    pub mode: Mode,
    pub labels: LabelMode,
    pub vsync: bool,
    pub require_gpu: bool,
    pub recovery: bool,
    pub maximum_ms: Option<f64>,
}

impl Configuration {
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut result = Self {
            constructions: 100,
            frames: 120,
            warmup: 20,
            trials: 3,
            mode: Mode::Prepared,
            labels: LabelMode::Off,
            vsync: false,
            require_gpu: false,
            recovery: false,
            maximum_ms: None,
        };
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--vsync" => result.vsync = true,
                "--require-gpu" => result.require_gpu = true,
                "--recovery" => result.recovery = true,
                _ => {
                    let value = arguments.next().ok_or("option requires a value")?;
                    match argument.as_str() {
                        "--constructions" => result.constructions = value.parse()?,
                        "--frames" => result.frames = value.parse()?,
                        "--warmup" => result.warmup = value.parse()?,
                        "--trials" => result.trials = value.parse()?,
                        "--mode" => {
                            result.mode = match value.as_str() {
                                "streaming" => Mode::Streaming,
                                "prepared" => Mode::Prepared,
                                "analytic" => Mode::Analytic,
                                _ => {
                                    return Err(
                                        "mode must be streaming, prepared or analytic".into()
                                    );
                                }
                            }
                        }
                        "--labels" => {
                            result.labels = match value.as_str() {
                                "off" => LabelMode::Off,
                                "runs" => LabelMode::Runs,
                                "batch" => LabelMode::Batch,
                                _ => return Err("labels must be off, runs or batch".into()),
                            }
                        }
                        "--maximum-ms" => result.maximum_ms = Some(value.parse()?),
                        _ => return Err(format!("unknown argument: {argument}").into()),
                    }
                }
            }
        }
        if ![100, 500, 2000].contains(&result.constructions)
            || !(1..=10_000).contains(&result.frames)
            || !(1..=1000).contains(&result.warmup)
            || !(1..=20).contains(&result.trials)
            || result
                .maximum_ms
                .is_some_and(|ms| !ms.is_finite() || ms <= 0.0)
        {
            return Err("constructions=100/500/2000; frames=1..10000; warmup=1..1000; trials=1..20; maximum-ms positive finite".into());
        }
        Ok(result)
    }
}
