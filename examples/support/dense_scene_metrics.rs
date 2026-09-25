//! Raw paired-frame records with explicit CPU/GPU and warm/cold boundaries.

use super::workload::Result;
use sim_engine::{FrameReport, GpuTimingBatch, GpuTimingId, GpuTimingSource};
use std::io::{self, Write};

pub fn percentile(values: &[f64], percentage: usize) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[(values.len() * percentage).div_ceil(100).saturating_sub(1)]
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub frame: usize,
    pub cold: bool,
    pub interval_ms: f64,
    pub callback_ms: f64,
    pub update_ms: f64,
    pub geometry_update_ms: f64,
    pub glyph_update_ms: f64,
    pub geometry_projection_ms: Option<f64>,
    pub geometry_engine_update_ms: Option<f64>,
    pub glyph_projection_ms: Option<f64>,
    pub glyph_engine_update_ms: Option<f64>,
    pub collection_ms: f64,
    pub work_ms: f64,
    pub preparation_ms: f64,
    pub tessellation_ms: Option<f64>,
    pub validation_ms: Option<f64>,
    pub retained_ms: Option<f64>,
    pub other_ms: Option<f64>,
    pub upload_ms: f64,
    pub acquire_ms: f64,
    pub uploaded_bytes: usize,
    pub label_upload_bytes: usize,
    pub geometry_update_upload_bytes: usize,
    pub vertices: usize,
    pub streaming_vertices: usize,
    pub draws: usize,
    pub frame_items: usize,
    pub commands: usize,
    pub retained_cpu_bytes: usize,
    pub retained_gpu_bytes: usize,
    pub cache_cpu_bytes: usize,
    pub cache_uniform_bytes: usize,
    pub created_buffers: usize,
    pub gpu_id: Option<GpuTimingId>,
    pub gpu_ms: Option<f64>,
}

impl Sample {
    pub fn from_report(frame: usize, cold: bool, report: FrameReport) -> Self {
        let metrics = report.metrics();
        let statistics = report.statistics();
        let stages = metrics.preparation();
        Self {
            frame,
            cold,
            interval_ms: 0.0,
            callback_ms: 0.0,
            update_ms: 0.0,
            geometry_update_ms: 0.0,
            glyph_update_ms: 0.0,
            geometry_projection_ms: None,
            geometry_engine_update_ms: None,
            glyph_projection_ms: None,
            glyph_engine_update_ms: None,
            collection_ms: 0.0,
            work_ms: metrics
                .total_cpu()
                .saturating_sub(metrics.surface_acquire())
                .as_secs_f64()
                * 1000.0,
            preparation_ms: metrics.tessellation().as_secs_f64() * 1000.0,
            tessellation_ms: stages.map(|s| s.tessellation().as_secs_f64() * 1000.0),
            validation_ms: stages.map(|s| s.geometry_validation().as_secs_f64() * 1000.0),
            retained_ms: stages.map(|s| s.retained_draws().as_secs_f64() * 1000.0),
            other_ms: stages.map(|s| s.other().as_secs_f64() * 1000.0),
            upload_ms: metrics.upload().as_secs_f64() * 1000.0,
            acquire_ms: metrics.surface_acquire().as_secs_f64() * 1000.0,
            uploaded_bytes: statistics.upload_bytes(),
            label_upload_bytes: 0,
            geometry_update_upload_bytes: 0,
            vertices: statistics.vertex_count(),
            streaming_vertices: statistics.streaming_vertex_count(),
            draws: statistics.draw_calls(),
            frame_items: statistics.pass_count(),
            commands: statistics.command_count(),
            retained_cpu_bytes: statistics.retained_cpu_bytes(),
            retained_gpu_bytes: statistics
                .retained_buffer_bytes()
                .saturating_add(statistics.texture_bytes()),
            cache_cpu_bytes: 0,
            cache_uniform_bytes: 0,
            created_buffers: 0,
            gpu_id: report.gpu_timing_id(),
            gpu_ms: None,
        }
    }
}

pub struct Samples {
    pub records: Vec<Sample>,
    pub collection_ms: f64,
}

impl Samples {
    pub fn new(frames: usize) -> Self {
        Self {
            records: Vec::with_capacity(frames + 1),
            collection_ms: 0.0,
        }
    }

    pub fn collect(&mut self, batch: GpuTimingBatch) -> Result<()> {
        self.collection_ms += batch.collection_cpu().as_secs_f64() * 1000.0;
        for sample in batch.samples() {
            // Warmup, pre-map and discarded-generation samples are not part of
            // the measured distribution. Match exact IDs, not completion order.
            if let Some(record) = self
                .records
                .iter_mut()
                .find(|record| record.gpu_id == Some(sample.id()))
            {
                if sample.source() != GpuTimingSource::FrameComposer || record.gpu_ms.is_some() {
                    return Err("duplicate or incorrectly attributed GPU sample".into());
                }
                record.gpu_ms = Some(sample.elapsed().as_secs_f64() * 1000.0);
            }
        }
        Ok(())
    }

    pub fn missing(&self) -> usize {
        self.records
            .iter()
            .filter(|sample| sample.gpu_ms.is_none())
            .count()
    }

    fn write_records(
        &self,
        output: &mut impl Write,
        prefix: &str,
        context: &str,
    ) -> io::Result<()> {
        for sample in &self.records {
            writeln!(
                output,
                "{prefix} {context} frame={} cold={} interval_ms={:.6} callback_ms={:.6} update_ms={:.6} geometry_update_ms={:.6} glyph_update_ms={:.6} geometry_projection_ms={:?} geometry_engine_update_ms={:?} glyph_projection_ms={:?} glyph_engine_update_ms={:?} collection_ms={:.6} work_ms={:.6} preparation_ms={:.6} tessellation_ms={:?} validation_ms={:?} retained_ms={:?} other_ms={:?} upload_ms={:.6} acquire_ms={:.6} vertices={} streaming_vertices={} upload_bytes={} label_update_bytes={} geometry_update_bytes={} draws={} items={} commands={} retained_cpu_bytes={} retained_gpu_bytes={} cache_cpu_bytes={} cache_uniform_bytes={} created_buffers={} gpu_id={:?} gpu_ms={:?}",
                sample.frame,
                sample.cold,
                sample.interval_ms,
                sample.callback_ms,
                sample.update_ms,
                sample.geometry_update_ms,
                sample.glyph_update_ms,
                sample.geometry_projection_ms,
                sample.geometry_engine_update_ms,
                sample.glyph_projection_ms,
                sample.glyph_engine_update_ms,
                sample.collection_ms,
                sample.work_ms,
                sample.preparation_ms,
                sample.tessellation_ms,
                sample.validation_ms,
                sample.retained_ms,
                sample.other_ms,
                sample.upload_ms,
                sample.acquire_ms,
                sample.vertices,
                sample.streaming_vertices,
                sample.uploaded_bytes,
                sample.label_upload_bytes,
                sample.geometry_update_upload_bytes,
                sample.draws,
                sample.frame_items,
                sample.commands,
                sample.retained_cpu_bytes,
                sample.retained_gpu_bytes,
                sample.cache_cpu_bytes,
                sample.cache_uniform_bytes,
                sample.created_buffers,
                sample.gpu_id.map(GpuTimingId::get),
                sample.gpu_ms,
            )?;
        }
        Ok(())
    }

    /// Preserve partial evidence without publishing a complete distribution.
    /// Missing GPU values stay explicit; failure reporting never waits for GPU.
    pub fn write_incomplete(
        &self,
        output: &mut impl Write,
        context: &str,
        reason: &str,
    ) -> io::Result<()> {
        let context = format!("{context} status=incomplete reason={reason}");
        writeln!(
            output,
            "DENSE_PARTIAL {context} records={} missing_gpu={}",
            self.records.len(),
            self.missing()
        )?;
        self.write_records(output, "DENSE_PARTIAL_SAMPLE", &context)
    }

    pub fn print(&self, context: &str) {
        self.write_records(&mut io::stdout().lock(), "DENSE_SAMPLE", context)
            .expect("write complete raw samples");
        let warm: Vec<_> = self.records.iter().filter(|record| !record.cold).collect();
        for (name, values) in [
            (
                "end_to_end_ms",
                warm.iter().map(|s| s.interval_ms).collect::<Vec<_>>(),
            ),
            (
                "callback_cpu_ms",
                warm.iter().map(|s| s.callback_ms).collect(),
            ),
            ("host_update_ms", warm.iter().map(|s| s.update_ms).collect()),
            (
                "geometry_update_ms",
                warm.iter().map(|s| s.geometry_update_ms).collect(),
            ),
            (
                "glyph_update_ms",
                warm.iter().map(|s| s.glyph_update_ms).collect(),
            ),
            (
                "geometry_projection_ms",
                warm.iter()
                    .filter_map(|s| s.geometry_projection_ms)
                    .collect(),
            ),
            (
                "geometry_engine_update_ms",
                warm.iter()
                    .filter_map(|s| s.geometry_engine_update_ms)
                    .collect(),
            ),
            (
                "glyph_projection_ms",
                warm.iter().filter_map(|s| s.glyph_projection_ms).collect(),
            ),
            (
                "glyph_engine_update_ms",
                warm.iter()
                    .filter_map(|s| s.glyph_engine_update_ms)
                    .collect(),
            ),
            ("renderer_work_ms", warm.iter().map(|s| s.work_ms).collect()),
            (
                "prepare_ms",
                warm.iter().map(|s| s.preparation_ms).collect(),
            ),
            (
                "tessellation_ms",
                warm.iter().filter_map(|s| s.tessellation_ms).collect(),
            ),
            (
                "validation_ms",
                warm.iter().filter_map(|s| s.validation_ms).collect(),
            ),
            (
                "retained_draws_ms",
                warm.iter().filter_map(|s| s.retained_ms).collect(),
            ),
            (
                "other_prepare_ms",
                warm.iter().filter_map(|s| s.other_ms).collect(),
            ),
            ("upload_ms", warm.iter().map(|s| s.upload_ms).collect()),
            ("acquire_ms", warm.iter().map(|s| s.acquire_ms).collect()),
            (
                "hardware_gpu_ms",
                warm.iter().filter_map(|s| s.gpu_ms).collect(),
            ),
        ] {
            if values.is_empty() {
                println!("DENSE_DISTRIBUTION {context} stage={name} samples=0 unavailable=true");
            } else {
                println!(
                    "DENSE_DISTRIBUTION {context} stage={name} samples={} p50={:.6} p95={:.6} p99={:.6}",
                    values.len(),
                    percentile(&values, 50),
                    percentile(&values, 95),
                    percentile(&values, 99)
                );
            }
        }
        println!(
            "DENSE_GPU {context} requested={} correlated={} missing={} collection_cpu_ms={:.6}",
            self.records.len(),
            self.records.len() - self.missing(),
            self.missing(),
            self.collection_ms
        );
    }

    pub fn target_met(&self, target_ms: f64) -> bool {
        let values: Vec<_> = self
            .records
            .iter()
            .filter(|s| !s.cold)
            .map(|s| s.interval_ms)
            .collect();
        !values.is_empty() && percentile(&values, 95) <= target_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_keeps_the_slow_tail() {
        assert_eq!(percentile(&[1.0, 4.0, 100.0], 95), 100.0);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 50), 2.0);
    }

    #[test]
    fn interrupted_trials_preserve_raw_samples_without_complete_distributions() {
        let mut samples = Samples::new(120);
        samples.records.push(Sample {
            cold: true,
            interval_ms: 20.0,
            ..Sample::default()
        });
        samples.records.push(Sample {
            frame: 7,
            interval_ms: 12.0,
            gpu_ms: Some(3.0),
            ..Sample::default()
        });
        let mut output = Vec::new();
        samples
            .write_incomplete(&mut output, "generation=2", "surface_transition")
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.contains("status=incomplete reason=surface_transition records=2 missing_gpu=1")
        );
        assert_eq!(output.matches("DENSE_PARTIAL_SAMPLE ").count(), 2);
        assert!(output.contains("frame=7 cold=false interval_ms=12.000000"));
        assert!(output.contains("gpu_ms=Some(3.0)"));
        assert!(output.contains("gpu_ms=None"));
        assert!(!output.contains("DENSE_SAMPLE "));
        assert!(!output.contains("DENSE_DISTRIBUTION "));
        assert!(!output.contains("status=complete"));
    }
}
