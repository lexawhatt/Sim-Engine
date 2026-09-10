use super::*;

fn viewport() -> ResolvedViewport {
    ResolvedViewport {
        viewport: LogicalViewport::new(1280.0, 720.0).unwrap(),
        origin: Vec2::ZERO,
        scissor: ScissorRect {
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
        },
        item_clip: None,
        item_clipped_out: false,
    }
}

fn panel(offset: f32) -> Scene {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for index in 0..4 {
        scene
            .set_screen_clip(Some(
                ScreenClipRect::from_min_size(
                    LogicalScreenPosition::new(index as f32, 0.0),
                    crate::LogicalScreenVector::new(80.0, 80.0),
                )
                .unwrap(),
            ))
            .unwrap();
        scene
            .try_rect(
                Rect::from_center_size(Vec2::new(offset, index as f32 * 12.0), Vec2::splat(8.0)),
                0.0,
                ShapeStyle::filled(Color::WHITE.with_alpha(0.5)),
            )
            .unwrap();
    }
    scene
}

#[derive(Default)]
struct Frame<'a> {
    vertices: Vec<Vertex>,
    ready: Vec<ReadyItem<'a>>,
    statistics: FrameStatistics,
    aggregate: TessellationStats,
}

impl<'a> Frame<'a> {
    fn push(
        &mut self,
        scene: &'a Scene,
        uniform: CameraUniform,
        viewport: ResolvedViewport,
        proofs: Option<&mut StreamingSceneProofs<'a>>,
    ) {
        let mut batches = Vec::new();
        if let Some(proofs) = proofs {
            proofs
                .prepare(
                    scene,
                    uniform,
                    viewport,
                    &mut self.vertices,
                    &mut self.ready,
                    &mut self.statistics,
                    &mut self.aggregate,
                    &mut batches,
                )
                .unwrap();
        } else {
            prepare_streaming_scene_resolved(
                scene,
                uniform,
                viewport,
                &mut self.vertices,
                &mut self.ready,
                &mut self.statistics,
                &mut self.aggregate,
                &mut batches,
            )
            .unwrap();
        }
        assert!(batches.is_empty());
    }

    fn compare(&self, other: &Self) {
        let without_uploads = |statistics: FrameStatistics| FrameStatistics {
            upload_bytes: 0,
            streaming_upload_bytes: 0,
            ..statistics
        };
        assert_eq!(
            without_uploads(self.statistics),
            without_uploads(other.statistics)
        );
        for frame in [self, other] {
            assert_eq!(
                frame.statistics.streaming_upload_bytes,
                frame.vertices.len() * std::mem::size_of::<Vertex>()
            );
            assert_eq!(
                frame.statistics.upload_bytes,
                frame.statistics.streaming_upload_bytes
                    + frame.ready.len() * std::mem::size_of::<CameraUniform>()
            );
        }
        assert_eq!(self.aggregate, other.aggregate);
        assert_eq!(self.ready.len(), other.ready.len());
        for (left, right) in self.ready.iter().zip(&other.ready) {
            let (ReadyItem::Geometry(left), ReadyItem::Geometry(right)) = (left, right) else {
                panic!("unexpected non-geometry item");
            };
            assert!(matches!(left.source, ReadySource::Streaming));
            assert_eq!(left.vertex_count, right.vertex_count);
            assert_eq!(
                bytemuck::bytes_of(&left.camera_uniform),
                bytemuck::bytes_of(&right.camera_uniform)
            );
            assert_eq!(left.viewport.scissor, right.viewport.scissor);
            assert_eq!(left.viewport.item_clip, right.viewport.item_clip);
            assert_eq!(
                left.viewport.item_clipped_out,
                right.viewport.item_clipped_out
            );
            assert_eq!(left.batches.len(), right.batches.len());
            for (left, right) in left.batches.iter().zip(&right.batches) {
                let left_vertices = &self.vertices
                    [left.vertex_range.start as usize..left.vertex_range.end as usize];
                let right_vertices = &other.vertices
                    [right.vertex_range.start as usize..right.vertex_range.end as usize];
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(left_vertices),
                    bytemuck::cast_slice::<_, u8>(right_vertices)
                );
                assert_eq!(left.screen_clip, right.screen_clip);
            }
        }
    }
}

#[test]
fn repeated_streaming_matches_uncached_bytes_order_clips_and_accounting() {
    let scene = panel(3.0);
    let other = panel(-6.0);
    let empty = Scene::new(Color::WHITE).unwrap();
    let mut baseline = Frame::default();
    let mut cached = Frame::default();
    let mut proofs = StreamingSceneProofs::default();
    let mut region = viewport();
    let uniform = CameraUniform::new(Camera2d::default(), region.viewport).unwrap();
    for index in 0..40 {
        // Source batch clips and per-item clip/scissor must remain independent.
        region.item_clipped_out = index % 4 == 0;
        region.item_clip = (index % 2 == 0).then_some(ScissorRect {
            x: index,
            y: 1,
            width: 100,
            height: 80,
        });
        let source = [&scene, &other, &empty][index as usize % 3];
        baseline.push(source, uniform, region, None);
        cached.push(source, uniform, region, Some(&mut proofs));
        cached.compare(&baseline);
    }
    assert_eq!(proofs.next, 3, "repeated draws should hit existing proofs");
    assert!(cached.vertices.len() < baseline.vertices.len());
    assert!(cached.statistics.streaming_upload_bytes < baseline.statistics.streaming_upload_bytes);
}

#[test]
fn streaming_proof_keys_include_identity_and_every_uniform_bit_with_bounded_eviction() {
    let scenes: Vec<_> = (0..10).map(|_| panel(3.0)).collect();
    let mut baseline = Frame::default();
    let mut cached = Frame::default();
    let mut proofs = StreamingSceneProofs::default();
    let region = viewport();
    let uniform = CameraUniform::new(Camera2d::default(), region.viewport).unwrap();
    for scene in scenes.iter().chain(scenes.iter().rev()) {
        baseline.push(scene, uniform, region, None);
        cached.push(scene, uniform, region, Some(&mut proofs));
    }
    assert_eq!(proofs.entries.iter().flatten().count(), 8);
    for component in 0..16 {
        let mut changed = uniform;
        let values =
            bytemuck::cast_slice_mut::<CameraUniform, f32>(std::slice::from_mut(&mut changed));
        values[component] = if values[component] == 0.0 {
            -0.0
        } else {
            values[component] * 1.001
        };
        baseline.push(&scenes[0], changed, region, None);
        cached.push(&scenes[0], changed, region, Some(&mut proofs));
    }
    cached.compare(&baseline);
}

#[test]
fn invalid_camera_cannot_reuse_or_publish_streaming_proof() {
    let scene = panel(3.0);
    let mut frame = Frame::default();
    let mut proofs = StreamingSceneProofs::default();
    let region = viewport();
    let uniform = CameraUniform::new(Camera2d::default(), region.viewport).unwrap();
    frame.push(&scene, uniform, region, Some(&mut proofs));
    let mut invalid = uniform;
    invalid.world_to_screen_x[0] = f32::MAX;
    let previous = bytemuck::cast_slice::<_, u8>(&frame.vertices).to_vec();
    let previous_stats = frame.statistics;
    let previous_aggregate = frame.aggregate;
    assert!(
        proofs
            .prepare(
                &scene,
                invalid,
                region,
                &mut frame.vertices,
                &mut frame.ready,
                &mut frame.statistics,
                &mut frame.aggregate,
                &mut Vec::new()
            )
            .is_err()
    );
    assert_eq!(bytemuck::cast_slice::<_, u8>(&frame.vertices), previous);
    assert_eq!(frame.statistics, previous_stats);
    assert_eq!(frame.aggregate, previous_aggregate);
    assert_eq!(proofs.next, 1);
    assert_eq!(frame.ready.len(), 1);
    frame.push(&scene, uniform, region, Some(&mut proofs));
    assert_eq!(proofs.next, 1);
}

#[test]
fn streaming_proof_cannot_survive_scene_mutation_between_frames() {
    let mut scene = panel(3.0);
    let region = viewport();
    let uniform = CameraUniform::new(Camera2d::default(), region.viewport).unwrap();
    let first_vertices = {
        let mut frame = Frame::default();
        frame.push(
            &scene,
            uniform,
            region,
            Some(&mut StreamingSceneProofs::default()),
        );
        frame.vertices
    };
    scene
        .try_circle(Vec2::ZERO, 3.0, ShapeStyle::filled(Color::rgb8(255, 0, 0)))
        .unwrap();
    let mut frame = Frame::default();
    let mut baseline = Frame::default();
    frame.push(
        &scene,
        uniform,
        region,
        Some(&mut StreamingSceneProofs::default()),
    );
    baseline.push(&scene, uniform, region, None);
    frame.compare(&baseline);
    assert!(frame.vertices.len() > first_vertices.len());
}
