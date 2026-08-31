use std::sync::OnceLock;

use super::*;

pub(super) fn tessellate_scene(
    scene: &Scene,
    vertices: &mut Vec<Vertex>,
    draw_batches: &mut Vec<PreparedDrawBatch>,
) -> Result<TessellationStats, TessellationError> {
    let initial_vertex_count = vertices.len();
    let initial_batch_count = draw_batches.len();
    let estimate = scene.statistics();
    reserve_items(vertices, estimate.estimated_tessellated_vertices())?;
    reserve_items(draw_batches, estimate.estimated_draw_batches())?;
    let mut stats = TessellationStats {
        command_count: scene.command_count(),
        command_counts: scene.statistics().accepted_by_primitive(),
        ..TessellationStats::default()
    };
    for scene_command in scene.commands() {
        let primitive = scene_command.command().primitive();
        let screen_clip = scene_command.screen_clip();
        let vertex_start = vertices.len();

        match scene_command.command() {
            DrawCommand::Circle(circle) => tessellate_circle(circle, vertices)?,
            DrawCommand::Rect(rectangle) => tessellate_rect(
                rectangle.rect(),
                rectangle.corner_radius(),
                rectangle.style(),
                vertices,
            )?,
            DrawCommand::Line(line) => tessellate_line(line, vertices)?,
            DrawCommand::Polyline(polyline) => {
                tessellate_polyline(polyline, vertices)?;
            }
        }

        for vertex in &mut vertices[vertex_start..] {
            vertex.depth = scene_command.depth();
        }

        let vertex_end = vertices.len();
        if vertices[vertex_start..]
            .iter()
            .any(|vertex| !vertex.is_finite())
        {
            vertices.truncate(vertex_start);
            stats.dropped_command_count += 1;
            stats.dropped_counts.increment(primitive);
            continue;
        }
        if vertex_end == vertex_start {
            stats.dropped_command_count += 1;
            stats.dropped_counts.increment(primitive);
            continue;
        }

        stats.rendered_command_count += 1;
        stats.rendered_counts.increment(primitive);

        let vertex_start =
            u32::try_from(vertex_start).map_err(|_| TessellationError::CapacityTooLarge)?;
        let vertex_end =
            u32::try_from(vertex_end).map_err(|_| TessellationError::CapacityTooLarge)?;
        match draw_batches.last_mut() {
            Some(batch)
                if batch.screen_clip == screen_clip && batch.vertex_range.end == vertex_start =>
            {
                batch.vertex_range.end = vertex_end;
            }
            _ => draw_batches.push(PreparedDrawBatch {
                vertex_range: vertex_start..vertex_end,
                screen_clip,
            }),
        }
    }
    stats.vertex_count = vertices.len().saturating_sub(initial_vertex_count);
    stats.draw_batch_count = draw_batches.len().saturating_sub(initial_batch_count);
    stats.upload_bytes = stats
        .vertex_count
        .saturating_mul(std::mem::size_of::<Vertex>());
    validate_tessellated_budget(scene, stats)?;
    Ok(stats)
}

fn reserve_items<T>(items: &mut Vec<T>, additional: usize) -> Result<(), TessellationError> {
    items
        .try_reserve(additional)
        .map_err(|_| TessellationError::AllocationFailed {
            requested_bytes: additional.saturating_mul(std::mem::size_of::<T>()),
        })
}

fn validate_tessellated_budget(
    scene: &Scene,
    stats: TessellationStats,
) -> Result<(), TessellationError> {
    let Some(budget) = scene.budget() else {
        return Ok(());
    };
    let actual = [
        (
            SceneBudgetResource::TessellatedVertices,
            budget.max_tessellated_vertices(),
            stats.vertex_count,
        ),
        (
            SceneBudgetResource::UploadBytes,
            budget.max_upload_bytes(),
            stats.upload_bytes,
        ),
        (
            SceneBudgetResource::DrawBatches,
            budget.max_draw_batches(),
            stats.draw_batch_count,
        ),
    ];
    for (resource, limit, actual) in actual {
        if actual > limit {
            return Err(TessellationError::BudgetExceeded {
                resource,
                limit,
                actual,
            });
        }
    }
    Ok(())
}

pub(super) fn screen_clip_to_scissor(
    screen_clip: ScreenClipRect,
    viewport: LogicalViewport,
    scale_factor: f32,
) -> Option<ScissorRect> {
    let rect = screen_clip.rect();
    if !rect.min.is_finite()
        || !rect.max.is_finite()
        || !scale_factor.is_finite()
        || scale_factor <= 0.0
    {
        return None;
    }
    let rect = rect.normalized();
    let physical_width = (viewport.width() * scale_factor).round();
    let physical_height = (viewport.height() * scale_factor).round();

    let min_x = (rect.min.x * scale_factor)
        .floor()
        .clamp(0.0, physical_width);
    let min_y = (rect.min.y * scale_factor)
        .floor()
        .clamp(0.0, physical_height);
    let max_x = (rect.max.x * scale_factor)
        .ceil()
        .clamp(0.0, physical_width);
    let max_y = (rect.max.y * scale_factor)
        .ceil()
        .clamp(0.0, physical_height);
    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some(ScissorRect {
        x: min_x as u32,
        y: min_y as u32,
        width: (max_x - min_x) as u32,
        height: (max_y - min_y) as u32,
    })
}

pub(super) fn logical_viewport_scissor(
    origin: Vec2,
    viewport: LogicalViewport,
    scale_factor: f32,
    target_width: u32,
    target_height: u32,
) -> Option<ScissorRect> {
    if !origin.is_finite() || !scale_factor.is_finite() || scale_factor <= 0.0 {
        return None;
    }
    let max = origin + viewport.size();
    if !max.is_finite() {
        return None;
    }
    let min_x = (origin.x * scale_factor)
        .floor()
        .clamp(0.0, target_width as f32);
    let min_y = (origin.y * scale_factor)
        .floor()
        .clamp(0.0, target_height as f32);
    let max_x = (max.x * scale_factor)
        .ceil()
        .clamp(0.0, target_width as f32);
    let max_y = (max.y * scale_factor)
        .ceil()
        .clamp(0.0, target_height as f32);
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some(ScissorRect {
        x: min_x as u32,
        y: min_y as u32,
        width: (max_x - min_x) as u32,
        height: (max_y - min_y) as u32,
    })
}

pub(super) fn offset_scissor(local: ScissorRect, viewport: ScissorRect) -> Option<ScissorRect> {
    let x = viewport.x.saturating_add(local.x);
    let y = viewport.y.saturating_add(local.y);
    let viewport_max_x = viewport.x.saturating_add(viewport.width);
    let viewport_max_y = viewport.y.saturating_add(viewport.height);
    let max_x = x.saturating_add(local.width).min(viewport_max_x);
    let max_y = y.saturating_add(local.height).min(viewport_max_y);
    (max_x > x && max_y > y).then_some(ScissorRect {
        x,
        y,
        width: max_x - x,
        height: max_y - y,
    })
}

fn tessellate_circle(circle: &Circle, vertices: &mut Vec<Vertex>) -> Result<(), TessellationError> {
    if circle.radius() <= 0.0 {
        return Ok(());
    }

    if let Some(shadow) = circle.style().shadow() {
        push_circle_shadow_world(circle.center(), circle.radius(), shadow, vertices)?;
    }

    if let Some(fill) = circle.style().fill() {
        push_circle_fill_world(circle.center(), circle.radius(), fill, Vec2::ZERO, vertices);
    }

    if let Some(stroke) = circle.style().stroke() {
        push_circle_stroke_world(circle.center(), circle.radius(), stroke, vertices)?;
    }
    Ok(())
}

fn push_circle_shadow_world(
    center_world: Vec2,
    radius_world: f32,
    shadow: Shadow,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    push_circle_fill_world(
        center_world,
        radius_world,
        Fill::Solid(shadow.color()),
        shadow.offset().to_vec2(),
        vertices,
    );

    if shadow.spread() > 0.0 {
        let points = circle_world_points(center_world, radius_world)?;
        push_closed_polyline_world(
            &points,
            shadow.spread() * 2.0,
            shadow.color(),
            shadow.offset().to_vec2(),
            vertices,
        )?;
    }
    Ok(())
}

fn push_circle_stroke_world(
    center_world: Vec2,
    radius_world: f32,
    stroke: Stroke,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let points = circle_world_points(center_world, radius_world)?;
    push_closed_polyline_world(
        &points,
        stroke.width(),
        stroke.color(),
        Vec2::ZERO,
        vertices,
    )
}

fn tessellate_rect(
    rect: Rect,
    corner_radius: f32,
    style: ShapeStyle,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let rect = rect.normalized();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return Ok(());
    }

    if let Some(shadow) = style.shadow() {
        push_rect_world(
            rect,
            corner_radius,
            Fill::Solid(shadow.color()),
            shadow.offset().to_vec2(),
            vertices,
        )?;
        if shadow.spread() > 0.0 {
            let points = rounded_rect_points(rect, corner_radius)?;
            push_closed_polyline_world(
                &points,
                shadow.spread() * 2.0,
                shadow.color(),
                shadow.offset().to_vec2(),
                vertices,
            )?;
        }
    }

    if let Some(fill) = style.fill() {
        push_rect_world(rect, corner_radius, fill, Vec2::ZERO, vertices)?;
    }

    if let Some(stroke) = style.stroke() {
        let points = rounded_rect_points(rect, corner_radius)?;
        push_closed_polyline_world(
            &points,
            stroke.width(),
            stroke.color(),
            Vec2::ZERO,
            vertices,
        )?;
    }
    Ok(())
}

fn tessellate_line(line: &Line, vertices: &mut Vec<Vertex>) -> Result<(), TessellationError> {
    tessellate_open_stroke(&[line.from(), line.to()], line.stroke_style(), vertices)
}

fn tessellate_polyline(
    polyline: &Polyline,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let mut points = Vec::new();
    reserve_items(&mut points, polyline.points().len())?;
    for point in polyline.points() {
        if points.last().is_none_or(|previous| *previous != *point) {
            points.push(*point);
        }
    }
    if points.len() < 2 {
        return Ok(());
    }
    tessellate_open_stroke(&points, polyline.stroke_style(), vertices)
}

fn tessellate_open_stroke(
    points: &[Vec2],
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let color = style.stroke().color();
    if let Some(dash) = style.dash_pattern() {
        tessellate_dashed_stroke(points, dash, style, vertices)?;
    } else {
        tessellate_stroke_run(points, style, true, true, vertices);
    }

    if let Some(marker) = style.start_marker() {
        push_stroke_marker(
            points[0],
            points[1] - points[0],
            true,
            marker,
            color,
            vertices,
        );
    }
    if let Some(marker) = style.end_marker() {
        let last = points.len() - 1;
        push_stroke_marker(
            points[last],
            points[last] - points[last - 1],
            false,
            marker,
            color,
            vertices,
        );
    }

    debug_assert!(style.stroke().width().is_finite() && style.stroke().width() > 0.0);
    Ok(())
}

fn tessellate_dashed_stroke(
    points: &[Vec2],
    dash: crate::StrokeDashPattern2d,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let lengths = dash.lengths();
    let total: f64 = lengths.iter().map(|length| f64::from(*length)).sum();
    let mut phase = f64::from(dash.phase()).rem_euclid(total);
    let mut pattern_index = 0usize;
    while phase >= f64::from(lengths[pattern_index]) {
        phase -= f64::from(lengths[pattern_index]);
        pattern_index = (pattern_index + 1) % lengths.len();
    }
    let mut pattern_remaining = f64::from(lengths[pattern_index]) - phase;
    let mut run = Vec::new();

    for (segment_index, pair) in points.windows(2).enumerate() {
        let from = pair[0];
        let to = pair[1];
        let delta = to - from;
        let segment_length = f64::from(delta.x).hypot(f64::from(delta.y));
        let mut segment_consumed = 0.0;
        while segment_consumed < segment_length {
            let segment_remaining = segment_length - segment_consumed;
            let consumed = segment_remaining.min(pattern_remaining);
            let finishes_segment = consumed >= segment_remaining;
            let finishes_pattern = consumed >= pattern_remaining;
            let finishes_path = finishes_segment && segment_index + 2 == points.len();
            let visible = pattern_index.is_multiple_of(2);
            if visible && consumed > 0.0 {
                let piece_from = segment_point(from, to, segment_consumed / segment_length);
                let piece_to =
                    segment_point(from, to, (segment_consumed + consumed) / segment_length);
                if run.last().is_none_or(|point| *point != piece_from) {
                    reserve_items(&mut run, 1)?;
                    run.push(piece_from);
                }
                if run.last().is_none_or(|point| *point != piece_to) {
                    reserve_items(&mut run, 1)?;
                    run.push(piece_to);
                }
                if finishes_pattern || finishes_path {
                    if run.len() >= 2 {
                        let starts_path = run[0] == points[0];
                        let ends_path = run[run.len() - 1] == points[points.len() - 1];
                        tessellate_stroke_run(&run, style, starts_path, ends_path, vertices);
                    }
                    run.clear();
                }
            }
            segment_consumed += consumed;
            pattern_remaining -= consumed;
            if pattern_remaining <= 0.0 {
                pattern_index = (pattern_index + 1) % lengths.len();
                pattern_remaining = f64::from(lengths[pattern_index]);
            }
        }
    }
    Ok(())
}

fn segment_point(from: Vec2, to: Vec2, amount: f64) -> Vec2 {
    Vec2::new(
        (f64::from(from.x) * (1.0 - amount) + f64::from(to.x) * amount) as f32,
        (f64::from(from.y) * (1.0 - amount) + f64::from(to.y) * amount) as f32,
    )
}

fn tessellate_stroke_run(
    points: &[Vec2],
    style: crate::StrokeStyle2d,
    starts_path: bool,
    ends_path: bool,
    vertices: &mut Vec<Vertex>,
) {
    match style.width_mode() {
        crate::StrokeWidthMode2d::LogicalPixels => {
            push_logical_stroke_run(points, style, starts_path, ends_path, vertices)
        }
        crate::StrokeWidthMode2d::WorldUnits => {
            push_world_stroke_run(points, style, starts_path, ends_path, vertices)
        }
    }
}

fn endpoint_tangent(style: crate::StrokeStyle2d, start: bool, is_path_endpoint: bool) -> f32 {
    let marker = if start {
        style.start_marker()
    } else {
        style.end_marker()
    };
    if is_path_endpoint && marker.is_some() {
        return 0.0;
    }
    if style.cap() == crate::StrokeCap2d::Square {
        let half_width = style.stroke().width() * 0.5;
        if start { -half_width } else { half_width }
    } else {
        0.0
    }
}

fn endpoint_has_round_cap(
    style: crate::StrokeStyle2d,
    start: bool,
    is_path_endpoint: bool,
) -> bool {
    let marker = if start {
        style.start_marker()
    } else {
        style.end_marker()
    };
    style.cap() == crate::StrokeCap2d::Round && (!is_path_endpoint || marker.is_none())
}

fn logical_join_role(join: crate::StrokeJoin2d) -> f32 {
    match join {
        crate::StrokeJoin2d::Bevel => 1.0,
        crate::StrokeJoin2d::Miter => 2.0,
        crate::StrokeJoin2d::Round => 3.0,
    }
}

fn push_logical_stroke_run(
    points: &[Vec2],
    style: crate::StrokeStyle2d,
    starts_path: bool,
    ends_path: bool,
    vertices: &mut Vec<Vertex>,
) {
    let half_width = style.stroke().width() * 0.5;
    let color = style.stroke().color();
    for (index, pair) in points.windows(2).enumerate() {
        let from = pair[0];
        let to = pair[1];
        let direction = to - from;
        let start_joint = index > 0;
        let end_joint = index + 2 < points.len();
        let previous = if start_joint {
            from - points[index - 1]
        } else {
            direction
        };
        let next = if end_joint {
            points[index + 2] - to
        } else {
            direction
        };
        let start_tangent = if start_joint {
            0.0
        } else {
            endpoint_tangent(style, true, starts_path)
        };
        let end_tangent = if end_joint {
            0.0
        } else {
            endpoint_tangent(style, false, ends_path)
        };
        let start_role = if start_joint {
            logical_join_role(style.join())
        } else {
            0.0
        };
        let end_role = if end_joint {
            logical_join_role(style.join())
        } else {
            0.0
        };
        let vertex = |world,
                      previous_direction,
                      next_direction,
                      normal_distance,
                      tangent_distance,
                      role,
                      parameter| {
            stroke_vertex_with_role(
                world,
                Vec2::ZERO,
                previous_direction,
                next_direction,
                normal_distance,
                tangent_distance,
                style.miter_limit(),
                role,
                parameter,
                color,
            )
        };
        let start_positive = vertex(
            from,
            previous,
            direction,
            half_width,
            start_tangent,
            start_role,
            1.0,
        );
        let end_positive = vertex(to, direction, next, half_width, end_tangent, end_role, -1.0);
        let end_negative = vertex(
            to,
            direction,
            next,
            -half_width,
            end_tangent,
            end_role,
            -1.0,
        );
        let start_negative = vertex(
            from,
            previous,
            direction,
            -half_width,
            start_tangent,
            start_role,
            1.0,
        );
        vertices.extend_from_slice(&[
            start_positive,
            end_positive,
            end_negative,
            start_positive,
            end_negative,
            start_negative,
        ]);
    }
    for window in points.windows(3) {
        push_logical_join_fill(window[0], window[1], window[2], style, vertices);
    }
    let start_direction = points[1] - points[0];
    if endpoint_has_round_cap(style, true, starts_path) {
        push_logical_round_cap(
            points[0],
            start_direction,
            half_width,
            true,
            color,
            vertices,
        );
    }
    let last = points.len() - 1;
    let end_direction = points[last] - points[last - 1];
    if endpoint_has_round_cap(style, false, ends_path) {
        push_logical_round_cap(
            points[last],
            end_direction,
            half_width,
            false,
            color,
            vertices,
        );
    }
}

fn push_logical_join_fill(
    previous: Vec2,
    center: Vec2,
    next: Vec2,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) {
    let incoming = center - previous;
    let outgoing = next - center;
    let half_width = style.stroke().width() * 0.5;
    let color = style.stroke().color();
    let (corner_role, inner_role) = match style.join() {
        crate::StrokeJoin2d::Bevel => (4.0, 5.0),
        crate::StrokeJoin2d::Miter => (6.0, 7.0),
        crate::StrokeJoin2d::Round => (8.0, 9.0),
    };
    for candidate_side in [-1.0, 1.0] {
        let inner = stroke_vertex_with_role(
            center,
            Vec2::ZERO,
            incoming,
            outgoing,
            -candidate_side * half_width,
            0.0,
            style.miter_limit(),
            inner_role,
            candidate_side,
            color,
        );
        if style.join() == crate::StrokeJoin2d::Round {
            for index in 0..ROUND_CAP_SEGMENTS {
                let start = index as f32 / ROUND_CAP_SEGMENTS as f32;
                let end = (index + 1) as f32 / ROUND_CAP_SEGMENTS as f32;
                vertices.extend_from_slice(&[
                    inner,
                    stroke_vertex_with_role(
                        center,
                        Vec2::ZERO,
                        incoming,
                        outgoing,
                        candidate_side * half_width,
                        0.0,
                        style.miter_limit(),
                        corner_role,
                        start,
                        color,
                    ),
                    stroke_vertex_with_role(
                        center,
                        Vec2::ZERO,
                        incoming,
                        outgoing,
                        candidate_side * half_width,
                        0.0,
                        style.miter_limit(),
                        corner_role,
                        end,
                        color,
                    ),
                ]);
            }
        } else {
            vertices.extend_from_slice(&[
                inner,
                stroke_vertex_with_role(
                    center,
                    Vec2::ZERO,
                    incoming,
                    outgoing,
                    candidate_side * half_width,
                    0.0,
                    style.miter_limit(),
                    corner_role,
                    -1.0,
                    color,
                ),
                stroke_vertex_with_role(
                    center,
                    Vec2::ZERO,
                    incoming,
                    outgoing,
                    candidate_side * half_width,
                    0.0,
                    style.miter_limit(),
                    corner_role,
                    1.0,
                    color,
                ),
            ]);
        }
    }
}

fn push_world_stroke_run(
    points: &[Vec2],
    style: crate::StrokeStyle2d,
    starts_path: bool,
    ends_path: bool,
    vertices: &mut Vec<Vertex>,
) {
    let half_width = style.stroke().width() * 0.5;
    let color = style.stroke().color();
    for (index, pair) in points.windows(2).enumerate() {
        let from = pair[0];
        let to = pair[1];
        let direction = to - from;
        let start_joint = index > 0;
        let end_joint = index + 2 < points.len();
        let start_center = if !start_joint
            && endpoint_tangent(style, true, starts_path) < 0.0
            && !(starts_path && style.start_marker().is_some())
        {
            from - precise_unit(direction) * half_width
        } else {
            from
        };
        let end_center = if !end_joint
            && endpoint_tangent(style, false, ends_path) > 0.0
            && !(ends_path && style.end_marker().is_some())
        {
            to + precise_unit(direction) * half_width
        } else {
            to
        };
        let start_offset = |side| {
            if start_joint {
                world_join_endpoint_offset(
                    from - points[index - 1],
                    direction,
                    side,
                    half_width,
                    style,
                    false,
                )
            } else {
                precise_unit(direction).perp() * (side * half_width)
            }
        };
        let end_offset = |side| {
            if end_joint {
                world_join_endpoint_offset(
                    direction,
                    points[index + 2] - to,
                    side,
                    half_width,
                    style,
                    true,
                )
            } else {
                precise_unit(direction).perp() * (side * half_width)
            }
        };
        let start_trim = if starts_path && index == 0 && style.start_marker().is_some() {
            endpoint_tangent(style, true, true)
        } else {
            0.0
        };
        let end_trim = if ends_path && index + 2 == points.len() && style.end_marker().is_some() {
            endpoint_tangent(style, false, true)
        } else {
            0.0
        };
        let vertex = |world, tangent| {
            stroke_vertex(
                world,
                Vec2::ZERO,
                direction,
                direction,
                0.0,
                tangent,
                1.0,
                color,
            )
        };
        let start_positive = vertex(start_center + start_offset(1.0), start_trim);
        let end_positive = vertex(end_center + end_offset(1.0), end_trim);
        let end_negative = vertex(end_center + end_offset(-1.0), end_trim);
        let start_negative = vertex(start_center + start_offset(-1.0), start_trim);
        vertices.extend_from_slice(&[
            start_positive,
            end_positive,
            end_negative,
            start_positive,
            end_negative,
            start_negative,
        ]);
    }
    for window in points.windows(3) {
        push_world_join_fill(window[0], window[1], window[2], style, vertices);
    }
    let start_direction = points[1] - points[0];
    if endpoint_has_round_cap(style, true, starts_path) {
        push_world_round_cap(
            points[0],
            start_direction,
            half_width,
            true,
            color,
            vertices,
        );
    }
    let last = points.len() - 1;
    let end_direction = points[last] - points[last - 1];
    if endpoint_has_round_cap(style, false, ends_path) {
        push_world_round_cap(
            points[last],
            end_direction,
            half_width,
            false,
            color,
            vertices,
        );
    }
}

fn world_miter_offset(
    incoming: Vec2,
    outgoing: Vec2,
    side: f32,
    half_width: f32,
) -> Option<(Vec2, f32)> {
    let incoming_normal = precise_unit(incoming).perp();
    let outgoing_normal = precise_unit(outgoing).perp();
    let miter = precise_unit(incoming_normal + outgoing_normal);
    let denominator = miter.dot(outgoing_normal);
    (denominator.abs() > 0.001).then(|| {
        (
            miter * (side * half_width / denominator),
            (1.0 / denominator).abs(),
        )
    })
}

fn world_join_endpoint_offset(
    incoming: Vec2,
    outgoing: Vec2,
    side: f32,
    half_width: f32,
    style: crate::StrokeStyle2d,
    incoming_segment: bool,
) -> Vec2 {
    let incoming_unit = precise_unit(incoming);
    let outgoing_unit = precise_unit(outgoing);
    let turn = incoming_unit
        .x
        .mul_add(outgoing_unit.y, -incoming_unit.y * outgoing_unit.x);
    if turn.abs() <= 0.000001 {
        if incoming_unit.dot(outgoing_unit) < 0.0 {
            return Vec2::ZERO;
        }
        return outgoing_unit.perp() * (side * half_width);
    }
    let outer_side = -turn.signum();
    let miter = world_miter_offset(incoming, outgoing, side, half_width);
    if side != outer_side {
        return miter.map_or(Vec2::ZERO, |(offset, multiple)| {
            if multiple <= style.miter_limit() {
                offset
            } else {
                Vec2::ZERO
            }
        });
    }
    if style.join() == crate::StrokeJoin2d::Miter
        && let Some((offset, multiple)) = miter
        && multiple <= style.miter_limit()
    {
        return offset;
    }
    let normal = if incoming_segment {
        incoming_unit.perp()
    } else {
        outgoing_unit.perp()
    };
    normal * (side * half_width)
}

fn push_world_join_fill(
    previous: Vec2,
    center: Vec2,
    next: Vec2,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) {
    let incoming = center - previous;
    let outgoing = next - center;
    let incoming_unit = precise_unit(incoming);
    let outgoing_unit = precise_unit(outgoing);
    let turn = incoming_unit
        .x
        .mul_add(outgoing_unit.y, -incoming_unit.y * outgoing_unit.x);
    if turn.abs() <= 0.000001 {
        return;
    }
    let outer_side = -turn.signum();
    let half_width = style.stroke().width() * 0.5;
    let color = style.stroke().color();
    let miter = world_miter_offset(incoming, outgoing, outer_side, half_width);
    if style.join() == crate::StrokeJoin2d::Miter
        && miter.is_some_and(|(_, multiple)| multiple <= style.miter_limit())
    {
        return;
    }
    let inner = center
        + world_miter_offset(incoming, outgoing, -outer_side, half_width).map_or(
            Vec2::ZERO,
            |(offset, multiple)| {
                if multiple <= style.miter_limit() {
                    offset
                } else {
                    Vec2::ZERO
                }
            },
        );
    let incoming_outer = center + incoming_unit.perp() * (outer_side * half_width);
    let outgoing_outer = center + outgoing_unit.perp() * (outer_side * half_width);
    if style.join() == crate::StrokeJoin2d::Round {
        let start = incoming_unit.perp() * outer_side;
        let finish = outgoing_unit.perp() * outer_side;
        let angle = start
            .x
            .mul_add(finish.y, -start.y * finish.x)
            .atan2(start.dot(finish));
        for index in 0..ROUND_CAP_SEGMENTS {
            let rotate = |amount: f32| {
                let radians = angle * amount;
                Vec2::new(
                    start.x.mul_add(radians.cos(), -start.y * radians.sin()),
                    start.x.mul_add(radians.sin(), start.y * radians.cos()),
                )
            };
            vertices.extend_from_slice(&[
                world_vertex(inner, Vec2::ZERO, color),
                world_vertex(
                    center + rotate(index as f32 / ROUND_CAP_SEGMENTS as f32) * half_width,
                    Vec2::ZERO,
                    color,
                ),
                world_vertex(
                    center + rotate((index + 1) as f32 / ROUND_CAP_SEGMENTS as f32) * half_width,
                    Vec2::ZERO,
                    color,
                ),
            ]);
        }
    } else {
        vertices.extend_from_slice(&[
            world_vertex(inner, Vec2::ZERO, color),
            world_vertex(incoming_outer, Vec2::ZERO, color),
            world_vertex(outgoing_outer, Vec2::ZERO, color),
        ]);
    }
}

fn precise_unit(direction: Vec2) -> Vec2 {
    let length = f64::from(direction.x).hypot(f64::from(direction.y));
    if length == 0.0 || !length.is_finite() {
        Vec2::ZERO
    } else {
        Vec2::new(
            (f64::from(direction.x) / length) as f32,
            (f64::from(direction.y) / length) as f32,
        )
    }
}

fn push_logical_round_cap(
    center: Vec2,
    direction: Vec2,
    radius: f32,
    start: bool,
    color: Color,
    vertices: &mut Vec<Vertex>,
) {
    let start_angle = if start {
        std::f32::consts::FRAC_PI_2
    } else {
        -std::f32::consts::FRAC_PI_2
    };
    let center_vertex = stroke_vertex(
        center,
        Vec2::ZERO,
        direction,
        direction,
        0.0,
        0.0,
        1.0,
        color,
    );
    for index in 0..ROUND_CAP_SEGMENTS {
        let amount_start = index as f32 / ROUND_CAP_SEGMENTS as f32;
        let amount_end = (index + 1) as f32 / ROUND_CAP_SEGMENTS as f32;
        let angle_start = start_angle + amount_start * std::f32::consts::PI;
        let angle_end = start_angle + amount_end * std::f32::consts::PI;
        vertices.push(center_vertex);
        for angle in [angle_start, angle_end] {
            vertices.push(stroke_vertex(
                center,
                Vec2::ZERO,
                direction,
                direction,
                angle.sin() * radius,
                angle.cos() * radius,
                1.0,
                color,
            ));
        }
    }
}

fn push_world_round_cap(
    center: Vec2,
    direction: Vec2,
    radius: f32,
    start: bool,
    color: Color,
    vertices: &mut Vec<Vertex>,
) {
    let tangent = precise_unit(direction);
    let normal = tangent.perp();
    let start_angle = if start {
        std::f32::consts::FRAC_PI_2
    } else {
        -std::f32::consts::FRAC_PI_2
    };
    let center_vertex = world_vertex(center, Vec2::ZERO, color);
    for index in 0..ROUND_CAP_SEGMENTS {
        let amount_start = index as f32 / ROUND_CAP_SEGMENTS as f32;
        let amount_end = (index + 1) as f32 / ROUND_CAP_SEGMENTS as f32;
        let angle_start = start_angle + amount_start * std::f32::consts::PI;
        let angle_end = start_angle + amount_end * std::f32::consts::PI;
        vertices.push(center_vertex);
        for angle in [angle_start, angle_end] {
            let offset = tangent * (angle.cos() * radius) + normal * (angle.sin() * radius);
            vertices.push(world_vertex(center + offset, Vec2::ZERO, color));
        }
    }
}

fn push_stroke_marker(
    center: Vec2,
    direction: Vec2,
    start: bool,
    marker: crate::StrokeMarker2d,
    color: Color,
    vertices: &mut Vec<Vertex>,
) {
    let tip_tangent = if start {
        -marker.length().get()
    } else {
        marker.length().get()
    };
    let half_width = marker.width().get() * 0.5;
    vertices.extend_from_slice(&[
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            0.0,
            tip_tangent,
            1.0,
            color,
        ),
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            half_width,
            0.0,
            1.0,
            color,
        ),
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            -half_width,
            0.0,
            1.0,
            color,
        ),
    ]);
}

fn push_circle_fill_world(
    center_world: Vec2,
    radius_world: f32,
    fill: Fill,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if radius_world <= 0.0 {
        return;
    }

    let center_vertex = world_vertex(center_world, screen_offset, fill.color_at(center_world));

    for pair in unit_circle_points().windows(2) {
        let world_start = center_world + pair[0] * radius_world;
        let world_end = center_world + pair[1] * radius_world;

        vertices.push(center_vertex);
        vertices.push(world_vertex(
            world_start,
            screen_offset,
            fill.color_at(world_start),
        ));
        vertices.push(world_vertex(
            world_end,
            screen_offset,
            fill.color_at(world_end),
        ));
    }
}

fn circle_world_points(
    center_world: Vec2,
    radius_world: f32,
) -> Result<Vec<Vec2>, TessellationError> {
    let mut points = Vec::new();
    reserve_items(&mut points, CIRCLE_SEGMENTS + 1)?;
    for point in unit_circle_points() {
        points.push(center_world + *point * radius_world);
    }
    Ok(points)
}

fn unit_circle_points() -> &'static [Vec2] {
    static POINTS: OnceLock<[Vec2; CIRCLE_SEGMENTS + 1]> = OnceLock::new();
    POINTS.get_or_init(|| {
        std::array::from_fn(|index| {
            let angle = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
            Vec2::new(angle.cos(), angle.sin())
        })
    })
}

fn unit_quarter_circle_points() -> &'static [Vec2] {
    static POINTS: OnceLock<[Vec2; CORNER_SEGMENTS + 1]> = OnceLock::new();
    POINTS.get_or_init(|| {
        std::array::from_fn(|step| {
            let angle = step as f32 / CORNER_SEGMENTS as f32 * std::f32::consts::FRAC_PI_2;
            Vec2::new(angle.cos(), angle.sin())
        })
    })
}

fn push_rect_world(
    rect: Rect,
    corner_radius: f32,
    fill: Fill,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    let points = rounded_rect_points(rect, corner_radius)?;
    if points.len() < 3 {
        return Ok(());
    }

    let center = rect.center();
    for index in 0..points.len() - 1 {
        vertices.push(world_vertex(
            center,
            screen_offset,
            fill.color_at(rect.center()),
        ));
        vertices.push(world_vertex(
            points[index],
            screen_offset,
            fill.color_at(points[index]),
        ));
        vertices.push(world_vertex(
            points[index + 1],
            screen_offset,
            fill.color_at(points[index + 1]),
        ));
    }
    Ok(())
}

fn rounded_rect_points(rect: Rect, corner_radius: f32) -> Result<Vec<Vec2>, TessellationError> {
    let radius = corner_radius
        .max(0.0)
        .min(rect.width().abs() * 0.5)
        .min(rect.height().abs() * 0.5);

    if radius <= f32::EPSILON {
        return Ok(vec![
            Vec2::new(rect.max.x, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y),
            Vec2::new(rect.min.x, rect.max.y),
            Vec2::new(rect.min.x, rect.min.y),
            Vec2::new(rect.max.x, rect.min.y),
        ]);
    }

    let corners = [
        Vec2::new(rect.max.x - radius, rect.max.y - radius),
        Vec2::new(rect.min.x + radius, rect.max.y - radius),
        Vec2::new(rect.min.x + radius, rect.min.y + radius),
        Vec2::new(rect.max.x - radius, rect.min.y + radius),
    ];

    let mut points = Vec::new();
    reserve_items(&mut points, CORNER_SEGMENTS * 4 + 1)?;
    for (corner_index, center) in corners.into_iter().enumerate() {
        for unit in unit_quarter_circle_points() {
            let unit = match corner_index {
                0 => *unit,
                1 => Vec2::new(-unit.y, unit.x),
                2 => Vec2::new(-unit.x, -unit.y),
                _ => Vec2::new(unit.y, -unit.x),
            };
            points.push(center + unit * radius);
        }
    }
    points.push(points[0]);

    Ok(points)
}

fn push_closed_polyline_world(
    points: &[Vec2],
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) -> Result<(), TessellationError> {
    if points.len() < 4 || width <= 0.0 || !width.is_finite() {
        return Ok(());
    }

    let mut unique_points = Vec::new();
    reserve_items(&mut unique_points, points.len() - 1)?;
    for point in &points[..points.len() - 1] {
        if unique_points
            .last()
            .is_none_or(|previous| (*point - *previous).length_squared() > f32::EPSILON)
        {
            unique_points.push(*point);
        }
    }
    if unique_points.len() > 1 {
        let first = unique_points[0];
        let last = unique_points[unique_points.len() - 1];
        if (first - last).length_squared() <= f32::EPSILON {
            unique_points.pop();
        }
    }
    if unique_points.len() < 3 {
        return Ok(());
    }

    let point_count = unique_points.len();
    let half_width = width * 0.5;
    for index in 0..point_count {
        let next = (index + 1) % point_count;
        let previous = (index + point_count - 1) % point_count;
        let after_next = (next + 1) % point_count;
        let current_previous_direction = unique_points[index] - unique_points[previous];
        let current_next_direction = unique_points[next] - unique_points[index];
        let next_next_direction = unique_points[after_next] - unique_points[next];

        vertices.push(legacy_stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(legacy_stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            half_width,
            color,
        ));
        vertices.push(legacy_stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(legacy_stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(legacy_stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(legacy_stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            -half_width,
            color,
        ));
    }
    Ok(())
}

pub(super) fn world_vertex(world: Vec2, screen_offset: Vec2, color: Color) -> Vertex {
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [0.0; 2],
        next_direction: [0.0; 2],
        normal_distance: 0.0,
        tangent_distance: 0.0,
        miter_limit: 1.0,
        stroke_role: 0.0,
        stroke_parameter: 0.0,
        color: color.to_array(),
    }
}

fn legacy_stroke_vertex(
    world: Vec2,
    screen_offset: Vec2,
    previous_direction: Vec2,
    next_direction: Vec2,
    normal_distance: f32,
    color: Color,
) -> Vertex {
    stroke_vertex(
        world,
        screen_offset,
        previous_direction,
        next_direction,
        normal_distance,
        0.0,
        1_000.0,
        color,
    )
}

#[allow(clippy::too_many_arguments)]
fn stroke_vertex(
    world: Vec2,
    screen_offset: Vec2,
    previous_direction: Vec2,
    next_direction: Vec2,
    normal_distance: f32,
    tangent_distance: f32,
    miter_limit: f32,
    color: Color,
) -> Vertex {
    stroke_vertex_with_role(
        world,
        screen_offset,
        previous_direction,
        next_direction,
        normal_distance,
        tangent_distance,
        miter_limit,
        0.0,
        0.0,
        color,
    )
}

#[allow(clippy::too_many_arguments)]
fn stroke_vertex_with_role(
    world: Vec2,
    screen_offset: Vec2,
    previous_direction: Vec2,
    next_direction: Vec2,
    normal_distance: f32,
    tangent_distance: f32,
    miter_limit: f32,
    stroke_role: f32,
    stroke_parameter: f32,
    color: Color,
) -> Vertex {
    let previous_direction = precise_unit(previous_direction);
    let next_direction = precise_unit(next_direction);
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [previous_direction.x, previous_direction.y],
        next_direction: [next_direction.x, next_direction.y],
        normal_distance,
        tangent_distance,
        miter_limit,
        stroke_role,
        stroke_parameter,
        color: color.to_array(),
    }
}
