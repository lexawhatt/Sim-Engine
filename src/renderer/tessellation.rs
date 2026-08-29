use super::*;

pub(super) fn tessellate_scene(
    scene: &Scene,
    vertices: &mut Vec<Vertex>,
    draw_batches: &mut Vec<PreparedDrawBatch>,
) -> Result<TessellationStats, TessellationError> {
    let estimate = scene.statistics();
    reserve_items(
        vertices,
        estimate
            .estimated_tessellated_vertices()
            .saturating_sub(vertices.len()),
    )?;
    reserve_items(
        draw_batches,
        estimate
            .estimated_draw_batches()
            .saturating_sub(draw_batches.len()),
    )?;
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
    stats.vertex_count = vertices.len();
    stats.draw_batch_count = draw_batches.len();
    stats.upload_bytes = vertices.len().saturating_mul(std::mem::size_of::<Vertex>());
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
    let width = style.stroke().width();
    let color = style.stroke().color();
    if let Some(dash) = style.dash_pattern() {
        visit_dash_stroke(points, dash, |event| match event {
            DashStrokeEvent::Segment {
                from,
                to,
                start_cap,
                end_cap,
            } => push_stroke_segment(from, to, style, start_cap, end_cap, vertices),
            DashStrokeEvent::Join {
                previous,
                center,
                next,
            } => push_stroke_join(previous, center, next, style, vertices),
        });
    } else if style.width_mode() == crate::StrokeWidthMode2d::LogicalPixels
        && style.join() == crate::StrokeJoin2d::Miter
    {
        for (index, pair) in points.windows(2).enumerate() {
            push_logical_miter_segment(points, index, pair[0], pair[1], style, vertices);
        }
    } else {
        for (index, pair) in points.windows(2).enumerate() {
            let from = pair[0];
            let to = pair[1];
            let start_cap = index == 0;
            let end_cap = index + 2 == points.len();
            push_stroke_segment(from, to, style, start_cap, end_cap, vertices);
            if !end_cap {
                push_stroke_join(from, to, points[index + 2], style, vertices);
            }
        }
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

    debug_assert!(width.is_finite() && width > 0.0);
    Ok(())
}

fn push_logical_miter_segment(
    points: &[Vec2],
    index: usize,
    from: Vec2,
    to: Vec2,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) {
    let direction = to - from;
    let previous = index
        .checked_sub(1)
        .map_or(direction, |previous| from - points[previous]);
    let next = points.get(index + 2).map_or(direction, |next| *next - to);
    let start_cap = index == 0;
    let end_cap = index + 2 == points.len();
    let half_width = style.stroke().width() * 0.5;
    let start_tangent = if start_cap && style.cap() == crate::StrokeCap2d::Square {
        -half_width
    } else {
        0.0
    };
    let end_tangent = if end_cap && style.cap() == crate::StrokeCap2d::Square {
        half_width
    } else {
        0.0
    };
    let color = style.stroke().color();
    let vertex = |world, previous_direction, next_direction, normal_distance, tangent_distance| {
        stroke_vertex(
            world,
            Vec2::ZERO,
            previous_direction,
            next_direction,
            normal_distance,
            tangent_distance,
            style.miter_limit(),
            color,
        )
    };
    let start_positive = vertex(from, previous, direction, half_width, start_tangent);
    let end_positive = vertex(to, direction, next, half_width, end_tangent);
    let end_negative = vertex(to, direction, next, -half_width, end_tangent);
    let start_negative = vertex(from, previous, direction, -half_width, start_tangent);
    vertices.extend_from_slice(&[
        start_positive,
        end_positive,
        end_negative,
        start_positive,
        end_negative,
        start_negative,
    ]);
    if style.cap() == crate::StrokeCap2d::Round {
        if start_cap {
            push_logical_round_cap(from, direction, half_width, true, color, vertices);
        }
        if end_cap {
            push_logical_round_cap(to, direction, half_width, false, color, vertices);
        }
    }
}

fn push_stroke_segment(
    from: Vec2,
    to: Vec2,
    style: crate::StrokeStyle2d,
    start_cap: bool,
    end_cap: bool,
    vertices: &mut Vec<Vertex>,
) {
    let width = style.stroke().width();
    let color = style.stroke().color();
    let half_width = width * 0.5;
    let direction = to - from;
    match style.width_mode() {
        crate::StrokeWidthMode2d::LogicalPixels => {
            let start_tangent = if start_cap && style.cap() == crate::StrokeCap2d::Square {
                -half_width
            } else {
                0.0
            };
            let end_tangent = if end_cap && style.cap() == crate::StrokeCap2d::Square {
                half_width
            } else {
                0.0
            };
            push_logical_stroke_quad(
                from,
                to,
                direction,
                half_width,
                start_tangent,
                end_tangent,
                color,
                vertices,
            );
            if style.cap() == crate::StrokeCap2d::Round {
                if start_cap {
                    push_logical_round_cap(from, direction, half_width, true, color, vertices);
                }
                if end_cap {
                    push_logical_round_cap(to, direction, half_width, false, color, vertices);
                }
            }
        }
        crate::StrokeWidthMode2d::WorldUnits => {
            push_world_stroke_quad(
                from,
                to,
                half_width,
                start_cap && style.cap() == crate::StrokeCap2d::Square,
                end_cap && style.cap() == crate::StrokeCap2d::Square,
                color,
                vertices,
            );
            if style.cap() == crate::StrokeCap2d::Round {
                if start_cap {
                    push_world_round_cap(from, direction, half_width, true, color, vertices);
                }
                if end_cap {
                    push_world_round_cap(to, direction, half_width, false, color, vertices);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DashStrokeEvent {
    Segment {
        from: Vec2,
        to: Vec2,
        start_cap: bool,
        end_cap: bool,
    },
    Join {
        previous: Vec2,
        center: Vec2,
        next: Vec2,
    },
}

fn visit_dash_stroke(
    points: &[Vec2],
    dash: crate::StrokeDashPattern2d,
    mut visit: impl FnMut(DashStrokeEvent),
) {
    let lengths = dash.lengths();
    let total: f64 = lengths.iter().map(|length| f64::from(*length)).sum();
    let mut phase = f64::from(dash.phase()).rem_euclid(total);
    let mut pattern_index = 0usize;
    while phase >= f64::from(lengths[pattern_index]) {
        phase -= f64::from(lengths[pattern_index]);
        pattern_index = (pattern_index + 1) % lengths.len();
    }
    let mut pattern_remaining = f64::from(lengths[pattern_index]) - phase;
    let mut visible_run_start = pattern_index.is_multiple_of(2);

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
                visit(DashStrokeEvent::Segment {
                    from: segment_point(from, to, segment_consumed / segment_length),
                    to: segment_point(from, to, (segment_consumed + consumed) / segment_length),
                    start_cap: visible_run_start,
                    end_cap: finishes_pattern || finishes_path,
                });
                visible_run_start = false;
                if finishes_segment && !finishes_path && !finishes_pattern {
                    visit(DashStrokeEvent::Join {
                        previous: from,
                        center: to,
                        next: points[segment_index + 2],
                    });
                }
            }
            segment_consumed += consumed;
            pattern_remaining -= consumed;
            if pattern_remaining <= 0.0 {
                pattern_index = (pattern_index + 1) % lengths.len();
                pattern_remaining = f64::from(lengths[pattern_index]);
                visible_run_start = pattern_index.is_multiple_of(2);
            }
        }
    }
}

fn segment_point(from: Vec2, to: Vec2, amount: f64) -> Vec2 {
    Vec2::new(
        (f64::from(from.x) * (1.0 - amount) + f64::from(to.x) * amount) as f32,
        (f64::from(from.y) * (1.0 - amount) + f64::from(to.y) * amount) as f32,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_logical_stroke_quad(
    from: Vec2,
    to: Vec2,
    direction: Vec2,
    half_width: f32,
    start_tangent: f32,
    end_tangent: f32,
    color: Color,
    vertices: &mut Vec<Vertex>,
) {
    let vertex = |world, normal_distance, tangent_distance| {
        stroke_vertex(
            world,
            Vec2::ZERO,
            direction,
            direction,
            normal_distance,
            tangent_distance,
            1.0,
            color,
        )
    };
    let start_positive = vertex(from, half_width, start_tangent);
    let end_positive = vertex(to, half_width, end_tangent);
    let end_negative = vertex(to, -half_width, end_tangent);
    let start_negative = vertex(from, -half_width, start_tangent);
    vertices.extend_from_slice(&[
        start_positive,
        end_positive,
        end_negative,
        start_positive,
        end_negative,
        start_negative,
    ]);
}

fn push_world_stroke_quad(
    from: Vec2,
    to: Vec2,
    half_width: f32,
    extend_start: bool,
    extend_end: bool,
    color: Color,
    vertices: &mut Vec<Vertex>,
) {
    let unit = precise_unit(to - from);
    let normal = unit.perp() * half_width;
    let start = from - unit * if extend_start { half_width } else { 0.0 };
    let end = to + unit * if extend_end { half_width } else { 0.0 };
    let start_positive = world_vertex(start + normal, Vec2::ZERO, color);
    let end_positive = world_vertex(end + normal, Vec2::ZERO, color);
    let end_negative = world_vertex(end - normal, Vec2::ZERO, color);
    let start_negative = world_vertex(start - normal, Vec2::ZERO, color);
    vertices.extend_from_slice(&[
        start_positive,
        end_positive,
        end_negative,
        start_positive,
        end_negative,
        start_negative,
    ]);
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

fn push_stroke_join(
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
    match style.width_mode() {
        crate::StrokeWidthMode2d::LogicalPixels => match style.join() {
            crate::StrokeJoin2d::Round => {
                push_circle_screen_at_world(center, half_width, color, Vec2::ZERO, vertices);
            }
            crate::StrokeJoin2d::Bevel | crate::StrokeJoin2d::Miter => {
                push_logical_join(center, incoming, outgoing, half_width, style, vertices);
            }
        },
        crate::StrokeWidthMode2d::WorldUnits => match style.join() {
            crate::StrokeJoin2d::Round => {
                push_circle_fill_world(center, half_width, Fill::Solid(color), Vec2::ZERO, vertices)
            }
            crate::StrokeJoin2d::Bevel | crate::StrokeJoin2d::Miter => {
                push_world_join(center, incoming, outgoing, half_width, style, vertices);
            }
        },
    }
}

fn push_logical_join(
    center: Vec2,
    incoming: Vec2,
    outgoing: Vec2,
    half_width: f32,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) {
    let color = style.stroke().color();
    let center_vertex = stroke_vertex(center, Vec2::ZERO, incoming, outgoing, 0.0, 0.0, 1.0, color);
    for side in [-1.0, 1.0] {
        let incoming_corner = stroke_vertex(
            center,
            Vec2::ZERO,
            incoming,
            incoming,
            side * half_width,
            0.0,
            1.0,
            color,
        );
        let outgoing_corner = stroke_vertex(
            center,
            Vec2::ZERO,
            outgoing,
            outgoing,
            side * half_width,
            0.0,
            1.0,
            color,
        );
        if style.join() == crate::StrokeJoin2d::Miter {
            let miter = stroke_vertex(
                center,
                Vec2::ZERO,
                incoming,
                outgoing,
                side * half_width,
                0.0,
                style.miter_limit(),
                color,
            );
            vertices.extend_from_slice(&[
                center_vertex,
                incoming_corner,
                miter,
                center_vertex,
                miter,
                outgoing_corner,
            ]);
        } else {
            vertices.extend_from_slice(&[center_vertex, incoming_corner, outgoing_corner]);
        }
    }
}

fn push_world_join(
    center: Vec2,
    incoming: Vec2,
    outgoing: Vec2,
    half_width: f32,
    style: crate::StrokeStyle2d,
    vertices: &mut Vec<Vertex>,
) {
    let incoming_normal = precise_unit(incoming).perp();
    let outgoing_normal = precise_unit(outgoing).perp();
    let color = style.stroke().color();
    let center_vertex = world_vertex(center, Vec2::ZERO, color);
    for side in [-1.0, 1.0] {
        let incoming_corner = center + incoming_normal * (side * half_width);
        let outgoing_corner = center + outgoing_normal * (side * half_width);
        if style.join() == crate::StrokeJoin2d::Miter {
            let combined = incoming_normal + outgoing_normal;
            let miter_direction = precise_unit(combined);
            let denominator = miter_direction.dot(outgoing_normal);
            let multiple = if denominator.abs() > 0.001 {
                (1.0 / denominator).abs()
            } else {
                f32::INFINITY
            };
            let miter = if multiple <= style.miter_limit() {
                center + miter_direction * (side * half_width / denominator)
            } else {
                outgoing_corner
            };
            vertices.extend_from_slice(&[
                center_vertex,
                world_vertex(incoming_corner, Vec2::ZERO, color),
                world_vertex(miter, Vec2::ZERO, color),
                center_vertex,
                world_vertex(miter, Vec2::ZERO, color),
                world_vertex(outgoing_corner, Vec2::ZERO, color),
            ]);
        } else {
            vertices.extend_from_slice(&[
                center_vertex,
                world_vertex(incoming_corner, Vec2::ZERO, color),
                world_vertex(outgoing_corner, Vec2::ZERO, color),
            ]);
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
    let base_tangent = if start {
        marker.length().get()
    } else {
        -marker.length().get()
    };
    let half_width = marker.width().get() * 0.5;
    vertices.extend_from_slice(&[
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            0.0,
            0.0,
            1.0,
            color,
        ),
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            half_width,
            base_tangent,
            1.0,
            color,
        ),
        stroke_vertex(
            center,
            Vec2::ZERO,
            direction,
            direction,
            -half_width,
            base_tangent,
            1.0,
            color,
        ),
    ]);
}

fn push_circle_screen_at_world(
    center_world: Vec2,
    radius: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if radius <= 0.0 {
        return;
    }

    let center_vertex = world_vertex(center_world, screen_offset, color);
    for index in 0..ROUND_CAP_SEGMENTS {
        let angle_start = index as f32 / ROUND_CAP_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / ROUND_CAP_SEGMENTS as f32 * std::f32::consts::TAU;
        vertices.push(center_vertex);
        vertices.push(world_vertex(
            center_world,
            screen_offset + Vec2::new(angle_start.cos(), -angle_start.sin()) * radius,
            color,
        ));
        vertices.push(world_vertex(
            center_world,
            screen_offset + Vec2::new(angle_end.cos(), -angle_end.sin()) * radius,
            color,
        ));
    }
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

    for index in 0..CIRCLE_SEGMENTS {
        let angle_start = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let world_start =
            center_world + Vec2::new(angle_start.cos(), angle_start.sin()) * radius_world;
        let world_end = center_world + Vec2::new(angle_end.cos(), angle_end.sin()) * radius_world;

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
    for index in 0..=CIRCLE_SEGMENTS {
        let angle = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        points.push(center_world + Vec2::new(angle.cos(), angle.sin()) * radius_world);
    }
    Ok(points)
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
        (
            Vec2::new(rect.max.x - radius, rect.max.y - radius),
            0.0,
            std::f32::consts::FRAC_PI_2,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.max.y - radius),
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.min.y + radius),
            std::f32::consts::PI,
            std::f32::consts::PI * 1.5,
        ),
        (
            Vec2::new(rect.max.x - radius, rect.min.y + radius),
            std::f32::consts::PI * 1.5,
            std::f32::consts::TAU,
        ),
    ];

    let mut points = Vec::new();
    reserve_items(&mut points, CORNER_SEGMENTS * 4 + 1)?;
    for (center, start_angle, end_angle) in corners {
        for step in 0..=CORNER_SEGMENTS {
            let amount = step as f32 / CORNER_SEGMENTS as f32;
            let angle = start_angle + (end_angle - start_angle) * amount;
            points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
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
        color: color.to_array(),
    }
}
