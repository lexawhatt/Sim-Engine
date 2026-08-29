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
        ..TessellationStats::default()
    };
    for scene_command in scene.commands() {
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
            DrawCommand::Line(line) => tessellate_line(line, vertices),
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
            continue;
        }
        if vertex_end == vertex_start {
            stats.dropped_command_count += 1;
            continue;
        }

        stats.rendered_command_count += 1;

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

fn tessellate_line(line: &Line, vertices: &mut Vec<Vertex>) {
    push_round_line_world(
        line.from(),
        line.to(),
        line.stroke().width(),
        line.stroke().color(),
        Vec2::ZERO,
        vertices,
    );
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

    let width = polyline.stroke().width();
    let color = polyline.stroke().color();
    let half_width = width * 0.5;
    for index in 0..points.len() - 1 {
        let from = points[index];
        let to = points[index + 1];
        let current_direction = to - from;
        let previous_direction = index
            .checked_sub(1)
            .map_or(current_direction, |previous| from - points[previous]);
        let next_direction = points
            .get(index + 2)
            .map_or(current_direction, |next| *next - to);

        vertices.push(stroke_vertex(
            from,
            Vec2::ZERO,
            previous_direction,
            current_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            to,
            Vec2::ZERO,
            current_direction,
            next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            to,
            Vec2::ZERO,
            current_direction,
            next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            from,
            Vec2::ZERO,
            previous_direction,
            current_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            to,
            Vec2::ZERO,
            current_direction,
            next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            from,
            Vec2::ZERO,
            previous_direction,
            current_direction,
            -half_width,
            color,
        ));
    }

    let radius = half_width;
    push_circle_screen_at_world(points[0], radius, color, Vec2::ZERO, vertices);
    push_circle_screen_at_world(
        points[points.len() - 1],
        radius,
        color,
        Vec2::ZERO,
        vertices,
    );
    Ok(())
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

fn push_round_line_world(
    from: Vec2,
    to: Vec2,
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if push_line_body_world(from, to, width, color, screen_offset, vertices) {
        let radius = width * 0.5;
        push_circle_screen_at_world(from, radius, color, screen_offset, vertices);
        push_circle_screen_at_world(to, radius, color, screen_offset, vertices);
    }
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

        vertices.push(stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
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

fn push_line_body_world(
    from: Vec2,
    to: Vec2,
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) -> bool {
    if width <= 0.0 {
        return false;
    }

    let delta = to - from;
    if !delta.is_finite() || (delta.x == 0.0 && delta.y == 0.0) {
        return false;
    }

    let half_width = width * 0.5;
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    true
}

pub(super) fn world_vertex(world: Vec2, screen_offset: Vec2, color: Color) -> Vertex {
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [0.0; 2],
        next_direction: [0.0; 2],
        normal_distance: 0.0,
        color: color.to_array(),
    }
}

fn stroke_vertex(
    world: Vec2,
    screen_offset: Vec2,
    previous_direction: Vec2,
    next_direction: Vec2,
    normal_distance: f32,
    color: Color,
) -> Vertex {
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [previous_direction.x, previous_direction.y],
        next_direction: [next_direction.x, next_direction.y],
        normal_distance,
        color: color.to_array(),
    }
}
