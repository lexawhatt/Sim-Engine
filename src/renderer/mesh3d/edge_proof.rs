//! Conservative homogeneous edge clipping and raster arithmetic.

use super::*;

pub(super) type ClipPlaneRanges = [(f64, f64); 6];
pub(super) type EdgeClipPlaneRanges = [ClipPlaneRanges; 2];

pub(super) type ClipComponentRange = (f64, f64);
pub(super) type ClipPointRanges = [ClipComponentRange; 4];
pub(super) type EdgeClipRanges = [ClipPointRanges; 2];

#[derive(Clone, Copy)]
pub(super) struct NormalizedEdgeRanges {
    pub(super) clip: EdgeClipRanges,
    pub(super) planes: EdgeClipPlaneRanges,
}

pub(super) fn clip_ranges_share_outside_plane(
    start: ClipPlaneRanges,
    end: ClipPlaneRanges,
) -> bool {
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    start
        .into_iter()
        .zip(end)
        .any(|(start, end)| start.1 <= -minimum_normal && end.1 <= -minimum_normal)
}

pub(super) fn validate_edge_homogeneous_classification(
    start: [ShaderValueRange; 4],
    end: [ShaderValueRange; 4],
) -> Result<NormalizedEdgeRanges, Mesh3dRenderError> {
    let minimum_magnitude = start
        .into_iter()
        .chain(end)
        .map(shader_range_minimum_magnitude)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let maximum_magnitude = start
        .into_iter()
        .chain(end)
        .map(|value| value.minimum.abs().max(value.maximum.abs()))
        .fold(1.0_f64, f64::max);
    if !minimum_magnitude.is_finite() || !maximum_magnitude.is_finite() {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let scale = wgsl_division_range(
        (1.0, 1.0),
        (minimum_magnitude.max(1.0), maximum_magnitude.max(1.0)),
    )
    .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;

    let mut normalized_clip = [[(0.0, 0.0); 4]; 2];
    let mut normalized_planes = [[(0.0, 0.0); 6]; 2];
    for (endpoint_index, clip) in [start, end].into_iter().enumerate() {
        let normalized = clip.map(|component| {
            rounded_f32_product_range((component.minimum, component.maximum), scale)
        });
        if normalized.iter().any(Option::is_none) {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        let normalized = normalized.map(Option::unwrap);
        normalized_clip[endpoint_index] = normalized;
        let x = normalized[0];
        let y = normalized[1];
        let z = normalized[2];
        let w = normalized[3];
        let planes = [
            rounded_f32_add_range(w, x, false),
            rounded_f32_add_range(w, x, true),
            rounded_f32_add_range(w, y, false),
            rounded_f32_add_range(w, y, true),
            Some(z),
            rounded_f32_add_range(w, z, true),
        ];
        if planes.iter().any(Option::is_none) {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        let planes = planes.map(Option::unwrap);
        normalized_planes[endpoint_index] = planes;
    }
    validate_normalized_plane_sides([start, end], normalized_planes)?;
    Ok(NormalizedEdgeRanges {
        clip: normalized_clip,
        planes: normalized_planes,
    })
}

pub(super) fn validate_normalized_plane_sides(
    raw_clip: [[ShaderValueRange; 4]; 2],
    normalized_planes: EdgeClipPlaneRanges,
) -> Result<(), Mesh3dRenderError> {
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    for (raw_clip, normalized_planes) in raw_clip.into_iter().zip(normalized_planes) {
        for (raw, normalized) in clip_plane_ranges(raw_clip)?
            .into_iter()
            .zip(normalized_planes)
        {
            let raw_inside = raw.0 >= 0.0;
            let raw_outside = raw.1 <= -minimum_normal;
            let normalized_inside = normalized.0 >= 0.0;
            let normalized_outside = normalized.1 <= -minimum_normal;
            if (raw_inside && !normalized_inside) || (raw_outside && !normalized_outside) {
                // Surface clipping classifies the unscaled homogeneous
                // coordinates. The edge shader scales every component in f32
                // first and only then evaluates the plane. Reject any case in
                // which those separately rounded operations can change side,
                // including a negative distance becoming inclusive `-0`.
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
        }
    }
    Ok(())
}

pub(super) fn interval_clip_edge_to_frustum(
    normalized: NormalizedEdgeRanges,
) -> Result<Option<EdgeClipRanges>, Mesh3dRenderError> {
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    let mut enter = (0.0_f64, 0.0_f64);
    let mut exit = (1.0_f64, 1.0_f64);
    for (start_distance, end_distance) in normalized.planes[0].into_iter().zip(normalized.planes[1])
    {
        let start_outside = start_distance.1 <= -minimum_normal;
        let end_outside = end_distance.1 <= -minimum_normal;
        if start_outside && end_outside {
            return Ok(None);
        }
        if start_outside || end_outside {
            let denominator = rounded_f32_add_range(start_distance, end_distance, true)
                .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
            let amount = wgsl_signed_division_range(start_distance, denominator)
                .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
            if amount.0 < 0.0 || amount.1 > 1.0 {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
            if start_outside {
                enter = (enter.0.max(amount.0), enter.1.max(amount.1));
            } else {
                exit = (exit.0.min(amount.0), exit.1.min(amount.1));
            }
        }
    }
    if enter.0 > exit.1 {
        return Ok(None);
    }
    if enter.1 > exit.0 {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let clipped_start = std::array::from_fn(|axis| {
        interval_lerp_range(normalized.clip[0][axis], normalized.clip[1][axis], enter)
    });
    let clipped_end = std::array::from_fn(|axis| {
        interval_lerp_range(normalized.clip[0][axis], normalized.clip[1][axis], exit)
    });
    if clipped_start.iter().any(Option::is_none) || clipped_end.iter().any(Option::is_none) {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let clipped = [
        clipped_start.map(Option::unwrap),
        clipped_end.map(Option::unwrap),
    ];
    if clipped[0][3].0 < minimum_normal || clipped[1][3].0 < minimum_normal {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    Ok(Some(clipped))
}

pub(super) fn projected_screen_axis_range(
    numerator: (f64, f64),
    denominator: (f64, f64),
    dimension: f32,
    inverted: bool,
) -> Option<(f64, f64)> {
    let ndc = wgsl_division_range(numerator, denominator)?;
    let half_ndc = rounded_f32_product_range(ndc, (0.5, 0.5))?;
    let normalized = rounded_f32_add_range((0.5, 0.5), half_ndc, inverted)?;
    rounded_f32_product_range(normalized, {
        let dimension = f64::from(dimension.max(1.0));
        (dimension, dimension)
    })
}

pub(super) fn validate_edge_projection_range(
    start: [(f64, f64); 4],
    end: [(f64, f64); 4],
    viewport: [f32; 4],
    fixed_delta: [f32; 2],
) -> Result<(), Mesh3dRenderError> {
    let start_x = projected_screen_axis_range(start[0], start[3], viewport[0], false)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let start_y = projected_screen_axis_range(start[1], start[3], viewport[1], true)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let end_x = projected_screen_axis_range(end[0], end[3], viewport[0], false)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let end_y = projected_screen_axis_range(end[1], end[3], viewport[1], true)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let delta_x = rounded_f32_add_range(end_x, start_x, true)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let delta_y = rounded_f32_add_range(end_y, start_y, true)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let component_minimum = |range: (f64, f64)| {
        if range.0 > 0.0 {
            range.0
        } else if range.1 < 0.0 {
            -range.1
        } else {
            0.0
        }
    };
    let minimum_length = component_minimum(delta_x).hypot(component_minimum(delta_y));
    let maximum_length = delta_x
        .0
        .abs()
        .max(delta_x.1.abs())
        .hypot(delta_y.0.abs().max(delta_y.1.abs()));
    let extrusion_threshold = 0.0001_f64;
    if !minimum_length.is_finite()
        || !maximum_length.is_finite()
        || (minimum_length <= extrusion_threshold && maximum_length > extrusion_threshold)
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    if maximum_length <= extrusion_threshold {
        return Ok(());
    }

    let fixed_x = f64::from(fixed_delta[0]);
    let fixed_y = f64::from(fixed_delta[1]);
    let fixed_length = fixed_x.hypot(fixed_y);
    let minimum_dot = fixed_x * if fixed_x >= 0.0 { delta_x.0 } else { delta_x.1 }
        + fixed_y * if fixed_y >= 0.0 { delta_y.0 } else { delta_y.1 };
    if !fixed_length.is_finite()
        || fixed_length <= extrusion_threshold
        || !minimum_dot.is_finite()
        || minimum_dot <= 0.0
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    Ok(())
}

pub(super) fn validate_edge_projection(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
    style: WireframeStyle3d,
    viewport: [f32; 4],
) -> Result<(), Mesh3dRenderError> {
    let style_sources = [
        style.visible_width().get(),
        style.hidden_width().map_or(0.0, LogicalPixels::get),
        style
            .hidden_pattern()
            .map_or(0.0, |pattern| pattern.0.get()),
        style
            .hidden_pattern()
            .map_or(0.0, |pattern| pattern.1.get()),
    ];
    if !viewport
        .into_iter()
        .chain(style_sources)
        .all(is_portable_shader_source)
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let pixels_per_logical = PhysicalPerLogical::new(viewport[2])
        .map_err(|_| Mesh3dRenderError::InvalidEdgeProjection)?;
    let visible_raster = edge_raster_envelope(style.visible_width(), pixels_per_logical)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let hidden_raster = match style.hidden_width() {
        Some(width) => Some(
            edge_raster_envelope(width, pixels_per_logical)
                .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?,
        ),
        None => None,
    };
    let hidden_period = style.hidden_pattern().map(|(dash, gap)| {
        let dash = dash.get();
        let gap = gap.get();
        if is_nonzero_subnormal(dash) || is_nonzero_subnormal(gap) {
            return f32::NAN;
        }
        let period = f64::from(dash) + f64::from(gap);
        if period > f64::from(MAX_PORTABLE_SHADER_VALUE) {
            f32::NAN
        } else {
            dash + gap
        }
    });
    if hidden_period.is_some_and(|period| !period.is_finite()) {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    validate_hidden_dash_envelope(viewport, hidden_period)?;

    for edge in mesh.display_edges() {
        let mut clip_ranges = [[ShaderValueRange::exact(0.0); 4]; 2];
        let mut clip = [[0.0_f32; 4]; 2];
        for (endpoint_index, vertex_index) in [edge.start(), edge.end()].into_iter().enumerate() {
            let vertex = mesh.vertices()[vertex_index as usize];
            clip_ranges[endpoint_index] =
                shader_clip_point_ranges(vertex, model_rows, camera_rows)?;
            validate_clip_classification(clip_ranges[endpoint_index])?;
            clip[endpoint_index] = clip_ranges[endpoint_index].map(|value| value.fixed);
        }
        let normalized_ranges =
            validate_edge_homogeneous_classification(clip_ranges[0], clip_ranges[1])?;
        if clip_ranges_share_outside_plane(normalized_ranges.planes[0], normalized_ranges.planes[1])
        {
            continue;
        }
        let Some(clipped_ranges) = interval_clip_edge_to_frustum(normalized_ranges)? else {
            continue;
        };
        let Some(clipped) = clip_edge_to_frustum(clip[0], clip[1])? else {
            // The interval proof established stable visibility; disagreement
            // with the host diagnostic fold is not portable.
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        };
        let mut screen = [[0.0_f32; 2]; 2];
        for (endpoint_index, clip) in clipped.into_iter().enumerate() {
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            screen[endpoint_index] = [
                (ndc_x * 0.5 + 0.5) * viewport[0],
                (0.5 - ndc_y * 0.5) * viewport[1],
            ];
            if !screen[endpoint_index][0].is_finite() || !screen[endpoint_index][1].is_finite() {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
        }
        let delta_x = screen[1][0] - screen[0][0];
        let delta_y = screen[1][1] - screen[0][1];
        let length_squared = delta_x * delta_x + delta_y * delta_y;
        if !delta_x.is_finite() || !delta_y.is_finite() || !length_squared.is_finite() {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        let threshold_squared = 0.0001_f32 * 0.0001_f32;
        if length_squared > threshold_squared * 0.99 && length_squared < threshold_squared * 1.01 {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        let screen_length = length_squared.sqrt();
        validate_edge_projection_range(
            clipped_ranges[0],
            clipped_ranges[1],
            viewport,
            [delta_x, delta_y],
        )?;
        let screen_normal = if screen_length > 0.0001 {
            [-delta_y / screen_length, delta_x / screen_length]
        } else {
            [0.0, 0.0]
        };
        if !screen_normal.into_iter().all(f32::is_finite) {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        validate_edge_shader_arithmetic(clipped_ranges, viewport, visible_raster, None)?;
        if let Some(hidden_raster) = hidden_raster {
            validate_edge_shader_arithmetic(
                clipped_ranges,
                viewport,
                hidden_raster,
                hidden_period,
            )?;
        }
    }
    Ok(())
}

pub(super) fn validate_hidden_dash_envelope(
    viewport: [f32; 4],
    dash_period: Option<f32>,
) -> Result<(), Mesh3dRenderError> {
    let Some(period) = dash_period else {
        return Ok(());
    };
    if period <= 0.0 || is_nonzero_subnormal(period) {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }

    // Frustum-clipped endpoints lie inside the physical target, regardless of
    // which legal association/FMA choice produced their clip coordinates.
    // Validate dash arithmetic against that complete envelope rather than the
    // one fixed CPU fold used by the diagnostic clipping mirror below.
    let width = f64::from(viewport[0]);
    let height = f64::from(viewport[1]);
    let pixels_per_logical = f64::from(viewport[2]);
    let screen_diagonal = width.hypot(height) * (1.0 + 8.0 * f64::from(f32::EPSILON));
    let maximum_logical_distance = screen_diagonal / pixels_per_logical;
    let maximum_repetition = maximum_logical_distance / f64::from(period);
    if !screen_diagonal.is_finite()
        || !maximum_logical_distance.is_finite()
        || !maximum_repetition.is_finite()
        || maximum_logical_distance > f64::from(MAX_PORTABLE_SHADER_VALUE)
        || maximum_repetition > f64::from(MAX_PORTABLE_SHADER_VALUE)
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    Ok(())
}

pub(super) fn clip_edge_to_frustum(
    start: [f32; 4],
    end: [f32; 4],
) -> Result<Option<[[f32; 4]; 2]>, Mesh3dRenderError> {
    Ok(clip_edge_to_frustum_details(start, end)?.map(|clipped| clipped.clip))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ClippedEdgeCpu {
    pub(super) clip: [[f32; 4]; 2],
    pub(super) enter: f32,
    pub(super) exit: f32,
}

pub(super) fn clip_edge_to_frustum_details(
    start: [f32; 4],
    end: [f32; 4],
) -> Result<Option<ClippedEdgeCpu>, Mesh3dRenderError> {
    let pair_max = start
        .into_iter()
        .chain(end)
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if !pair_max.is_finite() {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    if pair_max > MAX_PORTABLE_SHADER_VALUE {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let homogeneous_scale = 1.0 / pair_max.max(1.0);
    let start = start.map(|component| component * homogeneous_scale);
    let end = end.map(|component| component * homogeneous_scale);
    let start_distances = clip_plane_distances(start)?;
    let end_distances = clip_plane_distances(end)?;
    let mut enter = 0.0_f32;
    let mut exit = 1.0_f32;
    for (start_distance, end_distance) in start_distances.into_iter().zip(end_distances) {
        if start_distance < 0.0 && end_distance < 0.0 {
            return Ok(None);
        }
        if start_distance < 0.0 || end_distance < 0.0 {
            let denominator = start_distance - end_distance;
            let amount = start_distance / denominator;
            if !denominator.is_finite() || !amount.is_finite() {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
            if start_distance < 0.0 {
                enter = enter.max(amount);
            } else {
                exit = exit.min(amount);
            }
        }
    }
    if enter > exit {
        return Ok(None);
    }
    let clipped_start = shader_lerp_clip(start, end, enter)?;
    let clipped_end = shader_lerp_clip(start, end, exit)?;
    if clipped_start[3] <= 0.0 || clipped_end[3] <= 0.0 {
        return Ok(None);
    }
    Ok(Some(ClippedEdgeCpu {
        clip: [clipped_start, clipped_end],
        enter,
        exit,
    }))
}

pub(super) fn clip_plane_distances(clip: [f32; 4]) -> Result<[f32; 6], Mesh3dRenderError> {
    let add = |left: f32, right: f32| {
        let result = left + right;
        result
            .is_finite()
            .then_some(result)
            .ok_or(Mesh3dRenderError::InvalidEdgeProjection)
    };
    let subtract = |left: f32, right: f32| {
        let result = left - right;
        result
            .is_finite()
            .then_some(result)
            .ok_or(Mesh3dRenderError::InvalidEdgeProjection)
    };
    Ok([
        add(clip[3], clip[0])?,
        subtract(clip[3], clip[0])?,
        add(clip[3], clip[1])?,
        subtract(clip[3], clip[1])?,
        clip[2],
        subtract(clip[3], clip[2])?,
    ])
}

pub(super) fn shader_lerp_clip(
    start: [f32; 4],
    end: [f32; 4],
    amount: f32,
) -> Result<[f32; 4], Mesh3dRenderError> {
    let mut output = [0.0_f32; 4];
    for axis in 0..4 {
        let delta = end[axis] - start[axis];
        let value = start[axis] + delta * amount;
        if !delta.is_finite() || !value.is_finite() {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        output[axis] = value;
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeRasterEnvelope {
    pub(super) physical_half_width: f32,
    pub(super) raster_half_width: f32,
}

pub(super) fn edge_raster_envelope(
    logical_width: LogicalPixels,
    pixels_per_logical: PhysicalPerLogical,
) -> Option<EdgeRasterEnvelope> {
    let logical_width = logical_width.get();
    let pixels_per_logical = pixels_per_logical.get();
    if !is_portable_shader_source(logical_width) || !is_portable_shader_source(pixels_per_logical) {
        return None;
    }
    let physical_half_width_f64 = f64::from(logical_width) * f64::from(pixels_per_logical) * 0.5;
    let raster_half_width_f64 = (physical_half_width_f64 + 0.5).max(1.0);
    let maximum = f64::from(MAX_PORTABLE_SHADER_VALUE);
    if !physical_half_width_f64.is_finite()
        || physical_half_width_f64 <= 0.0
        || raster_half_width_f64 * 2.0 > maximum
    {
        return None;
    }
    let physical_half_width = logical_width * pixels_per_logical * 0.5;
    let raster_half_width = (physical_half_width + 0.5).max(1.0);
    let doubled_raster_width = raster_half_width * 2.0;
    (physical_half_width.is_finite()
        && physical_half_width > 0.0
        && raster_half_width.is_finite()
        && doubled_raster_width.is_finite())
    .then_some(EdgeRasterEnvelope {
        physical_half_width,
        raster_half_width,
    })
}

pub(super) fn validate_edge_shader_arithmetic(
    clipped: EdgeClipRanges,
    viewport: [f32; 4],
    raster: EdgeRasterEnvelope,
    dash_period: Option<f32>,
) -> Result<(), Mesh3dRenderError> {
    let maximum_screen_length = f64::from(viewport[0]).hypot(f64::from(viewport[1]))
        * (1.0 + 8.0 * f64::from(f32::EPSILON));
    let logical_distance = maximum_screen_length / f64::from(viewport[2]);
    if !maximum_screen_length.is_finite()
        || !logical_distance.is_finite()
        || logical_distance > f64::from(MAX_PORTABLE_SHADER_VALUE)
        || !(raster.physical_half_width + 0.5).is_finite()
        || !(raster.raster_half_width * 2.0).is_finite()
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    if let Some(period) = dash_period {
        let repetition = logical_distance / f64::from(period);
        let repeated_distance = repetition.floor() * f64::from(period);
        let within_period = logical_distance - repeated_distance;
        if !period.is_finite()
            || period <= 0.0
            || !repetition.is_finite()
            || !repeated_distance.is_finite()
            || !within_period.is_finite()
            || repetition.abs() > f64::from(MAX_PORTABLE_SHADER_VALUE)
            || repeated_distance.abs() > f64::from(MAX_PORTABLE_SHADER_VALUE)
            || within_period.abs() > f64::from(MAX_PORTABLE_SHADER_VALUE)
        {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
    }

    let dimensions = [viewport[0].max(1.0), viewport[1].max(1.0)];
    for axis in 0..2 {
        // A normalized screen direction can place the full raster half-width
        // on either axis. Use that complete envelope rather than one host
        // normalization result.
        let normalized_component_bound = 1.0 + 4.0 * f64::from(f32::EPSILON);
        let screen_offset = rounded_f32_product_range(
            (-normalized_component_bound, normalized_component_bound),
            (
                f64::from(raster.raster_half_width),
                f64::from(raster.raster_half_width),
            ),
        )
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
        let doubled_screen_offset = rounded_f32_product_range(screen_offset, (2.0, 2.0))
            .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
        let ndc_offset = wgsl_division_range(
            doubled_screen_offset,
            (f64::from(dimensions[axis]), f64::from(dimensions[axis])),
        )
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
        for endpoint in clipped {
            let offset = rounded_f32_product_range(ndc_offset, endpoint[3])
                .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
            if rounded_f32_add_range(endpoint[axis], offset, false).is_none() {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
        }
    }
    Ok(())
}
