struct Camera3dUniform {
    clip_row_0: vec4<f32>,
    clip_row_1: vec4<f32>,
    clip_row_2: vec4<f32>,
    clip_row_3: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera3d: Camera3dUniform;

struct Mesh3dVertexIn {
    @location(0) model_position: vec3<f32>,
    @location(1) model_row_0: vec4<f32>,
    @location(2) model_row_1: vec4<f32>,
    @location(3) model_row_2: vec4<f32>,
    @location(4) color: vec4<f32>,
};

struct Mesh3dVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn mesh3d_vs_main(input: Mesh3dVertexIn) -> Mesh3dVertexOut {
    let model = vec4<f32>(input.model_position, 1.0);
    let world = vec4<f32>(
        dot(input.model_row_0, model),
        dot(input.model_row_1, model),
        dot(input.model_row_2, model),
        1.0,
    );
    var output: Mesh3dVertexOut;
    output.position = vec4<f32>(
        dot(camera3d.clip_row_0, world),
        dot(camera3d.clip_row_1, world),
        dot(camera3d.clip_row_2, world),
        dot(camera3d.clip_row_3, world),
    );
    output.color = input.color;
    return output;
}

@fragment
fn mesh3d_fs_main(input: Mesh3dVertexOut) -> @location(0) vec4<f32> {
    return input.color;
}

struct Mesh3dEdgeIn {
    @location(0) model_start: vec3<f32>,
    @location(1) model_end: vec3<f32>,
};

struct EdgeObjectUniform {
    model_row_0: vec4<f32>,
    model_row_1: vec4<f32>,
    model_row_2: vec4<f32>,
    visible_color: vec4<f32>,
    hidden_color: vec4<f32>,
    edge_style: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> edge_object: EdgeObjectUniform;

struct Mesh3dEdgeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(linear) logical_distance: f32,
    @location(2) dash_gap: vec2<f32>,
    @location(3) @interpolate(linear) normal_distance: f32,
    @location(4) @interpolate(flat) physical_half_width: f32,
};

fn project_edge_point(model_position: vec3<f32>) -> vec4<f32> {
    let model = vec4<f32>(model_position, 1.0);
    let world = vec4<f32>(
        dot(edge_object.model_row_0, model),
        dot(edge_object.model_row_1, model),
        dot(edge_object.model_row_2, model),
        1.0,
    );
    return vec4<f32>(
        dot(camera3d.clip_row_0, world),
        dot(camera3d.clip_row_1, world),
        dot(camera3d.clip_row_2, world),
        dot(camera3d.clip_row_3, world),
    );
}

struct ClippedEdge {
    start_clip: vec4<f32>,
    end_clip: vec4<f32>,
    enter: f32,
    exit: f32,
    visible: bool,
};

fn clip_edge_to_frustum(start_clip: vec4<f32>, end_clip: vec4<f32>) -> ClippedEdge {
    let start_max = max(max(abs(start_clip.x), abs(start_clip.y)), max(abs(start_clip.z), abs(start_clip.w)));
    let end_max = max(max(abs(end_clip.x), abs(end_clip.y)), max(abs(end_clip.z), abs(end_clip.w)));
    let pair_max = max(start_max, end_max);
    // Keep the common homogeneous scale normal. Some Vulkan implementations
    // flush subnormal reciprocals of f32::MAX to zero, which would erase the
    // segment even though uniform homogeneous scaling is valid.
    let homogeneous_scale = max(1.0 / max(pair_max, 1.0), 1.17549435e-38);
    let normalized_start = start_clip * homogeneous_scale;
    let normalized_end = end_clip * homogeneous_scale;
    let start_distances = array<f32, 6>(
        normalized_start.w + normalized_start.x,
        normalized_start.w - normalized_start.x,
        normalized_start.w + normalized_start.y,
        normalized_start.w - normalized_start.y,
        normalized_start.z,
        normalized_start.w - normalized_start.z,
    );
    let end_distances = array<f32, 6>(
        normalized_end.w + normalized_end.x,
        normalized_end.w - normalized_end.x,
        normalized_end.w + normalized_end.y,
        normalized_end.w - normalized_end.y,
        normalized_end.z,
        normalized_end.w - normalized_end.z,
    );
    var enter = 0.0;
    var exit = 1.0;
    var visible = true;
    for (var plane = 0u; plane < 6u; plane += 1u) {
        let start_distance = start_distances[plane];
        let end_distance = end_distances[plane];
        if start_distance < 0.0 && end_distance < 0.0 {
            visible = false;
        } else if start_distance < 0.0 {
            enter = max(enter, start_distance / (start_distance - end_distance));
        } else if end_distance < 0.0 {
            exit = min(exit, start_distance / (start_distance - end_distance));
        }
    }
    visible = visible && enter <= exit;
    if !visible {
        let fallback = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return ClippedEdge(fallback, fallback, enter, exit, false);
    }
    let delta = normalized_end - normalized_start;
    let clipped_start = normalized_start + delta * enter;
    let clipped_end = normalized_start + delta * exit;
    if clipped_start.w <= 0.0 || clipped_end.w <= 0.0 {
        let fallback = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return ClippedEdge(fallback, fallback, enter, exit, false);
    }
    return ClippedEdge(clipped_start, clipped_end, enter, exit, true);
}

struct ClipProbeInput {
    start_clip: vec4<f32>,
    end_clip: vec4<f32>,
};

struct ClipProbeOutput {
    start_clip: vec4<f32>,
    end_clip: vec4<f32>,
    range: vec2<f32>,
    visible: u32,
    padding: u32,
};

@group(2) @binding(0)
var<storage, read> clip_probe_inputs: array<ClipProbeInput>;

@group(2) @binding(1)
var<storage, read_write> clip_probe_outputs: array<ClipProbeOutput>;

@compute @workgroup_size(64)
fn mesh3d_clip_probe_main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= arrayLength(&clip_probe_inputs) {
        return;
    }
    let input = clip_probe_inputs[index];
    let clipped = clip_edge_to_frustum(input.start_clip, input.end_clip);
    clip_probe_outputs[index] = ClipProbeOutput(
        clipped.start_clip,
        clipped.end_clip,
        vec2<f32>(clipped.enter, clipped.exit),
        select(0u, 1u, clipped.visible),
        0u,
    );
}

fn edge_vertex(
    input: Mesh3dEdgeIn,
    vertex_index: u32,
    logical_width: f32,
    color: vec4<f32>,
) -> Mesh3dEdgeOut {
    let clipped = clip_edge_to_frustum(
        project_edge_point(input.model_start),
        project_edge_point(input.model_end),
    );
    let start_clip = clipped.start_clip;
    let end_clip = clipped.end_clip;
    let start_ndc = start_clip.xy / start_clip.w;
    let end_ndc = end_clip.xy / end_clip.w;
    let dimensions = max(camera3d.viewport.xy, vec2<f32>(1.0, 1.0));
    let start_screen = vec2<f32>(
        (start_ndc.x * 0.5 + 0.5) * dimensions.x,
        (0.5 - start_ndc.y * 0.5) * dimensions.y,
    );
    let end_screen = vec2<f32>(
        (end_ndc.x * 0.5 + 0.5) * dimensions.x,
        (0.5 - end_ndc.y * 0.5) * dimensions.y,
    );
    let screen_direction = end_screen - start_screen;
    let screen_length = length(screen_direction);
    var screen_normal = vec2<f32>(0.0, 0.0);
    if screen_length > 0.0001 {
        let direction = screen_direction / screen_length;
        screen_normal = vec2<f32>(-direction.y, direction.x);
    }
    let endpoint_pattern = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
    let side_pattern = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let use_end = endpoint_pattern[vertex_index] == 1u;
    var clip = select(start_clip, end_clip, use_end);
    let physical_half_width = logical_width * camera3d.viewport.z * 0.5;
    let raster_half_width = max(physical_half_width + 0.5, 1.0);
    let screen_offset = screen_normal * side_pattern[vertex_index] * raster_half_width;
    let ndc_offset = vec2<f32>(
        screen_offset.x * 2.0 / dimensions.x,
        -screen_offset.y * 2.0 / dimensions.y,
    );
    clip.x += ndc_offset.x * clip.w;
    clip.y += ndc_offset.y * clip.w;
    var output: Mesh3dEdgeOut;
    output.position = clip;
    output.color = vec4<f32>(color.rgb, select(0.0, color.a, clipped.visible));
    output.logical_distance = select(0.0, screen_length / camera3d.viewport.z, use_end);
    output.dash_gap = edge_object.edge_style.zw;
    output.normal_distance = side_pattern[vertex_index] * raster_half_width;
    output.physical_half_width = physical_half_width;
    return output;
}

fn edge_coverage(input: Mesh3dEdgeOut) -> f32 {
    return clamp(
        input.physical_half_width + 0.5 - abs(input.normal_distance),
        0.0,
        1.0,
    );
}

@vertex
fn mesh3d_visible_edge_vs_main(
    input: Mesh3dEdgeIn,
    @builtin(vertex_index) vertex_index: u32,
) -> Mesh3dEdgeOut {
    return edge_vertex(
        input,
        vertex_index,
        edge_object.edge_style.x,
        edge_object.visible_color,
    );
}

@vertex
fn mesh3d_hidden_edge_vs_main(
    input: Mesh3dEdgeIn,
    @builtin(vertex_index) vertex_index: u32,
) -> Mesh3dEdgeOut {
    return edge_vertex(
        input,
        vertex_index,
        edge_object.edge_style.y,
        edge_object.hidden_color,
    );
}

@fragment
fn mesh3d_visible_edge_fs_main(input: Mesh3dEdgeOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color.rgb, input.color.a * edge_coverage(input));
}

@fragment
fn mesh3d_hidden_edge_fs_main(input: Mesh3dEdgeOut) -> @location(0) vec4<f32> {
    let period = input.dash_gap.x + input.dash_gap.y;
    let within_period = input.logical_distance - floor(input.logical_distance / period) * period;
    if period <= 0.0 || within_period > input.dash_gap.x {
        discard;
    }
    return vec4<f32>(input.color.rgb, input.color.a * edge_coverage(input));
}
