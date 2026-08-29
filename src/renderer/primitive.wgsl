struct CameraUniform {
    camera_center: vec4<f32>,
    world_to_screen_x: vec4<f32>,
    world_to_screen_y: vec4<f32>,
    screen_to_clip: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexIn {
    @location(0) world_position: vec2<f32>,
    @location(1) depth: f32,
    @location(2) screen_offset: vec2<f32>,
    @location(3) previous_direction: vec2<f32>,
    @location(4) next_direction: vec2<f32>,
    @location(5) normal_distance: f32,
    @location(6) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) particle_direction: vec2<f32>,
    @location(2) particle_mask: f32,
};

fn safe_normal(direction: vec2<f32>) -> vec2<f32> {
    let length_squared = dot(direction, direction);
    if length_squared <= 0.000001 {
        return vec2<f32>(0.0, 0.0);
    }
    let normalized_direction = direction * inverseSqrt(length_squared);
    return vec2<f32>(-normalized_direction.y, normalized_direction.x);
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let relative_world = input.world_position - camera.camera_center.xy;
    var screen = vec2<f32>(
        dot(camera.world_to_screen_x.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_x.w,
        dot(camera.world_to_screen_y.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_y.w,
    );

    if abs(input.normal_distance) > 0.0 {
        let previous_screen = vec2<f32>(
            dot(camera.world_to_screen_x.xy, input.previous_direction),
            dot(camera.world_to_screen_y.xy, input.previous_direction),
        );
        let next_screen = vec2<f32>(
            dot(camera.world_to_screen_x.xy, input.next_direction),
            dot(camera.world_to_screen_y.xy, input.next_direction),
        );
        let previous_normal = safe_normal(previous_screen);
        let next_normal = safe_normal(next_screen);
        let combined_normal = previous_normal + next_normal;
        var extrusion = next_normal * input.normal_distance;

        if dot(combined_normal, combined_normal) > 0.000001 {
            let miter = normalize(combined_normal);
            let denominator = dot(miter, next_normal);
            if abs(denominator) > 0.001 {
                extrusion = miter * (input.normal_distance / denominator);
            }
        }
        screen += extrusion;
    }

    screen += input.screen_offset;
    let clip = vec2<f32>(
        screen.x * camera.screen_to_clip.x + camera.screen_to_clip.z,
        screen.y * camera.screen_to_clip.y + camera.screen_to_clip.w,
    );
    var output: VertexOut;
    output.position = vec4<f32>(clip, 0.0, 1.0);
    output.color = input.color;
    output.particle_direction = vec2<f32>(0.0, 0.0);
    output.particle_mask = 0.0;
    return output;
}

struct DynamicIn {
    @location(0) world_position: vec2<f32>,
    @location(1) depth: f32,
    @location(2) color: vec4<f32>,
};

@vertex
fn dynamic_vs_main(input: DynamicIn) -> VertexOut {
    let relative_world = input.world_position - camera.camera_center.xy;
    let screen = vec2<f32>(
        dot(camera.world_to_screen_x.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_x.w,
        dot(camera.world_to_screen_y.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_y.w,
    );
    let clip = vec2<f32>(
        screen.x * camera.screen_to_clip.x + camera.screen_to_clip.z,
        screen.y * camera.screen_to_clip.y + camera.screen_to_clip.w,
    );
    var output: VertexOut;
    output.position = vec4<f32>(clip, 0.0, 1.0);
    output.color = input.color;
    output.particle_direction = vec2<f32>(0.0, 0.0);
    output.particle_mask = 0.0;
    return output;
}

struct ParticleIn {
    @location(0) unit_direction: vec2<f32>,
    @location(1) world_position: vec2<f32>,
    @location(2) depth: f32,
    @location(3) radius: f32,
    @location(4) color: vec4<f32>,
};

@vertex
fn particle_vs_main(input: ParticleIn) -> VertexOut {
    let relative_world = input.world_position - camera.camera_center.xy;
    let projected_screen = vec2<f32>(
        dot(camera.world_to_screen_x.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_x.w,
        dot(camera.world_to_screen_y.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_y.w,
    );
    let screen = projected_screen + input.unit_direction * input.radius;
    let clip = vec2<f32>(
        screen.x * camera.screen_to_clip.x + camera.screen_to_clip.z,
        screen.y * camera.screen_to_clip.y + camera.screen_to_clip.w,
    );
    var output: VertexOut;
    output.position = vec4<f32>(clip, 0.0, 1.0);
    output.color = input.color;
    output.particle_direction = input.unit_direction;
    output.particle_mask = 1.0;
    return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    if input.particle_mask > 0.5 && dot(input.particle_direction, input.particle_direction) > 1.0 {
        discard;
    }
    return input.color;
}

struct HeatmapUniform {
    value_range: vec4<f32>,
    dimensions: vec4<u32>,
    destination: vec4<f32>,
};

@group(0) @binding(0) var scalar_field: texture_2d<f32>;
@group(0) @binding(1) var color_map: texture_2d<f32>;
@group(0) @binding(2) var<uniform> heatmap: HeatmapUniform;

struct HeatmapOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn heatmap_vs_main(@builtin(vertex_index) index: u32) -> HeatmapOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let position = positions[index];
    var output: HeatmapOut;
    output.position = vec4<f32>(
        position * heatmap.destination.xy + heatmap.destination.zw,
        0.0,
        1.0,
    );
    output.uv = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    return output;
}

@fragment
fn heatmap_fs_main(input: HeatmapOut) -> @location(0) vec4<f32> {
    let dimensions = max(heatmap.dimensions.xy, vec2<u32>(1u, 1u));
    let texel = min(vec2<u32>(input.uv * vec2<f32>(dimensions)), dimensions - vec2<u32>(1u, 1u));
    var scalar = textureLoad(scalar_field, vec2<i32>(i32(texel.x), i32(texel.y)), 0).x;
    if heatmap.dimensions.z == 1u {
        let source = input.uv * vec2<f32>(dimensions) - vec2<f32>(0.5, 0.5);
        let base = floor(source);
        let amount = fract(source);
        let low = vec2<i32>(max(base, vec2<f32>(0.0, 0.0)));
        let high = min(low + vec2<i32>(1, 1), vec2<i32>(i32(dimensions.x) - 1, i32(dimensions.y) - 1));
        let top = mix(textureLoad(scalar_field, low, 0).x, textureLoad(scalar_field, vec2<i32>(high.x, low.y), 0).x, amount.x);
        let bottom = mix(textureLoad(scalar_field, vec2<i32>(low.x, high.y), 0).x, textureLoad(scalar_field, vec2<i32>(high.x, high.y), 0).x, amount.x);
        scalar = mix(top, bottom, amount.y);
    }
    let normalized = clamp((scalar - heatmap.value_range.x) / heatmap.value_range.y, 0.0, 1.0);
    let color_index = min(u32(round(normalized * 255.0)), 255u);
    return textureLoad(color_map, vec2<i32>(i32(color_index), 0), 0);
}

struct CompositeUniform {
    opacity: vec4<f32>,
    destination: vec4<f32>,
};

@group(0) @binding(0) var composition_source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> composition: CompositeUniform;

@vertex
fn composite_vs_main(@builtin(vertex_index) index: u32) -> HeatmapOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let position = positions[index];
    var output: HeatmapOut;
    output.position = vec4<f32>(
        position * composition.destination.xy + composition.destination.zw,
        0.0,
        1.0,
    );
    output.uv = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    return output;
}

@fragment
fn composite_fs_main(input: HeatmapOut) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(composition_source);
    let texel = min(
        vec2<u32>(input.uv * vec2<f32>(dimensions)),
        dimensions - vec2<u32>(1u, 1u),
    );
    let source = textureLoad(composition_source, vec2<i32>(texel), 0);
    // Render targets use source-alpha blending and therefore retain
    // premultiplied RGB. The presentation and temporal alpha pipelines expect
    // straight RGB, so unpremultiply exactly once at this boundary.
    var straight_rgb = vec3<f32>(0.0, 0.0, 0.0);
    if source.a > 0.0 {
        straight_rgb = source.rgb / source.a;
    }
    return vec4<f32>(straight_rgb, source.a * composition.opacity.x);
}
