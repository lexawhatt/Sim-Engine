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
    return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
