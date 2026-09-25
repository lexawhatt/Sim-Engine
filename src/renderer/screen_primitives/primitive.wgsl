struct CameraUniform {
    camera_center: vec4<f32>,
    world_to_screen_x: vec4<f32>,
    world_to_screen_y: vec4<f32>,
    screen_to_clip: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct Instance {
    @location(0) anchor: vec2<f32>,
    @location(1) corner_0: vec2<f32>,
    @location(2) corner_1: vec2<f32>,
    @location(3) corner_2: vec2<f32>,
    @location(4) corner_3: vec2<f32>,
    @location(5) shape: vec2<f32>,
    @location(6) color: vec4<f32>,
};
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(linear, sample) local: vec2<f32>,
    @location(1) @interpolate(flat) shape: vec2<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vs_main(instance: Instance, @builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let indices = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let corners = array<vec2<f32>, 4>(instance.corner_0, instance.corner_1, instance.corner_2, instance.corner_3);
    let local_corners = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let index = indices[vertex_index];
    let relative = instance.anchor - camera.camera_center.xy;
    let world = vec4<f32>(relative, 0.0, 1.0);
    let screen = vec2<f32>(dot(camera.world_to_screen_x, world), dot(camera.world_to_screen_y, world))
        + vec2<f32>(dot(camera.world_to_screen_x.xy, corners[index]), dot(camera.world_to_screen_y.xy, corners[index]));
    var output: VertexOut;
    output.position = vec4<f32>(screen * camera.screen_to_clip.xy + camera.screen_to_clip.zw, 0.0, 1.0);
    output.local = local_corners[index];
    if instance.shape.x == 1.0 {
        output.local.x *= instance.shape.y + 1.0;
    }
    output.shape = instance.shape;
    output.color = instance.color;
    return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    if input.shape.x != 2.0 {
        var horizontal = abs(input.local.x);
        if input.shape.x == 1.0 { horizontal = max(horizontal - input.shape.y, 0.0); }
        // Bound the product even for a long capsule's interior interpolation.
        if horizontal > 1.0 || abs(input.local.y) > 1.0 { discard; }
        if horizontal * horizontal + input.local.y * input.local.y > 1.0 { discard; }
    }
    return input.color;
}
