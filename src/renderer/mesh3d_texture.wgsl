struct Camera3dUniform {
    clip_row_0: vec4<f32>, clip_row_1: vec4<f32>,
    clip_row_2: vec4<f32>, clip_row_3: vec4<f32>, viewport: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera3d: Camera3dUniform;
@group(1) @binding(0) var material_texture: texture_2d<f32>;
@group(1) @binding(1) var material_sampler: sampler;

struct SurfaceOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) surface: vec2<f32>,
};

fn retained_transform(
    position: vec3<f32>,
    row0: vec4<f32>, row1: vec4<f32>,
    row2: vec4<f32>, color: vec4<f32>,
    uv: vec2<f32>, surface: vec4<f32>,
) -> SurfaceOut {
    let model = vec4<f32>(position, 1.0);
    let world = vec4<f32>(dot(row0, model), dot(row1, model), dot(row2, model), 1.0);
    let clip = vec4<f32>(dot(camera3d.clip_row_0, world), dot(camera3d.clip_row_1, world),
        dot(camera3d.clip_row_2, world), dot(camera3d.clip_row_3, world));
    return SurfaceOut(clip, color, uv, surface.xy);
}

@vertex
fn retained_vs_main(
    @location(0) position: vec3<f32>, @location(1) row0: vec4<f32>, @location(2) row1: vec4<f32>,
    @location(3) row2: vec4<f32>, @location(4) color: vec4<f32>, @location(5) uv: vec2<f32>, @location(7) surface: vec4<f32>,
) -> SurfaceOut {
    return retained_transform(position, row0, row1, row2, color, uv, surface);
}
@vertex
fn colored_retained_vs_main(
    @location(0) position: vec3<f32>, @location(1) row0: vec4<f32>, @location(2) row1: vec4<f32>,
    @location(3) row2: vec4<f32>, @location(4) color: vec4<f32>, @location(5) uv: vec2<f32>, @location(7) surface: vec4<f32>,
    @location(6) vertex_color: vec4<f32>,
) -> SurfaceOut {
    return retained_transform(position, row0, row1, row2, color * vertex_color, uv, surface);
}
@vertex
fn colored_clipped_vs_main(@location(0) clip: vec4<f32>, @location(4) color: vec4<f32>, @location(5) uv: vec2<f32>, @location(7) surface: vec4<f32>, @location(6) vertex_color: vec4<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color * vertex_color, uv, surface.xy);
}

@vertex
fn clipped_vs_main(@location(0) clip: vec4<f32>, @location(4) color: vec4<f32>, @location(5) uv: vec2<f32>, @location(7) surface: vec4<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color, uv, surface.xy);
}

@fragment
fn fs_main(input: SurfaceOut) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(material_texture, material_sampler, input.uv, 0.0) * input.color;
    if input.surface.x == 1.0 && color.a < input.surface.y { discard; }
    if input.surface.x == 2.0 { return color; }
    return vec4<f32>(color.rgb, 1.0);
}
