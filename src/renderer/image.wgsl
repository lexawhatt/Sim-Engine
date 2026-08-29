struct ImageUniform {
    destination: vec4<f32>,
    uv_rect: vec4<f32>,
    tint: vec4<f32>,
    world_clip_x: vec4<f32>,
    world_clip_y: vec4<f32>,
    world_mode: vec4<f32>,
};

@group(0) @binding(0) var image_texture: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> image: ImageUniform;

struct ImageOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

struct ImageInstance {
    @location(0) destination: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
    @location(2) tint: vec4<f32>,
};

@vertex
fn image_vs_main(@builtin(vertex_index) index: u32) -> ImageOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let position = positions[index];
    let amount = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    var output: ImageOut;
    if image.world_mode.x > 0.5 {
        let top = mix(
            vec2<f32>(image.world_clip_x.x, image.world_clip_y.x),
            vec2<f32>(image.world_clip_x.y, image.world_clip_y.y),
            amount.x,
        );
        let bottom = mix(
            vec2<f32>(image.world_clip_x.z, image.world_clip_y.z),
            vec2<f32>(image.world_clip_x.w, image.world_clip_y.w),
            amount.x,
        );
        output.position = vec4<f32>(mix(top, bottom, amount.y), 0.0, 1.0);
    } else {
        output.position = vec4<f32>(
            position * image.destination.xy + image.destination.zw,
            0.0,
            1.0,
        );
    }
    output.uv = mix(image.uv_rect.xy, image.uv_rect.zw, amount);
    output.tint = image.tint;
    return output;
}

@vertex
fn image_batch_vs_main(
    instance: ImageInstance,
    @builtin(vertex_index) index: u32,
) -> ImageOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0),
    );
    let amount = positions[index];
    let screen = image.uv_rect.xy + instance.destination.xy + amount * instance.destination.zw;
    var output: ImageOut;
    output.position = vec4<f32>(
        screen.x * image.destination.x + image.destination.z,
        screen.y * image.destination.y + image.destination.w,
        0.0,
        1.0,
    );
    output.uv = mix(instance.uv_rect.xy, instance.uv_rect.zw, amount);
    output.tint = instance.tint;
    return output;
}

@fragment
fn image_fs_main(input: ImageOut) -> @location(0) vec4<f32> {
    return textureSample(image_texture, image_sampler, input.uv) * input.tint;
}
