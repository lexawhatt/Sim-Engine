@group(0) @binding(0) var low_color: texture_2d<f32>;
@group(0) @binding(1) var low_depth: texture_depth_2d;
@group(0) @binding(2) var base_color: texture_2d<f32>;
@group(0) @binding(3) var base_depth: texture_depth_2d;
@group(0) @binding(4) var detail_color: texture_2d<f32>;
@group(0) @binding(5) var detail_depth: texture_depth_2d;
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    return vec4<f32>(positions[index],0.0,1.0);
}
struct Output { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 }
@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> Output {
    let pixel = vec2<i32>(position.xy);
    let extent = textureDimensions(base_color);
    let low_extent = textureDimensions(low_color);
    let low_pixel = min(vec2<i32>(position.xy * vec2<f32>(low_extent) / vec2<f32>(extent)), vec2<i32>(low_extent)-vec2<i32>(1));
    let base_z = textureLoad(base_depth, pixel, 0);
    let detail_z = textureLoad(detail_depth, pixel, 0);
    if detail_z < base_z { return Output(textureLoad(detail_color, pixel, 0), detail_z); }
    // Native coverage/depth decides visibility. At a low-resolution silhouette
    // with no covered low texel, preserve accepted geometry using native color.
    if base_z == 1.0 || textureLoad(low_depth, low_pixel, 0) == 1.0 {
        return Output(textureLoad(base_color, pixel, 0), base_z);
    }
    return Output(textureLoad(low_color, low_pixel, 0), base_z);
}
