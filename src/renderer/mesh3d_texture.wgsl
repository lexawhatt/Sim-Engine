@group(1) @binding(0) var material_texture: texture_2d<f32>;
@group(1) @binding(1) var material_sampler: sampler;
@vertex fn retained_vs_main(input: SurfaceVertexIn, @location(5) uv: vec2<f32>) -> SurfaceOut {
    return surface_retained(input, uv, vec4<f32>(1.0), vec3<f32>(0.0));
}
@vertex fn retained_enhanced_vs_main(input: SurfaceVertexIn, @location(5) uv: vec2<f32>, @location(8) normal: vec3<f32>) -> SurfaceOut {
    return surface_retained(input, uv, vec4<f32>(1.0), normal);
}
@vertex fn colored_retained_vs_main(input: SurfaceVertexIn, @location(5) uv: vec2<f32>, @location(6) vertex_color: vec4<f32>) -> SurfaceOut {
    return surface_retained(input, uv, vertex_color, vec3<f32>(0.0));
}
@vertex fn colored_retained_enhanced_vs_main(input: SurfaceVertexIn, @location(5) uv: vec2<f32>, @location(6) vertex_color: vec4<f32>, @location(8) normal: vec3<f32>) -> SurfaceOut {
    return surface_retained(input, uv, vertex_color, normal);
}
@vertex fn clipped_vs_main(@location(0) clip: vec4<f32>, @location(12) uv_transform: vec4<f32>, @location(4) color: vec4<f32>, @location(7) surface: vec4<f32>, @location(5) uv: vec2<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color * vec4<f32>(1.0), uv * uv_transform.xy + uv_transform.zw, surface, vec4<f32>(0.0));
}
@vertex fn clipped_enhanced_vs_main(@location(0) clip: vec4<f32>, @location(12) uv_transform: vec4<f32>, @location(4) color: vec4<f32>, @location(7) surface: vec4<f32>, @location(5) uv: vec2<f32>, @location(8) auxiliary: vec4<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color * vec4<f32>(1.0), uv * uv_transform.xy + uv_transform.zw, surface, auxiliary);
}
@vertex fn colored_clipped_vs_main(@location(0) clip: vec4<f32>, @location(12) uv_transform: vec4<f32>, @location(4) color: vec4<f32>, @location(7) surface: vec4<f32>, @location(5) uv: vec2<f32>, @location(6) vertex_color: vec4<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color * vertex_color, uv * uv_transform.xy + uv_transform.zw, surface, vec4<f32>(0.0));
}
@vertex fn colored_clipped_enhanced_vs_main(@location(0) clip: vec4<f32>, @location(12) uv_transform: vec4<f32>, @location(4) color: vec4<f32>, @location(7) surface: vec4<f32>, @location(5) uv: vec2<f32>, @location(6) vertex_color: vec4<f32>, @location(8) auxiliary: vec4<f32>) -> SurfaceOut {
    return SurfaceOut(clip, color * vertex_color, uv * uv_transform.xy + uv_transform.zw, surface, auxiliary);
}
@fragment fn fs_main(input: SurfaceOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let color = textureSample(material_texture, material_sampler, input.uv) * input.color;
    return surface_fragment(color, input.surface, input.normal_depth, front);
}
