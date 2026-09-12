// Shared retained/clipped, textured/plain surface RGB contract.
struct SurfaceEnvironmentGpu {
    depth_row: vec4<f32>, ambient: vec4<f32>, sunlight: vec4<f32>,
    direction: vec4<f32>, fog_color_start: vec4<f32>, fog_density: vec4<f32>,
};
struct Camera3dUniform {
    clip_row_0: vec4<f32>, clip_row_1: vec4<f32>, clip_row_2: vec4<f32>, clip_row_3: vec4<f32>,
    viewport: vec4<f32>, environment: SurfaceEnvironmentGpu,
};
@group(0) @binding(0) var<uniform> camera3d: Camera3dUniform;
struct SurfaceVertexIn {
    @location(0) position: vec3<f32>,
    @location(1) row0: vec4<f32>, @location(2) row1: vec4<f32>, @location(3) row2: vec4<f32>,
    @location(4) color: vec4<f32>, @location(7) surface: vec4<f32>,
    @location(9) normal0: vec4<f32>, @location(10) normal1: vec4<f32>, @location(11) normal2: vec4<f32>,
    @location(12) uv_transform: vec4<f32>,
};
struct SurfaceOut {
    @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) surface: vec4<f32>, @location(3) normal_depth: vec4<f32>,
};
fn surface_retained(input: SurfaceVertexIn, uv: vec2<f32>, vertex_color: vec4<f32>, normal: vec3<f32>) -> SurfaceOut {
    let model = vec4<f32>(input.position, 1.0);
    let world = vec4<f32>(dot(input.row0, model), dot(input.row1, model), dot(input.row2, model), 1.0);
    let clip = vec4<f32>(dot(camera3d.clip_row_0, world), dot(camera3d.clip_row_1, world),
        dot(camera3d.clip_row_2, world), dot(camera3d.clip_row_3, world));
    var auxiliary = vec4<f32>(0.0);
    if input.surface.z == 1.0 && camera3d.environment.direction.w > 0.0 {
        let n = vec4<f32>(normal, 0.0);
        auxiliary = vec4<f32>(dot(input.normal0, n), dot(input.normal1, n), dot(input.normal2, n), 0.0);
    }
    if input.surface.w == 1.0 && camera3d.environment.fog_density.x > 0.0 {
        auxiliary.w = dot(camera3d.environment.depth_row, world);
    }
    return SurfaceOut(clip, input.color * vertex_color, uv * input.uv_transform.xy + input.uv_transform.zw, input.surface, auxiliary);
}
fn surface_fragment(color: vec4<f32>, surface: vec4<f32>, auxiliary: vec4<f32>, front: bool) -> vec4<f32> {
    if surface.x == 1.0 && color.a < surface.y { discard; }
    var rgb = color.rgb;
    if surface.z == 1.0 {
        var illumination = camera3d.environment.ambient.rgb;
        if camera3d.environment.direction.w > 0.0 {
            var normal = auxiliary.xyz;
            if !front { normal = -normal; }
            let extent = max(max(abs(normal.x), abs(normal.y)), abs(normal.z));
            if extent >= 1.1754943508222875e-38 {
                let scaled = normal / extent;
                let unit = scaled / sqrt(dot(scaled, scaled));
                illumination += camera3d.environment.sunlight.rgb * max(dot(unit, camera3d.environment.direction.xyz), 0.0);
            }
        }
        rgb = clamp(rgb * illumination, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let density = camera3d.environment.fog_density.x;
    let start = camera3d.environment.fog_color_start.w;
    // Branch before subtraction/product: WGSL select evaluates unsafe operands.
    if surface.w == 1.0 && density > 0.0 && auxiliary.w > start {
        let distance = auxiliary.w - start;
        var amount = 1.0;
        if density > 1.0 {
            if distance < 80.0 / density { amount = 1.0 - exp(-(density * distance)); }
        } else {
            amount = 1.0 - exp(-(density * distance));
        }
        rgb = mix(rgb, camera3d.environment.fog_color_start.rgb, amount);
    }
    if surface.x == 2.0 { return vec4<f32>(rgb, color.a); }
    return vec4<f32>(rgb, 1.0);
}
