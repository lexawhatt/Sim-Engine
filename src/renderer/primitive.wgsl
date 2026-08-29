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
    @location(6) tangent_distance: f32,
    @location(7) miter_limit: f32,
    @location(8) stroke_role: f32,
    @location(9) stroke_parameter: f32,
    @location(10) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) particle_direction: vec2<f32>,
    @location(2) particle_mask: f32,
};

fn safe_unit(direction: vec2<f32>) -> vec2<f32> {
    let scale = max(abs(direction.x), abs(direction.y));
    if scale <= 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    let scaled = direction / scale;
    let length_squared = dot(scaled, scaled);
    if length_squared <= 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    return scaled * inverseSqrt(length_squared);
}

fn safe_normal(direction: vec2<f32>) -> vec2<f32> {
    let normalized_direction = safe_unit(direction);
    return vec2<f32>(-normalized_direction.y, normalized_direction.x);
}

fn cross_2d(left: vec2<f32>, right: vec2<f32>) -> f32 {
    return left.x * right.y - left.y * right.x;
}

fn same_side(left: f32, right: f32) -> bool {
    return left * right > 0.0;
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let relative_world = input.world_position - camera.camera_center.xy;
    var screen = vec2<f32>(
        dot(camera.world_to_screen_x.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_x.w,
        dot(camera.world_to_screen_y.xyz, vec3<f32>(relative_world, input.depth)) + camera.world_to_screen_y.w,
    );

    if abs(input.normal_distance) > 0.0 || abs(input.tangent_distance) > 0.0 {
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
        let previous_tangent = safe_unit(previous_screen);
        let next_tangent = safe_unit(next_screen);
        let combined_normal = previous_normal + next_normal;
        let turn = cross_2d(previous_tangent, next_tangent);
        let reverses = abs(turn) <= 0.000001 && dot(previous_tangent, next_tangent) < 0.0;
        let side = sign(input.normal_distance);
        let outer_side = -sign(turn);
        var extrusion = next_normal * input.normal_distance + next_tangent * input.tangent_distance;
        var miter_offset = vec2<f32>(0.0, 0.0);
        var miter_multiple = 1e30;
        var miter_valid = false;

        if dot(combined_normal, combined_normal) > 0.000001 {
            let miter = safe_unit(combined_normal);
            let denominator = dot(miter, next_normal);
            if abs(denominator) > 0.001 {
                miter_multiple = abs(1.0 / denominator);
                miter_offset = miter * (input.normal_distance / denominator);
                miter_valid = true;
            }
        }

        // Roles 1..3 are the two sides of a segment endpoint at a join.
        // Inner endpoints meet at the miter intersection. The outer endpoint
        // uses the selected corner, except for an in-limit miter join. This
        // makes adjacent segment quads disjoint in their interiors.
        if input.stroke_role >= 1.0 && input.stroke_role <= 3.0 {
            if reverses {
                // Scene validation rejects exact source retraces. Collapsing a
                // projection-induced reversal still prevents a crossed quad
                // from injecting NaN or backend-dependent winding.
                extrusion = vec2<f32>(0.0, 0.0);
            } else if abs(turn) <= 0.000001 {
                extrusion = next_normal * input.normal_distance;
            } else if !same_side(side, outer_side) {
                if miter_valid && miter_multiple <= input.miter_limit {
                    extrusion = miter_offset;
                } else {
                    extrusion = vec2<f32>(0.0, 0.0);
                }
            } else if input.stroke_role == 2.0 && miter_valid && miter_multiple <= input.miter_limit {
                extrusion = miter_offset;
            } else if input.stroke_parameter < 0.0 {
                extrusion = previous_normal * input.normal_distance;
            } else {
                extrusion = next_normal * input.normal_distance;
            }
        } else if input.stroke_role >= 4.0 {
            // Join fill vertices carry a candidate outer side. Only the fan
            // matching the actual projected turn survives; the other fan is
            // collapsed to the path center and therefore cannot blend twice.
            var candidate_side = side;
            if input.stroke_role == 5.0 || input.stroke_role == 7.0 || input.stroke_role == 9.0 {
                candidate_side = -side;
            }
            var join_active = abs(turn) > 0.000001 && same_side(candidate_side, outer_side);
            if input.stroke_role == 6.0 || input.stroke_role == 7.0 {
                join_active = join_active && (!miter_valid || miter_multiple > input.miter_limit);
            }
            if !join_active {
                extrusion = vec2<f32>(0.0, 0.0);
            } else if input.stroke_role == 5.0 || input.stroke_role == 7.0 || input.stroke_role == 9.0 {
                if miter_valid && miter_multiple <= input.miter_limit {
                    extrusion = miter_offset;
                } else {
                    extrusion = vec2<f32>(0.0, 0.0);
                }
            } else if input.stroke_role == 8.0 {
                let start = previous_normal * candidate_side;
                let finish = next_normal * candidate_side;
                let angle = atan2(cross_2d(start, finish), dot(start, finish));
                let amount_angle = angle * input.stroke_parameter;
                let rotated = vec2<f32>(
                    start.x * cos(amount_angle) - start.y * sin(amount_angle),
                    start.x * sin(amount_angle) + start.y * cos(amount_angle),
                );
                extrusion = rotated * abs(input.normal_distance);
            } else if input.stroke_parameter < 0.0 {
                extrusion = previous_normal * input.normal_distance;
            } else {
                extrusion = next_normal * input.normal_distance;
            }
        } else if miter_valid && miter_multiple <= input.miter_limit {
            extrusion = miter_offset + next_tangent * input.tangent_distance;
        }

        if input.stroke_role >= 1.0 && input.stroke_role <= 3.0 {
            extrusion += next_tangent * input.tangent_distance;
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
