#version 450

layout(location = 0) in vec2 center;
layout(location = 1) in vec2 corner;
layout(location = 2) in float radius;
layout(location = 3) in vec3 color;
layout(location = 4) in float glow;

layout(push_constant) uniform Push {
    vec4 transform;
    vec4 params;
} pc;

layout(location = 0) out vec2 v_corner;
layout(location = 1) out vec3 v_color;
layout(location = 2) out float v_glow;
layout(location = 3) out vec2 v_world;

void main() {
    v_corner = corner;
    v_color = color;
    v_glow = glow;
    vec2 world = center + corner * radius;
    v_world = world;
    gl_Position = vec4(
        world.x * pc.transform.x + pc.transform.z,
        -(world.y * pc.transform.y + pc.transform.w),
        0.0,
        1.0
    );
}

