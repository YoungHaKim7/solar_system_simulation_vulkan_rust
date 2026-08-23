#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 color;
layout(location = 2) in float alpha;

layout(push_constant) uniform Push {
    vec4 transform;
    vec4 params;
} pc;

layout(location = 0) out vec3 v_color;
layout(location = 1) out float v_alpha;

void main() {
    v_color = color;
    v_alpha = alpha;
    gl_Position = vec4(
        pos.x * pc.transform.x + pc.transform.z,
        -(pos.y * pc.transform.y + pc.transform.w),
        0.0,
        1.0
    );
}
