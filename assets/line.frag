#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in float v_alpha;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(v_color, v_alpha);
}

