#version 450

layout(push_constant) uniform Push {
    vec4 transform;
    vec4 params;
} pc;

layout(location = 0) in vec2 v_corner;
layout(location = 1) in vec3 v_color;
layout(location = 2) in float v_glow;
layout(location = 3) in vec2 v_world;
layout(location = 0) out vec4 f_color;

void main() {
    float d = length(v_corner);
    float disc = 1.0 - smoothstep(0.86, 1.0, d);
    vec2 n = v_corner / max(d, 1e-4);
    vec2 to_sun = pc.params.xy - v_world;
    float dist = max(length(to_sun), 1e-4);
    float lam = clamp(dot(n, to_sun / dist), 0.0, 1.0);
    vec3 col = v_color * (0.45 + 0.55 * lam + 0.15);
    float alpha = disc;
    if (v_glow > 0.5) {
        float halo = exp(-d * 2.2) * 0.85;
        col = mix(col, vec3(1.0, 0.97, 0.88), smoothstep(0.6, 0.0, d));
        col += vec3(1.0, 0.70, 0.30) * halo;
        alpha = max(alpha, clamp(halo, 0.0, 1.0));
    }
    f_color = vec4(col, alpha);
}
