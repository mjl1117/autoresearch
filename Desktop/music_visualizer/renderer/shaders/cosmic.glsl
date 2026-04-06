#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;
uniform vec3  u_color_b;
uniform float u_ring_times[4];
uniform int   u_ring_count;
void main() { fragColor = vec4(u_color_a * (0.5 + u_intensity * 0.5), 1.0); }
