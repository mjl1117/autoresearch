#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_prev_frame;
uniform vec2  u_resolution;
uniform float u_motion_blur;
uniform float u_bloom_intensity;
void main() { fragColor = texture(u_scene, v_uv); }
