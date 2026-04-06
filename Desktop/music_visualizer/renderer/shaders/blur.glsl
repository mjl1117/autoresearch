#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform vec2 u_direction;
uniform vec2 u_resolution;
void main() { fragColor = texture(u_tex, v_uv); }
