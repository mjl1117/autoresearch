#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
void main() { fragColor = texture(u_scene, v_uv); }
