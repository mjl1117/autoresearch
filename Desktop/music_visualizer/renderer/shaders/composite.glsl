#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex_a;
uniform sampler2D u_tex_b;
uniform float u_blend;
void main() {
    vec4 a = texture(u_tex_a, v_uv);
    vec4 b = texture(u_tex_b, v_uv);
    fragColor = mix(a, b, u_blend);
}
