#version 330
uniform sampler2D u_overlay;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    // pygame surfaces are top-down; OpenGL UVs are bottom-up — flip Y
    fragColor = texture(u_overlay, vec2(v_uv.x, 1.0 - v_uv.y));
}
