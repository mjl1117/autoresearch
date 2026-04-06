#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform vec2 u_direction;    // (1,0) for horizontal, (0,1) for vertical
uniform vec2 u_resolution;

const float WEIGHTS[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162);

void main() {
    vec2 texel = u_direction / u_resolution;
    vec3 col = texture(u_tex, v_uv).rgb * WEIGHTS[0];
    for (int i = 1; i < 5; i++) {
        col += texture(u_tex, v_uv + texel * float(i)).rgb * WEIGHTS[i];
        col += texture(u_tex, v_uv - texel * float(i)).rgb * WEIGHTS[i];
    }
    fragColor = vec4(col, 1.0);
}
