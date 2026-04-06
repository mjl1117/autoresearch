#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;

void main() {
    vec3 col = texture(u_scene, v_uv).rgb;
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float threshold = 0.65;
    float knee = 0.1;
    float extracted = smoothstep(threshold - knee, threshold + knee, luma);
    fragColor = vec4(col * extracted, 1.0);
}
