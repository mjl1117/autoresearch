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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float sdTriangle(vec2 p, float r) {
    const float k = 1.7320508;
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) * 0.5;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

void main() {
    vec2 uv = v_uv;

    float glitchStrength = u_dissonance * u_pulse;
    float slice = floor(uv.y * 24.0);
    float timeSlot = floor(u_time * 28.0);
    float sliceOffset = (hash(vec2(slice, timeSlot)) - 0.5) * glitchStrength * 0.18;
    uv.x = fract(uv.x + sliceOffset);

    vec2 p = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    vec3 col = vec3(0.015, 0.015, 0.03);
    vec2 grid = abs(fract(p * 5.0 + 0.5) - 0.5);
    float gridLine = min(grid.x, grid.y);
    col += 0.04 * (1.0 - smoothstep(0.0, 0.03, gridLine)) * vec3(0.4, 0.6, 1.0);

    float rotA = u_time * 0.25 + u_intensity * 0.5;
    vec2 tp = p * rot2(rotA);
    float triSize = 0.28 + u_intensity * 0.12;

    float aberr = u_dissonance * 0.012 + u_pulse * 0.006;
    float tri_r = sdTriangle(p * rot2(rotA) + vec2(aberr, 0.0), triSize);
    float tri_c = sdTriangle(tp, triSize);
    float tri_b = sdTriangle(p * rot2(rotA) - vec2(aberr, 0.0), triSize);
    col.r += u_color_a.r * (1.0 - smoothstep(0.0, 0.009, abs(tri_r))) * 0.95;
    col.g += 0.15 * (1.0 - smoothstep(0.0, 0.011, abs(tri_c)));
    col.b += u_color_b.b * (1.0 - smoothstep(0.0, 0.009, abs(tri_b))) * 0.7;
    col += u_color_a * max(0.0, -tri_c) * 0.04;

    vec2 tpG = p * rot2(rotA + 0.08);
    float triG = sdTriangle(tpG, triSize * 0.98);
    col += u_color_a * (1.0 - smoothstep(0.0, 0.007, abs(triG))) * 0.25;

    vec2 bp = p * rot2(-u_time * 0.4 + u_intensity * 0.3);
    float boxSize = 0.16 + u_brightness * 0.06;
    float box = sdBox(bp, vec2(boxSize));
    col.r += u_color_b.r * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.5;
    col.g += u_color_b.g * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.8;
    col.b += u_color_b.b * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.9;

    vec2 bp2 = p * rot2(u_time * 0.6 - 1.2);
    float box2 = sdBox(bp2, vec2(0.08 + u_intensity * 0.04));
    col += u_color_a * (1.0 - smoothstep(0.0, 0.006, abs(box2))) * 0.4;

    float barCount = 20.0;
    float barIdx = floor(uv.x * barCount);
    float barX   = fract(uv.x * barCount);
    float hSeed  = hash(vec2(barIdx * 0.13, 1.0));
    float barH   = clamp((0.05 + hSeed * 0.15) * (0.3 + u_intensity * 1.4), 0.0, 0.55);
    float barFill = step(1.0 - uv.y * 0.8, barH);
    float gap = smoothstep(0.0, 0.06, barX) * smoothstep(1.0, 0.94, barX);
    vec3 barCol = mix(u_color_a, u_color_b, barIdx / (barCount - 1.0));
    col += barCol * barFill * gap * 0.85;

    float scanline = mod(gl_FragCoord.y, 3.0) < 1.0 ? 0.80 : 1.0;
    col *= scanline;

    col += u_color_a * u_pulse * 0.18 * (1.0 - length(p) * 0.6);

    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.6;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.5);
    fragColor = vec4(col, 1.0);
}
