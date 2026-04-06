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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

void main() {
    vec2 uv = v_uv;
    float ar = u_resolution.x / u_resolution.y;
    vec2 p = vec2((uv.x - 0.5) * ar, uv.y - 0.5);

    vec3 col = vec3(0.008, 0.008, 0.018);

    vec2 starGrid = uv * vec2(70.0, 48.0);
    vec2 starCell = floor(starGrid);
    vec2 starFrac = fract(starGrid);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 cell = starCell + vec2(float(dx), float(dy));
            float h = hash(cell);
            if (h > 0.72) {
                vec2 starPos = vec2(hash(cell + 1.3), hash(cell + 2.7));
                vec2 offset = starFrac - starPos - vec2(float(dx), float(dy));
                float starSize = 0.035 + hash(cell + 5.1) * 0.045;
                float brightness_mod = 0.35 + hash(cell + 3.3) * 0.65;
                float twinkle = 0.65 + 0.35 * sin(
                    u_time * (0.8 + hash(cell) * 2.5) + hash(cell + 9.1) * 6.28
                );
                float dist = length(offset);
                float star = smoothstep(starSize, 0.0, dist);
                float spike_h = smoothstep(starSize * 3.0, 0.0, abs(offset.y)) *
                                smoothstep(0.5, 0.0, abs(offset.x)) * 0.3;
                float spike_v = smoothstep(starSize * 3.0, 0.0, abs(offset.x)) *
                                smoothstep(0.5, 0.0, abs(offset.y)) * 0.3;
                vec3 starCol = mix(vec3(0.9, 0.95, 1.0), u_color_a * 1.5, hash(cell + 7.1) * 0.5);
                col += starCol * (star + spike_h + spike_v) * brightness_mod * twinkle;
            }
        }
    }

    float warpT = u_time * 0.05;
    float nx = noise(p * 1.5 + vec2(warpT, 0.0)) * 0.04;
    float ny = noise(p * 1.5 + vec2(0.0, warpT * 0.8)) * 0.04;

    float nebula1 = exp(-length(p - vec2(-0.18 + nx, 0.05)) * (1.8 - u_intensity * 0.6));
    float nebula2 = exp(-length(p - vec2(0.25 + ny, -0.08)) * (2.5 - u_brightness * 0.5));
    float nebula3 = exp(-length(p - vec2(-0.05, 0.20)) * 3.2) * 0.5;

    col += u_color_a * nebula1 * (0.22 + u_intensity * 0.18);
    col += u_color_b * nebula2 * (0.15 + u_brightness * 0.12);
    col += mix(u_color_a, u_color_b, 0.5) * nebula3 * 0.12;

    for (int ri = 0; ri < 4; ri++) {
        if (ri >= u_ring_count) break;
        float age = u_time - u_ring_times[ri];
        if (age < 0.0 || age > 7.0) continue;
        float radius = age * 0.22 + 0.02;
        float fade = pow(1.0 - age / 7.0, 1.5);
        float dist = abs(length(p) - radius);
        float ringWidth = 0.018 + age * 0.003;
        float ring = smoothstep(ringWidth, 0.0, dist) * fade;
        float secondaryRing = smoothstep(ringWidth * 2.5, 0.0, abs(length(p) - radius * 0.7)) * fade * 0.3;
        vec3 ringCol = mix(u_color_a, vec3(0.85, 0.9, 1.0), 0.4);
        col += ringCol * (ring + secondaryRing) * (0.45 + u_intensity * 0.3);
    }

    float coreRadius = 4.0 - u_intensity * 2.5;
    float core = exp(-length(p) * coreRadius);
    col += u_color_a * core * (0.25 + u_intensity * 0.45);
    col += vec3(0.95, 0.97, 1.0) * core * core * (0.08 + u_intensity * 0.18);

    col += u_color_a * u_pulse * exp(-length(p) * 2.8) * 0.45;

    float grain = hash(uv + fract(u_time * 0.39)) * 0.045 - 0.0225;
    col += grain;

    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.9;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.4);
    fragColor = vec4(col, 1.0);
}
