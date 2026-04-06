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

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
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

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8775, 0.4794, -0.4794, 0.8775);
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = rot * p * 2.1 + vec2(31.7, 17.3);
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 uv = v_uv;
    float ar = u_resolution.x / u_resolution.y;
    vec2 p = vec2((uv.x - 0.5) * ar, uv.y - 0.5);
    float t = u_time * 0.12;

    vec2 q = vec2(fbm(p + t), fbm(p + vec2(1.73, 9.24) + t * 0.85));
    vec2 r = vec2(
        fbm(p + 1.6 * q + vec2(1.7, 9.2) + t * 0.28),
        fbm(p + 1.0 * q + vec2(8.3, 2.8) + t * 0.44)
    );
    float f = fbm(p + r);

    vec3 col = vec3(0.02, 0.02, 0.05);
    col = mix(col, u_color_a * 0.35, clamp(f * 2.5, 0.0, 1.0));
    col = mix(col, u_color_b * 0.45, clamp(length(q) * 0.7, 0.0, 1.0));
    col = mix(col, u_color_a * 0.6,  clamp(f * f * 3.5, 0.0, 1.0));
    col *= 0.4 + 0.6 * u_intensity;

    for (int wi = 0; wi < 4; wi++) {
        float fi = float(wi);
        float freq = 1.0 + fi * 0.55;
        float speed = 0.25 + fi * 0.08;
        float amp = (0.07 - fi * 0.014) * (0.6 + u_intensity * 1.0);
        float warp = fbm(vec2(p.x * 1.8 + fi * 2.1, t * 0.4)) * 0.08;
        float wave_y = sin(p.x * freq * 3.14159 + u_time * speed + warp) * amp;
        float dist = abs(p.y - wave_y);
        float line = smoothstep(0.022 - fi * 0.002, 0.0, dist);
        float tint = (fi + uv.x) / 5.0;
        col += mix(u_color_a, u_color_b, tint) * line * (1.0 - fi * 0.18) * 0.9;
    }

    vec2 orb_pos[3];
    orb_pos[0] = vec2(sin(u_time * 0.22) * 0.45 * ar, cos(u_time * 0.17) * 0.28);
    orb_pos[1] = vec2(cos(u_time * 0.13) * 0.38 * ar, sin(u_time * 0.27) * 0.35);
    orb_pos[2] = vec2(sin(u_time * 0.18 + 1.57) * 0.28 * ar, cos(u_time * 0.11 + 2.09) * 0.30);
    float orb_sz[3];
    orb_sz[0] = 0.30 + u_intensity * 0.12;
    orb_sz[1] = 0.24 + u_intensity * 0.09;
    orb_sz[2] = 0.18 + u_intensity * 0.07;
    vec3 orb_col[3];
    orb_col[0] = u_color_a;
    orb_col[1] = u_color_b;
    orb_col[2] = u_color_a * 0.6 + u_color_b * 0.4;

    for (int oi = 0; oi < 3; oi++) {
        float d = length(p - orb_pos[oi]);
        float s = orb_sz[oi];
        col += orb_col[oi] * exp(-d * d / (s * s)) * 0.13;
    }

    for (int pi = 0; pi < 48; pi++) {
        vec2 seed = vec2(float(pi) * 0.3742, float(pi) * 0.7373);
        vec2 base = hash2(seed) * 2.0 - 1.0;
        base.x *= ar;
        float drift_x = sin(u_time * (0.15 + hash(seed) * 0.2) + seed.x * 6.28) * 0.18;
        float drift_y = cos(u_time * (0.12 + hash(seed + 1.0) * 0.15) + seed.y * 6.28) * 0.15;
        vec2 pos = base + vec2(drift_x, drift_y);
        pos.x = mod(pos.x + ar, 2.0 * ar) - ar;
        pos.y = mod(pos.y + 0.5, 1.0) - 0.5;
        float pd = length(p - pos);
        float psize = 0.007 + hash(seed + 2.0) * 0.009;
        float particle = smoothstep(psize, 0.0, pd);
        float pbrightness = 0.4 + hash(seed + floor(u_time * 0.5)) * 0.6;
        col += mix(u_color_a, u_color_b, hash(seed + 3.0)) * particle * pbrightness * 0.65;
    }

    col += u_color_a * u_pulse * 0.35 * exp(-length(p) * (2.5 - u_intensity));
    col += vec3(u_dissonance * u_pulse * 0.12);

    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.3;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.6);
    fragColor = vec4(col, 1.0);
}
