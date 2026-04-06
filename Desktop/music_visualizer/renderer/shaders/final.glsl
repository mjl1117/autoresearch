#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_prev_frame;
uniform vec2  u_resolution;
uniform float u_motion_blur;      // 0.0 = off, 0.15 = organic motion blur
uniform float u_bloom_intensity;

vec3 fxaa(sampler2D tex, vec2 uv, vec2 resolution) {
    vec2 px = 1.0 / resolution;
    vec3 rgbNW = texture(tex, uv + vec2(-1.0, -1.0) * px).rgb;
    vec3 rgbNE = texture(tex, uv + vec2( 1.0, -1.0) * px).rgb;
    vec3 rgbSW = texture(tex, uv + vec2(-1.0,  1.0) * px).rgb;
    vec3 rgbSE = texture(tex, uv + vec2( 1.0,  1.0) * px).rgb;
    vec3 rgbM  = texture(tex, uv).rgb;

    vec3 luma_w = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma_w);
    float lumaNE = dot(rgbNE, luma_w);
    float lumaSW = dot(rgbSW, luma_w);
    float lumaSE = dot(rgbSE, luma_w);
    float lumaM  = dot(rgbM,  luma_w);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    if (lumaMax - lumaMin < 0.1) return rgbM;

    vec2 dir = vec2(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
         ((lumaNW + lumaSW) - (lumaNE + lumaSE))
    );
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125, 0.0078125);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;

    vec3 rgbA = 0.5 * (
        texture(tex, uv + dir * (1.0/3.0 - 0.5)).rgb +
        texture(tex, uv + dir * (2.0/3.0 - 0.5)).rgb
    );
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, uv + dir * -0.5).rgb +
        texture(tex, uv + dir *  0.5).rgb
    );
    float lumaB = dot(rgbB, luma_w);
    return (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
}

void main() {
    vec2 uv = v_uv;

    vec3 col = fxaa(u_scene, uv, u_resolution);

    if (u_motion_blur > 0.0) {
        vec3 prev = texture(u_prev_frame, uv).rgb;
        col = mix(col, prev, u_motion_blur);
    }

    vec3 bloom = texture(u_bloom, uv).rgb;
    col += bloom * u_bloom_intensity;

    col = col / (col + vec3(1.0));
    col = pow(col, vec3(1.0 / 2.2));

    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 0.5;
    col *= vig;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
