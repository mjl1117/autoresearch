from __future__ import annotations
from pathlib import Path

import moderngl
import numpy as np

from engine.render_params import RenderParams, Style

_SHADER_DIR = Path(__file__).parent / "shaders"
_QUAD_VERTS = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype="f4")


def _load(name: str) -> str:
    return (_SHADER_DIR / name).read_text()


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._w = width
        self._h = height
        self._ring_times: list[float] = []

        vbo = ctx.buffer(_QUAD_VERTS)
        vert_src = _load("quad.vert")

        self._prog: dict[Style, moderngl.Program] = {
            Style.GEOMETRIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("geometric.glsl"),
            ),
            Style.ORGANIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("organic.glsl"),
            ),
            Style.COSMIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("cosmic.glsl"),
            ),
        }
        self._vaos: dict[Style, moderngl.VertexArray] = {
            style: ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])
            for style, prog in self._prog.items()
        }

        self._prog_composite = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("composite.glsl"),
        )
        self._vao_composite = ctx.vertex_array(
            self._prog_composite, [(vbo, "2f", "in_vert")]
        )

        self._prog_bloom_extract = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("bloom_extract.glsl"),
        )
        self._vao_bloom_extract = ctx.vertex_array(
            self._prog_bloom_extract, [(vbo, "2f", "in_vert")]
        )

        self._prog_blur = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("blur.glsl"),
        )
        self._vao_blur = ctx.vertex_array(
            self._prog_blur, [(vbo, "2f", "in_vert")]
        )

        self._prog_final = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("final.glsl"),
        )
        self._vao_final = ctx.vertex_array(
            self._prog_final, [(vbo, "2f", "in_vert")]
        )

        def _make_fbo() -> tuple[moderngl.Framebuffer, moderngl.Texture]:
            tex = ctx.texture((width, height), 3)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            fbo = ctx.framebuffer(color_attachments=[tex])
            return fbo, tex

        self._fbo_a, self._tex_a = _make_fbo()
        self._fbo_b, self._tex_b = _make_fbo()
        self._fbo_composite, self._tex_composite = _make_fbo()
        self._fbo_prev, self._tex_prev = _make_fbo()
        self._fbo_bright, self._tex_bright = _make_fbo()
        self._fbo_blur_h, self._tex_blur_h = _make_fbo()
        self._fbo_blur_v, self._tex_blur_v = _make_fbo()
        self._fbo_final, self._tex_final = _make_fbo()

    def render_frame(self, params: RenderParams, elapsed_time: float) -> None:
        # Track ring pulses for cosmic mode
        if params.pulse > 0.6:
            self._ring_times.append(elapsed_time)
            if len(self._ring_times) > 4:
                self._ring_times.pop(0)

        # 1. Render active style → fbo_a
        self._render_style(params.active_style, params, elapsed_time, self._fbo_a)

        # 2. Render blend target → fbo_b (only if actively blending)
        if params.blend_weight > 0.001:
            self._render_style(params.blend_target, params, elapsed_time, self._fbo_b)

        su = self._set_uniform

        # 3. Composite blend
        self._fbo_composite.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_a.use(location=0)
        self._tex_b.use(location=1)
        p = self._prog_composite
        su(p, "u_tex_a", 0)
        su(p, "u_tex_b", 1)
        su(p, "u_blend", params.blend_weight)
        self._vao_composite.render(moderngl.TRIANGLE_STRIP)

        # 4. Bloom extraction
        self._fbo_bright.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_composite.use(location=0)
        su(self._prog_bloom_extract, "u_scene", 0)
        self._vao_bloom_extract.render(moderngl.TRIANGLE_STRIP)

        # 5. Horizontal blur
        self._fbo_blur_h.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_bright.use(location=0)
        su(self._prog_blur, "u_tex", 0)
        su(self._prog_blur, "u_direction", (1.0, 0.0))
        su(self._prog_blur, "u_resolution", (float(self._w), float(self._h)))
        self._vao_blur.render(moderngl.TRIANGLE_STRIP)

        # 6. Vertical blur
        self._fbo_blur_v.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_blur_h.use(location=0)
        su(self._prog_blur, "u_tex", 0)
        su(self._prog_blur, "u_direction", (0.0, 1.0))
        self._vao_blur.render(moderngl.TRIANGLE_STRIP)

        # 7. Final pass
        self._fbo_final.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_composite.use(location=0)
        self._tex_blur_v.use(location=1)
        self._tex_prev.use(location=2)
        p = self._prog_final
        su(p, "u_scene", 0)
        su(p, "u_bloom", 1)
        su(p, "u_prev_frame", 2)
        su(p, "u_resolution", (float(self._w), float(self._h)))
        su(p, "u_motion_blur", 0.15 if params.active_style == Style.ORGANIC else 0.0)
        su(p, "u_bloom_intensity", 0.4)
        self._vao_final.render(moderngl.TRIANGLE_STRIP)

        # 8. Copy final → prev for next frame
        self._ctx.copy_framebuffer(dst=self._fbo_prev, src=self._fbo_final)

    def read_pixels(self) -> np.ndarray:
        """Return current frame as (H, W, 3) uint8 array."""
        raw = self._fbo_final.read(components=3)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self._h, self._w, 3)
        return np.flipud(arr)

    def release(self) -> None:
        for fbo, tex in [
            (self._fbo_a, self._tex_a),
            (self._fbo_b, self._tex_b),
            (self._fbo_composite, self._tex_composite),
            (self._fbo_prev, self._tex_prev),
            (self._fbo_bright, self._tex_bright),
            (self._fbo_blur_h, self._tex_blur_h),
            (self._fbo_blur_v, self._tex_blur_v),
            (self._fbo_final, self._tex_final),
        ]:
            fbo.release()
            tex.release()
        for vao in self._vaos.values():
            vao.release()
        for prog in self._prog.values():
            prog.release()
        for vao in (self._vao_composite, self._vao_bloom_extract, self._vao_blur, self._vao_final):
            vao.release()
        for prog in (self._prog_composite, self._prog_bloom_extract, self._prog_blur, self._prog_final):
            prog.release()

    @staticmethod
    def _set_uniform(prog: moderngl.Program, name: str, value: object) -> None:
        """Set a uniform only if it is active (not optimized away by the compiler)."""
        if name in prog:
            prog[name].value = value

    def _render_style(
        self,
        style: Style,
        params: RenderParams,
        elapsed_time: float,
        fbo: moderngl.Framebuffer,
    ) -> None:
        fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        prog = self._prog[style]
        su = self._set_uniform
        su(prog, "u_time", elapsed_time)
        su(prog, "u_resolution", (float(self._w), float(self._h)))
        su(prog, "u_intensity", params.intensity)
        su(prog, "u_brightness", params.brightness)
        su(prog, "u_pulse", params.pulse)
        su(prog, "u_dissonance", params.dissonance_raw)
        su(prog, "u_color_a", params.color_a)
        su(prog, "u_color_b", params.color_b)
        if style == Style.COSMIC:
            ring_times = (self._ring_times + [0.0, 0.0, 0.0, 0.0])[:4]
            su(prog, "u_ring_times", tuple(ring_times))
            su(prog, "u_ring_count", len(self._ring_times))
        self._vaos[style].render(moderngl.TRIANGLE_STRIP)
