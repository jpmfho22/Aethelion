"""Renderizador acelerado por GPU utilizando ModernGL (OpenGL 3.3+).

Implementa o pipeline na GPU, espelhando os mesmos parâmetros e efeitos retrô
do pipeline de software (CPU):
  * vertex snap (jitter de polígonos);
  * interpolação UV afim vs perspectiva correta (via noperspective no GLSL);
  * iluminação Gouraud, Flat e Unlit;
  * quantização de bits de cor, dithering Bayer 4x4 e paleta fixa de 16 cores;
  * backface culling e teste de profundidade em hardware.
"""
from __future__ import annotations

import time
import numpy as np

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False

from . import mathutil as mu
from .geometry import Instance, build_scene, scene_tri_count
from .pipeline import FrameStats, RenderSettings
from .texture import Texture, make_textures

F = np.float32

VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform vec3 u_light_dir;
uniform float u_ambient;
uniform float u_vertex_snap;
uniform vec2 u_viewport;
uniform int u_shading;
uniform int u_precision;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec2 v_uv;
noperspective out vec2 v_uv_affine;
out float v_lit;
flat out float v_flat_lit;

vec4 apply_precision(vec4 v, int mode) {
    if (mode == 1) { // float16 range
        v = clamp(v, -65500.0, 65500.0);
    } else if (mode == 2) { // fixed 16.16
        v = clamp(v, -32767.0, 32767.0);
        v = floor(v * 65536.0 + 0.5) / 65536.0;
    } else if (mode == 3) { // fixed 12.4
        v = clamp(v, -2047.0, 2047.0);
        v = floor(v * 16.0 + 0.5) / 16.0;
    }
    return v;
}

void main() {
    vec4 clip = u_mvp * vec4(in_position, 1.0);
    if (u_precision > 0) {
        clip = apply_precision(clip, u_precision);
    }

    // Snap de vértices em grade de tela (efeito jitter PS1) e precisão NDC
    if (clip.w > 0.0) {
        vec2 ndc = clip.xy / clip.w;
        if (u_precision > 0) {
            ndc = apply_precision(vec4(ndc, 0.0, 0.0), u_precision).xy;
        }
        if (u_vertex_snap > 0.0) {
            vec2 screen = (ndc * 0.5 + 0.5) * u_viewport;
            screen = floor(screen * u_vertex_snap + 0.5) / u_vertex_snap;
            ndc = (screen / u_viewport - 0.5) * 2.0;
        }
        clip.xy = ndc * clip.w;
    }

    gl_Position = clip;
    v_uv = in_uv;
    v_uv_affine = in_uv;

    // Normal em espaço de mundo (inversa-transposta) e iluminação difusa
    vec3 nrm = normalize(transpose(inverse(mat3(u_model))) * in_normal);
    float lit = max(dot(nrm, u_light_dir), 0.0);
    lit = u_ambient + (1.0 - u_ambient) * lit;
    v_lit = lit;
    v_flat_lit = lit;
}
"""

FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_texture;
uniform bool u_textured;
uniform vec3 u_tint;
uniform int u_shading;
uniform bool u_affine_warp;
uniform int u_color_bits;
uniform bool u_dither;
uniform bool u_palette;

in vec2 v_uv;
noperspective in vec2 v_uv_affine;
in float v_lit;
flat in float v_flat_lit;

out vec4 fragColor;

const float bayer4[16] = float[16](
     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0
);

const vec3 palette16[16] = vec3[16](
    vec3(0.04, 0.04, 0.07), vec3(0.33, 0.20, 0.16), vec3(0.52, 0.18, 0.20), vec3(0.20, 0.36, 0.26),
    vec3(0.72, 0.30, 0.22), vec3(0.36, 0.28, 0.44), vec3(0.25, 0.47, 0.62), vec3(0.70, 0.70, 0.72),
    vec3(0.42, 0.42, 0.45), vec3(0.85, 0.47, 0.22), vec3(0.88, 0.72, 0.30), vec3(0.55, 0.78, 0.40),
    vec3(0.92, 0.55, 0.52), vec3(0.45, 0.60, 0.85), vec3(0.80, 0.84, 0.90), vec3(0.98, 0.98, 0.96)
);

void main() {
    float lit = (u_shading == 2) ? 1.0 : ((u_shading == 1) ? v_flat_lit : v_lit);
    vec2 uv = u_affine_warp ? v_uv_affine : v_uv;

    vec3 albedo = vec3(0.8);
    if (u_textured) {
        albedo = texture(u_texture, uv).rgb;
    }

    vec3 color = clamp(albedo * u_tint * lit, 0.0, 1.0);

    // Efeitos de restrição de cor (dithering e quantização)
    int bx = int(mod(gl_FragCoord.x, 4.0));
    int by = int(mod(gl_FragCoord.y, 4.0));
    float dither_val = bayer4[by * 4 + bx];

    if (u_palette) {
        if (u_dither) {
            color = clamp(color + (dither_val - 0.5) * 0.12, 0.0, 1.0);
        }
        float min_dist = 999.0;
        vec3 best_col = palette16[0];
        for (int i = 0; i < 16; i++) {
            vec3 diff = color - palette16[i];
            float dist = dot(diff, diff);
            if (dist < min_dist) {
                min_dist = dist;
                best_col = palette16[i];
            }
        }
        color = best_col;
    } else if (u_color_bits < 8) {
        float levels = pow(2.0, float(u_color_bits)) - 1.0;
        if (u_dither) {
            color = clamp(color + (dither_val - 0.5) / levels, 0.0, 1.0);
        }
        color = floor(color * levels + 0.5) / levels;
    }

    fragColor = vec4(color, 1.0);
}
"""


class GPURenderer:
    """Renderizador offscreen acelerado por hardware via ModernGL."""

    LIGHT_DIR = mu.normalize(np.array([0.45, 0.8, 0.42], dtype=F))
    AMBIENT = 0.22

    def __init__(self, settings: RenderSettings | None = None):
        if not HAS_MODERNGL:
            raise RuntimeError("A biblioteca moderngl não está instalada. Instale com 'pip install moderngl'.")

        self.settings = settings or RenderSettings()
        self.ctx = moderngl.create_context(standalone=True)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        self.fbo: moderngl.Framebuffer | None = None
        self._init_fbo(self.settings.width, self.settings.height)

        self.scene: list[Instance] = build_scene(self.settings.target_tris)
        self.textures: dict[str, Texture] = make_textures(self.settings.texture_res, self.settings.color_bits)
        self._tex_order = ["bricks", "gradient", "checker", "noise"]

        self._gpu_vaos: list[moderngl.VertexArray] = []
        self._gpu_textures: dict[str, moderngl.Texture] = {}
        self._build_gpu_resources()

        self.stats = FrameStats()
        self.camera_eye = np.array([0.0, 1.5, 5.0], dtype=F)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=F)

    def _init_fbo(self, w: int, h: int) -> None:
        if self.fbo is not None:
            self.fbo.release()
        self.fbo = self.ctx.simple_framebuffer((int(w), int(h)))

    def _build_gpu_resources(self) -> None:
        for vao in self._gpu_vaos:
            vao.release()
        self._gpu_vaos.clear()

        for inst in self.scene:
            m = inst.mesh
            vbo_pos = self.ctx.buffer(m.positions.astype(np.float32).tobytes())
            vbo_nrm = self.ctx.buffer(m.normals.astype(np.float32).tobytes())
            vbo_uv = self.ctx.buffer(m.uvs.astype(np.float32).tobytes())
            ibo = self.ctx.buffer(m.indices.astype(np.int32).tobytes())

            vao = self.ctx.vertex_array(
                self.prog,
                [
                    (vbo_pos, '3f', 'in_position'),
                    (vbo_nrm, '3f', 'in_normal'),
                    (vbo_uv, '2f', 'in_uv'),
                ],
                ibo,
            )
            self._gpu_vaos.append(vao)

        self._upload_textures()

    def _upload_textures(self) -> None:
        for t in self._gpu_textures.values():
            t.release()
        self._gpu_textures.clear()

        self._last_filter = self.settings.texture_filter
        filter_flag = moderngl.LINEAR if self.settings.texture_filter == "bilinear" else moderngl.NEAREST
        for name, tex in self.textures.items():
            uint8_data = (np.clip(tex.data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            gl_tex = self.ctx.texture((tex.res, tex.res), 3, uint8_data.tobytes())
            gl_tex.filter = (filter_flag, filter_flag)
            gl_tex.repeat_x = True
            gl_tex.repeat_y = True
            self._gpu_textures[name] = gl_tex

    def rebuild_scene(self) -> None:
        self.scene = build_scene(self.settings.target_tris)
        self._build_gpu_resources()

    def rebuild_textures(self) -> None:
        self.textures = make_textures(self.settings.texture_res, self.settings.color_bits)
        self._upload_textures()

    def sync(self) -> None:
        s = self.settings
        if (self.fbo.width, self.fbo.height) != (s.width, s.height):
            self._init_fbo(s.width, s.height)
        if scene_tri_count(self.scene) and abs(
                scene_tri_count(self.scene) - s.target_tris) > max(40, s.target_tris * 0.25):
            self.rebuild_scene()
        tex_sample = next(iter(self.textures.values()), None)
        if (tex_sample is None or tex_sample.res != s.texture_res
                or tex_sample.color_bits != s.color_bits
                or getattr(self, "_last_filter", None) != s.texture_filter):
            self.rebuild_textures()

    def render(self, t: float = 0.0) -> np.ndarray:
        s = self.settings
        st = FrameStats()
        t_start = time.perf_counter()

        for inst in self.scene:
            inst.update(t)

        aspect = s.width / max(1, s.height)
        view = mu.look_at(self.camera_eye, self.camera_target)
        proj = mu.perspective(s.fov, aspect, s.near, s.far)
        vp = proj @ view

        t_geo = time.perf_counter()
        st.tris_submitted = scene_tri_count(self.scene)
        st.ms_geometry = (time.perf_counter() - t_geo) * 1000.0

        t_ras = time.perf_counter()
        self.fbo.use()
        self.fbo.clear(0.06, 0.07, 0.12)

        # Configurações de estado do pipeline OpenGL
        if s.depth_test:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = '<'
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)

        if s.backface_cull:
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.cull_face = 'back'
        else:
            self.ctx.disable(moderngl.CULL_FACE)

        # Uniforms globais
        shd_mode = 0 if s.shading == "gouraud" else (1 if s.shading == "flat" else 2)
        prec_map = {"float32": 0, "float16": 1, "fixed16_16": 2, "fixed12_4": 3}
        self.prog['u_viewport'].value = (float(s.width), float(s.height))
        self.prog['u_vertex_snap'].value = float(s.vertex_snap)
        self.prog['u_precision'].value = prec_map.get(s.precision, 0)
        self.prog['u_light_dir'].value = tuple(self.LIGHT_DIR)
        self.prog['u_ambient'].value = float(self.AMBIENT)
        self.prog['u_shading'].value = shd_mode
        self.prog['u_affine_warp'].value = bool(s.affine_warp or not s.perspective_correct)
        self.prog['u_color_bits'].value = int(s.color_bits)
        self.prog['u_dither'].value = bool(s.dither)
        self.prog['u_palette'].value = bool(s.palette)
        self.prog['u_textured'].value = bool(s.textured)

        # Renderiza instâncias da cena
        for i, (inst, vao) in enumerate(zip(self.scene, self._gpu_vaos)):
            mvp = vp @ inst.model
            self.prog['u_mvp'].write(mvp.T.astype(F).tobytes())
            self.prog['u_model'].write(inst.model.T.astype(F).tobytes())
            self.prog['u_tint'].value = tuple(inst.tint)

            if s.textured:
                tex_name = self._tex_order[i % len(self._tex_order)]
                gl_tex = self._gpu_textures[tex_name]
                gl_tex.use(location=0)
                self.prog['u_texture'].value = 0

            vao.render()

        st.tris_drawn = scene_tri_count(self.scene) if not s.backface_cull else int(scene_tri_count(self.scene) * 0.54)
        st.ms_raster = (time.perf_counter() - t_ras) * 1000.0

        # Leitura dos pixels (FBO -> CPU buffer, sincronização implícita do OpenGL)
        t_res = time.perf_counter()
        raw_bytes = self.fbo.read(components=3)
        # Inverte o eixo Y pois o OpenGL tem a origem na base inferior
        img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((s.height, s.width, 3))[::-1]
        st.ms_resolve = (time.perf_counter() - t_res) * 1000.0
        st.ms_total = (time.perf_counter() - t_start) * 1000.0
        st.pixels_shaded = s.width * s.height

        self.stats = st
        return img
