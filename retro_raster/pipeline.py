"""Pipeline de rasterização por software.

Fluxo implementado (eq. 1 do documento):

    Geometria -> Transformação -> Projeção -> Rasterização -> Framebuffer -> Imagem

Restrições de arquiteturas legadas reproduzidas de forma controlada:
  * resolução do framebuffer reduzida (com upscale nearest para exibição);
  * quantidade de geometria da cena variável;
  * resolução de textura reduzida;
  * precisão numérica (float32 / float16 / ponto fixo tipo PS1);
  * snap de vértices em grade inteira (jitter característico do PS1);
  * ausência de correção de perspectiva na interpolação de UV (afim);
  * profundidade de cor reduzida, dithering ordenado e paleta fixa;
  * z-buffer opcional (alternativa: ordenação por profundidade média).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import mathutil as mu
from .framebuffer import Framebuffer
from .geometry import Instance, build_scene, scene_tri_count
from .texture import Texture, make_textures

F = np.float32


# --------------------------------------------------------------------------- #
# Configuração dos experimentos
# --------------------------------------------------------------------------- #
@dataclass
class RenderSettings:
    # --- Dispositivo ---
    device: str = "cpu"                  # cpu | gpu

    # --- Framebuffer ---
    width: int = 320
    height: int = 240

    # --- Geometria ---
    target_tris: int = 2000

    # --- Textura ---
    texture_res: int = 64
    texture_filter: str = "nearest"      # nearest | bilinear
    textured: bool = True

    # --- Precisão numérica ---
    precision: str = "float32"           # float32 | float16 | fixed16_16 | fixed12_4
    vertex_snap: int = 0                 # 0 = off; N = snap em grade de N subpixels

    # --- Rasterização ---
    perspective_correct: bool = True
    depth_test: bool = True              # False -> painter's algorithm
    backface_cull: bool = True

    # --- Cor ---
    color_bits: int = 8                  # 8 | 5 | 4 | 3 | 2 | 1
    dither: bool = False
    palette: bool = False

    # --- Sombreamento ---
    shading: str = "gouraud"             # flat | gouraud | unlit
    affine_warp: bool = False            # força UV afim (artefato PS1)

    # --- Câmera ---
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameStats:
    """Métricas coletadas por quadro para os experimentos."""
    tris_submitted: int = 0
    tris_drawn: int = 0
    tris_culled_back: int = 0
    tris_clipped: int = 0
    pixels_shaded: int = 0
    ms_geometry: float = 0.0
    ms_raster: float = 0.0
    ms_resolve: float = 0.0
    ms_total: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms_total if self.ms_total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        return d


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Emulação de precisão numérica
# --------------------------------------------------------------------------- #
def apply_precision(arr: np.ndarray, mode: str) -> np.ndarray:
    """Reduz a precisão numérica dos dados conforme o modo escolhido."""
    if mode == "float32":
        return arr.astype(F, copy=False)
    if mode == "float16":
        # Evita overflow para +/-inf quando arr excede o limite do float16 (~65504)
        c = np.clip(arr, -65500.0, 65500.0)
        return c.astype(np.float16).astype(F)
    if mode == "fixed16_16":          # 16.16 -> 1/65536 de resolução
        c = np.clip(arr, -32767.0, 32767.0)
        return (np.round(c * 65536.0) / 65536.0).astype(F)
    if mode == "fixed12_4":           # 12.4 -> 1/16, bem grosseiro (estilo PS1)
        c = np.clip(arr, -2047.0, 2047.0)
        return (np.round(c * 16.0) / 16.0).astype(F)
    return arr.astype(F, copy=False)


# --------------------------------------------------------------------------- #
# Recorte contra o plano near (W > epsilon), em espaço de clip
# --------------------------------------------------------------------------- #
def clip_near(poly_clip: np.ndarray, attrs: list[np.ndarray], eps: float = 1e-5):
    """Sutherland-Hodgman em 1 plano (w > eps) para um único triângulo.

    poly_clip: (N, 4) posições homogêneas; attrs: lista de (N, k) atributos.
    Retorna (poly, attrs) recortados — polígono convexo com N' vértices.
    """
    if not np.all(np.isfinite(poly_clip)):
        return None, None
    n = poly_clip.shape[0]
    out_p, out_a = [], [[] for _ in attrs]
    for i in range(n):
        j = (i + 1) % n
        wi, wj = poly_clip[i, 3], poly_clip[j, 3]
        ini, inj = wi > eps, wj > eps
        if ini:
            out_p.append(poly_clip[i])
            for k, a in enumerate(attrs):
                out_a[k].append(a[i])
        if ini != inj:
            denom = wj - wi
            if abs(denom) < 1e-9:
                denom = 1e-9 if denom >= 0 else -1e-9
            t = float(np.clip((eps - wi) / denom, 0.0, 1.0))
            out_p.append(poly_clip[i] + (poly_clip[j] - poly_clip[i]) * t)
            for k, a in enumerate(attrs):
                out_a[k].append(a[i] + (a[j] - a[i]) * t)
    if len(out_p) < 3:
        return None, None
    return np.array(out_p, dtype=F), [np.array(a, dtype=F) for a in out_a]


# --------------------------------------------------------------------------- #
# Renderizador
# --------------------------------------------------------------------------- #
class Renderer:
    LIGHT_DIR = mu.normalize(np.array([0.45, 0.8, 0.42], dtype=F))
    AMBIENT = F(0.22)

    def __init__(self, settings: RenderSettings | None = None):
        self.settings = settings or RenderSettings()
        self.fb = Framebuffer(self.settings.width, self.settings.height)
        self.scene: list[Instance] = build_scene(self.settings.target_tris)
        self.textures: dict[str, Texture] = make_textures(self.settings.texture_res,
                                                          self.settings.color_bits)
        self._tex_order = ["bricks", "gradient", "checker", "noise"]
        self.stats = FrameStats()
        self.camera_eye = np.array([0.0, 1.5, 5.0], dtype=F)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=F)
        self._gpu_renderer = None
        if self.settings.device == "gpu":
            self._ensure_gpu()

    def _ensure_gpu(self):
        if self._gpu_renderer is None:
            from .gpu_pipeline import GPURenderer
            self._gpu_renderer = GPURenderer(self.settings)
        self._gpu_renderer.camera_eye = self.camera_eye
        self._gpu_renderer.camera_target = self.camera_target
        return self._gpu_renderer

    # ------------------------------------------------------------------ #
    def rebuild_scene(self) -> None:
        self.scene = build_scene(self.settings.target_tris)
        if self._gpu_renderer is not None:
            self._gpu_renderer.rebuild_scene()

    def rebuild_textures(self) -> None:
        self.textures = make_textures(self.settings.texture_res, self.settings.color_bits)
        if self._gpu_renderer is not None:
            self._gpu_renderer.rebuild_textures()

    def sync(self) -> None:
        """Reaplica configurações que exigem reconstrução de recursos."""
        s = self.settings
        if s.device == "gpu":
            gpu = self._ensure_gpu()
            gpu.sync()
            return
        self.fb.resize(s.width, s.height)
        if scene_tri_count(self.scene) and abs(
                scene_tri_count(self.scene) - s.target_tris) > max(40, s.target_tris * 0.25):
            self.rebuild_scene()
        tex_sample = next(iter(self.textures.values()), None)
        if tex_sample is None or tex_sample.res != s.texture_res or tex_sample.color_bits != s.color_bits:
            self.rebuild_textures()

    # ------------------------------------------------------------------ #
    def render(self, t: float = 0.0) -> np.ndarray:
        """Renderiza um quadro e devolve a imagem uint8 (H, W, 3)."""
        s = self.settings
        if s.device == "gpu":
            gpu = self._ensure_gpu()
            gpu.sync()
            img = gpu.render(t)
            self.stats = gpu.stats
            return img

        st = FrameStats()
        t_start = time.perf_counter()

        self.fb.clear()
        for inst in self.scene:
            inst.update(t)

        aspect = s.width / max(1, s.height)
        view = mu.look_at(self.camera_eye, self.camera_target)
        proj = mu.perspective(s.fov, aspect, s.near, s.far)
        vp = proj @ view

        # ---------------- Etapa de geometria ---------------- #
        t_geo = time.perf_counter()
        batches = []
        for inst in self.scene:
            mesh = inst.mesh
            mvp = vp @ inst.model

            pos_h = np.concatenate(
                [mesh.positions, np.ones((mesh.positions.shape[0], 1), dtype=F)], axis=1)
            clip = (mvp @ pos_h.T).T                      # (V, 4)
            clip = apply_precision(clip, s.precision)

            # Matriz normal (inversa-transposta) para suportar escalas não-uniformes
            m3 = inst.model[:3, :3]
            try:
                norm_mat = np.linalg.inv(m3).T
            except np.linalg.LinAlgError:
                norm_mat = m3
            nrm_world = (norm_mat @ mesh.normals.T).T
            nl = np.linalg.norm(nrm_world, axis=1, keepdims=True)
            nrm_world = nrm_world / np.maximum(nl, 1e-9)

            # Iluminação difusa por vértice (Gouraud) ou sem luz
            if s.shading == "unlit":
                lit = np.ones((mesh.positions.shape[0],), dtype=F)
            else:
                lit = np.clip(nrm_world @ self.LIGHT_DIR, 0.0, 1.0)
                lit = self.AMBIENT + (1.0 - self.AMBIENT) * lit
            batches.append((inst, clip, lit.astype(F), mesh.uvs, mesh.indices))
            st.tris_submitted += mesh.tri_count
        st.ms_geometry = (time.perf_counter() - t_geo) * 1000.0

        # ---------------- Rasterização ---------------- #
        t_ras = time.perf_counter()
        painter: list[tuple] = []   # usado quando depth_test=False
        W, H = s.width, s.height

        for bi, (inst, clip, lit, uvs, indices) in enumerate(batches):
            tex = self.textures[self._tex_order[bi % len(self._tex_order)]] \
                if s.textured else None
            tint = np.array(inst.tint, dtype=F)

            # ---- Triagem vetorizada de todos os triângulos do lote ---- #
            cw = clip[:, 3]
            tri_w = cw[indices]                        # (T, 3)
            needs_clip = np.any(tri_w <= 1e-5, axis=1) | np.any(~np.isfinite(tri_w), axis=1)

            # Triângulos totalmente à frente do near: setup em lote.
            safe = ~needs_clip
            safe_idx = np.nonzero(safe)[0]
            if safe_idx.size:
                ti = indices[safe_idx]                 # (N, 3)
                w = tri_w[safe_idx]
                ndc = clip[:, :3][ti] / w[..., None]   # (N, 3, 3)
                ndc = apply_precision(ndc, s.precision)

                finite_mask = np.all(np.isfinite(ndc), axis=(1, 2))

                sx = (ndc[..., 0] * 0.5 + 0.5) * W
                sy = (1.0 - (ndc[..., 1] * 0.5 + 0.5)) * H
                if s.vertex_snap > 0:
                    g = F(s.vertex_snap)
                    sx = np.round(sx * g) / g
                    sy = np.round(sy * g) / g

                area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
                        - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))

                keep = finite_mask & np.isfinite(area) & (np.abs(area) >= 1e-7)
                if s.backface_cull:
                    back = area >= 0.0
                    st.tris_culled_back += int(np.count_nonzero(back & keep))
                    keep &= ~back

                # Rejeição por bounding box fora da tela ou degenerada
                x0 = np.ceil(sx.min(axis=1) - 0.5)
                x1 = np.floor(sx.max(axis=1) - 0.5)
                y0 = np.ceil(sy.min(axis=1) - 0.5)
                y1 = np.floor(sy.max(axis=1) - 0.5)
                keep &= (x1 >= 0) & (x0 <= W - 1) & (y1 >= 0) & (y0 <= H - 1) & (x1 >= x0) & (y1 >= y0)

                # Rejeição de triângulos totalmente além do far plane ou aquém do near plane
                z_tri = ndc[..., 2]
                keep &= ~np.all(z_tri > 1.0, axis=1) & ~np.all(z_tri < -1.0, axis=1)

                vis = np.nonzero(keep)[0]
                if vis.size:
                    z_all = ndc[..., 2]
                    inv_w_all = (1.0 / w).astype(F)
                    lit_all = lit[ti]                  # (N, 3)
                    uv_all = uvs[ti]                   # (N, 3, 2)
                    bx0 = np.maximum(x0, 0).astype(np.int32)
                    bx1 = np.minimum(x1, W - 1).astype(np.int32)
                    by0 = np.maximum(y0, 0).astype(np.int32)
                    by1 = np.minimum(y1, H - 1).astype(np.int32)

                    for k in vis:
                        item = self._raster_setup(
                            sx[k], sy[k], z_all[k], inv_w_all[k], area[k],
                            lit_all[k], uv_all[k],
                            int(bx0[k]), int(bx1[k]), int(by0[k]), int(by1[k]),
                            tex, tint, st, collect=not s.depth_test)
                        if item is not None:
                            painter.append(item)

            # ---- Triângulos que cruzam o near: recorte individual ---- #
            for t_i in np.nonzero(needs_clip)[0]:
                tri = indices[t_i]
                cp = clip[tri]
                lv = lit[tri].reshape(3, 1)
                uvv = uvs[tri]

                poly, attrs = clip_near(cp, [lv, uvv])
                st.tris_clipped += 1
                if poly is None:
                    continue
                for k in range(1, poly.shape[0] - 1):
                    item = self._raster_triangle(
                        poly[[0, k, k + 1]], attrs[0][[0, k, k + 1]],
                        attrs[1][[0, k, k + 1]], tex, tint, st,
                        collect=not s.depth_test)
                    if item is not None:
                        painter.append(item)

        if not s.depth_test and painter:
            # Painter's algorithm: ordena por profundidade média decrescente.
            painter.sort(key=lambda it: -it[0])
            for _, xs, ys, cols in painter:
                self.fb.color[ys, xs] = cols
                st.pixels_shaded += xs.size
                st.tris_drawn += 1

        st.ms_raster = (time.perf_counter() - t_ras) * 1000.0

        # ---------------- Resolve / imagem final ---------------- #
        t_res = time.perf_counter()
        img = self.fb.resolve(s.color_bits, s.dither, s.palette)
        st.ms_resolve = (time.perf_counter() - t_res) * 1000.0
        st.ms_total = (time.perf_counter() - t_start) * 1000.0
        self.stats = st
        return img

    # ------------------------------------------------------------------ #
    def _raster_triangle(self, clip_pos, lit, uv, tex, tint, st: FrameStats,
                         collect: bool = False):
        """Rasteriza um triângulo em espaço de clip (usado após o recorte near)."""
        s = self.settings
        w = clip_pos[:, 3]
        if np.any(w <= 1e-5) or not np.all(np.isfinite(w)):
            return None

        # Divisão perspectiva -> NDC
        ndc = clip_pos[:, :3] / w[:, None]
        if not np.all(np.isfinite(ndc)):
            return None
        ndc = apply_precision(ndc, s.precision)
        if not np.all(np.isfinite(ndc)):
            return None

        # Viewport: NDC [-1,1] -> pixels
        W, H = s.width, s.height
        sx = (ndc[:, 0] * 0.5 + 0.5) * W
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

        if not np.all(np.isfinite(sx)) or not np.all(np.isfinite(sy)):
            return None

        # Snap de vértices em grade inteira (jitter típico do PS1)
        if s.vertex_snap > 0:
            g = F(s.vertex_snap)
            sx = np.round(sx * g) / g
            sy = np.round(sy * g) / g

        area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
        if not np.isfinite(area) or abs(area) < 1e-7:
            return None
        if s.backface_cull and area >= 0.0:
            st.tris_culled_back += 1
            return None

        x_min = max(int(np.ceil(sx.min() - 0.5)), 0)
        x_max = min(int(np.floor(sx.max() - 0.5)), W - 1)
        y_min = max(int(np.ceil(sy.min() - 0.5)), 0)
        y_max = min(int(np.floor(sy.max() - 0.5)), H - 1)
        if x_min > x_max or y_min > y_max:
            return None

        return self._raster_setup(sx, sy, ndc[:, 2], (1.0 / w).astype(F), area,
                                  lit.reshape(3), uv, x_min, x_max, y_min, y_max,
                                  tex, tint, st, collect)

    # ------------------------------------------------------------------ #
    def _raster_setup(self, sx, sy, z, inv_w, area, lit, uv,
                      x_min, x_max, y_min, y_max, tex, tint,
                      st: FrameStats, collect: bool = False):
        """Varredura por bounding box com coordenadas baricêntricas.

        Recebe o triângulo já projetado em espaço de tela e seu bounding box
        recortado contra a janela. As coordenadas baricêntricas são avaliadas
        por funções de aresta (edge functions) de forma vetorizada.
        """
        if x_min > x_max or y_min > y_max or not np.isfinite(area) or abs(area) < 1e-7:
            return None

        s = self.settings
        lit = np.asarray(lit, dtype=F).reshape(3)

        inv_area = F(1.0) / area
        px = np.arange(x_min, x_max + 1, dtype=F)[None, :] + F(0.5)
        py = np.arange(y_min, y_max + 1, dtype=F)[:, None] + F(0.5)

        # Avaliação de edge functions lineares (w0, w1) via broadcasting
        A0 = (sy[1] - sy[2]) * inv_area
        B0 = (sx[2] - sx[1]) * inv_area
        C0 = (sx[1] * sy[2] - sx[2] * sy[1]) * inv_area

        A1 = (sy[2] - sy[0]) * inv_area
        B1 = (sx[0] - sx[2]) * inv_area
        C1 = (sx[2] * sy[0] - sx[0] * sy[2]) * inv_area

        w0 = (A0 * px + C0) + (B0 * py)
        w1 = (A1 * px + C1) + (B1 * py)
        w2 = F(1.0) - w0 - w1

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            return None

        yy, xx = np.nonzero(mask)
        b0, b1, b2 = w0[mask], w1[mask], w2[mask]
        yy = yy + y_min
        xx = xx + x_min

        # Profundidade interpolada e teste contra limites NDC [-1, 1]
        zz = b0 * z[0] + b1 * z[1] + b2 * z[2]
        in_depth_range = (zz >= -1.0) & (zz <= 1.0)
        if not in_depth_range.any():
            return None
        if not in_depth_range.all():
            b0, b1, b2 = b0[in_depth_range], b1[in_depth_range], b2[in_depth_range]
            yy, xx, zz = yy[in_depth_range], xx[in_depth_range], zz[in_depth_range]

        if s.depth_test:
            prev = self.fb.depth[yy, xx]
            keep = zz < prev
            if not keep.any():
                return None
            b0, b1, b2 = b0[keep], b1[keep], b2[keep]
            yy, xx, zz = yy[keep], xx[keep], zz[keep]

        # ---- Interpolação de atributos ---- #
        persp = s.perspective_correct and not s.affine_warp
        if persp:
            iw = b0 * inv_w[0] + b1 * inv_w[1] + b2 * inv_w[2]
            iw = np.maximum(iw, 1e-9)
            c0 = b0 * inv_w[0] / iw
            c1 = b1 * inv_w[1] / iw
            c2 = b2 * inv_w[2] / iw
        else:
            c0, c1, c2 = b0, b1, b2

        if s.shading == "flat":
            shade = np.full(zz.shape, F(lit.mean()), dtype=F)
        elif s.shading == "unlit":
            shade = np.ones(zz.shape, dtype=F)
        else:
            shade = c0 * lit[0] + c1 * lit[1] + c2 * lit[2]

        if tex is not None:
            u = c0 * uv[0, 0] + c1 * uv[1, 0] + c2 * uv[2, 0]
            v = c0 * uv[0, 1] + c1 * uv[1, 1] + c2 * uv[2, 1]
            albedo = tex.sample(u, v, bilinear=(s.texture_filter == "bilinear"))
        else:
            albedo = np.ones((zz.size, 3), dtype=F) * 0.8

        cols = np.clip(albedo * tint * shade[:, None], 0.0, 1.0).astype(F)

        if collect:
            # Painter's algorithm: devolve o fragmento para ordenação posterior.
            return (float(zz.mean()), xx, yy, cols)

        self.fb.depth[yy, xx] = zz
        self.fb.color[yy, xx] = cols
        st.tris_drawn += 1
        st.pixels_shaded += zz.size
        return None
