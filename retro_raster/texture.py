"""Texturas procedurais com resolução e profundidade de cor controláveis.

A resolução de textura é um dos parâmetros experimentais: texturas pequenas
(32x32, 64x64) reproduzem a aparência de memória gráfica restrita típica de
consoles antigos.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def _checker(res: int) -> np.ndarray:
    cells = max(2, res // 8)
    yy, xx = np.mgrid[0:res, 0:res]
    c = ((xx * cells // res) + (yy * cells // res)) % 2
    a = np.array([0.92, 0.32, 0.26], dtype=F)
    b = np.array([0.96, 0.93, 0.84], dtype=F)
    return np.where(c[..., None].astype(bool), a, b).astype(F)


def _bricks(res: int) -> np.ndarray:
    yy, xx = np.mgrid[0:res, 0:res]
    rows = max(2, res // 10)
    row = yy * rows // res
    shift = (row % 2) * (res // (2 * rows) if rows else 0)
    cols = max(2, res // 12)
    col = ((xx + shift) * cols // res)
    mortar = (((yy * rows * 4 // res) % 4) == 0) | ((((xx + shift) * cols * 6 // res) % 6) == 0)
    rng = np.random.default_rng(7)
    tone = rng.uniform(0.78, 1.0, size=(rows + 2, cols + 2)).astype(F)
    base = np.array([0.72, 0.33, 0.24], dtype=F) * tone[row % (rows + 2), col % (cols + 2)][..., None]
    out = np.where(mortar[..., None], np.array([0.80, 0.78, 0.72], dtype=F), base)
    return out.astype(F)


def _gradient(res: int) -> np.ndarray:
    yy, xx = np.mgrid[0:res, 0:res]
    u = (xx / max(1, res - 1)).astype(F)
    v = (yy / max(1, res - 1)).astype(F)
    rgb = np.stack([0.25 + 0.6 * u, 0.3 + 0.5 * v, 0.85 - 0.45 * u], axis=-1)
    rings = (np.sin(12.0 * np.pi * np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)) * 0.5 + 0.5)
    return np.clip(rgb * (0.75 + 0.25 * rings[..., None]), 0.0, 1.0).astype(F)


def _noise(res: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    low = rng.uniform(0.3, 1.0, size=(max(2, res // 8), max(2, res // 8), 3)).astype(F)
    rep = int(np.ceil(res / low.shape[0]))
    big = np.repeat(np.repeat(low, rep, axis=0), rep, axis=1)[:res, :res]
    return np.clip(big * np.array([0.55, 0.75, 0.55], dtype=F), 0, 1).astype(F)


PATTERNS = {
    "checker": _checker,
    "bricks": _bricks,
    "gradient": _gradient,
    "noise": _noise,
}


class Texture:
    """Textura RGB float32 (res, res, 3) com amostragem nearest ou bilinear."""

    def __init__(self, pattern: str = "checker", res: int = 64, color_bits: int = 8):
        self.pattern = pattern if pattern in PATTERNS else "checker"
        self.res = int(max(2, res))
        self.color_bits = int(color_bits)
        self.data = PATTERNS[self.pattern](self.res)
        if self.color_bits < 8:
            levels = float(2 ** self.color_bits - 1)
            self.data = np.round(self.data * levels) / levels
        self.data = np.ascontiguousarray(self.data, dtype=F)

    # ------------------------------------------------------------------ #
    def sample(self, u: np.ndarray, v: np.ndarray, bilinear: bool = False) -> np.ndarray:
        """Amostra a textura em coordenadas UV (repetição/wrap)."""
        res = self.res
        if not bilinear:
            iu = np.floor(u * res).astype(np.int32) % res
            iv = np.floor(v * res).astype(np.int32) % res
            return self.data[iv, iu]

        fu = u * res - 0.5
        fv = v * res - 0.5
        u0 = np.floor(fu).astype(np.int32)
        v0 = np.floor(fv).astype(np.int32)
        du = (fu - u0).astype(F)[..., None]
        dv = (fv - v0).astype(F)[..., None]
        u0m, v0m = u0 % res, v0 % res
        u1m, v1m = (u0 + 1) % res, (v0 + 1) % res
        c00 = self.data[v0m, u0m]
        c10 = self.data[v0m, u1m]
        c01 = self.data[v1m, u0m]
        c11 = self.data[v1m, u1m]
        return (c00 * (1 - du) * (1 - dv) + c10 * du * (1 - dv)
                + c01 * (1 - du) * dv + c11 * du * dv)


def make_textures(res: int, color_bits: int = 8) -> dict[str, Texture]:
    return {name: Texture(name, res, color_bits) for name in PATTERNS}
