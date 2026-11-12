"""Álgebra linear mínima para o pipeline gráfico (coordenadas homogêneas).

Todas as matrizes são 4x4, column-major na convenção M @ v, com v = [x, y, z, 1].
"""
from __future__ import annotations

import numpy as np

F = np.float32


def identity() -> np.ndarray:
    return np.eye(4, dtype=F)


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=F)
    m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
    return m


def scale(sx: float, sy: float | None = None, sz: float | None = None) -> np.ndarray:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = np.eye(4, dtype=F)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def rotate_x(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Matriz de visualização (world -> view), sistema destro olhando para -Z."""
    eye = np.asarray(eye, dtype=F)
    target = np.asarray(target, dtype=F)
    up = np.asarray(up, dtype=F)

    f = normalize(target - eye)          # frente
    s = normalize(np.cross(f, up))       # direita
    u = np.cross(s, f)                   # cima real

    m = np.eye(4, dtype=F)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Transformação de perspectiva clássica (Blinn, 1993): NDC em [-1, 1]."""
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=F)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m
