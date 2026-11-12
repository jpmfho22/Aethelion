"""Geração de geometria: malhas de triângulos com densidade controlável.

A quantidade de triângulos da cena é um dos parâmetros experimentais, portanto
todas as primitivas são paramétricas e a cena sabe se ajustar a um "orçamento"
de triângulos informado pelo usuário.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu

F = np.float32


@dataclass
class Mesh:
    """Malha indexada: posições, normais, coordenadas de textura e índices."""
    positions: np.ndarray            # (V, 3) float32
    normals: np.ndarray              # (V, 3) float32
    uvs: np.ndarray                  # (V, 2) float32
    indices: np.ndarray              # (T, 3) int32

    @property
    def tri_count(self) -> int:
        return int(self.indices.shape[0])


@dataclass
class Instance:
    """Instância de malha com transformação de modelagem própria."""
    mesh: Mesh
    model: np.ndarray = field(default_factory=mu.identity)
    spin: float = 0.0                 # rad/s em torno de Y (animação da cena)
    base_model: np.ndarray = field(default_factory=mu.identity)
    tint: tuple = (1.0, 1.0, 1.0)

    def update(self, t: float) -> None:
        self.model = self.base_model @ mu.rotate_y(self.spin * t) if self.spin else self.base_model


# --------------------------------------------------------------------------- #
# Primitivas paramétricas
# --------------------------------------------------------------------------- #
def cube(subdiv: int = 1) -> Mesh:
    """Cubo subdividido: 12 * subdiv^2 triângulos."""
    subdiv = max(1, int(subdiv))
    pos, nrm, uv, idx = [], [], [], []

    faces = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    ]
    for n, u_ax, v_ax in faces:
        n = np.array(n, dtype=F)
        u_ax = np.array(u_ax, dtype=F)
        v_ax = np.array(v_ax, dtype=F)
        base = len(pos)
        for i in range(subdiv + 1):
            for j in range(subdiv + 1):
                a = (i / subdiv) * 2.0 - 1.0
                b = (j / subdiv) * 2.0 - 1.0
                pos.append(n + u_ax * a + v_ax * b)
                nrm.append(n)
                uv.append((i / subdiv, j / subdiv))
        for i in range(subdiv):
            for j in range(subdiv):
                a = base + i * (subdiv + 1) + j
                b = a + 1
                c = a + (subdiv + 1)
                d = c + 1
                idx.append((a, c, b))
                idx.append((b, c, d))

    return Mesh(np.array(pos, F) * 0.5, np.array(nrm, F), np.array(uv, F),
                np.array(idx, np.int32))


def sphere(stacks: int = 12, slices: int = 16, radius: float = 1.0) -> Mesh:
    """Esfera UV: 2 * stacks * slices triângulos (menos as tampas degeneradas)."""
    stacks, slices = max(2, int(stacks)), max(3, int(slices))
    pos, nrm, uv, idx = [], [], [], []

    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * np.pi * j / slices
            n = np.array([np.sin(phi) * np.cos(theta), np.cos(phi),
                          np.sin(phi) * np.sin(theta)], dtype=F)
            pos.append(n * radius)
            nrm.append(n)
            uv.append((j / slices, i / stacks))

    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + 1
            c = a + (slices + 1)
            d = c + 1
            if i != 0:
                idx.append((a, c, b))
            if i != stacks - 1:
                idx.append((b, c, d))

    return Mesh(np.array(pos, F), np.array(nrm, F), np.array(uv, F),
                np.array(idx, np.int32))


def torus(seg_u: int = 16, seg_v: int = 10, r_major: float = 1.0,
          r_minor: float = 0.35) -> Mesh:
    """Toro: 2 * seg_u * seg_v triângulos."""
    seg_u, seg_v = max(3, int(seg_u)), max(3, int(seg_v))
    pos, nrm, uv, idx = [], [], [], []

    for i in range(seg_u + 1):
        u = 2.0 * np.pi * i / seg_u
        cu, su = np.cos(u), np.sin(u)
        for j in range(seg_v + 1):
            v = 2.0 * np.pi * j / seg_v
            cv, sv = np.cos(v), np.sin(v)
            n = np.array([cu * cv, sv, su * cv], dtype=F)
            p = np.array([cu * (r_major + r_minor * cv), r_minor * sv,
                          su * (r_major + r_minor * cv)], dtype=F)
            pos.append(p)
            nrm.append(n)
            uv.append((i / seg_u * 3.0, j / seg_v))

    for i in range(seg_u):
        for j in range(seg_v):
            a = i * (seg_v + 1) + j
            b = a + 1
            c = a + (seg_v + 1)
            d = c + 1
            idx.append((a, c, b))
            idx.append((b, c, d))

    return Mesh(np.array(pos, F), np.array(nrm, F), np.array(uv, F),
                np.array(idx, np.int32))


def ground(size: float = 6.0, subdiv: int = 4, y: float = -1.2) -> Mesh:
    """Plano de chão: 2 * subdiv^2 triângulos."""
    subdiv = max(1, int(subdiv))
    pos, nrm, uv, idx = [], [], [], []
    for i in range(subdiv + 1):
        for j in range(subdiv + 1):
            x = (i / subdiv - 0.5) * 2.0 * size
            z = (j / subdiv - 0.5) * 2.0 * size
            pos.append((x, y, z))
            nrm.append((0.0, 1.0, 0.0))
            uv.append((i / subdiv * 4.0, j / subdiv * 4.0))
    for i in range(subdiv):
        for j in range(subdiv):
            a = i * (subdiv + 1) + j
            b = a + 1
            c = a + (subdiv + 1)
            d = c + 1
            idx.append((a, b, c))
            idx.append((b, d, c))
    return Mesh(np.array(pos, F), np.array(nrm, F), np.array(uv, F),
                np.array(idx, np.int32))


# --------------------------------------------------------------------------- #
# Cena
# --------------------------------------------------------------------------- #
def build_scene(target_tris: int = 2000) -> list[Instance]:
    """Monta a cena aproximando o orçamento de triângulos pedido.

    A cena é propositalmente simples (executa em GPU integrada / CPU), com a
    densidade geométrica escalada para o alvo informado.
    """
    target_tris = max(120, int(target_tris))

    # Distribuição do orçamento entre os objetos da cena.
    budget = {"torus": 0.42, "sphere": 0.26, "cube": 0.18, "ground": 0.14}

    t_tris = target_tris * budget["torus"]
    seg_u = max(3, int(np.sqrt(t_tris / 2.0 * 1.6)))
    seg_v = max(3, int(t_tris / (2.0 * seg_u)))

    s_tris = target_tris * budget["sphere"]
    stacks = max(2, int(np.sqrt(s_tris / 2.0 / 1.4)))
    slices = max(3, int(s_tris / (2.0 * stacks)))

    c_tris = target_tris * budget["cube"]
    c_sub = max(1, int(np.sqrt(c_tris / 12.0)))

    g_tris = target_tris * budget["ground"]
    g_sub = max(1, int(np.sqrt(g_tris / 2.0)))

    inst = [
        Instance(torus(seg_u, seg_v), base_model=mu.translate(0, 0.15, 0) @ mu.rotate_x(0.5),
                 spin=0.7, tint=(1.0, 0.85, 0.7)),
        Instance(sphere(stacks, slices, 0.62),
                 base_model=mu.translate(-1.85, 0.1, 0.4), spin=-0.9, tint=(0.75, 0.9, 1.0)),
        Instance(cube(c_sub),
                 base_model=mu.translate(1.9, 0.05, -0.3) @ mu.rotate_y(0.6) @ mu.scale(0.9),
                 spin=0.5, tint=(1.0, 1.0, 0.85)),
        Instance(ground(5.5, g_sub), base_model=mu.identity(), tint=(0.8, 0.85, 0.8)),
    ]
    for i in inst:
        i.update(0.0)
    return inst


def scene_tri_count(scene: list[Instance]) -> int:
    return sum(i.mesh.tri_count for i in scene)
