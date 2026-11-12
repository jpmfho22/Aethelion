"""Framebuffer: alvo de cor + buffer de profundidade, com pós-processamento retro."""
from __future__ import annotations

import numpy as np

F = np.float32

# Paleta fixa de 16 cores (estilo console antigo), usada no modo "palette".
PALETTE_16 = np.array([
    [0.04, 0.04, 0.07], [0.33, 0.20, 0.16], [0.52, 0.18, 0.20], [0.20, 0.36, 0.26],
    [0.72, 0.30, 0.22], [0.36, 0.28, 0.44], [0.25, 0.47, 0.62], [0.70, 0.70, 0.72],
    [0.42, 0.42, 0.45], [0.85, 0.47, 0.22], [0.88, 0.72, 0.30], [0.55, 0.78, 0.40],
    [0.92, 0.55, 0.52], [0.45, 0.60, 0.85], [0.80, 0.84, 0.90], [0.98, 0.98, 0.96],
], dtype=F)

# Matriz de Bayer 4x4 para dithering ordenado.
BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=F) / 16.0

PALETTE_16_SQ = np.sum(PALETTE_16 ** 2, axis=1)


class Framebuffer:
    """Buffer de cor float32 (H, W, 3) e profundidade float32 (H, W)."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.color = np.zeros((self.height, self.width, 3), dtype=F)
        self.depth = np.full((self.height, self.width), np.inf, dtype=F)
        # Pré-calcula matriz de dithering para evitar np.tile redundante a cada quadro
        tile = np.tile(BAYER4, (self.height // 4 + 1, self.width // 4 + 1))
        self.bayer_tile = tile[:self.height, :self.width][..., None].astype(F)

    def resize(self, width: int, height: int) -> None:
        if (int(width), int(height)) != (self.width, self.height):
            self.__init__(width, height)

    def clear(self, color=(0.06, 0.07, 0.12)) -> None:
        self.color[:] = np.asarray(color, dtype=F)
        self.depth[:] = np.inf

    # ------------------------------------------------------------------ #
    # Pós-processamento: quantização de cor, dithering e paleta fixa
    # ------------------------------------------------------------------ #
    def resolve(self, color_bits: int = 8, dither: bool = False,
                palette: bool = False) -> np.ndarray:
        """Converte o buffer float em imagem uint8 aplicando as restrições de cor."""
        img = np.clip(self.color, 0.0, 1.0)

        if palette:
            if dither:
                img = np.clip(img + (self.bayer_tile - 0.5) * 0.12, 0.0, 1.0)
            flat = img.reshape(-1, 3)
            # Otimização vetorial: argmin ||x - p||^2 == argmax (2 * x @ p.T - ||p||^2)
            # Processamento em lotes para evitar estouro de memória em resoluções altas
            chunk_size = 65536
            idx_list = []
            for start in range(0, flat.shape[0], chunk_size):
                chunk = flat[start:start + chunk_size]
                metric = 2.0 * (chunk @ PALETTE_16.T) - PALETTE_16_SQ[None, :]
                idx_list.append(np.argmax(metric, axis=1))
            idx = np.concatenate(idx_list) if len(idx_list) > 1 else idx_list[0]
            img = PALETTE_16[idx].reshape(self.height, self.width, 3)
        elif color_bits < 8:
            levels = float(2 ** color_bits - 1)
            if dither:
                img = np.clip(img + (self.bayer_tile - 0.5) / levels, 0.0, 1.0)
            img = np.round(img * levels) / levels

        return (img * 255.0 + 0.5).astype(np.uint8)
