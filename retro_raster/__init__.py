"""Aethelion (retro_raster) — pipeline de rasterização por software inspirado em consoles antigos.

Eixo 7: arquitetura de consoles antigos / rasterização legada.
Fluxo: Geometria -> Transformação -> Projeção -> Rasterização -> Framebuffer -> Imagem
"""
from aethelion.pipeline import Renderer, RenderSettings, FrameStats
from aethelion.framebuffer import Framebuffer
from aethelion.texture import Texture, make_textures
from aethelion import geometry, mathutil

__version__ = "1.0.0"
__all__ = [
    "Renderer", "RenderSettings", "FrameStats", "Framebuffer",
    "Texture", "make_textures", "geometry", "mathutil",
]
