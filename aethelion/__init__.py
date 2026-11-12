"""Aethelion — pipeline de rasterização por software inspirado em consoles antigos.

Eixo 7: arquitetura de consoles antigos / rasterização legada.
Fluxo: Geometria -> Transformação -> Projeção -> Rasterização -> Framebuffer -> Imagem
"""
from .pipeline import Renderer, RenderSettings, FrameStats
from .framebuffer import Framebuffer
from .texture import Texture, make_textures
from . import geometry, mathutil

__version__ = "1.0.0"
__all__ = [
    "Renderer", "RenderSettings", "FrameStats", "Framebuffer",
    "Texture", "make_textures", "geometry", "mathutil",
]
