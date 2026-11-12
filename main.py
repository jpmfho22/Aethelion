#!/usr/bin/env python3
"""Ponto de entrada da aplicação.

Modos de uso:
    python main.py                      # aplicação interativa (GUI)
    python main.py --headless -o out    # renderiza 1 quadro em PNG
    python main.py --experiments        # roda todos os experimentos
    python main.py --experiments --sweep geometry
    python main.py --help               # lista todos os parâmetros
"""
from __future__ import annotations

import argparse
import os
import sys

from aethelion.pipeline import RenderSettings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aethelion — Rasterizador por software inspirado em arquiteturas de consoles antigos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    m = p.add_argument_group("modo de execução")
    m.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="dispositivo de execução (cpu = software, gpu = moderngl)")
    m.add_argument("--headless", action="store_true",
                   help="renderiza quadros sem abrir janela (salva PNG)")
    m.add_argument("--experiments", action="store_true",
                   help="executa as varreduras de parâmetros e salva CSV/gráficos")
    m.add_argument("--sweep", choices=["resolution", "geometry", "texture",
                                        "precision", "color", "all"], default="all",
                   help="qual varredura executar com --experiments")
    m.add_argument("-o", "--outdir", default="results", help="diretório de saída")
    m.add_argument("--frames", type=int, default=10,
                   help="quadros medidos por configuração nos experimentos")

    fb = p.add_argument_group("framebuffer")
    fb.add_argument("--width", type=int, default=320)
    fb.add_argument("--height", type=int, default=240)

    g = p.add_argument_group("geometria")
    g.add_argument("--tris", type=int, default=2000, help="quantidade alvo de triângulos")

    t = p.add_argument_group("textura")
    t.add_argument("--texres", type=int, default=64, help="resolução da textura")
    t.add_argument("--filter", choices=["nearest", "bilinear"], default="nearest")
    t.add_argument("--no-texture", action="store_true")

    n = p.add_argument_group("precisão numérica")
    n.add_argument("--precision", choices=["float32", "float16", "fixed16_16", "fixed12_4"],
                   default="float32")
    n.add_argument("--vertex-snap", type=int, default=0,
                   help="snap de vértices em grade (0 = desligado)")

    r = p.add_argument_group("rasterização")
    r.add_argument("--affine", action="store_true", help="UV afim (artefato PS1)")
    r.add_argument("--no-perspective", action="store_true",
                   help="desativa correção de perspectiva")
    r.add_argument("--no-zbuffer", action="store_true",
                   help="usa painter's algorithm em vez de z-buffer")
    r.add_argument("--no-cull", action="store_true", help="desativa backface culling")
    r.add_argument("--shading", choices=["gouraud", "flat", "unlit"], default="gouraud")

    c = p.add_argument_group("cor")
    c.add_argument("--bits", type=int, choices=[8, 5, 4, 3, 2, 1], default=8,
                   help="bits por canal de cor")
    c.add_argument("--dither", action="store_true", help="dithering ordenado Bayer 4x4")
    c.add_argument("--palette", action="store_true", help="paleta fixa de 16 cores")

    return p


def settings_from_args(a) -> RenderSettings:
    return RenderSettings(
        device=a.device,
        width=a.width, height=a.height,
        target_tris=a.tris,
        texture_res=a.texres, texture_filter=a.filter, textured=not a.no_texture,
        precision=a.precision, vertex_snap=a.vertex_snap,
        perspective_correct=not a.no_perspective,
        depth_test=not a.no_zbuffer,
        backface_cull=not a.no_cull,
        color_bits=a.bits, dither=a.dither, palette=a.palette,
        shading=a.shading, affine_warp=a.affine,
    )


def main() -> int:
    args = build_parser().parse_args()
    settings = settings_from_args(args)

    # ---- Experimentos ---- #
    if args.experiments:
        from aethelion import experiments as ex
        if args.sweep == "all":
            ex.run_all(settings, args.outdir, args.frames)
        else:
            ex.run_sweep(args.sweep, settings, args.outdir, args.frames)
        return 0

    # ---- Headless ---- #
    if args.headless:
        from PIL import Image
        from aethelion.experiments import measure
        os.makedirs(args.outdir, exist_ok=True)
        stats, img = measure(settings, frames=5)
        up = max(1, 640 // img.shape[1])
        path = os.path.join(args.outdir, "frame.png")
        Image.fromarray(img).resize((img.shape[1] * up, img.shape[0] * up),
                                    Image.NEAREST).save(path)
        print(f"Imagem salva em {path}")
        print(f"  resolução  : {settings.width}x{settings.height}")
        print(f"  triângulos : {stats['tris_submitted']} enviados / "
              f"{stats['tris_drawn']} desenhados")
        print(f"  tempo      : {stats['ms_total']:.2f} ms  ({stats['fps']:.1f} FPS)")
        print(f"    geometria: {stats['ms_geometry']:.2f} ms")
        print(f"    raster   : {stats['ms_raster']:.2f} ms")
        print(f"    resolve  : {stats['ms_resolve']:.2f} ms")
        return 0

    # ---- GUI ---- #
    try:
        from aethelion.app import main as run_app
    except Exception as e:
        print(f"Erro ao iniciar a interface: {e}", file=sys.stderr)
        print("Use --headless ou --experiments em ambientes sem display.", file=sys.stderr)
        return 1

    if os.name != "nt" and not os.environ.get("DISPLAY"):
        print("Nenhum display detectado (DISPLAY vazio).", file=sys.stderr)
        print("Execute em um ambiente gráfico ou use:", file=sys.stderr)
        print("  python main.py --headless", file=sys.stderr)
        print("  python main.py --experiments", file=sys.stderr)
        return 1

    run_app(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
