"""Experimentos headless: varia um parâmetro por vez e mede o desempenho.

Relaciona alterações na configuração do sistema com métricas de desempenho,
conforme a organização experimental adotada como referência de estrutura.

Saídas: CSV com as métricas, PNGs comparativos e gráficos (se matplotlib existir).
"""
from __future__ import annotations

import csv
import os
import statistics
from dataclasses import replace

import numpy as np

from .pipeline import Renderer, RenderSettings


def measure(settings: RenderSettings, frames: int = 12, warmup: int = 3):
    """Renderiza N quadros e devolve (métricas medianas, última imagem)."""
    r = Renderer(settings)
    img = None
    samples = []
    for i in range(frames + warmup):
        t = 0.35 * i
        img = r.render(t)
        if i >= warmup:
            samples.append(r.stats.to_dict())
    agg = {k: statistics.median([s[k] for s in samples]) for k in samples[0]}
    agg["tris_submitted"] = int(agg["tris_submitted"])
    agg["tris_drawn"] = int(agg["tris_drawn"])
    agg["pixels_shaded"] = int(agg["pixels_shaded"])
    return agg, img


# --------------------------------------------------------------------------- #
SWEEPS = {
    "resolution": ("width/height", [(160, 120), (256, 192), (320, 240),
                                    (400, 300), (512, 384), (640, 480)]),
    "geometry":   ("target_tris", [250, 500, 1000, 2000, 4000, 8000, 14000]),
    "texture":    ("texture_res", [8, 16, 32, 64, 128, 256]),
    "precision":  ("precision", ["float32", "float16", "fixed16_16", "fixed12_4"]),
    "color":      ("color_bits", [8, 5, 4, 3, 2, 1]),
}


def run_sweep(name: str, base: RenderSettings, outdir: str, frames: int = 10):
    """Executa uma varredura de parâmetro e salva CSV + imagens."""
    param, values = SWEEPS[name]
    os.makedirs(outdir, exist_ok=True)
    rows = []

    print(f"\n=== Varredura: {name}  ({param}) ===")
    for v in values:
        if name == "resolution":
            s = replace(base, width=v[0], height=v[1])
            label = f"{v[0]}x{v[1]}"
            fname = f"{v[0]}-{v[1]}"
        else:
            s = replace(base, **{param: v})
            label = str(v)
            fname = label

        stats, img = measure(s, frames=frames)
        rows.append({"variant": label, **stats})

        try:
            from PIL import Image
            up = max(1, 512 // img.shape[1])
            Image.fromarray(img).resize(
                (img.shape[1] * up, img.shape[0] * up), Image.NEAREST
            ).save(os.path.join(outdir, f"{name}_{fname}.png"))
        except Exception as e:  # pragma: no cover
            print(f"  (aviso: falha ao salvar PNG: {e})")

        print(f"  {label:>12} | {stats['ms_total']:7.2f} ms | {stats['fps']:6.1f} fps "
              f"| tris {stats['tris_drawn']:6d} | px {stats['pixels_shaded']:8d}")

    csv_path = os.path.join(outdir, f"{name}.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  -> CSV: {csv_path}")

    _plot(name, param, rows, outdir)
    return rows


def _plot(name, param, rows, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [r["variant"] for r in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))

    ax1.bar(x - 0.2, [r["ms_raster"] for r in rows], 0.4, label="rasterização (ms)",
            color="#4c78a8")
    ax1.bar(x + 0.2, [r["ms_geometry"] for r in rows], 0.4, label="geometria (ms)",
            color="#9ecae9")
    ax1.set_ylabel("tempo por quadro (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_xlabel(param)

    ax2 = ax1.twinx()
    ax2.plot(x, [r["fps"] for r in rows], "o-", color="#e45756", label="FPS")
    ax2.set_ylabel("FPS")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=3)
    ax1.set_title(f"Impacto de '{param}' no desempenho do pipeline")
    fig.tight_layout()
    path = os.path.join(outdir, f"{name}_plot.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> gráfico: {path}")


def run_all(base: RenderSettings, outdir: str = "results", frames: int = 10):
    os.makedirs(outdir, exist_ok=True)
    all_rows = {}
    for name in SWEEPS:
        all_rows[name] = run_sweep(name, base, outdir, frames)

    # Comparação qualitativa: moderno vs. "console antigo"
    _qualitative(base, outdir)
    print(f"\nTodos os experimentos concluídos. Resultados em: {os.path.abspath(outdir)}")
    return all_rows


def _qualitative(base: RenderSettings, outdir: str):
    """Gera imagens lado a lado: pipeline sem restrições vs. restrito."""
    from dataclasses import replace
    presets = {
        "moderno": replace(base, width=512, height=384, texture_res=256,
                           texture_filter="bilinear", precision="float32",
                           color_bits=8, dither=False, palette=False,
                           perspective_correct=True, vertex_snap=0,
                           affine_warp=False, target_tris=8000),
        "ps1_like": replace(base, width=256, height=192, texture_res=32,
                            texture_filter="nearest", precision="fixed12_4",
                            color_bits=5, dither=True, palette=False,
                            perspective_correct=False, affine_warp=True,
                            vertex_snap=1, depth_test=False, target_tris=1200),
        "paleta_16": replace(base, width=256, height=192, texture_res=32,
                             texture_filter="nearest", precision="float16",
                             color_bits=4, dither=True, palette=True,
                             target_tris=1200),
    }
    try:
        from PIL import Image
    except Exception:
        return
    for name, s in presets.items():
        stats, img = measure(s, frames=6)
        up = max(1, 512 // img.shape[1])
        Image.fromarray(img).resize((img.shape[1] * up, img.shape[0] * up),
                                    Image.NEAREST).save(
            os.path.join(outdir, f"preset_{name}.png"))
        print(f"  preset {name:>10}: {stats['ms_total']:6.2f} ms  {stats['fps']:5.1f} fps")
