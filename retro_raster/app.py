"""Aplicação interativa (Tkinter): câmera orbital + painel de parâmetros.

Todos os parâmetros do pipeline são alteráveis pela interface, sem necessidade
de modificar o código-fonte, conforme exigido pela metodologia.

Controles de câmera:
  * arrastar com o botão esquerdo  -> orbitar (yaw/pitch)
  * roda do mouse / botão direito  -> zoom (distância)
  * setas                          -> orbitar pelo teclado
  * R                              -> reiniciar câmera
  * Espaço                         -> pausar/retomar animação
  * P                              -> salvar screenshot PNG
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from .pipeline import Renderer, RenderSettings


class App:
    DISPLAY_W, DISPLAY_H = 760, 570

    def __init__(self, settings: RenderSettings | None = None):
        self.renderer = Renderer(settings or RenderSettings())
        self.s = self.renderer.settings

        # Estado da câmera (coordenadas esféricas)
        self.yaw, self.pitch, self.dist = 0.6, 0.35, 5.4
        self.paused = False
        self.t = 0.0
        self._last = time.perf_counter()
        self._drag = None
        self._fps_hist: list[float] = []

        self.root = tk.Tk()
        self.root.title("Aethelion — Pipeline de Rasterização (Eixo 7)")
        self.root.configure(bg="#1b1b22")
        self._build_ui()
        self._bind_events()
        self._update_camera()
        self.root.after(16, self._loop)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg="#1b1b22")
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, width=self.DISPLAY_W, height=self.DISPLAY_H,
                                bg="black", highlightthickness=0, cursor="fleur")
        self.canvas.pack(side="left", padx=8, pady=8)
        self._img_id = self.canvas.create_image(0, 0, anchor="nw")
        self._photo = None

        panel = tk.Frame(main, bg="#23232c")
        panel.pack(side="right", fill="y", padx=(0, 8), pady=8)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background="#23232c", foreground="#e8e8f0", font=("TkDefaultFont", 9))
        style.configure("TCheckbutton", background="#23232c", foreground="#e8e8f0")
        style.configure("TCombobox", fieldbackground="#33333f", background="#33333f")
        style.configure("H.TLabel", background="#23232c", foreground="#8fd0ff",
                        font=("TkDefaultFont", 9, "bold"))

        row = {"n": 0}

        def header(text):
            ttk.Label(panel, text=text, style="H.TLabel").grid(
                row=row["n"], column=0, columnspan=2, sticky="w", padx=8, pady=(10, 2))
            row["n"] += 1

        def slider(label, lo, hi, init, cb, step=1, fmt="{:.0f}"):
            ttk.Label(panel, text=label).grid(row=row["n"], column=0, sticky="w", padx=8)
            val = ttk.Label(panel, text=fmt.format(init))
            val.grid(row=row["n"], column=1, sticky="e", padx=8)
            row["n"] += 1
            var = tk.DoubleVar(value=init)
            timer = [None]

            def on(_=None):
                v = var.get()
                v = round(v / step) * step
                val.config(text=fmt.format(v))
                if timer[0] is not None:
                    panel.after_cancel(timer[0])
                timer[0] = panel.after(80, lambda: cb(v))

            sc = tk.Scale(panel, from_=lo, to=hi, orient="horizontal", variable=var,
                          command=on, showvalue=False, length=210, resolution=step,
                          bg="#23232c", fg="#e8e8f0", troughcolor="#15151b",
                          highlightthickness=0, sliderrelief="flat", bd=0)
            sc.grid(row=row["n"], column=0, columnspan=2, sticky="we", padx=8)
            row["n"] += 1
            return var

        def combo(label, values, init, cb, width=18):
            ttk.Label(panel, text=label).grid(row=row["n"], column=0, sticky="w", padx=8, pady=2)
            var = tk.StringVar(value=init)
            cb_w = ttk.Combobox(panel, textvariable=var, values=values, state="readonly", width=width)
            cb_w.grid(row=row["n"], column=1, sticky="e", padx=8, pady=2)
            cb_w.bind("<<ComboboxSelected>>", lambda e: cb(var.get()))
            row["n"] += 1
            return var

        def check(label, init, cb):
            var = tk.BooleanVar(value=init)
            ttk.Checkbutton(panel, text=label, variable=var,
                            command=lambda: cb(var.get())).grid(
                row=row["n"], column=0, columnspan=2, sticky="w", padx=8, pady=1)
            row["n"] += 1
            return var

        # --- Dispositivo ---
        header("DISPOSITIVO")
        combo("Processamento", ["CPU (Software)", "GPU (ModernGL)"],
              "GPU (ModernGL)" if self.s.device == "gpu" else "CPU (Software)",
              self._set_device, width=18)

        # --- Framebuffer ---
        header("FRAMEBUFFER")
        self._res_presets = [
            (160, 120, "160x120 (Retro Low)"),
            (256, 192, "256x192 (PS1 / NDS)"),
            (320, 240, "320x240 (QVGA)"),
            (400, 300, "400x300"),
            (512, 384, "512x384"),
            (640, 480, "640x480 (VGA)"),
            (800, 600, "800x600 (SVGA)"),
            (1280, 720, "1280x720 (HD)"),
            (1920, 1080, "1920x1080 (Full HD)"),
            (2560, 1440, "2560x1440 (Quad HD)"),
            (3840, 2160, "3840x2160 (4K UHD)"),
        ]
        names = [item[2] for item in self._res_presets]
        match = next((item[2] for item in self._res_presets if item[0] == self.s.width and item[1] == self.s.height), None)
        cur = match or f"{self.s.width}x{self.s.height}"
        combo("Resolução", names, cur, self._set_res, width=18)

        # --- Geometria ---
        header("GEOMETRIA")
        slider("Triângulos (alvo)", 150, 20000, self.s.target_tris,
               self._set_tris, step=50)

        # --- Textura ---
        header("TEXTURA")
        slider("Resolução", 8, 256, self.s.texture_res, self._set_texres, step=8)
        combo("Filtro", ["nearest", "bilinear"], self.s.texture_filter,
              self._set_filter)
        check("Texturas ativas", self.s.textured,
              lambda v: setattr(self.s, "textured", v))

        # --- Precisão ---
        header("PRECISÃO NUMÉRICA")
        combo("Modo", ["float32", "float16", "fixed16_16", "fixed12_4"],
              self.s.precision, lambda v: setattr(self.s, "precision", v))
        slider("Vertex snap", 0, 8, self.s.vertex_snap,
               lambda v: setattr(self.s, "vertex_snap", int(v)), step=1)

        # --- Rasterização ---
        header("RASTERIZAÇÃO")
        check("Correção de perspectiva", self.s.perspective_correct,
              lambda v: setattr(self.s, "perspective_correct", v))
        check("UV afim (artefato PS1)", self.s.affine_warp,
              lambda v: setattr(self.s, "affine_warp", v))
        check("Z-buffer", self.s.depth_test,
              lambda v: setattr(self.s, "depth_test", v))
        check("Backface culling", self.s.backface_cull,
              lambda v: setattr(self.s, "backface_cull", v))
        combo("Sombreamento", ["gouraud", "flat", "unlit"], self.s.shading,
              lambda v: setattr(self.s, "shading", v))

        # --- Cor ---
        header("COR")
        combo("Bits por canal", ["8", "5", "4", "3", "2", "1"], str(self.s.color_bits),
              self._set_bits)
        check("Dithering (Bayer 4x4)", self.s.dither,
              lambda v: setattr(self.s, "dither", v))
        check("Paleta fixa (16 cores)", self.s.palette,
              lambda v: setattr(self.s, "palette", v))

        # --- Estatísticas ---
        header("ESTATÍSTICAS")
        self.stats_label = tk.Label(panel, text="", justify="left", anchor="w",
                                    bg="#15151b", fg="#9fe8a0", font=("TkFixedFont", 8),
                                    padx=6, pady=6)
        self.stats_label.grid(row=row["n"], column=0, columnspan=2, sticky="we", padx=8, pady=4)
        row["n"] += 1

        btns = tk.Frame(panel, bg="#23232c")
        btns.grid(row=row["n"], column=0, columnspan=2, pady=6)
        tk.Button(btns, text="Screenshot", command=self._screenshot,
                  bg="#33333f", fg="#e8e8f0", relief="flat", padx=8).pack(side="left", padx=3)
        tk.Button(btns, text="Reset câmera", command=self._reset_cam,
                  bg="#33333f", fg="#e8e8f0", relief="flat", padx=8).pack(side="left", padx=3)
        row["n"] += 1

        tk.Label(panel, text="Arraste = orbitar · roda = zoom\nEspaço = pausar · P = screenshot",
                 bg="#23232c", fg="#8a8a99", font=("TkDefaultFont", 8),
                 justify="left").grid(row=row["n"], column=0, columnspan=2, padx=8, pady=(0, 8))

    # ------------------------------------------------------------------ #
    def _set_device(self, name: str) -> None:
        self.s.device = "gpu" if "GPU" in name else "cpu"
        self.renderer.sync()

    def _set_res(self, name: str) -> None:
        clean = name.split()[0]
        w, h = (int(x) for x in clean.split("x"))
        self.s.width, self.s.height = w, h
        self.renderer.sync()

    def _set_tris(self, v: float) -> None:
        self.s.target_tris = int(v)
        self.renderer.rebuild_scene()

    def _set_texres(self, v: float) -> None:
        self.s.texture_res = int(v)
        self.renderer.rebuild_textures()

    def _set_filter(self, v: str) -> None:
        self.s.texture_filter = v
        self.renderer.rebuild_textures()

    def _set_bits(self, v: str) -> None:
        self.s.color_bits = int(v)
        self.renderer.rebuild_textures()

    # ------------------------------------------------------------------ #
    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._drag_move)
        c.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        c.bind("<MouseWheel>", lambda e: self._zoom(-e.delta / 120.0))
        c.bind("<Button-4>", lambda e: self._zoom(-1))
        c.bind("<Button-5>", lambda e: self._zoom(1))
        self.root.bind("<Left>", lambda e: self._orbit(-0.1, 0))
        self.root.bind("<Right>", lambda e: self._orbit(0.1, 0))
        self.root.bind("<Up>", lambda e: self._orbit(0, 0.08))
        self.root.bind("<Down>", lambda e: self._orbit(0, -0.08))
        self.root.bind("<space>", lambda e: setattr(self, "paused", not self.paused))
        self.root.bind("r", lambda e: self._reset_cam())
        self.root.bind("p", lambda e: self._screenshot())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _press(self, e):
        self._drag = (e.x, e.y)

    def _drag_move(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self._orbit(dx * 0.008, -dy * 0.006)

    def _orbit(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, -1.45, 1.45))
        self._update_camera()

    def _zoom(self, steps):
        self.dist = float(np.clip(self.dist * (1.1 ** steps), 1.4, 28.0))
        self._update_camera()

    def _reset_cam(self):
        self.yaw, self.pitch, self.dist = 0.6, 0.35, 5.4
        self._update_camera()

    def _update_camera(self):
        cp = np.cos(self.pitch)
        self.renderer.camera_eye = np.array([
            self.dist * cp * np.sin(self.yaw),
            self.dist * np.sin(self.pitch),
            self.dist * cp * np.cos(self.yaw)], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _screenshot(self):
        img = self.renderer.render(self.t)
        name = f"screenshot_{int(time.time())}.png"
        Image.fromarray(img).save(name)
        print(f"[Aethelion] screenshot salvo: {name}")

    # ------------------------------------------------------------------ #
    def _loop(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if not self.paused:
            self.t += dt

        img = self.renderer.render(self.t)

        # Upscale nearest para resoluções retrô; bilinear para resoluções altas
        pil = Image.fromarray(img)
        scale = min(self.DISPLAY_W / img.shape[1], self.DISPLAY_H / img.shape[0])
        size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
        filter_mode = Image.NEAREST if scale >= 1.0 else Image.BILINEAR
        pil = pil.resize(size, filter_mode)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.coords(self._img_id, (self.DISPLAY_W - size[0]) // 2,
                           (self.DISPLAY_H - size[1]) // 2)
        self.canvas.itemconfig(self._img_id, image=self._photo)

        st = self.renderer.stats
        self._fps_hist.append(st.fps)
        if len(self._fps_hist) > 30:
            self._fps_hist.pop(0)
        avg = sum(self._fps_hist) / len(self._fps_hist)

        dev_tag = "GPU (ModernGL)" if self.s.device == "gpu" else "CPU (Software)"
        self.stats_label.config(text=(
            f"Disp       : {dev_tag}\n"
            f"FPS        : {st.fps:6.1f}  (méd {avg:5.1f})\n"
            f"Frame      : {st.ms_total:6.2f} ms\n"
            f"  geometria: {st.ms_geometry:6.2f} ms\n"
            f"  raster   : {st.ms_raster:6.2f} ms\n"
            f"  resolve  : {st.ms_resolve:6.2f} ms\n"
            f"Tris enviad: {st.tris_submitted:6d}\n"
            f"Tris deseng: {st.tris_drawn:6d}\n"
            f"Backface   : {st.tris_culled_back:6d}\n"
            f"Pixels     : {st.pixels_shaded:6d}"
        ))

        elapsed_ms = (time.perf_counter() - now) * 1000.0
        delay = max(5, int(33 - elapsed_ms)) if not self.paused else 30
        self.root.after(delay, self._loop)

    def run(self):
        self.root.mainloop()


def main(settings: RenderSettings | None = None):
    App(settings).run()
