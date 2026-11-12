# Aethelion — Pipeline de Rasterização Inspirado em Consoles Antigos

Implementação de um **pipeline de renderização tridimensional por software**, desenvolvido
conforme as especificações da metodologia proposta (Eixo 7 — arquitetura de consoles antigos).

O fluxo implementado corresponde à equação (1) do documento:

```
Geometria → Transformação → Projeção → Rasterização → Framebuffer → Imagem
```

Nenhuma biblioteca gráfica acelerada é utilizada: toda a rasterização é feita em CPU com
NumPy, o que torna explícitas (e mensuráveis) cada etapa do pipeline.

---

## 1. Requisitos

| Dependência | Uso | Obrigatória |
|---|---|---|
| Python ≥ 3.9 | — | sim |
| `numpy` | núcleo do pipeline CPU | sim |
| `pillow` | gravação de PNG e exibição na GUI | sim |
| `moderngl` | aceleração por hardware (GPU) | opcional |
| `tkinter` | interface interativa | só para a GUI |
| `matplotlib` | gráficos dos experimentos | opcional |

```bash
pip install -r requirements.txt
```

> `tkinter` acompanha o Python oficial no Windows/macOS. No Linux, instale o pacote do
> sistema: `sudo apt install python3-tk`.

---

## 2. Execução

### 2.1 Aplicação interativa (GUI)

```bash
python main.py
```

Abre a janela com **câmera interativa** e painel lateral onde **todos** os parâmetros do
pipeline podem ser alterados em tempo real, sem editar o código-fonte.

**Controles de câmera**

| Entrada | Ação |
|---|---|
| Arrastar com botão esquerdo | orbitar (yaw / pitch) |
| Roda do mouse | zoom |
| Setas | orbitar pelo teclado |
| `R` | reiniciar câmera |
| `Espaço` | pausar / retomar animação |
| `P` | salvar screenshot PNG |
| `Esc` | sair |

### 2.2 Modo headless (sem display)

```bash
python main.py --headless -o results
python main.py --headless --width 256 --height 192 --tris 1200 --precision fixed12_4 \
               --vertex-snap 1 --affine --bits 5 --dither -o results
```

### 2.3 Experimentos automatizados

```bash
python main.py --experiments                  # todas as varreduras
python main.py --experiments --sweep geometry # apenas uma varredura
```

Gera, em `results/`: um **CSV por varredura**, **PNGs** de cada configuração,
**gráficos** de desempenho e três **presets comparativos**.

---

## 3. Parâmetros experimentais

Todos disponíveis na GUI e na linha de comando.

### 3.1 Dispositivo de processamento
| Parâmetro | CLI | Valores |
|---|---|---|
| Dispositivo | `--device` | `cpu` (software rasterizer puro via NumPy), `gpu` (hardware via ModernGL) |

### 3.2 Framebuffer
| Parâmetro | CLI | Valores |
|---|---|---|
| Resolução | `--width`, `--height` | 160×120 … 640×480 |

A imagem é renderizada na resolução interna escolhida e ampliada por **nearest neighbor**
para exibição, preservando o aspecto de pixels grandes.

### 3.2 Geometria
| Parâmetro | CLI | Valores |
|---|---|---|
| Quantidade de triângulos | `--tris` | 150 … 20000 |

A cena (toro, esfera, cubo e plano de chão) é **paramétrica**: a densidade de cada malha é
escalada automaticamente para aproximar o orçamento de triângulos solicitado.

### 3.3 Textura
| Parâmetro | CLI | Valores |
|---|---|---|
| Resolução da textura | `--texres` | 8 … 256 |
| Filtro | `--filter` | `nearest`, `bilinear` |
| Texturas ativas | `--no-texture` | — |

Quatro texturas procedurais (`checker`, `bricks`, `gradient`, `noise`), geradas na
resolução pedida. Resoluções baixas reproduzem a memória gráfica restrita de consoles.

### 3.4 Precisão numérica
| Parâmetro | CLI | Valores |
|---|---|---|
| Modo de precisão | `--precision` | `float32`, `float16`, `fixed16_16`, `fixed12_4` |
| Snap de vértices | `--vertex-snap` | 0 (desligado) … 8 |

- `fixed16_16` — ponto fixo 16.16, quantização de 1/65536.
- `fixed12_4` — ponto fixo 12.4, quantização de 1/16; precisão grosseira típica do PS1.
- `--vertex-snap` força os vértices projetados a uma grade inteira, reproduzindo o
  **jitter / tremulação de polígonos** característico daquela geração.

### 3.5 Rasterização
| Parâmetro | CLI | Efeito |
|---|---|---|
| Correção de perspectiva | `--no-perspective` | interpolação afim dos atributos |
| UV afim | `--affine` | artefato clássico de *texture warping* |
| Z-buffer | `--no-zbuffer` | alterna para *painter's algorithm* (ordenação por profundidade) |
| Backface culling | `--no-cull` | desativa descarte de faces traseiras |
| Sombreamento | `--shading` | `gouraud`, `flat`, `unlit` |

### 3.6 Cor
| Parâmetro | CLI | Valores |
|---|---|---|
| Bits por canal | `--bits` | 8, 5, 4, 3, 2, 1 |
| Dithering | `--dither` | Bayer 4×4 ordenado |
| Paleta fixa | `--palette` | 16 cores, por vizinho mais próximo |

---

## 4. Estrutura do código

```
Aethelion/
├── main.py                  # CLI: GUI, headless e experimentos
├── requirements.txt
├── README.md
├── RESULTADOS.md            # discussão dos resultados experimentais
└── aethelion/
    ├── __init__.py
    ├── mathutil.py          # matrizes 4x4, look_at, perspective (coord. homogêneas)
    ├── geometry.py          # malhas paramétricas e montagem da cena
    ├── texture.py           # texturas procedurais, amostragem nearest/bilinear
    ├── framebuffer.py       # buffers de cor/profundidade, quantização, dithering, paleta
    ├── pipeline.py          # transformação, recorte, rasterização, sombreamento, métricas
    ├── experiments.py       # varreduras de parâmetros, CSV e gráficos
    └── app.py               # GUI Tkinter: câmera interativa + painel de parâmetros
```

### Etapas do pipeline (`pipeline.py`)

1. **Transformação** — vértices levados a espaço de clip por `MVP = Projeção · Visão · Modelo`,
   em coordenadas homogêneas. Normais transformadas para iluminação por vértice.
2. **Projeção** — perspectiva clássica; a precisão numérica configurada é aplicada aqui.
3. **Recorte** — Sutherland-Hodgman contra o plano *near* (`w > ε`), com re-triangulação
   em leque, evitando divisão por zero e artefatos atrás da câmera.
4. **Rasterização** — triagem vetorizada em lote (área com sinal, backface culling,
   rejeição por bounding box) seguida de varredura por **funções de aresta** e coordenadas
   baricêntricas; teste de profundidade contra o z-buffer.
5. **Sombreamento** — interpolação com ou sem correção de perspectiva, amostragem de
   textura e iluminação difusa.
6. **Resolve** — quantização de cor, dithering e mapeamento de paleta no framebuffer.

---

## 5. Métricas coletadas

Por quadro, exibidas na GUI e exportadas nos CSVs:

- `ms_total`, `fps`
- `ms_geometry`, `ms_raster`, `ms_resolve` (tempo por etapa)
- `tris_submitted`, `tris_drawn`, `tris_culled_back`, `tris_clipped`
- `pixels_shaded`

Os experimentos reportam a **mediana** de N quadros, descartando quadros de aquecimento.

---

## 6. Observações de desempenho

O renderizador é **inteiramente por software**, propositalmente: o objetivo é medir o
custo de cada etapa do pipeline, não atingir taxas de quadros elevadas. O custo é dominado
pela rasterização e escala com a **área coberta em pixels**, não apenas com a contagem de
triângulos.

Para uso interativo fluido, recomenda-se 256×192 ou menos com ~1000–2000 triângulos —
configuração que, convenientemente, é também a mais próxima do hardware que o trabalho
busca estudar.

---

## 7. Referências

As referências abaixo são as do documento de especificação e orientam as escolhas de
implementação:

1. T. Akenine-Möller, E. Haines, N. Hoffman. *Real-Time Rendering*, 4ª ed. CRC Press, 2018.
   — organização geral do pipeline e rasterização.
2. J. F. Blinn. "A Trip Down the Graphics Pipeline: The Homogeneous Perspective Transform."
   *IEEE CG&A*, 13(3):75–80, 1993. — transformação de perspectiva e coordenadas homogêneas.
3. C. Everitt, M. J. Kilgard. "Practical and Robust Stenciled Shadow Volumes for
   Hardware-Accelerated Rendering." *GDC*, 2002. — referência principal do eixo.
4. M. Li, M. Magnor, H.-P. Seidel. "Online Accelerated Rendering of Visual Hulls in Real
   Scenes." *Journal of WSCG*, 11(1), 2003. — referência de estrutura experimental.
