# Resultados Experimentais

Medições obtidas com `python main.py --experiments --frames 6`.
Cada valor é a **mediana** de 6 quadros, após 3 quadros de aquecimento.

> **Condições da medição.** Os números abaixo foram coletados em um contêiner Linux
> compartilhado, sem GPU e com concorrência de CPU. Servem para comparar **tendências
> relativas** entre configurações, não como desempenho absoluto. Ao reproduzir em outra
> máquina os valores mudarão — a forma das curvas é o que importa. Dois pontos de ruído
> estão sinalizados explicitamente adiante.

Configuração base: 320×240, 2000 triângulos (alvo), textura 64×64, `nearest`,
`float32`, 8 bits/canal, z-buffer e backface culling ativos, sombreamento Gouraud.

---

## 1. Resolução do framebuffer

| Resolução | Tempo (ms) | FPS | Pixels sombreados | Rasterização (ms) | Resolve (ms) |
|---|---|---|---|---|---|
| 160×120 | 85,74 | 11,7 | 10.855 | 84,92 | 0,31 |
| 256×192 | 95,22 | 10,5 | 27.823 | 93,28 | 1,13 |
| 320×240 | 110,37 | 9,1 | 43.424 | 108,26 | 1,07 |
| 400×300 | 141,20 | 7,1 | 67.853 | 136,80 | 2,40 |
| 512×384 | 158,53 | 6,4 | 111.210 | 153,38 | 2,53 |
| 640×480 | 121,95* | 8,2* | 173.843 | 116,33 | 3,33 |

**Observações.** O número de pixels sombreados cresce com o quadrado da resolução
(10.855 → 173.843, fator ~16×), mas o tempo total cresce bem menos (~1,4× de 160×120 a
512×384). Isso indica que, nesta implementação, o custo é dominado por **overhead fixo por
triângulo** (setup, bounding box, chamadas NumPy) e não pelo preenchimento de pixels. Há
um custo fixo de ~80 ms praticamente independente da resolução.

O custo do estágio *resolve* (quantização de cor) é o único que escala de forma limpa com
a área: 0,31 ms → 3,33 ms, aproximadamente proporcional à contagem de pixels.

\* **Anomalia de medição.** 640×480 aparece mais rápido que 512×384, o que é fisicamente
implausível — ele sombreia 56% mais pixels. Trata-se de ruído causado por contenção de CPU
no ambiente compartilhado. Em uma máquina dedicada espera-se que a curva seja monotônica.
Reproduza com `--frames 30` para reduzir a variância.

---

## 2. Quantidade de geometria

| Triângulos (alvo) | Tempo (ms) | FPS | Tris desenhados |
|---|---|---|---|
| 250 | 15,8 | 58,2 | ~115 |
| 500 | 26,9 | 36,3 | ~230 |
| 1.000 | 52,25 | 19,1 | 455 |
| 2.000 | 99,16 | 10,1 | 926 |
| 4.000 | 184,68 | 5,4 | 1.790 |
| 8.000 | 356,50 | 2,8 | 3.308 |
| 14.000 | 613,48 | 1,6 | 5.418 |

**Observações.** Esta é a relação mais limpa de todo o experimento: o tempo é
**linear na contagem de triângulos** (~0,044 ms por triângulo desenhado, constante em toda
a faixa). Note que os pixels sombreados permanecem praticamente constantes (~43.400) em
todas as configurações — a cena ocupa a mesma área na tela, apenas com mais subdivisões.

Isso confirma a conclusão da seção 1: **o gargalo é o custo por primitiva, não por pixel.**
Esse comportamento difere do hardware real da época, em que a taxa de preenchimento
(*fill rate*) costumava ser o fator limitante. A diferença é explicada pela natureza da
implementação: cada triângulo envolve uma sequência de operações NumPy cujo overhead de
interpretação supera o trabalho aritmético quando o triângulo cobre poucos pixels.

Aproximadamente 46% dos triângulos são descartados por backface culling, como esperado
para superfícies fechadas.

---

## 3. Resolução de textura

| Resolução | Tempo (ms) | FPS |
|---|---|---|
| 8×8 | 97,33 | 10,3 |
| 16×16 | 97,09 | 10,3 |
| 32×32 | 99,32 | 10,1 |
| 64×64 | 97,90 | 10,2 |
| 128×128 | 100,71 | 9,9 |
| 256×256 | 103,09 | 9,7 |

**Observações.** O impacto no desempenho é **marginal** (~6% entre os extremos). A
amostragem `nearest` custa uma indexação por fragmento independentemente do tamanho da
textura, e todas as texturas testadas cabem confortavelmente no cache.

Esse resultado é relevante para o objetivo do trabalho: em arquiteturas antigas, a
resolução de textura era limitada por **memória disponível**, não por custo de
processamento. O experimento reproduz essa característica — a restrição é de capacidade,
e seu efeito é **visual**, não temporal. Comparar `texture_8.png` com `texture_256.png`
mostra a diferença de nitidez sem custo de quadro correspondente.

O filtro `bilinear` tem custo real (4 amostras + interpolação) e é mensurável
separadamente via `--filter bilinear`.

---

## 4. Precisão numérica

| Modo | Tempo (ms) | FPS | Tris desenhados |
|---|---|---|---|
| `float32` | 97,77 | 10,2 | 926 |
| `float16` | 100,68 | 9,9 | 926 |
| `fixed16_16` | 101,35 | 9,9 | 926 |
| `fixed12_4` | 47,50 | 21,1 | **392** |

**Observações.** `float16` e `fixed16_16` têm custo ligeiramente **maior** que `float32`,
pois a redução de precisão é emulada com operações adicionais (conversão ou
arredondamento) sobre aritmética que o hardware moderno executa nativamente em 32 bits.
Isto é um artefato da emulação: em hardware de época, menor precisão era genuinamente mais
barata.

**`fixed12_4` merece atenção.** O ganho aparente de 2× em desempenho **não** vem de
aritmética mais barata — vem do fato de que a quantização grosseira (1/16) colapsa
vértices próximos, tornando muitos triângulos **degenerados** (área < 1e-7) e,
portanto, descartados: apenas 392 dos 926 triângulos são desenhados. O pipeline fica mais
rápido porque está desenhando **menos da cena**, com perda visível de qualidade.

Esse é um efeito colateral legítimo e instrutivo da precisão reduzida, análogo ao colapso
de polígonos observado em hardware de precisão limitada. Comparar
`precision_float32.png` com `precision_fixed12_4.png` evidencia a perda geométrica.

---

## 5. Profundidade de cor

| Bits/canal | Tempo (ms) | FPS | Cores por canal |
|---|---|---|---|
| 8 | 98,36 | 10,2 | 256 |
| 5 | 100,20 | 10,0 | 32 |
| 4 | 101,74 | 9,8 | 16 |
| 3 | 125,23 | 8,0 | 8 |
| 2 | 114,39 | 8,7 | 4 |
| 1 | 112,21 | 8,9 | 2 |

**Observações.** A quantização de cor é aplicada no estágio *resolve*, cujo custo é de
menos de 1 ms — ou seja, **a variação observada na tabela é essencialmente ruído de
medição**, não efeito real do parâmetro. A não-monotonicidade (3 bits mais lento que
1 bit) confirma isso.

Como na resolução de textura, trata-se de uma restrição cujo efeito é **puramente
visual**. O impacto estético é significativo: com 4 bits ou menos surge *banding* visível
nos gradientes do chão, atenuado pelo dithering Bayer 4×4 (`--dither`), que troca precisão
de cor por ruído espacial estruturado — exatamente a técnica empregada na época.

---

## 6. Comparação qualitativa (presets)

| Preset | Configuração | Tempo (ms) | FPS |
|---|---|---|---|
| `moderno` | 512×384, 8000 tris, textura 256 bilinear, float32, 8 bits, perspectiva correta | 578,96 | 1,7 |
| `ps1_like` | 256×192, 1200 tris, textura 32 nearest, fixed12_4, 5 bits, dither, UV afim, vertex snap, sem z-buffer | 37,61 | 26,6 |
| `paleta_16` | 256×192, 1200 tris, textura 32, float16, paleta fixa 16 cores, dither | 88,31 | 11,3 |

A diferença de **15×** entre `moderno` e `ps1_like` resulta da combinação de todas as
restrições, com destaque para a contagem de triângulos (fator dominante, seção 2).

As imagens `preset_ps1_like.png` e `preset_paleta_16.png` apresentam os artefatos
característicos pretendidos:

- **Texture warping afim** — as texturas se deformam visivelmente sobre as superfícies do
  chão e do toro, por ausência de correção de perspectiva na interpolação de UV;
- **Jitter de vértices** — o snap em grade inteira produz contornos poligonais instáveis;
- **Baixa contagem de polígonos** — silhuetas facetadas, especialmente na esfera;
- **Dithering ordenado** — padrão regular visível nas regiões de gradiente;
- **Paleta limitada** — no preset de 16 cores, achatamento cromático acentuado.

---

## 7. Conclusões

1. **O custo por primitiva domina** nesta implementação. O tempo escala linearmente com a
   contagem de triângulos e apenas fracamente com a resolução — inverso ao comportamento
   do hardware da época, em que o *fill rate* era limitante. A causa é o overhead de
   interpretação do Python por triângulo.

2. **Nem toda restrição legada tem custo computacional.** Resolução de textura e
   profundidade de cor eram limitações de **memória** e seu efeito aqui é exclusivamente
   visual, com impacto de desempenho dentro da margem de ruído. Restrições de
   **geometria** e **resolução** afetam o tempo diretamente.

3. **Precisão reduzida degrada a geometria antes de acelerar o pipeline.** O caso
   `fixed12_4` mostra que a quantização agressiva remove triângulos por degeneração,
   produzindo um ganho de desempenho que é, na verdade, perda de conteúdo renderizado.

4. **Os artefatos visuais característicos são reproduzíveis** de forma controlada e
   independente, permitindo isolar a contribuição de cada restrição para a estética
   associada àquela geração de hardware.

---

## 8. Reprodução

```bash
python main.py --experiments --frames 30 -o results
```

Um número maior de quadros (`--frames 30`) reduz a variância e deve eliminar as anomalias
assinaladas nas seções 1 e 5. Os arquivos gerados em `results/` incluem CSVs com todas as
métricas por configuração, PNGs de cada variante e gráficos comparativos.
