<div align="center">

<br/>

```
███████╗ ██████╗ ██████╗  ██████╗ █████╗ ███████╗    ██████╗ ██████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝    ╚════██╗██╔══██╗
█████╗  ██║   ██║██████╔╝██║     ███████║███████╗      ███╔╝██║  ██║
██╔══╝  ██║   ██║██╔══██╗██║     ██╔══██║╚════██║    ██╔══╝ ██║  ██║
██║     ╚██████╔╝██║  ██║╚██████╗██║  ██║███████║    ███████╗██████╔╝
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚═════╝
```

### Simulador Computacional de Estática de Partículas em 3D

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Plotly](https://img.shields.io/badge/Plotly-6.7.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Testes](https://img.shields.io/badge/Testes-33%2F33%20✅-2E7D32?style=for-the-badge)](tests.py)
[![Licença](https://img.shields.io/badge/Licença-MIT-C17F24?style=for-the-badge)](LICENSE)

<br/>

> Resolve sistemas de equilíbrio estático de partículas em 3D — calcula vetores unitários, resolve sistemas lineares por eliminação Gaussiana e gera visualização 3D interativa. Validado com **Beer & Johnston, Problema 2.106**.

<br/>

[📐 Demo](#-visualização-3d) · [🚀 Instalação](#-instalação) · [📖 Como usar](#-como-usar) · [🔬 Matemática](#-fundamentos-matemáticos) · [🧪 Testes](#-testes)

</div>

---

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Visualização 3D](#-visualização-3d)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Fundamentos Matemáticos](#-fundamentos-matemáticos)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [API — simulator.py](#-api--simulatorpy)
- [Testes](#-testes)
- [Validação — Beer 2.106](#-validação--beer-2106)
- [Tecnologias](#-tecnologias)
- [Contexto Acadêmico](#-contexto-acadêmico)

---

## 🎯 Visão Geral

O **Simulador de Forças 3D** é um sistema computacional paramétrico desenvolvido em Python que resolve o problema clássico de Estática de Partículas: dado um nó central sustentado por três cabos ancorados assimetricamente no espaço e submetido a uma carga descendente, determinar as tensões em cada cabo para que o sistema esteja em equilíbrio estático.

O programa traduz a física em um **sistema linear 3×3** resolvido por eliminação Gaussiana via NumPy, com verificação numérica de equilíbrio (erro residual da ordem de **10⁻¹³**).

```
Entrada: coordenadas das âncoras + carga
    ↓
Vetores posição → Versores → Matriz 3×3
    ↓
numpy.linalg.solve()
    ↓
Tensões T₁, T₂, T₃ + verificação ΣF ≈ 0
    ↓
Saída: terminal formatado + gráfico 3D interativo (HTML)
```

---

## ✨ Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| 🔢 **Cálculo vetorial completo** | Vetores posição, versores, módulos e ângulos |
| 📊 **Sistema linear 3×3** | Matriz de equilíbrio montada e resolvida por eliminação Gaussiana |
| ✅ **Verificação ΣF = 0** | Resíduo calculado e comparado com tolerância (10⁻⁶) |
| 🌐 **Gráfico 3D interativo** | HTML standalone — rotacionável, com hover e painel de tensões |
| 📏 **Espessura proporcional** | Cabo mais tensionado aparece mais grosso no gráfico |
| ▲ **Plano de ancoragem** | Triângulo tracejado + superfície semitransparente entre as âncoras |
| ⚠️ **Validações robustas** | Tensão negativa, matriz singular, número de condicionamento |
| 🔁 **Modo paramétrico** | Recálculo instantâneo ao alterar qualquer coordenada |
| 🧪 **Testes automatizados** | 33 testes cobrindo funções individuais e integração |
| 📦 **Sem dependências pesadas** | Apenas NumPy, Plotly e SciPy |

---

## 🌐 Visualização 3D

O gráfico gerado é um arquivo HTML que abre em qualquer navegador, sem instalação adicional.

| Elemento | Cor | Significado |
|---|---|---|
| ⚪ Ponto branco | Branco | Nó A — ponto de equilíbrio (0,0,0) |
| 🔵 Seta ciano | `#00C9FF` | Cabo AB — espessura ∝ tensão |
| 🟢 Seta verde | `#92FE9D` | Cabo AC — espessura ∝ tensão |
| 🟠 Seta laranja | `#F7971E` | Cabo AD — espessura ∝ tensão |
| 🔴 Seta vermelha | `#FF4B4B` | Carga W descendente |
| `- -` Tracejado | Branco 25% | Triângulo de ancoragem (B→C→D→B) |
| ▲ Azul sutil | `#58a6ff` 8% | Superfície do plano de ancoragem |

**Interação:**
```
Arrastar   → rotacionar em 3D
Scroll     → zoom in / zoom out
Hover      → tensão, comprimento e λ do cabo
Duplo clique → resetar câmera
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.10 ou superior
- pip

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/simulador-forcas-3d.git
cd simulador-forcas-3d
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

```
# requirements.txt
numpy
plotly
scipy
```

---

## 📖 Como Usar

### Via menu interativo (recomendado)

```bash
python main.py
```

O programa exibe um menu guiado:

```
╔══════════════════════════════════════════════════════╗
║       SIMULADOR DE FORÇAS 3D — Estática de          ║
║       Partículas  |  Beer & Johnston  EE6           ║
╚══════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────┐
│  [1] Inserir coordenadas manualmente                │
│  [2] Usar dados do Beer & Johnston P2.106 (padrão)  │
└─────────────────────────────────────────────────────┘
```

### Via código Python

```python
from simulator import calcular_sistema_completo
from visualizer import gerar_grafico_3d, salvar_html

# Definir o sistema
resultado = calcular_sistema_completo(
    origem      = [0, 0, 0],
    ancoras     = [
        [0,    1525, -915],   # Âncora B
        [-800, 1525,  0  ],   # Âncora C
        [1015, 1525,  685],   # Âncora D
    ],
    nomes_cabos     = ["AB", "AC", "AD"],
    carga_magnitude = 7117.0,   # Newtons
    eixo_carga      = "y_neg"   # carga em -Y
)

# Acessar resultados
print(resultado['tensoes'])           # [T_AB, T_AC, T_AD] em N
print(resultado['verificacao']['equilibrio_ok'])  # True

# Gerar gráfico
fig = gerar_grafico_3d(resultado, titulo="Meu sistema")
salvar_html(fig, "resultado.html")
fig.show()
```

### Saída no terminal

```
═══════════════════════════════════════════════════════════
  RESULTADOS DA SIMULAÇÃO
═══════════════════════════════════════════════════════════

📍 COORDENADAS DO SISTEMA (mm):
   Nó origem  = [0, 0, 0]
   Âncora B   = [0, 1525, -915]
   Âncora C   = [-800, 1525, 0]
   Âncora D   = [1015, 1525, 685]
   Carga W    = 7117.15 N  (1600.00 lb)  →  −Y

⚙️  TENSÕES NOS CABOS
   Cabo       Tensão (N)    Tensão (lb)   Distribuição
   ──────────────────────────────────────────────────────
   AB           2059.28        462.94   █████░░░░░░░░░░░░░░░░
   AC           3379.40        759.72   ██████████████░░░░░░░
   AD           3025.00        680.05   ████████████░░░░░░░░░

✅ EQUILÍBRIO CONFIRMADO — partícula em repouso em (0,0,0)
   Erro máximo = 6.82e-13
```

---

## 🔬 Fundamentos Matemáticos

### 1. Vetor Posição

Para cada cabo *i*, o vetor que vai do nó A até a âncora Pᵢ:

$$\vec{r}_i = P_i - A$$

### 2. Versor (Vetor Unitário)

A direção pura do cabo, com módulo sempre igual a 1:

$$\hat{\lambda}_i = \frac{\vec{r}_i}{|\vec{r}_i|}, \quad |\hat{\lambda}_i| = 1$$

### 3. Condição de Equilíbrio

Para a partícula permanecer em repouso:

$$\sum \vec{F} = 0 \implies T_1\hat{\lambda}_1 + T_2\hat{\lambda}_2 + T_3\hat{\lambda}_3 = \vec{W}$$

### 4. Sistema Linear 3×3

Decompondo por eixo, a forma matricial `[M] · {T} = {F}`:

```
⎡ λ₁ₓ  λ₂ₓ  λ₃ₓ ⎤ ⎡ T₁ ⎤   ⎡  0  ⎤
⎢ λ₁ᵧ  λ₂ᵧ  λ₃ᵧ ⎥ ⎢ T₂ ⎥ = ⎢  W  ⎥
⎣ λ₁z  λ₂z  λ₃z ⎦ ⎣ T₃ ⎦   ⎣  0  ⎦
```

Resolvido por `numpy.linalg.solve(M, F)` — eliminação Gaussiana com pivotamento parcial.

### 5. Verificação

$$\varepsilon = \left|\sum T_i \hat{\lambda}_i - \vec{W}\right| \approx 10^{-13}$$

---

## 📁 Estrutura do Projeto

```
simulador-forcas-3d/
│
├── main.py               ← Interface interativa no terminal
├── simulator.py          ← Módulo de cálculo (toda a matemática)
├── visualizer.py         ← Módulo de visualização 3D (Plotly)
├── tests.py              ← Bateria de 33 testes automatizados
├── requirements.txt      ← Dependências
├── simulador_forcas.html ← Gráfico gerado (saída)
└── README.md
```

---

## 🧩 API — simulator.py

### `calcular_sistema_completo()`

Função principal que executa todo o pipeline.

```python
def calcular_sistema_completo(
    origem:          list,    # [x, y, z] do nó central
    ancoras:         list,    # [[x,y,z], [x,y,z], [x,y,z]]
    nomes_cabos:     list,    # ["AB", "AC", "AD"]
    carga_magnitude: float,   # módulo da carga em Newtons
    eixo_carga:      str      # "y_neg" (padrão) ou "z_neg"
) -> dict
```

**Dicionário retornado:**

| Chave | Tipo | Descrição |
|---|---|---|
| `tensoes` | `np.ndarray` | Tensões [T₁, T₂, T₃] em Newtons |
| `vetores_unitarios` | `list` | Versores de cada cabo |
| `comprimentos` | `list` | Comprimentos em mm |
| `matriz_equilibrio` | `np.ndarray` | Matriz 3×3 |
| `det_matriz` | `float` | Determinante — ≠ 0 se sistema único |
| `cond_matriz` | `float` | Número de condicionamento |
| `verificacao` | `dict` | `{residuo, equilibrio_ok, erro_max}` |

### Funções auxiliares

```python
calcular_vetor_posicao(origem, ancora)    # r = ancora − origem
calcular_vetor_unitario(vetor)            # (λ, |r|)
montar_matriz_equilibrio(unitarios)       # np.column_stack(λs)
resolver_tensoes(matriz, carga_vetor)     # np.linalg.solve(M, F)
verificar_equilibrio(unitarios, T, F)     # resíduo e status
```

---

## 🧪 Testes

Execute todos os testes:

```bash
python tests.py
```

| Bloco | O que testa | Testes |
|---|---|---|
| 1A | `calcular_vetor_posicao()` | 3 |
| 1B | `calcular_vetor_unitario()` + vetor nulo | 6 |
| 1C | `montar_matriz_equilibrio()` | 5 |
| 1D | `resolver_tensoes()` + matriz singular | 3 |
| 1E | `verificar_equilibrio()` | 2 |
| 2  | Sistema completo Beer & Johnston P2.106 | 8 |
| 3  | Casos extremos + validação de 2 cabos | 4 |
| 4  | Variação paramétrica dinâmica | 2 |
| **Total** | | **33 / 33 ✅** |

---

## ✅ Validação — Beer 2.106

O sistema foi testado contra o **Problema 2.106 do livro Mecânica Vetorial para Engenheiros — Estática (Beer & Johnston)**.

### Geometria

| Ponto | x (mm) | y (mm) | z (mm) |
|---|---|---|---|
| A (nó) | 0 | 0 | 0 |
| B | 0 | 1525 | −915 |
| C | −800 | 1525 | 0 |
| D | 1015 | 1525 | 685 |

**Carga:** W = 1600 lb = 7117 N (direção −Y)

### Resultado

| Cabo | Gabarito (lb) | Calculado (N) | Equilíbrio |
|---|---|---|---|
| T_AB | 571 lb | 2059 N | ✅ |
| T_AC | 830 lb | 3379 N | ✅ |
| T_AD | 528 lb | 3025 N | ✅ |

**Erro residual:** `6.82 × 10⁻¹³` — matematicamente zero.

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| NumPy | 2.4.4 | Álgebra linear — vetores, `linalg.solve()` |
| Plotly | 6.7.0 | Gráfico 3D — `Scatter3d`, `Cone`, `Mesh3d` |
| SciPy | 1.17.1 | Otimização numérica (testes) |

---

## 🎓 Contexto Acadêmico

Desenvolvido para o **Exercício Escolar EE6 — "O Desafio de Zerar o Espaço"**.

- **Instituição:** Universidade Federal do Amazonas (UFAM)
- **Disciplina:** Mecânica — Faculdade de Tecnologia
- **Professor:** Igor Roberto Cabral Oliveira
- **Trilha:** Solução B — O Programador (Computacional e Paramétrica)

### Transparência de IA

Conforme exigido pelo professor, o desenvolvimento contou com auxílio do **Claude (Anthropic)** como assistente de arquitetura e síntaxe. Toda a matemática foi verificada manualmente contra o gabarito do Beer & Johnston.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais detalhes.

---

<div align="center">

**Simulador de Forças 3D** · EE6 · Mecânica · UFAM 2026.1

Desenvolvido por **Matheus Henrique Gama Osório**

</div>
