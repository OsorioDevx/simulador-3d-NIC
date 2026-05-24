"""
simulator.py
============
Módulo de cálculo de Estática de Partículas 3D.

Melhorias implementadas:
- Validação de número de cabos (apenas 3 para sistema 3x3)
- Alerta de tensão negativa (cabo em compressão — fisicamente impossível)
- Número de condicionamento da matriz (estabilidade numérica)
- Verificação de equilíbrio com tolerância documentada
- Funções puras sem efeitos colaterais

Validado com: Beer & Johnston — Problema 2.106
Autor: Simulador EE6 · UFAM 2026.1
"""

import numpy as np


# ── Constantes ────────────────────────────────────────────────
TOLERANCIA_EQUILIBRIO = 1e-6   # resíduo máximo aceito para ΣF ≈ 0
COND_AVISO            = 1e8    # número de condicionamento que dispara aviso


# ─────────────────────────────────────────────────────────────
# 1. VETORES
# ─────────────────────────────────────────────────────────────

def calcular_vetor_posicao(origem: list, ancora: list) -> np.ndarray:
    """
    Calcula o vetor posição do nó de origem até o ponto de ancoragem.

    r = ancora − origem

    Args:
        origem: coordenadas [x, y, z] do nó central
        ancora: coordenadas [x, y, z] do ponto de ancoragem

    Returns:
        np.ndarray: vetor posição [rx, ry, rz]
    """
    return np.array(ancora, dtype=float) - np.array(origem, dtype=float)


def calcular_vetor_unitario(vetor: np.ndarray) -> tuple:
    """
    Calcula o vetor unitário (versor) e o módulo de um vetor posição.

    λ = r / |r|      |λ| = 1 sempre

    Args:
        vetor: vetor posição [rx, ry, rz]

    Returns:
        tuple: (vetor_unitario: np.ndarray, modulo: float)

    Raises:
        ValueError: se o vetor for nulo (origem == âncora)
    """
    modulo = np.linalg.norm(vetor)

    if modulo < 1e-10:
        raise ValueError(
            "Vetor nulo detectado — origem e âncora não podem ser o mesmo ponto."
        )

    return vetor / modulo, float(modulo)


# ─────────────────────────────────────────────────────────────
# 2. SISTEMA LINEAR
# ─────────────────────────────────────────────────────────────

def montar_matriz_equilibrio(vetores_unitarios: list) -> np.ndarray:
    """
    Monta a matriz de equilíbrio empilhando os versores como colunas.

    Cada coluna i = componentes (x, y, z) do versor λᵢ.
    Sistema a resolver: [M] · {T} = {F}

    Args:
        vetores_unitarios: lista de np.ndarray — versores de cada cabo

    Returns:
        np.ndarray: matriz 3×3
    """
    return np.column_stack(vetores_unitarios)


def resolver_tensoes(matriz: np.ndarray, carga_vetor: np.ndarray) -> np.ndarray:
    """
    Resolve o sistema linear [M] · {T} = {F} para as tensões.

    Usa eliminação Gaussiana com pivotamento parcial (numpy).
    Verifica o número de condicionamento antes de resolver.

    Args:
        matriz:       matriz 3×3 de equilíbrio
        carga_vetor:  vetor da carga [Fx, Fy, Fz]

    Returns:
        np.ndarray: tensões [T1, T2, T3]

    Raises:
        np.linalg.LinAlgError: se a matriz for singular (cabos coplanares)
    """
    # Verificar condicionamento
    cond = np.linalg.cond(matriz)
    if cond > COND_AVISO:
        print(
            f"\n  ⚠️  Atenção: matriz mal condicionada (cond = {cond:.2e}).\n"
            "     Os cabos podem estar próximos de coplanares — resultado pode ser impreciso."
        )

    try:
        return np.linalg.solve(matriz, carga_vetor)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            "Matriz singular: cabos coplanares — o sistema não tem solução única.\n"
            "Ajuste as coordenadas para que os cabos apontem em direções distintas nos 3 eixos."
        )


def verificar_equilibrio(
    vetores_unitarios: list,
    tensoes: np.ndarray,
    carga_vetor: np.ndarray,
    tolerancia: float = TOLERANCIA_EQUILIBRIO
) -> dict:
    """
    Verifica se a condição de equilíbrio ΣF = 0 é satisfeita.

    Resíduo = Σ(Tᵢ · λᵢ) − F_carga  →  deve ser ≈ [0, 0, 0]

    Args:
        vetores_unitarios: versores de cada cabo
        tensoes:           tensões calculadas
        carga_vetor:       vetor da carga aplicada
        tolerancia:        margem para aceitar equilíbrio (padrão 1e-6)

    Returns:
        dict com chaves: 'residuo', 'equilibrio_ok', 'erro_max'
    """
    forca_cabos = sum(t * u for t, u in zip(tensoes, vetores_unitarios))
    residuo     = forca_cabos - carga_vetor
    erro_max    = float(np.max(np.abs(residuo)))

    return {
        'residuo':       residuo,
        'equilibrio_ok': erro_max < tolerancia,
        'erro_max':      erro_max,
    }


# ─────────────────────────────────────────────────────────────
# 3. ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

def calcular_sistema_completo(
    origem:          list,
    ancoras:         list,
    nomes_cabos:     list,
    carga_magnitude: float,
    eixo_carga:      str = 'y_neg'
) -> dict:
    """
    Executa todo o pipeline de cálculo e retorna os resultados.

    Passos internos:
      1. Define vetor de carga conforme eixo
      2. Valida número de cabos (deve ser 3)
      3. Calcula vetores posição e unitários
      4. Monta e resolve a matriz 3×3
      5. Verifica tensões negativas (cabo em compressão)
      6. Verifica equilíbrio ΣF ≈ 0

    Args:
        origem:          coordenadas [x, y, z] do nó central
        ancoras:         lista de coordenadas [[x1,y1,z1], ...]
        nomes_cabos:     nomes dos cabos (ex: ['AB','AC','AD'])
        carga_magnitude: módulo da carga em Newtons
        eixo_carga:      'y_neg' (peso em −Y) ou 'z_neg' (peso em −Z)

    Returns:
        dict: resultado completo do sistema

    Raises:
        ValueError: número de cabos diferente de 3
    """
    # 1. Vetor de carga
    eixo = eixo_carga.lower().strip()
    if eixo == 'y_neg':
        carga_vetor = np.array([0.0, carga_magnitude, 0.0])
    elif eixo == 'z_neg':
        carga_vetor = np.array([0.0, 0.0, carga_magnitude])
    else:
        raise ValueError(
            f"Eixo '{eixo_carga}' não reconhecido. Use 'y_neg' ou 'z_neg'."
        )

    # 2. Validar número de cabos
    n = len(ancoras)
    if n != 3:
        raise ValueError(
            f"Este solver exige exatamente 3 cabos (sistema 3×3). "
            f"Recebidos: {n} cabo(s)."
        )

    # 3. Vetores posição e unitários
    vetores_posicao  = [calcular_vetor_posicao(origem, a) for a in ancoras]
    vetores_unitarios, modulos = [], []
    for vp in vetores_posicao:
        vu, mod = calcular_vetor_unitario(vp)
        vetores_unitarios.append(vu)
        modulos.append(mod)

    # 4. Montar e resolver
    matriz  = montar_matriz_equilibrio(vetores_unitarios)
    tensoes = resolver_tensoes(matriz, carga_vetor)

    # 5. Alertar tensões negativas
    negativos = [nomes_cabos[i] for i, t in enumerate(tensoes) if t < 0]
    if negativos:
        print(
            f"\n  ⚠️  Tensão negativa detectada nos cabos: {negativos}.\n"
            "     Cabos suportam apenas tração. Verifique a geometria do sistema."
        )

    # 6. Verificar equilíbrio
    verificacao = verificar_equilibrio(vetores_unitarios, tensoes, carga_vetor)

    return {
        'origem':             np.array(origem, dtype=float),
        'ancoras':            [np.array(a, dtype=float) for a in ancoras],
        'nomes_cabos':        nomes_cabos,
        'carga_magnitude':    carga_magnitude,
        'carga_vetor':        carga_vetor,
        'vetores_posicao':    vetores_posicao,
        'vetores_unitarios':  vetores_unitarios,
        'comprimentos':       modulos,
        'matriz_equilibrio':  matriz,
        'tensoes':            tensoes,
        'verificacao':        verificacao,
        'cond_matriz':        float(np.linalg.cond(matriz)),
        'det_matriz':         float(np.linalg.det(matriz)),
    }