"""
visualizer.py
=============
Módulo de visualização 3D interativa do sistema de forças.

Melhorias implementadas:
- Espessura dos cabos proporcional à tensão (3px → 14px)
- Cone da seta também proporcional à espessura
- Linha tracejada conectando as âncoras (triângulo de ancoragem)
- Plano semitransparente no triângulo de ancoragem (Mesh3d, opacity=0.08)
- Hover com espessura visual exibida
- Paleta expandida (suporta até 6 cabos)

Autor: Simulador EE6 · UFAM 2026.1
"""

import numpy as np
import plotly.graph_objects as go


# ── Paleta ─────────────────────────────────────────────────────
CORES_CABOS = [
    "#00C9FF",   # ciano
    "#92FE9D",   # verde
    "#F7971E",   # laranja
    "#C084FC",   # roxo
    "#FB7185",   # rosa
    "#34D399",   # esmeralda
]
COR_CARGA = "#FF4B4B"   # vermelho
COR_NO    = "#FFFFFF"   # branco

# ── Limites de espessura visual (px) ───────────────────────────
LARGURA_MIN = 3
LARGURA_MAX = 14


# ─────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────

def _calcular_espessuras(tensoes: np.ndarray) -> list:
    """
    Mapeia tensões para espessuras de linha proporcionais.

    Fórmula:
        w_i = LARGURA_MIN + (t_i − t_min) / (t_max − t_min)
              × (LARGURA_MAX − LARGURA_MIN)

    Tensões iguais → espessura média para todos (evita divisão por zero).

    Args:
        tensoes: array de tensões calculadas

    Returns:
        list[float]: espessura de cada cabo (mesma ordem)
    """
    t_min     = float(np.min(tensoes))
    t_max     = float(np.max(tensoes))
    amplitude = t_max - t_min

    if amplitude < 1e-10:
        media = (LARGURA_MIN + LARGURA_MAX) / 2
        return [media] * len(tensoes)

    return [
        LARGURA_MIN + (float(t) - t_min) / amplitude * (LARGURA_MAX - LARGURA_MIN)
        for t in tensoes
    ]


def _vetor_para_cone(origem, destino, escala: float = 1.0):
    """
    Converte dois pontos em coordenadas para Scatter3d + Cone.

    Returns:
        (xl, yl, zl, ux, uy, uz, cx, cy, cz)
    """
    o = np.array(origem, dtype=float)
    d = np.array(destino, dtype=float)
    direcao = d - o
    ponta   = o + direcao * escala
    base    = o + direcao * escala * 0.88   # base do cone a 88%

    xl = [o[0], ponta[0], None]
    yl = [o[1], ponta[1], None]
    zl = [o[2], ponta[2], None]

    return (
        xl, yl, zl,
        float(direcao[0]), float(direcao[1]), float(direcao[2]),
        float(base[0]),    float(base[1]),    float(base[2]),
    )


# ─────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────

def gerar_grafico_3d(resultado: dict, titulo: str = "Simulador de Forças 3D") -> go.Figure:
    """
    Gera o gráfico 3D interativo completo do sistema de forças.

    Elementos criados (na ordem de sobreposição):
      1. Cabos — linha + cone (espessura proporcional à tensão)
      2. Triângulo tracejado entre âncoras
      3. Plano de ancoragem semitransparente (Mesh3d)
      4. Carga descendente — linha + cone
      5. Nó central (origem)
      6. Painel de anotações com tensões

    Args:
        resultado: dicionário retornado por calcular_sistema_completo()
        titulo:    título do gráfico

    Returns:
        go.Figure: figura Plotly pronta para .show() ou .write_html()
    """
    origem       = resultado['origem']
    ancoras      = resultado['ancoras']
    nomes        = resultado['nomes_cabos']
    tensoes      = resultado['tensoes']
    carga_v      = resultado['carga_vetor']
    unitarios    = resultado['vetores_unitarios']
    comprimentos = resultado['comprimentos']
    verif        = resultado['verificacao']

    max_coord = max(np.max(np.abs(a)) for a in ancoras)
    esc_carga = max_coord * 0.5

    traces = []

    # ── 1. Cabos ─────────────────────────────────────────────
    espessuras = _calcular_espessuras(tensoes)

    for i, (ancora, nome, tensao, vu, cor, larg) in enumerate(
            zip(ancoras, nomes, tensoes, unitarios, CORES_CABOS, espessuras)):

        xl, yl, zl, ux, uy, uz, cx, cy, cz = _vetor_para_cone(origem, ancora)

        # Cone proporcional: 5% a 10% de max_coord
        fator_cone = 0.05 + (larg - LARGURA_MIN) / (LARGURA_MAX - LARGURA_MIN) * 0.05

        # Linha
        traces.append(go.Scatter3d(
            x=xl, y=yl, z=zl,
            mode='lines',
            name=f"Cabo {nome}",
            line=dict(color=cor, width=larg),
            hovertemplate=(
                f"<b>Cabo {nome}</b><br>"
                f"Tensão: {tensao:.1f} N ({tensao/4.44822:.1f} lb)<br>"
                f"Comprimento: {comprimentos[i]:.1f} mm<br>"
                f"λ = ({vu[0]:+.3f}, {vu[1]:+.3f}, {vu[2]:+.3f})<br>"
                f"Espessura visual: {larg:.1f} px"
                "<extra></extra>"
            )
        ))

        # Cone (seta)
        norm = np.linalg.norm([ux, uy, uz])
        traces.append(go.Cone(
            x=[cx], y=[cy], z=[cz],
            u=[ux/norm], v=[uy/norm], w=[uz/norm],
            sizemode='absolute',
            sizeref=max_coord * fator_cone,
            colorscale=[[0, cor], [1, cor]],
            showscale=False, hoverinfo='skip',
            name=f"Seta {nome}"
        ))

        # Rótulo da âncora
        traces.append(go.Scatter3d(
            x=[ancora[0]], y=[ancora[1]], z=[ancora[2]],
            mode='markers+text',
            marker=dict(size=7, color=cor, symbol='circle'),
            text=[f"  {nome[1]}({ancora[0]:.0f},{ancora[1]:.0f},{ancora[2]:.0f})"],
            textfont=dict(color=cor, size=11),
            textposition='middle right',
            showlegend=False,
            hovertemplate=(
                f"<b>Âncora {nome[1]}</b><br>"
                f"({ancora[0]:.0f}, {ancora[1]:.0f}, {ancora[2]:.0f}) mm"
                "<extra></extra>"
            )
        ))

    # ── 2. Triângulo tracejado entre âncoras ─────────────────
    pts = list(ancoras) + [ancoras[0]]   # fecha o triângulo
    traces.append(go.Scatter3d(
        x=[p[0] for p in pts],
        y=[p[1] for p in pts],
        z=[p[2] for p in pts],
        mode='lines',
        name='Plano de ancoragem',
        line=dict(color='rgba(255,255,255,0.25)', width=2, dash='dash'),
        hovertemplate=(
            "<b>Plano de ancoragem</b><br>"
            "Triângulo formado pelas âncoras"
            "<extra></extra>"
        )
    ))

    # ── 3. Superfície semitransparente (Mesh3d) ───────────────
    traces.append(go.Mesh3d(
        x=[a[0] for a in ancoras],
        y=[a[1] for a in ancoras],
        z=[a[2] for a in ancoras],
        i=[0], j=[1], k=[2],
        color='#58a6ff',
        opacity=0.08,
        name='Superfície de ancoragem',
        showlegend=True,
        hovertemplate=(
            "<b>Superfície de ancoragem</b><br>"
            "Plano definido pelas âncoras"
            "<extra></extra>"
        ),
        flatshading=True,
        lighting=dict(ambient=1.0),
    ))

    # ── 4. Carga descendente ──────────────────────────────────
    carga_dir    = -carga_v / np.linalg.norm(carga_v)
    dest_carga   = origem + carga_dir * esc_carga
    carga_mag    = float(np.linalg.norm(carga_v))

    xl, yl, zl, ux, uy, uz, cx, cy, cz = _vetor_para_cone(origem, dest_carga)

    traces.append(go.Scatter3d(
        x=xl, y=yl, z=zl,
        mode='lines',
        name=f"Carga W = {carga_mag:.1f} N",
        line=dict(color=COR_CARGA, width=6),
        hovertemplate=(
            f"<b>Carga W</b><br>"
            f"W = {carga_mag:.1f} N ({carga_mag/4.44822:.1f} lb)"
            "<extra></extra>"
        )
    ))

    norm = np.linalg.norm([ux, uy, uz])
    traces.append(go.Cone(
        x=[cx], y=[cy], z=[cz],
        u=[ux/norm], v=[uy/norm], w=[uz/norm],
        sizemode='absolute', sizeref=max_coord * 0.07,
        colorscale=[[0, COR_CARGA], [1, COR_CARGA]],
        showscale=False, hoverinfo='skip', name="Seta carga"
    ))

    # ── 5. Nó central ─────────────────────────────────────────
    # Anel externo semitransparente
    traces.append(go.Scatter3d(
        x=[origem[0]], y=[origem[1]], z=[origem[2]],
        mode='markers',
        marker=dict(size=22, color='rgba(255,255,255,0.1)', symbol='circle'),
        showlegend=False, hoverinfo='skip', name="Anel nó"
    ))
    # Ponto principal
    traces.append(go.Scatter3d(
        x=[origem[0]], y=[origem[1]], z=[origem[2]],
        mode='markers+text',
        marker=dict(size=12, color=COR_NO, symbol='circle',
                    line=dict(color='gray', width=2)),
        text=["  A (0,0,0)"],
        textfont=dict(color=COR_NO, size=12),
        textposition='middle right',
        name="Nó A (equilíbrio)",
        hovertemplate=(
            "<b>Nó Central A</b><br>"
            "Ponto de equilíbrio<br>(0, 0, 0)"
            "<extra></extra>"
        )
    ))

    # ── 6. Painel de anotações ────────────────────────────────
    annot = ["<b>⚖️ Tensões nos Cabos</b>", "─────────────────────"]
    for nome, tensao, cor in zip(nomes, tensoes, CORES_CABOS):
        annot.append(
            f"<span style='color:{cor}'>■</span> "
            f"T<sub>{nome}</sub> = <b>{tensao:.1f} N</b> "
            f"({tensao/4.44822:.1f} lb)"
        )
    annot += [
        "─────────────────────",
        f"<b>W = {carga_mag:.1f} N ({carga_mag/4.44822:.1f} lb)</b>",
        f"det(M) = {resultado.get('det_matriz', 0):.3e}",
        f"ΣF ≈ <b>{verif['erro_max']:.1e}</b> ✅"
    ]

    # ── 7. Layout ─────────────────────────────────────────────
    lim = max_coord * 1.2

    layout = go.Layout(
        title=dict(text=titulo, font=dict(color='white', size=17), x=0.5),
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0D1117',
        scene=dict(
            xaxis=dict(title='X (mm)', gridcolor='#30363d', zerolinecolor='#58a6ff',
                       range=[-lim, lim], backgroundcolor='#0D1117', color='white'),
            yaxis=dict(title='Y (mm)', gridcolor='#30363d', zerolinecolor='#58a6ff',
                       range=[-lim * 0.6, lim], backgroundcolor='#0D1117', color='white'),
            zaxis=dict(title='Z (mm)', gridcolor='#30363d', zerolinecolor='#58a6ff',
                       range=[-lim, lim], backgroundcolor='#0D1117', color='white'),
            bgcolor='#0D1117',
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)),
            aspectmode='cube',
        ),
        legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#30363d',
                    font=dict(color='white', size=11), x=0.01, y=0.99),
        annotations=[dict(
            text="<br>".join(annot),
            align='left', showarrow=False,
            xref='paper', yref='paper',
            x=1.0, y=0.98,
            bordercolor='#30363d', borderwidth=1,
            bgcolor='rgba(13,17,23,0.85)',
            font=dict(color='white', size=11),
            xanchor='right'
        )],
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return go.Figure(data=traces, layout=layout)


def salvar_html(fig: go.Figure, caminho: str = "simulador_forcas.html"):
    """
    Salva o gráfico como HTML standalone (abre em qualquer navegador).

    Args:
        fig:     figura Plotly
        caminho: caminho do arquivo de saída
    """
    fig.write_html(
        caminho,
        include_plotlyjs='cdn',
        full_html=True,
        config={'scrollZoom': True, 'displayModeBar': True}
    )
    print(f"✅ Gráfico salvo em: {caminho}")