"""
main.py
=======
Simulador de Forças 3D — Estática de Partículas
Beer & Johnston | Exercício EE6 · UFAM 2026.1

Execute: python main.py
"""

import sys
import os
import numpy as np
from simulator import calcular_sistema_completo
from visualizer import gerar_grafico_3d, salvar_html


# ══════════════════════════════════════════════════════════════
#  UTILITÁRIOS DE INTERFACE
# ══════════════════════════════════════════════════════════════

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')


def cabecalho():
    print("╔══════════════════════════════════════════════════════╗")
    print("║       SIMULADOR DE FORÇAS 3D — Estática de          ║")
    print("║       Partículas  |  Beer & Johnston  EE6           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def ler_float(mensagem: str, padrao: float = None) -> float:
    """Lê número float com validação e suporte a valor padrão."""
    while True:
        txt = mensagem
        if padrao is not None:
            txt += f" [padrão: {padrao}]"
        txt += ": "
        entrada = input(txt).strip()

        if entrada == "" and padrao is not None:
            return float(padrao)
        try:
            return float(entrada.replace(",", "."))
        except ValueError:
            print("  ⚠️  Valor inválido. Digite um número (ex: 1525 ou -800).\n")


def ler_coordenada(nome_ponto: str, padrao: list = None) -> list:
    """Lê as 3 coordenadas (x, y, z) de um ponto."""
    print(f"\n  📍 Coordenadas do ponto {nome_ponto}:")
    if padrao:
        print(f"     (ENTER para usar o padrão: {padrao})")
    x = ler_float("     x", padrao[0] if padrao else None)
    y = ler_float("     y", padrao[1] if padrao else None)
    z = ler_float("     z", padrao[2] if padrao else None)
    return [x, y, z]


def confirmar(mensagem: str) -> bool:
    """Pede confirmação s/n."""
    while True:
        r = input(f"\n{mensagem} (s/n): ").strip().lower()
        if r in ('s', 'sim', 'y', 'yes', ''):
            return True
        if r in ('n', 'nao', 'não', 'no'):
            return False
        print("  Digite s para sim ou n para não.")


def menu_eixo_carga() -> str:
    print("\n  Em qual direção a carga atua?")
    print("  [1] -Y  (para baixo, eixo Y vertical) ← padrão")
    print("  [2] -Z  (para baixo, eixo Z vertical)")
    while True:
        op = input("\n  Opção [1]: ").strip()
        if op in ("", "1"):
            return "y_neg"
        if op == "2":
            return "z_neg"
        print("  ⚠️  Opção inválida.")


def menu_unidade_carga() -> str:
    print("\n  Unidade da carga:")
    print("  [1] Newtons (N)  ← padrão")
    print("  [2] Libras  (lb)")
    while True:
        op = input("\n  Opção [1]: ").strip()
        if op in ("", "1"):
            return "N"
        if op == "2":
            return "lb"
        print("  ⚠️  Opção inválida.")


# ══════════════════════════════════════════════════════════════
#  IMPRESSÃO DOS RESULTADOS
# ══════════════════════════════════════════════════════════════

def barra_tensao(valor: float, maximo: float, largura: int = 20) -> str:
    """Gera barra visual proporcional à tensão."""
    preenchido = max(1, int(valor / maximo * largura)) if maximo > 0 else 1
    return '█' * preenchido + '░' * (largura - preenchido)


def imprimir_resultados(resultado: dict, eixo_carga: str):
    nomes        = resultado['nomes_cabos']
    tensoes      = resultado['tensoes']
    unitarios    = resultado['vetores_unitarios']
    comprimentos = resultado['comprimentos']
    carga_N      = resultado['carga_magnitude']
    verif        = resultado['verificacao']
    det          = resultado.get('det_matriz', 0)

    sep = "═" * 58

    print(f"\n{sep}")
    print("  RESULTADOS DA SIMULAÇÃO")
    print(f"{sep}")

    print("\n📍 COORDENADAS DO SISTEMA (mm):")
    print(f"   Nó origem  = {resultado['origem'].tolist()}")
    for nome, ancora in zip(nomes, resultado['ancoras']):
        print(f"   Âncora {nome[-1]}    = {ancora.tolist()}")

    eixo_str = "−Y" if eixo_carga == "y_neg" else "−Z"
    print(f"   Carga W    = {carga_N:.2f} N  ({carga_N/4.44822:.2f} lb)  →  {eixo_str}")

    print("\n📐 COMPRIMENTOS DOS CABOS:")
    for nome, comp in zip(nomes, comprimentos):
        print(f"   |r_{nome}| = {comp:>8.2f} mm")

    print("\n🔢 VETORES UNITÁRIOS (λ):")
    print(f"   {'Cabo':<8}  {'λx':>10}  {'λy':>10}  {'λz':>10}")
    print("   " + "─" * 42)
    for nome, vu in zip(nomes, unitarios):
        print(f"   {nome:<8}  {vu[0]:>+10.6f}  {vu[1]:>+10.6f}  {vu[2]:>+10.6f}")

    print("\n📊 MATRIZ DE EQUILÍBRIO [M] 3×3:")
    M = resultado['matriz_equilibrio']
    for i, eixo in enumerate(['x', 'y', 'z']):
        print(f"   {eixo}: [{M[i,0]:>+10.6f}  {M[i,1]:>+10.6f}  {M[i,2]:>+10.6f}]")
    print(f"   det(M) = {det:.4e}  {'✅ Sistema determinado' if abs(det) > 1e-10 else '❌ Singular'}")

    print(f"\n{sep}")
    print("  ⚙️  TENSÕES NOS CABOS")
    print(f"{sep}")
    print(f"\n   {'Cabo':<8}  {'Tensão (N)':>12}  {'Tensão (lb)':>12}  Distribuição")
    print("   " + "─" * 54)

    t_max = max(abs(t) for t in tensoes)
    for nome, t in zip(nomes, tensoes):
        barra = barra_tensao(abs(t), t_max)
        aviso = " ⚠️" if t < 0 else ""
        print(f"   {nome:<8}  {t:>12.2f}  {t/4.44822:>12.2f}  {barra}{aviso}")

    print(f"\n   Carga W    =  {carga_N:>10.2f} N  ({carga_N/4.44822:.2f} lb)")

    print(f"\n{sep}")
    print("  ✅ VERIFICAÇÃO DE EQUILÍBRIO  (ΣF deve ser ≈ 0)")
    print(f"{sep}")
    res = verif['residuo']
    print(f"\n   ΣFx = {res[0]:>+.4e}")
    print(f"   ΣFy = {res[1]:>+.4e}")
    print(f"   ΣFz = {res[2]:>+.4e}")
    print(f"\n   Erro máximo = {verif['erro_max']:.2e}")

    if verif['equilibrio_ok']:
        print("\n   ✅ EQUILÍBRIO CONFIRMADO — partícula em repouso em (0,0,0)")
    else:
        print("\n   ⚠️  ATENÇÃO: Resíduo acima da tolerância numérica!")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════
#  COLETA DE DADOS
# ══════════════════════════════════════════════════════════════

# Dados padrão — Beer & Johnston P2.106
_PAD_ANCORAS = [
    [0,    1525, -915],
    [-800, 1525,  0  ],
    [1015, 1525,  685],
]
_PAD_NOMES   = ["AB", "AC", "AD"]
_PAD_CARGA   = 1600 * 4.44822
_PAD_EIXO    = "y_neg"


def coletar_dados() -> dict:
    limpar_tela()
    cabecalho()
    print("  Dica: pressione ENTER para aceitar o valor padrão (Beer 2.106).\n")

    print("┌─────────────────────────────────────────────────────┐")
    print("│  MODO DE ENTRADA                                    │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  [1] Inserir coordenadas manualmente                │")
    print("│  [2] Usar dados do Beer & Johnston P2.106 (padrão)  │")
    print("└─────────────────────────────────────────────────────┘")

    while True:
        modo = input("\n  Escolha o modo [2]: ").strip()
        if modo in ("", "2"):
            print("\n  ✅ Usando coordenadas do Problema 2.106.")
            return dict(origem=[0,0,0], ancoras=_PAD_ANCORAS,
                        nomes_cabos=_PAD_NOMES, carga_N=_PAD_CARGA, eixo=_PAD_EIXO)
        if modo == "1":
            break
        print("  ⚠️  Opção inválida.")

    # ── Entrada manual ────────────────────────────────────────
    print("\n\n┌─────────────────────────────────────────────────────┐")
    print("│  PASSO 1 — ORIGEM (nó de equilíbrio)                │")
    print("└─────────────────────────────────────────────────────┘")
    origem = ler_coordenada("origem (A)", padrao=[0, 0, 0])

    print("\n\n┌─────────────────────────────────────────────────────┐")
    print("│  PASSO 2 — PONTOS DE ANCORAGEM  (3 cabos)           │")
    print("└─────────────────────────────────────────────────────┘")
    print("  O sistema resolve exatamente 3 cabos (sistema 3×3).")

    ancoras, nomes = [], []
    pads = _PAD_ANCORAS
    for i in range(3):
        letra = chr(ord('B') + i)
        nome  = f"A{letra}"
        ainda = ler_coordenada(f"{letra} (cabo {nome})", padrao=pads[i])
        ancoras.append(ainda)
        nomes.append(nome)

    print("\n\n┌─────────────────────────────────────────────────────┐")
    print("│  PASSO 3 — CARGA APLICADA                           │")
    print("└─────────────────────────────────────────────────────┘")
    unidade = menu_unidade_carga()

    if unidade == "N":
        carga_N = ler_float("  Magnitude da carga (N)", padrao=round(_PAD_CARGA, 2))
    else:
        carga_lb = ler_float("  Magnitude da carga (lb)", padrao=1600.0)
        carga_N  = carga_lb * 4.44822

    print("\n\n┌─────────────────────────────────────────────────────┐")
    print("│  PASSO 4 — DIREÇÃO DA CARGA                         │")
    print("└─────────────────────────────────────────────────────┘")
    eixo = menu_eixo_carga()

    return dict(origem=origem, ancoras=ancoras,
                nomes_cabos=nomes, carga_N=carga_N, eixo=eixo)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    while True:
        dados = coletar_dados()

        print("\n\n🔄 Calculando sistema de forças...")
        try:
            resultado = calcular_sistema_completo(
                origem          = dados['origem'],
                ancoras         = dados['ancoras'],
                nomes_cabos     = dados['nomes_cabos'],
                carga_magnitude = dados['carga_N'],
                eixo_carga      = dados['eixo']
            )
        except Exception as e:
            print(f"\n  ❌ Erro no cálculo: {e}")
            input("\n  Pressione ENTER para tentar novamente...")
            continue

        limpar_tela()
        cabecalho()
        imprimir_resultados(resultado, dados['eixo'])

        if confirmar("  🎨 Deseja gerar o gráfico 3D interativo?"):
            print("\n  Gerando gráfico...")
            carga_lb = dados['carga_N'] / 4.44822
            titulo   = (
                f"Simulador de Forças 3D  |  "
                f"W = {dados['carga_N']:.1f} N  ({carga_lb:.1f} lb)"
            )
            fig = gerar_grafico_3d(resultado, titulo=titulo)
            salvar_html(fig, "simulador_forcas.html")
            print("  🌐 Abrindo no navegador...")
            fig.show()

        if not confirmar("  🔁 Deseja simular outro sistema?"):
            print("\n  ✅ Simulação encerrada. Até logo!\n")
            break

    input("  Pressione ENTER para fechar...")
    sys.exit(0)


if __name__ == "__main__":
    main()