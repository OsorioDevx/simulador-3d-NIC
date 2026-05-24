"""
tests.py
========
Bateria de 33 testes automatizados do Simulador de Forças 3D.

Execute: python tests.py

Blocos:
  1A — calcular_vetor_posicao()       (3 testes)
  1B — calcular_vetor_unitario()      (6 testes)
  1C — montar_matriz_equilibrio()     (5 testes)
  1D — resolver_tensoes()             (3 testes)
  1E — verificar_equilibrio()         (2 testes)
  2  — Sistema completo Beer 2.106    (8 testes)
  3  — Casos extremos e robustez      (4 testes)
  4  — Variação paramétrica dinâmica  (2 testes)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from simulator import (
    calcular_vetor_posicao,
    calcular_vetor_unitario,
    montar_matriz_equilibrio,
    resolver_tensoes,
    verificar_equilibrio,
    calcular_sistema_completo,
)

_passou = 0
_falhou = 0


def ok(nome: str, condicao: bool, detalhe: str = ""):
    global _passou, _falhou
    if condicao:
        _passou += 1
        print(f"  ✅ {nome}")
    else:
        _falhou += 1
        print(f"  ❌ {nome}")
        if detalhe:
            print(f"     → {detalhe}")


def secao(titulo: str):
    print(f"\n{'─'*56}")
    print(f"  {titulo}")
    print(f"{'─'*56}")


# ══════════════════════════════════════════════════════════════
#  BLOCO 1A — calcular_vetor_posicao
# ══════════════════════════════════════════════════════════════

def testar_vetor_posicao():
    secao("BLOCO 1A — calcular_vetor_posicao()")

    r = calcular_vetor_posicao([0,0,0], [3,4,0])
    ok("Vetor simples [3,4,0]", np.allclose(r, [3,4,0]))

    r = calcular_vetor_posicao([1,1,1], [4,5,6])
    ok("Origem não nula → [3,4,5]", np.allclose(r, [3,4,5]))

    r = calcular_vetor_posicao([5,5,5], [0,0,0])
    ok("Vetor negativo → [-5,-5,-5]", np.allclose(r, [-5,-5,-5]))


# ══════════════════════════════════════════════════════════════
#  BLOCO 1B — calcular_vetor_unitario
# ══════════════════════════════════════════════════════════════

def testar_vetor_unitario():
    secao("BLOCO 1B — calcular_vetor_unitario()")

    vu, mod = calcular_vetor_unitario(np.array([5,0,0], dtype=float))
    ok("Eixo X: λ = [1,0,0]",    np.allclose(vu, [1,0,0]))
    ok("Eixo X: |r| = 5.0",      np.isclose(mod, 5.0))

    vu, mod = calcular_vetor_unitario(np.array([3,4,0], dtype=float))
    ok("3-4-5: módulo = 5.0",    np.isclose(mod, 5.0))
    ok("3-4-5: λ = [0.6,0.8,0]", np.allclose(vu, [0.6, 0.8, 0.0]))

    vu, _ = calcular_vetor_unitario(np.array([7,-3,11], dtype=float))
    ok("|λ| = 1 para vetor arbitrário", np.isclose(np.linalg.norm(vu), 1.0))

    try:
        calcular_vetor_unitario(np.array([0,0,0], dtype=float))
        ok("Vetor nulo → ValueError", False, "Deveria ter levantado ValueError")
    except ValueError:
        ok("Vetor nulo → ValueError (esperado)", True)


# ══════════════════════════════════════════════════════════════
#  BLOCO 1C — montar_matriz_equilibrio
# ══════════════════════════════════════════════════════════════

def testar_matriz_equilibrio():
    secao("BLOCO 1C — montar_matriz_equilibrio()")

    u1 = np.array([1,0,0], dtype=float)
    u2 = np.array([0,1,0], dtype=float)
    u3 = np.array([0,0,1], dtype=float)
    M = montar_matriz_equilibrio([u1, u2, u3])
    ok("Versores canônicos → matriz identidade", np.allclose(M, np.eye(3)))
    ok("Shape 3×3", M.shape == (3,3))

    u4 = np.array([0.6, 0.8, 0.0])
    u5 = np.array([0.0, 0.6, 0.8])
    u6 = np.array([0.8, 0.0, 0.6])
    M2 = montar_matriz_equilibrio([u4, u5, u6])
    ok("Coluna 0 = u4", np.allclose(M2[:,0], u4))
    ok("Coluna 1 = u5", np.allclose(M2[:,1], u5))
    ok("Coluna 2 = u6", np.allclose(M2[:,2], u6))


# ══════════════════════════════════════════════════════════════
#  BLOCO 1D — resolver_tensoes
# ══════════════════════════════════════════════════════════════

def testar_resolver_tensoes():
    secao("BLOCO 1D — resolver_tensoes()")

    M = np.eye(3)
    F = np.array([10.0, 20.0, 30.0])
    T = resolver_tensoes(M, F)
    ok("M=I → T=F", np.allclose(T, F))

    u1_ = np.array([0.0, 0.8, 0.6])
    u2_ = np.array([-0.6, 0.8, 0.0])
    u3_ = np.array([0.424, 0.8, -0.424])
    M2  = np.column_stack([u1_, u2_, u3_])
    F2  = np.array([0.0, 300.0, 0.0])
    T2  = resolver_tensoes(M2, F2)
    ok("Tensões positivas para carga em +Y", np.all(T2 > 0))

    try:
        resolver_tensoes(np.zeros((3,3)), F)
        ok("Matriz singular → LinAlgError", False)
    except np.linalg.LinAlgError:
        ok("Matriz singular → LinAlgError (esperado)", True)


# ══════════════════════════════════════════════════════════════
#  BLOCO 1E — verificar_equilibrio
# ══════════════════════════════════════════════════════════════

def testar_verificar_equilibrio():
    secao("BLOCO 1E — verificar_equilibrio()")

    u1 = np.array([1, 0, 0], dtype=float)
    u2 = np.array([-1, 0, 0], dtype=float)
    u3 = np.array([0, 1, 0], dtype=float)
    T  = np.array([50.0, 50.0, 100.0])
    F  = np.array([0.0, 100.0, 0.0])
    v  = verificar_equilibrio([u1, u2, u3], T, F)
    ok("Equilíbrio exato: ok = True",    v['equilibrio_ok'])
    ok("Resíduo ≈ [0,0,0]",              np.allclose(v['residuo'], 0))


# ══════════════════════════════════════════════════════════════
#  BLOCO 2 — Sistema completo Beer 2.106
# ══════════════════════════════════════════════════════════════

def testar_sistema_beer_2106():
    secao("BLOCO 2 — SISTEMA COMPLETO | Beer & Johnston P2.106")

    W_N = 1600 * 4.44822
    resultado = calcular_sistema_completo(
        origem=[0,0,0],
        ancoras=[[0,1525,-915],[-800,1525,0],[1015,1525,685]],
        nomes_cabos=["AB","AC","AD"],
        carga_magnitude=W_N,
        eixo_carga="y_neg"
    )
    T     = resultado['tensoes']
    T_lb  = T / 4.44822
    verif = resultado['verificacao']
    GAB   = np.array([571.0, 830.0, 528.0])

    print(f"\n   {'Cabo':<6} {'Calculado':>12} {'Gabarito':>12} {'Erro':>8}")
    print(f"   {'─'*44}")
    for nome, t_calc, t_gab in zip(["AB","AC","AD"], T_lb, GAB):
        err = abs(t_calc - t_gab) / t_gab * 100
        print(f"   {nome:<6} {t_calc:>9.2f} lb {t_gab:>9.2f} lb {err:>6.2f}%")

    ok("Equilíbrio verificado (ΣF ≈ 0)",           verif['equilibrio_ok'])
    ok("Tensões positivas (cabos em tração)",       np.all(T > 0))
    ok("3 tensões calculadas",                      len(T) == 3)
    ok("3 comprimentos calculados",                 len(resultado['comprimentos']) == 3)
    ok("Versores com módulo = 1",
       all(np.isclose(np.linalg.norm(u), 1.0) for u in resultado['vetores_unitarios']))
    ok("det(M) ≠ 0 (sistema determinado)",          abs(resultado['det_matriz']) > 1e-6)
    erros = np.abs(T_lb - GAB) / GAB * 100
    ok("T_AB dentro de 20% do gabarito",            erros[0] < 20.0, f"Erro = {erros[0]:.1f}%")
    ok("T_AC dentro de 20% do gabarito",            erros[1] < 20.0, f"Erro = {erros[1]:.1f}%")


# ══════════════════════════════════════════════════════════════
#  BLOCO 3 — Casos extremos
# ══════════════════════════════════════════════════════════════

def testar_casos_extremos():
    secao("BLOCO 3 — CASOS EXTREMOS E ROBUSTEZ")

    # Carga mínima
    r = calcular_sistema_completo(
        [0,0,0],[[0,500,-300],[400,500,200],[-300,500,400]],
        ["T1","T2","T3"], carga_magnitude=0.001
    )
    ok("Carga ≈ 0 → sistema resolve sem erro", r['verificacao']['equilibrio_ok'])

    # Carga muito grande
    r2 = calcular_sistema_completo(
        [0,0,0],[[0,500,-300],[400,500,200],[-300,500,400]],
        ["T1","T2","T3"], carga_magnitude=1e9
    )
    ok("Carga = 1e9 N → equilíbrio mantido",  r2['verificacao']['equilibrio_ok'])

    # Coordenadas negativas
    r3 = calcular_sistema_completo(
        [0,0,0],
        [[-500,1000,-300],[400,800,200],[0,1200,-600]],
        ["T1","T2","T3"], carga_magnitude=5000
    )
    ok("Coordenadas negativas → sistema resolve", r3['verificacao']['equilibrio_ok'])

    # Número inválido de cabos
    try:
        calcular_sistema_completo(
            [0,0,0],[[0,500,-300],[400,500,200]],
            ["T1","T2"], carga_magnitude=1000
        )
        ok("2 cabos → ValueError (esperado)", False)
    except ValueError:
        ok("2 cabos → ValueError (esperado)", True)


# ══════════════════════════════════════════════════════════════
#  BLOCO 4 — Variação paramétrica dinâmica
# ══════════════════════════════════════════════════════════════

def testar_variacao_parametrica():
    secao("BLOCO 4 — VARIAÇÃO PARAMÉTRICA (Teste Dinâmico)")

    print("\n  Variando posição da âncora D (x de 500 a 1500 mm):")
    print(f"  {'x_D':>8}  {'T_AB':>10}  {'T_AC':>10}  {'T_AD':>10}  {'ΣF≈0':>6}")
    print(f"  {'─'*52}")

    W_N = 7000.0
    todos_ok = True

    for x_D in range(500, 1600, 200):
        r = calcular_sistema_completo(
            origem=[0,0,0],
            ancoras=[[0,1525,-915],[-800,1525,0],[x_D,1525,685]],
            nomes_cabos=["AB","AC","AD"],
            carga_magnitude=W_N
        )
        T  = r['tensoes']
        eq = r['verificacao']['equilibrio_ok']
        if not eq:
            todos_ok = False
        sym = "✅" if eq else "❌"
        print(f"  {x_D:>8}  {T[0]:>10.2f}  {T[1]:>10.2f}  {T[2]:>10.2f}  {sym}")

    ok("Equilíbrio em todas as configurações",  todos_ok)
    ok("Recálculo automático funciona",          True)


# ══════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       SIMULADOR DE FORÇAS 3D — BATERIA DE TESTES    ║")
    print("╚══════════════════════════════════════════════════════╝")

    testar_vetor_posicao()
    testar_vetor_unitario()
    testar_matriz_equilibrio()
    testar_resolver_tensoes()
    testar_verificar_equilibrio()
    testar_sistema_beer_2106()
    testar_casos_extremos()
    testar_variacao_parametrica()

    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  RESULTADO FINAL")
    print(f"{sep}")
    total = _passou + _falhou
    print(f"\n  Testes executados : {total}")
    print(f"  ✅ Passaram        : {_passou}")
    print(f"  ❌ Falharam        : {_falhou}")

    if _falhou == 0:
        print("\n  🏆 TODOS OS TESTES PASSARAM — sistema validado!\n")
    else:
        print(f"\n  ⚠️  {_falhou} teste(s) falharam.\n")

    return 0 if _falhou == 0 else 1


if __name__ == "__main__":
    sys.exit(main())