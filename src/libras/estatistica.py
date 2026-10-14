"""Testes estatisticos para comparar condicoes avaliadas por validacao cruzada.

Por que nao um teste t comum: as particoes compartilham dados de treino, entao os
resultados por particao nao sao independentes e o teste t comum subestima a variancia
(Nadeau; Bengio, 2003). Usamos:

1. Teste t corrigido para validacao cruzada repetida (Nadeau; Bengio, 2003, na forma
   de Bouckaert; Frank, 2004): diferencas pareadas por (seed, particao), variancia
   multiplicada por (1/(k*r) + n_teste/n_treino), k*r - 1 graus de liberdade.
2. Wilcoxon pareado sobre as medias por particao (nao parametrico). Com poucas
   particoes ele tem um p minimo alto (com 5 particoes, 0,0625 bilateral): nesse caso
   nunca podera indicar significancia, e o relatorio diz isso.
3. Correcao de Holm (1979) para multiplas comparacoes contra o baseline.
"""

from __future__ import annotations

import math

import numpy as np


def t_corrigido(diferencas: list[float], n_treino: float, n_teste: float, k: int, r: int) -> dict:
    d = np.asarray([x for x in diferencas if np.isfinite(x)], float)
    n = len(d)
    out: dict = {"n": n, "media": float(d.mean()) if n else float("nan")}
    if n < 2:
        out.update(
            t=float("nan"),
            gl=0,
            p=float("nan"),
            ic95=(float("nan"), float("nan")),
            obs="menos de 2 diferencas: teste nao aplicavel",
        )
        return out
    from scipy import stats

    var = float(d.var(ddof=1))
    fator = 1.0 / (k * r) + (n_teste / n_treino if n_treino > 0 else 0.0)
    gl = n - 1
    if var == 0:
        p = 1.0 if out["media"] == 0 else 0.0
        return {
            **out,
            "t": float("inf") if out["media"] else 0.0,
            "gl": gl,
            "p": p,
            "ic95": (out["media"], out["media"]),
            "obs": "variancia nula entre particoes",
        }
    ep = math.sqrt(fator * var)
    t = out["media"] / ep
    p = float(2 * stats.t.sf(abs(t), gl))
    q = float(stats.t.ppf(0.975, gl))
    return {**out, "t": float(t), "gl": gl, "p": p, "ic95": (out["media"] - q * ep, out["media"] + q * ep)}


def wilcoxon_pareado(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    p_min = 2.0 / (2**n) if n else float("nan")  # p bilateral minimo possivel (todas as diferencas no mesmo sentido)
    if n < 2 or np.allclose(a, b):
        return {"n": n, "p": 1.0 if n >= 2 else float("nan"), "p_minimo_possivel": p_min}
    from scipy import stats

    try:
        r = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact")
        p = float(r.pvalue)
    except ValueError:
        p = float("nan")
    return {"n": n, "p": p, "p_minimo_possivel": p_min}


def holm(pvalores: list[float]) -> list[float]:
    """Valores-p ajustados por Holm (monotonos, limitados a 1). NaN permanece NaN."""
    idx = [i for i, p in enumerate(pvalores) if np.isfinite(p)]
    m = len(idx)
    ordem = sorted(idx, key=lambda i: pvalores[i])
    ajust = [float("nan")] * len(pvalores)
    corrente = 0.0
    for pos, i in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - pos) * pvalores[i]))
        ajust[i] = corrente
    return ajust
