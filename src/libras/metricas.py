"""Metricas de classificacao. Calculadas SOMENTE sobre dados de teste da particao."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def metricas(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Acuracia, precisao/recall/F1 por classe, macro-F1 e matriz de confusao.

    A matriz e conferida: a soma tem de ser igual ao numero de amostras de teste.
    Classes sem amostra no teste ficam com suporte 0 e nao entram na media macro.
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    if cm.sum() != len(y_true):
        raise AssertionError("Matriz de confusao nao soma o numero de amostras de teste.")
    por_classe = {}
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prev = cm[:, i].sum()
        sup = cm[i, :].sum()
        prec = tp / prev if prev else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        por_classe[c] = {"precisao": float(prec), "recall": float(rec), "f1": float(f1), "suporte": int(sup)}
        if sup:
            f1s.append(f1)
    return {
        "n": int(len(y_true)),
        "acuracia": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "f1_macro": float(np.mean(f1s)) if f1s else float("nan"),
        "por_classe": por_classe,
        "matriz_confusao": cm.tolist(),
    }


def agregar_por_video(
    probs: np.ndarray, videos: np.ndarray, y: np.ndarray
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Media das probabilidades das janelas de cada video -> uma predicao por video."""
    grupos: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(videos):
        grupos[v].append(i)
    ids = sorted(grupos)
    p = np.stack([probs[grupos[v]].mean(axis=0) for v in ids])
    yt = np.array([y[grupos[v][0]] for v in ids])
    return ids, p, yt, p.argmax(axis=1)


def media_dp(valores: list[float]) -> tuple[float, float]:
    v = np.array([x for x in valores if np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
