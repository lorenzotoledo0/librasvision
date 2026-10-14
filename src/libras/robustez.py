"""Teste de robustez: avalia o modelo em copias DEGRADADAS do conjunto de teste.

Dois tipos:
- degradacoes de landmarks (oclusao de mao, ruido, queda de FPS, perda da face),
  aplicadas de forma deterministica a cada janela de teste;
- degradacoes de video (ex.: escuro), em que os frames dos videos de teste foram
  degradados e os landmarks reextraidos (scripts/aumentar_videos.py).

O modelo nunca e treinado nessas copias de teste.
"""

from __future__ import annotations

import numpy as np

from .augmentation.transformacoes import ParamRuido, remover_frames, ruido
from .decisao import FRACAO_MINIMA_POSE_PADRAO
from .normalizacao import EspecFeatures, desmontar, montar


def degradar_janela(
    X: np.ndarray,
    espec: EspecFeatures,
    nome: str,
    params: dict,
    rng: np.random.Generator,
    fracao_pose: float = FRACAO_MINIMA_POSE_PADRAO,
) -> np.ndarray:
    d = desmontar(X, espec)
    L = len(X)
    if nome == "oclusao_mao":
        maos = [n for n, k in (("mao_esq", 1), ("mao_dir", 2)) if n in d and (d["presenca"][:, k] > 0.5).any()]
        if not maos:
            return X
        n = min(int(params.get("frames", 10)), L)
        ini = int(rng.integers(0, L - n + 1))
        d = ruido(d, ParamRuido(oclusao=(maos[int(rng.integers(0, len(maos)))], ini, ini + n)), fracao_pose)
    elif nome == "ruido":
        d = ruido(
            d,
            ParamRuido(sigma=float(params.get("sigma", 0.03)), semente_jitter=int(rng.integers(0, 2**31))),
            fracao_pose,
        )
    elif nome == "queda_fps":
        n = int(round(float(params.get("fracao", 0.5)) * (L - 2)))
        rem = rng.choice(np.arange(1, L - 1), size=n, replace=False) if n > 0 else np.array([], int)
        d = remover_frames(d, rem)
    elif nome == "sem_face":
        if "face" not in d:
            return X
        d = ruido(d, ParamRuido(perda_face=(0, L)), fracao_pose)
    else:
        from .erros import ErroConfig

        raise ErroConfig(f"Degradacao de landmarks desconhecida: {nome}")
    return montar(d, espec)


def degradar_lote(
    X: np.ndarray,
    espec: EspecFeatures,
    nome: str,
    params: dict,
    seed: int,
    fracao_pose: float = FRACAO_MINIMA_POSE_PADRAO,
) -> np.ndarray:
    """Deterministico: a janela i usa seed + i, entao todas as condicoes veem o mesmo teste degradado."""
    return np.stack(
        [
            degradar_janela(X[i].astype(np.float32), espec, nome, params, np.random.default_rng(seed + i), fracao_pose)
            for i in range(len(X))
        ]
    )
