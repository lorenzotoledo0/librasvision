"""Data augmentation de sequencias de landmarks (Fase 4).

Adaptacao, para landmarks normalizados pelo corpo, da ideia de Islam, Hossain et al.
(2019), que aplicaram rotacao, zoom e deslocamento a imagens para uma CNN. Ver
docs/decisoes.md (D8) para a correspondencia entre as transformacoes de imagem e as
de landmarks, e por que zoom e deslocamento globais nao tem efeito aqui.

Uso no treino (online, so nas janelas de treino):

    aum = Aumentador(cfg, espec)          # ou Aumentador(cfg, espec, familias=["espacial"])
    ds = criar_tf_dataset(conj, idx_treino, ..., transformar=aum)

Custo zero na inferencia: nada deste pacote e chamado no tempo real.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..erros import ErroConfig
from ..normalizacao import EspecFeatures, desmontar, montar
from .transformacoes import espacial, ruido, sortear_espacial, sortear_ruido, sortear_temporal, temporal

FAMILIAS_ONLINE = ("espacial", "temporal", "ruido")
INTENSIDADES = {"fraca": 0.5, "media": 1.0, "forte": 1.5}


class Aumentador:
    """Aplica as familias ativas a um lote (B, L, D) de janelas normalizadas.

    Chamavel como transformar(x, rng) pelo gerador do dataset. Conta quantas janelas
    receberam cada familia, para registrar o efeito real da augmentation na execucao.
    """

    def __init__(self, cfg, espec: EspecFeatures, familias: list[str] | None = None, intensidade: str | None = None):
        ca = cfg["augmentation"]
        self.cfg_aug = ca
        self.espec = espec
        self.fracao_pose = float(cfg["normalizacao"]["fracao_minima_pose"])
        intensidade = intensidade or ca.get("intensidade", "media")
        if intensidade not in INTENSIDADES:
            raise ErroConfig(f"augmentation.intensidade deve ser uma de {list(INTENSIDADES)}.")
        self.intensidade = intensidade
        self.m = INTENSIDADES[intensidade]
        if familias is None:
            familias = [f for f in FAMILIAS_ONLINE if ca[f].get("ativo", False)]
        desconhecidas = [f for f in familias if f not in FAMILIAS_ONLINE]
        if desconhecidas:
            raise ErroConfig(
                f"Familias desconhecidas: {desconhecidas}. Opcoes: {list(FAMILIAS_ONLINE)} "
                "(a familia 'video' e offline: scripts/aumentar_videos.py)."
            )
        if "espacial" in familias and ca["espacial"].get("espelhamento") and len(espec.face_idx):
            raise ErroConfig("augmentation.espacial.espelhamento exige features.face.modo: nenhuma.")
        self.familias = list(familias)
        self.contagem: Counter = Counter()

    def descricao(self) -> dict:
        return {
            "familias": self.familias,
            "intensidade": self.intensidade,
            "multiplicador": self.m,
            "parametros": {f: self.cfg_aug[f] for f in self.familias},
        }

    def janela(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Uma janela (L, D) -> janela aumentada. Ordem: temporal, espacial, ruido."""
        if not self.familias:
            return X
        d = desmontar(X, self.espec)
        L = len(X)
        aplicou = False
        if "temporal" in self.familias and rng.random() < self.cfg_aug["temporal"]["prob"]:
            d = temporal(d, sortear_temporal(rng, self.cfg_aug["temporal"], self.m, L))
            self.contagem["temporal"] += 1
            aplicou = True
        if "espacial" in self.familias and rng.random() < self.cfg_aug["espacial"]["prob"]:
            d = espacial(d, sortear_espacial(rng, self.cfg_aug["espacial"], self.m), self.fracao_pose)
            self.contagem["espacial"] += 1
            aplicou = True
        if "ruido" in self.familias and rng.random() < self.cfg_aug["ruido"]["prob"]:
            d = ruido(d, sortear_ruido(rng, self.cfg_aug["ruido"], self.m, L, d), self.fracao_pose)
            self.contagem["ruido"] += 1
            aplicou = True
        self.contagem["janelas"] += 1
        if aplicou:
            self.contagem["alteradas"] += 1
        return montar(d, self.espec) if aplicou else X

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.familias:
            return x
        return np.stack([self.janela(xi, rng) for xi in x]).astype(np.float32)

    def resumo_contagem(self) -> dict:
        n = max(1, self.contagem["janelas"])
        return {
            "janelas_vistas": self.contagem["janelas"],
            "fracao_alteradas": round(self.contagem["alteradas"] / n, 4),
            **{f"fracao_{f}": round(self.contagem[f] / n, 4) for f in self.familias},
        }


def gerar_copias_offline(X: np.ndarray, aum: Aumentador, copias: int, seed: int) -> np.ndarray:
    """Modo offline: copias fixas e reprodutiveis das janelas de treino (fator = 1 + copias)."""
    rng = np.random.default_rng(seed)
    out = [
        aum(X[i : i + 256].astype(np.float32), rng).astype(np.float16)
        for _ in range(copias)
        for i in range(0, len(X), 256)
    ]
    return np.concatenate(out) if out else np.zeros((0,) + X.shape[1:], np.float16)
