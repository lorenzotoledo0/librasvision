"""Medidas de enquadramento e de imagem para o app real (13.4-b e 13.9-A), calculadas NO SERVICO de tempo
real a partir do que ele ja tem (frame e landmarks do MediaPipe). A interface so le e formata.

Nada aqui altera a decisao (D15): o Transcritor, o limiar, a media movel e o minimo de maos continuam
como estao, e o que o modelo recebe nao muda. As fracoes minimas de pose e de maos sao as do modelo,
importadas de libras.decisao (nunca copiadas); os limiares novos (borda, brilho, visibilidade do ombro)
vem da configuracao do app real (configs/app_real.yaml) e sao valores iniciais a calibrar.

Coordenadas: normalizadas pela imagem ORIGINAL, nao espelhada (como o MediaPipe as entrega). Quem
desenha na previa espelhada converte (x -> 1 - x).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .decisao import fracao_minima_pose, min_fracao_mao

I_NARIZ_A_BOCA = range(0, 11)  # pose 0 a 10: nariz, olhos, orelhas e boca
I_ORELHAS = (7, 8)
I_OMBROS = (11, 12)
LADOS = ("esquerda", "direita", "topo", "base")  # lados da imagem ORIGINAL (nao espelhada)
TAMANHO_BRILHO = (64, 36)  # copia reduzida para medir a luminancia (custo < 1 ms)


PESOS_CINZA_BGR = np.array([0.114, 0.587, 0.299], np.float32)  # luminancia (ITU-R BT.601), ordem BGR


def brilho_medio(frame_bgr: np.ndarray) -> float:
    """Luminancia media (0-255) de uma copia reduzida em cinza: amostra uma grade de cerca de 64x36
    pontos (salto fixo nas linhas e colunas). Medido: cerca de 0,05 ms por frame 640x480, contra cerca
    de 1 ms do cv2.resize com INTER_AREA nesta maquina."""
    alt, larg = frame_bgr.shape[:2]
    passo_y = max(1, alt // TAMANHO_BRILHO[1])
    passo_x = max(1, larg // TAMANHO_BRILHO[0])
    peq = frame_bgr[::passo_y, ::passo_x]
    return float((peq @ PESOS_CINZA_BGR).mean())


def caixa_cabeca(pose: np.ndarray) -> list[float] | None:
    """Caixa [x0, y0, x1, y1] da cabeca estimada pelos pontos 0 a 10 da pose. Os pontos da pose vao do
    nariz as orelhas; a testa e o cabelo ficam acima deles, entao a caixa sobe 0,7 x a distancia entre as
    orelhas (proporcao aproximada da cabeca; so posiciona a legenda, nao entra no modelo)."""
    pts = pose[list(I_NARIZ_A_BOCA), :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    orelhas = pose[list(I_ORELHAS), :2]
    larg = float(abs(orelhas[0, 0] - orelhas[1, 0])) if np.isfinite(orelhas).all() else float(x1 - x0)
    larg = max(larg, float(x1 - x0), 1e-3)
    return [float(x0 - 0.15 * larg), float(y0 - 0.7 * larg), float(x1 + 0.15 * larg), float(y1 + 0.25 * larg)]


def caixa_ombros(pose: np.ndarray) -> list[float] | None:
    pts = pose[list(I_OMBROS), :2]
    if not np.isfinite(pts).all():
        return None
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def situacao_ombros(pose: np.ndarray, presente: bool, visibilidade_minima: float) -> str:
    """'ok' (os dois visiveis e dentro da imagem), 'parcial' (so um) ou 'fora' (nenhum)."""
    if not presente:
        return "fora"
    n = 0
    for i in I_OMBROS:
        x, y, _z, vis = pose[i]
        if np.isfinite([x, y, vis]).all() and vis >= visibilidade_minima and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            n += 1
    return ("fora", "parcial", "ok")[n]


def lados_mao_cortada(maos: list[np.ndarray], borda: float) -> list[str]:
    """Lados da imagem em que alguma mao DETECTADA chega a menos de `borda` da borda. Mao nao detectada
    (repouso fora da imagem) nao conta: e uso normal."""
    lados = set()
    for m in maos:
        pts = m[:, :2]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if not len(pts):
            continue
        if (pts[:, 0] < borda).any():
            lados.add("esquerda")
        if (pts[:, 0] > 1 - borda).any():
            lados.add("direita")
        if (pts[:, 1] < borda).any():
            lados.add("topo")
        if (pts[:, 1] > 1 - borda).any():
            lados.add("base")
    return [l for l in LADOS if l in lados]


class MedidorEnquadramento:
    """Uma medida por frame processado. As fracoes de pose e de maos usam os ultimos `janela_s` segundos
    e os minimos DO MODELO (os mesmos da decisao)."""

    def __init__(self, params: dict, meta_modelo: dict | None, cfg_tr: dict | None):
        self.borda = float(params["borda_mao"])
        self.brilho_minimo = float(params["brilho_minimo"])
        self.vis_ombro = float(params["visibilidade_ombro"])
        self.janela_s = float(params["janela_s"])
        self.min_pose = fracao_minima_pose(meta_modelo)
        self.min_mao = min_fracao_mao(cfg_tr)
        self._hist: deque = deque()

    def medir(self, frame_bgr: np.ndarray, lm, t: float | None = None) -> dict:
        t0 = time.perf_counter()
        t = t0 if t is None else t
        pose_ok, mao_ok = bool(lm.presenca[0]), bool(lm.presenca[1] or lm.presenca[2])
        self._hist.append((t, pose_ok, mao_ok))
        while self._hist and t - self._hist[0][0] > self.janela_s:
            self._hist.popleft()
        fr_pose = float(np.mean([h[1] for h in self._hist]))
        fr_mao = float(np.mean([h[2] for h in self._hist]))
        brilho = brilho_medio(frame_bgr)
        maos = [m for m, p in ((lm.mao_esq, lm.presenca[1]), (lm.mao_dir, lm.presenca[2])) if p]
        out = {
            "pessoa": fr_pose >= self.min_pose,
            "fracao_pose": round(fr_pose, 3),
            "fracao_maos": round(fr_mao, 3),
            "maos_suficientes": fr_mao >= self.min_mao,
            "ombros": situacao_ombros(lm.pose, pose_ok, self.vis_ombro),
            "mao_cortada": lados_mao_cortada(maos, self.borda),
            "brilho": round(brilho, 1),
            "pouca_luz": brilho < self.brilho_minimo,
            "cabeca": caixa_cabeca(lm.pose) if pose_ok else None,
            "ombros_caixa": caixa_ombros(lm.pose) if pose_ok else None,
        }
        out["ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out
