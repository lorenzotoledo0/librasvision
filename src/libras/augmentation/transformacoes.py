"""Transformacoes de data augmentation sobre UMA janela normalizada e desmontada.

Todas as funcoes sao puras: recebem o dicionario de partes (ver
normalizacao.desmontar) e parametros explicitos, e devolvem um dicionario novo.
O sorteio dos parametros fica separado (sortear_*), o que permite testar cada
transformacao com valores conhecidos e garantir reprodutibilidade por seed.

Convencoes preservadas por todas as transformacoes:
- partes ausentes (flag de presenca 0) continuam com coordenadas 0;
- o canal de visibility da pose nao e alterado;
- o rotulo nao e tocado (as funcoes nem o recebem).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..decisao import FRACAO_MINIMA_POSE_PADRAO
from ..normalizacao import I_FACE, I_MAO_D, I_MAO_E, I_POSE, renormalizar, zerar_ausentes

# pontos da pose ligados a cada mao (pulso, mindinho, indicador, polegar)
POSE_MAO = {"mao_esq": [15, 17, 19, 21], "mao_dir": [16, 18, 20, 22]}
# pares esquerda/direita da pose do MediaPipe (para espelhamento)
PARES_POSE = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]
PARTES = {"pose": I_POSE, "mao_esq": I_MAO_E, "mao_dir": I_MAO_D, "face": I_FACE}


def _copia(d: dict) -> dict:
    return {k: v.copy() for k, v in d.items()}


def _presente(d: dict, nome: str) -> np.ndarray:
    return d["presenca"][:, PARTES[nome]] > 0.5


# ======================================================================
# A. Espaciais
# ======================================================================
@dataclass
class ParamEspacial:
    angulo_rad: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    cisalhamento: float = 0.0
    escala_mao_esq: float = 1.0
    escala_mao_dir: float = 1.0
    desloc_maos: tuple[float, float] = (0.0, 0.0)
    escala_z: float = 1.0
    espelhar: bool = False

    def matriz(self) -> np.ndarray:
        c, s = np.cos(self.angulo_rad), np.sin(self.angulo_rad)
        R = np.array([[c, -s], [s, c]])
        Sh = np.array([[1.0, self.cisalhamento], [0.0, 1.0]])
        S = np.diag([self.sx, self.sy])
        return R @ Sh @ S


def sortear_espacial(rng: np.random.Generator, c: dict, m: float) -> ParamEspacial:
    """Sorteia parametros dentro das faixas do config multiplicadas pela intensidade m."""
    u = lambda v: float(rng.uniform(-v * m, v * m))  # noqa: E731
    return ParamEspacial(
        angulo_rad=np.deg2rad(u(c["rotacao_graus"])),
        sx=1 + u(c["escala_xy"]),
        sy=1 + u(c["escala_xy"]),
        cisalhamento=u(c["cisalhamento"]),
        escala_mao_esq=1 + u(c["escala_maos"]),
        escala_mao_dir=1 + u(c["escala_maos"]),
        desloc_maos=(u(c["deslocamento_maos"]), u(c["deslocamento_maos"])),
        escala_z=1 + u(c["escala_z"]),
        espelhar=bool(c.get("espelhamento", False)) and bool(rng.random() < 0.5),
    )


def espacial(d: dict, p: ParamEspacial, fracao_minima_pose: float = FRACAO_MINIMA_POSE_PADRAO) -> dict:
    """Afim 2D global (rotacao, cisalhamento, escala anisotropica) em torno do centro dos ombros,
    tamanho e posicao das maos, escala de z, espelhamento opcional; depois renormaliza."""
    d = _copia(d)
    A = p.matriz()
    for nome in PARTES:
        if nome not in d:
            continue
        arr, pres = d[nome], _presente(d, nome)
        arr[pres, :, :2] = arr[pres, :, :2] @ A.T
        arr[pres, :, 2] *= p.escala_z
    dx, dy = p.desloc_maos
    for nome, esc in (("mao_esq", p.escala_mao_esq), ("mao_dir", p.escala_mao_dir)):
        if nome not in d:
            continue
        arr, pres = d[nome], _presente(d, nome)
        pulso = arr[pres, :1, :]
        arr[pres] = pulso + (arr[pres] - pulso) * esc
        arr[pres, :, 0] += dx
        arr[pres, :, 1] += dy
        if "pose" in d:  # leva junto os pontos da mao que a pose tambem tem
            pp = _presente(d, "pose")
            d["pose"][np.ix_(pp, POSE_MAO[nome], [0])] += dx
            d["pose"][np.ix_(pp, POSE_MAO[nome], [1])] += dy
    if p.espelhar:
        d = espelhar(d)
    renormalizar(d, fracao_minima_pose)
    zerar_ausentes(d)
    return d


def espelhar(d: dict) -> dict:
    """Espelhamento horizontal com troca esquerda/direita (pose e maos).

    So e linguisticamente valido para sinais simetricos; desabilitado por padrao. A face
    nao tem mapeamento de pares disponivel, por isso o espelhamento exige face desligada.
    """
    if "face" in d:
        from ..erros import ErroConfig

        raise ErroConfig("Espelhamento exige features.face.modo: nenhuma (nao ha mapa de pares da malha facial).")
    d = _copia(d)
    for nome in ("pose", "mao_esq", "mao_dir"):
        if nome in d:
            d[nome][..., 0] *= -1
    if "pose" in d:
        pose = d["pose"].copy()
        for a, b in PARES_POSE:
            d["pose"][:, a], d["pose"][:, b] = pose[:, b], pose[:, a]
    if "mao_esq" in d:
        d["mao_esq"], d["mao_dir"] = d["mao_dir"], d["mao_esq"]
        pres = d["presenca"].copy()
        d["presenca"][:, I_MAO_E], d["presenca"][:, I_MAO_D] = pres[:, I_MAO_D], pres[:, I_MAO_E]
    return d


# ======================================================================
# B. Temporais
# ======================================================================
def _interpolar(d: dict, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> dict:
    """Frame t de saida = mistura dos frames a[t] e b[t] com peso w[t].

    Para cada parte: interpola se presente nos dois; senao copia o mais proximo
    (inclusive a ausencia). Mesma regra de normalizacao.reamostrar.
    """
    out = {}
    perto = np.where(w < 0.5, a, b)
    pres_nova = np.zeros_like(d["presenca"])
    for nome, k in PARTES.items():
        if nome not in d:
            continue
        arr = d[nome]
        ambos = (d["presenca"][a, k] > 0.5) & (d["presenca"][b, k] > 0.5)
        ww = w[:, None, None].astype(np.float32)
        lerp = arr[a] * (1 - ww) + arr[b] * ww
        out[nome] = np.where(ambos[:, None, None], lerp, arr[perto]).astype(np.float32)
        pres_nova[:, k] = np.where(ambos, 1.0, d["presenca"][perto, k])
    out["presenca"] = pres_nova.astype(np.float32)
    zerar_ausentes(out)
    return out


def amostrar_posicoes(d: dict, pos: np.ndarray) -> dict:
    """Reamostra a janela nas posicoes (fracionarias) pos, com bordas repetidas."""
    L = len(d["presenca"])
    pos = np.clip(np.asarray(pos, dtype=np.float64), 0, L - 1)
    a = np.floor(pos).astype(int)
    b = np.minimum(a + 1, L - 1)
    return _interpolar(d, a, b, pos - a)


def time_warp(d: dict, fator: float) -> dict:
    """fator > 1: sinal mais rapido (cobre mais tempo de origem); < 1: mais lento.
    Sempre devolve o mesmo numero de frames."""
    L = len(d["presenca"])
    c = (L - 1) / 2.0
    return amostrar_posicoes(d, c + (np.arange(L) - c) * fator)


def deslocar(d: dict, k: int) -> dict:
    L = len(d["presenca"])
    return amostrar_posicoes(d, np.arange(L) + k)


def remover_frames(d: dict, removidos: np.ndarray) -> dict:
    """Remove frames internos e os reconstroi interpolando entre os vizinhos mantidos (queda de FPS)."""
    L = len(d["presenca"])
    removidos = np.unique(np.asarray(removidos, dtype=int))
    removidos = removidos[(removidos > 0) & (removidos < L - 1)]
    if len(removidos) == 0:
        return _copia(d)
    mantidos = np.setdiff1d(np.arange(L), removidos)
    t = np.arange(L)
    j = np.searchsorted(mantidos, t, side="right") - 1
    a = mantidos[np.clip(j, 0, len(mantidos) - 1)]
    b = mantidos[np.clip(j + 1, 0, len(mantidos) - 1)]
    eh_mantido = np.isin(t, mantidos)
    a = np.where(eh_mantido, t, a)
    b = np.where(eh_mantido, t, b)
    w = np.where(b > a, (t - a) / np.maximum(b - a, 1), 0.0)
    return _interpolar(d, a, b, w)


@dataclass
class ParamTemporal:
    fator_warp: float = 1.0
    deslocamento: int = 0
    removidos: tuple[int, ...] = ()


def sortear_temporal(rng: np.random.Generator, c: dict, m: float, L: int) -> ParamTemporal:
    w = c["warp"] * m
    dmax = int(round(c["deslocamento_max"] * m))
    n_rem = int(round(min(0.5, c["remocao_frames"] * m) * L))
    rem = (
        tuple(sorted(rng.choice(np.arange(1, L - 1), size=min(n_rem, L - 2), replace=False).tolist()))
        if n_rem > 0 and L > 2
        else ()
    )
    return ParamTemporal(float(rng.uniform(1 - w, 1 + w)), int(rng.integers(-dmax, dmax + 1)) if dmax else 0, rem)


def temporal(d: dict, p: ParamTemporal) -> dict:
    d = time_warp(d, p.fator_warp) if p.fator_warp != 1.0 else _copia(d)
    if p.deslocamento:
        d = deslocar(d, p.deslocamento)
    if p.removidos:
        d = remover_frames(d, np.array(p.removidos))
    return d


# ======================================================================
# C. Ruido e falhas de captura
# ======================================================================
@dataclass
class ParamRuido:
    sigma: float = 0.0
    oclusao: tuple[str, int, int] | None = None  # (mao, inicio, fim)
    perda_face: tuple[int, int] | None = None  # (inicio, fim)
    semente_jitter: int = 0


def sortear_ruido(rng: np.random.Generator, c: dict, m: float, L: int, d: dict) -> ParamRuido:
    p = ParamRuido(sigma=float(c["jitter_sigma"] * m), semente_jitter=int(rng.integers(0, 2**31 - 1)))
    nmax = max(1, int(round(c["oclusao_max_frames"] * m)))
    if rng.random() < c["oclusao_mao_prob"]:
        maos = [n for n in ("mao_esq", "mao_dir") if n in d and _presente(d, n).any()]
        if maos:
            n = int(rng.integers(1, min(nmax, L) + 1))
            ini = int(rng.integers(0, L - n + 1))
            p.oclusao = (maos[int(rng.integers(0, len(maos)))], ini, ini + n)
    if "face" in d and rng.random() < c["perda_face_prob"]:
        n = int(rng.integers(1, min(nmax, L) + 1))
        ini = int(rng.integers(0, L - n + 1))
        p.perda_face = (ini, ini + n)
    return p


def ruido(d: dict, p: ParamRuido, fracao_minima_pose: float = FRACAO_MINIMA_POSE_PADRAO) -> dict:
    d = _copia(d)
    if p.sigma > 0:
        g = np.random.default_rng(p.semente_jitter)
        for nome in PARTES:
            if nome in d:
                pres = _presente(d, nome)
                d[nome][pres, :, :3] += g.normal(0, p.sigma, size=d[nome][pres, :, :3].shape).astype(np.float32)
    if p.oclusao is not None:
        nome, i, f = p.oclusao
        d["presenca"][i:f, PARTES[nome]] = 0.0
    if p.perda_face is not None:
        i, f = p.perda_face
        d["presenca"][i:f, I_FACE] = 0.0
    zerar_ausentes(d)
    if p.sigma > 0:
        renormalizar(d, fracao_minima_pose)
    return d
