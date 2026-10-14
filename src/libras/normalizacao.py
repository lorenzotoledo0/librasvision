"""Normalizacao e montagem do vetor de features. MODULO UNICO.

O mesmo codigo transforma landmarks em features no treino, na avaliacao e no tempo
real. Qualquer mudanca aqui vale para os tres, o que evita divergencia entre o que
o modelo viu no treino e o que ele recebe ao vivo.

Fluxo (identico para um video do dataset e para o buffer da webcam):

    landmarks crus + instantes (s)
      -> reamostrar(...)         grade uniforme de fps_alvo (30 FPS)
      -> preparar_janela(...)    para cada janela de `comprimento` frames:
           1. coordenadas em pixels (corrige a proporcao da imagem)
           2. interpola lacunas internas curtas sem deteccao
           3. origem = mediana do centro dos ombros na janela
              escala = mediana da largura dos ombros na janela
           4. partes ausentes viram 0; presenca vai junto como 4 flags
           5. seleciona as features do config (pose, maos, subconjunto da face)

Nada aqui espelha a imagem ou troca esquerda/direita.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decisao import FRACAO_MINIMA_POSE_PADRAO
from .landmarks import N_FACE, N_MAO, N_POSE

OMBRO_E, OMBRO_D = 11, 12
I_POSE, I_MAO_E, I_MAO_D, I_FACE = range(4)

# Indices da malha facial (468 pontos) por regiao, derivados das conexoes oficiais
# do MediaPipe 0.10.21 (face_mesh_connections). Copiados aqui para nao depender do
# MediaPipe no treino; o teste test_indices_face_iguais_ao_mediapipe confere a copia.
FACE_REGIOES: dict[str, list[int]] = {
    "labios": [
        0,
        13,
        14,
        17,
        37,
        39,
        40,
        61,
        78,
        80,
        81,
        82,
        84,
        87,
        88,
        91,
        95,
        146,
        178,
        181,
        185,
        191,
        267,
        269,
        270,
        291,
        308,
        310,
        311,
        312,
        314,
        317,
        318,
        321,
        324,
        375,
        402,
        405,
        409,
        415,
    ],
    "sobrancelhas": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
    "contorno": [
        10,
        21,
        54,
        58,
        67,
        93,
        103,
        109,
        127,
        132,
        136,
        148,
        149,
        150,
        152,
        162,
        172,
        176,
        234,
        251,
        284,
        288,
        297,
        323,
        332,
        338,
        356,
        361,
        365,
        377,
        378,
        379,
        389,
        397,
        400,
        454,
    ],
    "olhos": [
        7,
        33,
        133,
        144,
        145,
        153,
        154,
        155,
        157,
        158,
        159,
        160,
        161,
        163,
        173,
        246,
        249,
        263,
        362,
        373,
        374,
        380,
        381,
        382,
        384,
        385,
        386,
        387,
        388,
        390,
        398,
        466,
    ],
    "nariz": [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440],
}


# ======================================================================
# Estrutura de dados crua (independente do MediaPipe)
# ======================================================================
@dataclass
class Partes:
    """Landmarks crus de T frames, no formato do MediaPipe (x, y normalizados pela imagem).

    Ausencias sao NaN nas coordenadas e False em `presenca`.
    """

    pose: np.ndarray  # (T, 33, 4)  x, y, z, visibility
    mao_esq: np.ndarray  # (T, 21, 3)
    mao_dir: np.ndarray  # (T, 21, 3)
    face: np.ndarray  # (T, 468, 3)
    presenca: np.ndarray  # (T, 4) bool

    @property
    def n(self) -> int:
        return int(self.presenca.shape[0])

    def fatia(self, ini: int, fim: int) -> "Partes":
        return Partes(
            self.pose[ini:fim], self.mao_esq[ini:fim], self.mao_dir[ini:fim], self.face[ini:fim], self.presenca[ini:fim]
        )

    def copia(self) -> "Partes":
        return Partes(
            self.pose.copy(), self.mao_esq.copy(), self.mao_dir.copy(), self.face.copy(), self.presenca.copy()
        )

    def lista(self) -> list[np.ndarray]:
        return [self.pose, self.mao_esq, self.mao_dir, self.face]

    @staticmethod
    def de_lista(arrs: list[np.ndarray], presenca: np.ndarray) -> "Partes":
        return Partes(arrs[0], arrs[1], arrs[2], arrs[3], presenca)

    @staticmethod
    def de_sequencia(seq) -> "Partes":
        """Converte libras.landmarks.SequenciaLandmarks (ou npz carregado) em Partes."""
        return Partes(
            np.asarray(seq.pose, np.float32),
            np.asarray(seq.mao_esq, np.float32),
            np.asarray(seq.mao_dir, np.float32),
            np.asarray(seq.face, np.float32),
            np.asarray(seq.presenca, bool),
        )

    @staticmethod
    def vazias(t: int) -> "Partes":
        nan = np.nan
        return Partes(
            np.full((t, N_POSE, 4), nan, np.float32),
            np.full((t, N_MAO, 3), nan, np.float32),
            np.full((t, N_MAO, 3), nan, np.float32),
            np.full((t, N_FACE, 3), nan, np.float32),
            np.zeros((t, 4), bool),
        )


# ======================================================================
# Especificacao do vetor de features
# ======================================================================
@dataclass
class EspecFeatures:
    usar_pose: bool
    usar_visibility: bool
    usar_maos: bool
    face_idx: np.ndarray
    # deslocamentos (inicio, fim) de cada bloco dentro do vetor
    blocos: dict[str, tuple[int, int]] = field(default_factory=dict)
    dim: int = 0

    def __post_init__(self):
        o = 0
        b = {}
        if self.usar_pose:
            n = N_POSE * (4 if self.usar_visibility else 3)
            b["pose"] = (o, o + n)
            o += n
        if self.usar_maos:
            b["mao_esq"] = (o, o + N_MAO * 3)
            o += N_MAO * 3
            b["mao_dir"] = (o, o + N_MAO * 3)
            o += N_MAO * 3
        if len(self.face_idx):
            b["face"] = (o, o + len(self.face_idx) * 3)
            o += len(self.face_idx) * 3
        b["presenca"] = (o, o + 4)
        o += 4
        self.blocos, self.dim = b, o

    @property
    def dim_coordenadas(self) -> int:
        return self.dim - 4

    def descricao(self) -> str:
        partes = [f"{k}={f - i}" for k, (i, f) in self.blocos.items()]
        return f"{self.dim} dimensoes por frame ({', '.join(partes)})"

    def como_dict(self) -> dict:
        return {
            "usar_pose": self.usar_pose,
            "usar_visibility": self.usar_visibility,
            "usar_maos": self.usar_maos,
            "face_idx": [int(i) for i in self.face_idx],
            "blocos": {k: list(v) for k, v in self.blocos.items()},
            "dim": self.dim,
        }

    @staticmethod
    def de_dict(d: dict) -> "EspecFeatures":
        return EspecFeatures(d["usar_pose"], d["usar_visibility"], d["usar_maos"], np.array(d["face_idx"], dtype=int))


def espec_de_config(cfg) -> EspecFeatures:
    f = cfg["features"]
    modo = f["face"]["modo"]
    if modo == "nenhuma":
        idx: list[int] = []
    elif modo == "completa":
        idx = list(range(N_FACE))
    else:
        idx_set: set[int] = set()
        for r in f["face"]["subconjunto"]:
            if r not in FACE_REGIOES:
                from .erros import ErroConfig

                raise ErroConfig(f"Regiao de face desconhecida: {r!r}. Opcoes: {sorted(FACE_REGIOES)}")
            idx_set.update(FACE_REGIOES[r])
        idx = sorted(idx_set)
    return EspecFeatures(
        bool(f["pose"]["usar"]),
        bool(f["pose"].get("usar_visibility", True)),
        bool(f["maos"]["usar"]),
        np.array(idx, dtype=int),
    )


# ======================================================================
# Reamostragem temporal
# ======================================================================
def tempos_uniformes(n: int, fps: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / float(fps)


def reamostrar(p: Partes, tempos: np.ndarray, fps_alvo: float, t_fim: float | None = None) -> Partes:
    """Leva a sequencia para uma grade uniforme de fps_alvo, ancorada no ULTIMO instante.

    Ancorar no fim garante que, no tempo real, o frame mais recente sempre entre na
    janela. Quando os dois frames vizinhos tem a parte detectada, interpola
    linearmente; senao usa o frame mais proximo (inclusive a ausencia).
    """
    tempos = np.asarray(tempos, dtype=np.float64)
    if p.n == 0:
        return p.copia()
    if p.n == 1:
        return p.copia()
    t_fim = tempos[-1] if t_fim is None else t_fim
    dt = 1.0 / float(fps_alvo)
    n_alvo = int(np.floor((t_fim - tempos[0]) / dt + 1e-6)) + 1
    alvo = t_fim - dt * np.arange(n_alvo)[::-1]
    b = np.searchsorted(tempos, alvo, side="left").clip(1, p.n - 1)
    a = b - 1
    denom = np.maximum(tempos[b] - tempos[a], 1e-9)
    w = ((alvo - tempos[a]) / denom).clip(0.0, 1.0)
    perto = np.where(w < 0.5, a, b)
    novas, pres = [], np.zeros((n_alvo, 4), bool)
    for k, arr in enumerate(p.lista()):
        ambos = p.presenca[a, k] & p.presenca[b, k]
        ww = w[:, None, None].astype(np.float32)
        lerp = arr[a] * (1 - ww) + arr[b] * ww
        out = np.where(ambos[:, None, None], lerp, arr[perto]).astype(np.float32)
        pres[:, k] = np.where(ambos, True, p.presenca[perto, k])
        novas.append(out)
    return Partes.de_lista(novas, pres)


# ======================================================================
# Lacunas
# ======================================================================
def _trechos_ausentes(pres: np.ndarray) -> list[tuple[int, int]]:
    """Trechos [ini, fim) com presenca False."""
    out, ini = [], None
    for i, v in enumerate(pres):
        if not v and ini is None:
            ini = i
        elif v and ini is not None:
            out.append((ini, i))
            ini = None
    if ini is not None:
        out.append((ini, len(pres)))
    return out


def interpolar_lacunas(p: Partes, max_frames: int) -> Partes:
    """Interpola linearmente lacunas INTERNAS de ate max_frames (bordas nao sao preenchidas)."""
    q = p.copia()
    if max_frames <= 0:
        return q
    arrs = q.lista()
    for k, arr in enumerate(arrs):
        pres = q.presenca[:, k]
        for ini, fim in _trechos_ausentes(pres):
            if ini == 0 or fim == len(pres) or (fim - ini) > max_frames:
                continue
            a, b = arr[ini - 1], arr[fim]
            n = fim - ini + 1
            for j, t in enumerate(range(ini, fim), start=1):
                arr[t] = a + (b - a) * (j / n)
            pres[ini:fim] = True
    return q


# ======================================================================
# Janela -> features
# ======================================================================
def referencia_ombros(ombro_e: np.ndarray, ombro_d: np.ndarray) -> tuple[np.ndarray, float]:
    """Centro (mediana do ponto medio dos ombros) e escala (mediana da largura dos ombros).

    Unica definicao da referencia corporal, usada na normalizacao e na renormalizacao
    apos data augmentation.
    """
    centro = np.median((ombro_e + ombro_d) / 2.0, axis=0)
    escala = float(np.median(np.linalg.norm(ombro_e - ombro_d, axis=1)))
    return centro, escala


@dataclass
class ResultadoJanela:
    X: np.ndarray | None  # (T, dim) float32, ou None se invalida
    valida: bool
    motivo: str = ""
    centro: tuple[float, float] = (0.0, 0.0)
    escala: float = 0.0


def preparar_janela(p: Partes, largura: int, altura: int, espec: EspecFeatures, cfg_norm: dict) -> ResultadoJanela:
    """Transforma T frames crus em uma matriz (T, dim) normalizada. Ver docstring do modulo."""
    if p.n == 0:
        return ResultadoJanela(None, False, "janela vazia")
    q = p.copia()
    # 1. pixels: x*largura, y*altura, z*largura (z do MediaPipe usa aproximadamente a escala de x)
    W, H = float(largura), float(altura)
    for arr in q.lista():
        arr[..., 0] *= W
        arr[..., 1] *= H
        arr[..., 2] *= W
    # 2. lacunas curtas
    q = interpolar_lacunas(q, int(cfg_norm.get("interpolar_lacunas_max_frames", 0)))
    # 3. referencia nos ombros
    tem_pose = q.presenca[:, I_POSE]
    if tem_pose.mean() < float(cfg_norm.get("fracao_minima_pose", FRACAO_MINIMA_POSE_PADRAO)) or not tem_pose.any():
        return ResultadoJanela(None, False, "pose ausente na maior parte da janela")
    centro, escala = referencia_ombros(q.pose[tem_pose, OMBRO_E, :2], q.pose[tem_pose, OMBRO_D, :2])
    if not np.isfinite(escala) or escala < 1e-3 * max(W, H):
        return ResultadoJanela(None, False, "largura dos ombros invalida")
    for arr in q.lista():
        arr[..., 0] = (arr[..., 0] - centro[0]) / escala
        arr[..., 1] = (arr[..., 1] - centro[1]) / escala
        arr[..., 2] = arr[..., 2] / escala
    # 4. ausentes -> 0
    for k, arr in enumerate(q.lista()):
        arr[~q.presenca[:, k]] = 0.0
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # 5. selecao
    T = q.n
    blocos = []
    if espec.usar_pose:
        pose = q.pose if espec.usar_visibility else q.pose[..., :3]
        blocos.append(pose.reshape(T, -1))
    if espec.usar_maos:
        blocos.append(q.mao_esq.reshape(T, -1))
        blocos.append(q.mao_dir.reshape(T, -1))
    if len(espec.face_idx):
        blocos.append(q.face[:, espec.face_idx, :].reshape(T, -1))
    blocos.append(q.presenca.astype(np.float32))
    X = np.concatenate(blocos, axis=1).astype(np.float32)
    return ResultadoJanela(X, True, "", (float(centro[0]), float(centro[1])), escala)


def energia_maos(X: np.ndarray, espec: EspecFeatures) -> float:
    """Movimento medio das maos na janela (unidades de largura de ombro por frame)."""
    if "mao_esq" not in espec.blocos:
        return 0.0
    pi = espec.blocos["presenca"][0]
    total, cont = 0.0, 0
    for nome, k in (("mao_esq", I_MAO_E), ("mao_dir", I_MAO_D)):
        i, f = espec.blocos[nome]
        m = X[:, i:f].reshape(len(X), N_MAO, 3)[..., :2]
        pres = X[:, pi + k] > 0.5
        ok = pres[1:] & pres[:-1]
        if ok.any():
            d = np.linalg.norm(m[1:] - m[:-1], axis=2).mean(axis=1)
            total += float(d[ok].sum())
            cont += int(ok.sum())
    return total / cont if cont else 0.0


def movimento_maos_por_frame(p: Partes, largura: int, altura: int, suavizar: int = 5) -> np.ndarray:
    """Curva de movimento das maos por frame (editor de corte, item 9).

    Mesma medida de energia_maos (deslocamento medio dos 21 pontos da mao entre frames, em xy, em
    unidades de largura de ombro), mas por frame e com a referencia de ombros do take inteiro.
    Por frame vale a mao que mais se moveu; frame sem a mao nos dois instantes vale 0. Media movel
    curta para a curva ficar legivel. So para exibicao: nao entra no treino."""
    n = p.n
    if n < 2:
        return np.zeros(n, np.float32)
    esc = np.array([largura, altura], np.float64)
    tem_pose = p.presenca[:, I_POSE]
    if tem_pose.any():
        _, escala = referencia_ombros(p.pose[tem_pose, OMBRO_E, :2] * esc, p.pose[tem_pose, OMBRO_D, :2] * esc)
    else:
        escala = 0.0
    escala = escala if np.isfinite(escala) and escala > 1e-6 else float(max(largura, altura)) / 4.0
    curva = np.zeros(n, np.float64)
    for mao, k in ((p.mao_esq, I_MAO_E), (p.mao_dir, I_MAO_D)):
        xy = mao[..., :2] * esc
        pres = p.presenca[:, k]
        ok = pres[1:] & pres[:-1]
        d = np.linalg.norm(xy[1:] - xy[:-1], axis=2).mean(axis=1) / escala
        d = np.where(ok, d, 0.0)
        curva[1:] = np.maximum(curva[1:], np.nan_to_num(d))
    if suavizar > 1:
        curva = np.convolve(curva, np.ones(suavizar) / suavizar, mode="same")
    return curva.astype(np.float32)


# ======================================================================
# Video inteiro -> janelas
# ======================================================================
@dataclass
class JanelasVideo:
    X: np.ndarray  # (n, L, dim) float32
    inicios: np.ndarray  # inicio de cada janela (em frames na grade de fps_alvo)
    energias: np.ndarray
    n_frames_reamostrado: int
    descartadas: dict = field(default_factory=dict)  # motivo -> quantidade
    avisos: list[str] = field(default_factory=list)


def inicios_janelas(n: int, comprimento: int, passo: int) -> list[int]:
    if n <= comprimento:
        return [0]
    ini = list(range(0, n - comprimento + 1, passo))
    if ini[-1] != n - comprimento:
        ini.append(n - comprimento)
    return ini


def _preencher_bordas(p: Partes, comprimento: int) -> Partes:
    """Video curto: repete o primeiro e o ultimo frame (pessoa parada) ate completar a janela."""
    falta = comprimento - p.n
    antes, depois = falta // 2, falta - falta // 2
    arrs = [np.concatenate([np.repeat(a[:1], antes, 0), a, np.repeat(a[-1:], depois, 0)]) for a in p.lista()]
    pres = np.concatenate([np.repeat(p.presenca[:1], antes, 0), p.presenca, np.repeat(p.presenca[-1:], depois, 0)])
    return Partes.de_lista(arrs, pres)


def janelas_do_video(
    p: Partes,
    tempos: np.ndarray,
    largura: int,
    altura: int,
    espec: EspecFeatures,
    cfg,
    eh_nenhum_sinal: bool,
    passo: int | None = None,
) -> JanelasVideo:
    """Gera as janelas normalizadas de um video, usando exatamente o fluxo do tempo real."""
    cj, cn = cfg["janela"], cfg["normalizacao"]
    L = int(cj["comprimento"])
    passo = int(passo or cj["passo"])
    r = reamostrar(p, tempos, float(cj["fps_alvo"]))
    n_reamostrado = r.n
    avisos = []
    if r.n < L:
        avisos.append(f"video_curto({r.n}<{L}_frames)")
        r = _preencher_bordas(r, L)
    Xs: list[np.ndarray] = []
    inis: list[int] = []
    energias: list[float] = []
    descartes: dict[str, int] = {}
    for ini in inicios_janelas(r.n, L, passo):
        res = preparar_janela(r.fatia(ini, ini + L), largura, altura, espec, cn)
        if not res.valida:
            descartes[res.motivo] = descartes.get(res.motivo, 0) + 1
            continue
        assert res.X is not None  # janela valida sempre tem X
        Xs.append(res.X)
        inis.append(ini)
        energias.append(energia_maos(res.X, espec))
    if not Xs:
        return JanelasVideo(
            np.zeros((0, L, espec.dim), np.float32),
            np.zeros(0, int),
            np.zeros(0),
            n_reamostrado,
            descartes,
            avisos + ["nenhuma_janela_valida"],
        )
    X, ini_ok, en = np.stack(Xs), np.array(inis), np.array(energias)
    if cj.get("selecao", "todas") == "atividade" and not eh_nenhum_sinal and len(X) > 1:
        mx = en.max()
        if mx <= 0:
            avisos.append("sem_movimento_de_mao")
        else:
            manter = en >= float(cj.get("limiar_atividade", 0.5)) * mx
            n_fora = int((~manter).sum())
            if n_fora:
                descartes["pouco movimento (selecao por atividade)"] = n_fora
            X, ini_ok, en = X[manter], ini_ok[manter], en[manter]
    return JanelasVideo(X, ini_ok, en, n_reamostrado, descartes, avisos)


# ======================================================================
# Acesso estruturado a uma janela ja normalizada (usado pela augmentation)
# ======================================================================
def desmontar(X: np.ndarray, espec: EspecFeatures) -> dict[str, np.ndarray]:
    """(L, dim) -> dict com copias: pose (L,33,4|3), mao_esq/mao_dir (L,21,3), face (L,F,3), presenca (L,4)."""
    L = len(X)
    out: dict[str, np.ndarray] = {}
    for nome, (i, f) in espec.blocos.items():
        bloco = np.array(X[:, i:f], dtype=np.float32, copy=True)
        if nome == "pose":
            out[nome] = bloco.reshape(L, N_POSE, 4 if espec.usar_visibility else 3)
        elif nome in ("mao_esq", "mao_dir"):
            out[nome] = bloco.reshape(L, N_MAO, 3)
        elif nome == "face":
            out[nome] = bloco.reshape(L, len(espec.face_idx), 3)
        else:
            out[nome] = bloco
    return out


def montar(d: dict[str, np.ndarray], espec: EspecFeatures) -> np.ndarray:
    L = len(d["presenca"])
    return np.concatenate([d[n].reshape(L, -1) for n in espec.blocos], axis=1).astype(np.float32)


_PARTE_IDX = {"pose": I_POSE, "mao_esq": I_MAO_E, "mao_dir": I_MAO_D, "face": I_FACE}


def zerar_ausentes(d: dict[str, np.ndarray]) -> None:
    """Garante a convencao: parte ausente => coordenadas 0 (em todos os frames)."""
    for nome, k in _PARTE_IDX.items():
        if nome in d:
            d[nome][d["presenca"][:, k] < 0.5] = 0.0


def renormalizar(d: dict[str, np.ndarray], fracao_minima_pose: float = FRACAO_MINIMA_POSE_PADRAO) -> bool:
    """Reaplica a referencia dos ombros (mesma regra de preparar_janela) a uma janela desmontada.

    Retorna False se nao houver pose suficiente (a janela fica como estava).
    """
    if "pose" not in d:
        return True
    tem = d["presenca"][:, I_POSE] > 0.5
    if not tem.any() or tem.mean() < fracao_minima_pose:
        return False
    centro, escala = referencia_ombros(d["pose"][tem, OMBRO_E, :2], d["pose"][tem, OMBRO_D, :2])
    if not np.isfinite(escala) or escala < 1e-6:
        return False
    for nome, k in _PARTE_IDX.items():
        if nome not in d:
            continue
        pres = d["presenca"][:, k] > 0.5
        arr = d[nome]
        arr[pres, :, 0] = (arr[pres, :, 0] - centro[0]) / escala
        arr[pres, :, 1] = (arr[pres, :, 1] - centro[1]) / escala
        arr[pres, :, 2] = arr[pres, :, 2] / escala
    zerar_ausentes(d)
    return True
