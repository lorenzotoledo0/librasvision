"""Extracao de landmarks com MediaPipe Holistic. MODULO UNICO usado por coleta, treino e tempo real.

Regra de lateralidade: o frame entregue ao MediaPipe NUNCA e espelhado, em nenhuma
etapa. O espelhamento existe apenas na exibicao para o sinalizador. Assim os rotulos
mao_esq/mao_dir sao gerados da mesma forma na coleta, no treino e na inferencia.

Coordenadas sao guardadas "cruas" (normalizadas pela imagem, como o MediaPipe as
entrega) junto com largura/altura do video. A normalizacao corporal (centro dos
ombros, escala pela largura dos ombros, correcao de proporcao da imagem) fica em
libras.normalizacao e e aplicada depois, tambem por um unico modulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

N_POSE = 33
N_MAO = 21
N_FACE = 468
PARTES = ("pose", "mao_esq", "mao_dir", "face")


def _mp():
    """Importa o MediaPipe sob demanda com mensagem clara se faltar a API legada."""
    try:
        import mediapipe as mp

        return mp, mp.solutions.holistic
    except AttributeError as e:  # mediapipe >= 0.10.30 removeu mp.solutions
        from .erros import ErroConfig

        raise ErroConfig(
            "Esta versao do MediaPipe nao tem mais a API Holistic legada (mp.solutions). "
            "Instale a versao fixada: pip install -r requirements.txt"
        ) from e


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------
@dataclass
class LandmarksFrame:
    pose: np.ndarray  # (33, 4) x, y, z, visibility  | NaN se ausente
    mao_esq: np.ndarray  # (21, 3)
    mao_dir: np.ndarray  # (21, 3)
    face: np.ndarray  # (468, 3)
    presenca: np.ndarray  # (4,) bool na ordem de PARTES

    @property
    def tem_mao(self) -> bool:
        return bool(self.presenca[1] or self.presenca[2])


@dataclass
class SequenciaLandmarks:
    pose: np.ndarray  # (T, 33, 4)
    mao_esq: np.ndarray  # (T, 21, 3)
    mao_dir: np.ndarray  # (T, 21, 3)
    face: np.ndarray  # (T, 468, 3)
    presenca: np.ndarray  # (T, 4) bool
    largura: int
    altura: int
    fps: float

    @property
    def n_frames(self) -> int:
        return int(self.presenca.shape[0])

    def taxa(self, parte: str) -> float:
        if self.n_frames == 0:
            return 0.0
        return float(self.presenca[:, PARTES.index(parte)].mean())

    @property
    def taxa_alguma_mao(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return float((self.presenca[:, 1] | self.presenca[:, 2]).mean())


def _vazio(n: int, d: int) -> np.ndarray:
    return np.full((n, d), np.nan, dtype=np.float32)


def _converter(lista, n: int, com_vis: bool) -> tuple[np.ndarray, bool]:
    if lista is None:
        return _vazio(n, 4 if com_vis else 3), False
    if com_vis:
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lista.landmark], dtype=np.float32)
    else:
        arr = np.array([[p.x, p.y, p.z] for p in lista.landmark], dtype=np.float32)
    return arr, True


def resultado_para_frame(res) -> LandmarksFrame:
    """Converte o resultado do Holistic para arrays numpy de forma fixa."""
    pose, p0 = _converter(res.pose_landmarks, N_POSE, True)
    me, p1 = _converter(res.left_hand_landmarks, N_MAO, False)
    md, p2 = _converter(res.right_hand_landmarks, N_MAO, False)
    face, p3 = _converter(res.face_landmarks, N_FACE, False)
    if face.shape[0] != N_FACE:  # refine_face_landmarks=True devolve 478; cortamos para manter a forma
        face = face[:N_FACE]
    return LandmarksFrame(pose, me, md, face, np.array([p0, p1, p2, p3], dtype=bool))


def empilhar(frames: list[LandmarksFrame], largura: int, altura: int, fps: float) -> SequenciaLandmarks:
    if not frames:
        return SequenciaLandmarks(
            np.zeros((0, N_POSE, 4), np.float32),
            np.zeros((0, N_MAO, 3), np.float32),
            np.zeros((0, N_MAO, 3), np.float32),
            np.zeros((0, N_FACE, 3), np.float32),
            np.zeros((0, 4), bool),
            largura,
            altura,
            fps,
        )
    return SequenciaLandmarks(
        np.stack([f.pose for f in frames]),
        np.stack([f.mao_esq for f in frames]),
        np.stack([f.mao_dir for f in frames]),
        np.stack([f.face for f in frames]),
        np.stack([f.presenca for f in frames]),
        largura,
        altura,
        fps,
    )


# --------------------------------------------------------------------------
# Extrator
# --------------------------------------------------------------------------
class ExtratorHolistic:
    """Envolve mp.solutions.holistic.Holistic com os parametros do config.

    modo_estatico=True trata cada frame de forma independente (usado para amostrar
    frames soltos na validacao de importacao). Para video continuo use False.
    """

    def __init__(self, cfg_mediapipe: dict, modo_estatico: bool = False):
        self._mp, holistic = _mp()
        self._holistic_mod = holistic
        self.parametros = {
            "static_image_mode": bool(modo_estatico),
            "model_complexity": int(cfg_mediapipe.get("model_complexity", 1)),
            "min_detection_confidence": float(cfg_mediapipe.get("min_detection_confidence", 0.5)),
            "min_tracking_confidence": float(cfg_mediapipe.get("min_tracking_confidence", 0.5)),
            "refine_face_landmarks": bool(cfg_mediapipe.get("refine_face_landmarks", False)),
        }
        self._h = holistic.Holistic(**self.parametros)

    @property
    def versao_mediapipe(self) -> str:
        return self._mp.__version__

    def processar(self, frame_bgr: np.ndarray):
        """Processa UM frame BGR (nao espelhado). Retorna (LandmarksFrame, resultado_bruto)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._h.process(rgb)
        return resultado_para_frame(res), res

    def processar_frames(self, frames: list[np.ndarray], fps: float) -> SequenciaLandmarks:
        if not frames:
            return empilhar([], 0, 0, fps)
        alt, larg = frames[0].shape[:2]
        return empilhar([self.processar(f)[0] for f in frames], larg, alt, fps)

    def desenhar(self, frame_bgr: np.ndarray, resultado) -> np.ndarray:
        """Desenha landmarks sobre uma copia do frame (apenas para feedback visual)."""
        du = self._mp.solutions.drawing_utils
        h = self._holistic_mod
        img = frame_bgr.copy()
        if resultado.pose_landmarks:
            du.draw_landmarks(img, resultado.pose_landmarks, h.POSE_CONNECTIONS)
        for mao in (resultado.left_hand_landmarks, resultado.right_hand_landmarks):
            if mao:
                du.draw_landmarks(img, mao, h.HAND_CONNECTIONS)
        return img

    def fechar(self) -> None:
        if self._h is not None:
            self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


# --------------------------------------------------------------------------
# Leitura de video
# --------------------------------------------------------------------------
def ler_frames(caminho: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Itera sobre os frames BGR de um arquivo de video, sem espelhar."""
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    try:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            n += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        cap.release()


def extrair_video(caminho: Path, cfg_mediapipe: dict) -> SequenciaLandmarks:
    """Extrai landmarks de todos os frames de um video (modo de video continuo)."""
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    frames_lm: list[LandmarksFrame] = []
    larg = alt = 0
    with ExtratorHolistic(cfg_mediapipe, modo_estatico=False) as ext:
        for frame in ler_frames(caminho):
            alt, larg = frame.shape[:2]
            frames_lm.append(ext.processar(frame)[0])
    return empilhar(frames_lm, larg, alt, fps)


# --------------------------------------------------------------------------
# Subconjuntos da face (para reduzir dimensionalidade na Fase 3)
# --------------------------------------------------------------------------
def indices_face(subconjuntos: list[str]) -> list[int]:
    """Indices dos pontos da malha facial para os subconjuntos pedidos."""
    from mediapipe.python.solutions import face_mesh_connections as fmc

    mapa = {
        "labios": [fmc.FACEMESH_LIPS],
        "sobrancelhas": [fmc.FACEMESH_LEFT_EYEBROW, fmc.FACEMESH_RIGHT_EYEBROW],
        "contorno": [fmc.FACEMESH_FACE_OVAL],
        "olhos": [fmc.FACEMESH_LEFT_EYE, fmc.FACEMESH_RIGHT_EYE],
        "nariz": [fmc.FACEMESH_NOSE],
    }
    idx: set[int] = set()
    for s in subconjuntos:
        if s not in mapa:
            from .erros import ErroConfig

            raise ErroConfig(f"Subconjunto de face desconhecido: {s!r}. Opcoes: {sorted(mapa)}")
        for conexoes in mapa[s]:
            for a, b in conexoes:
                idx.update((a, b))
    return sorted(idx)


# --------------------------------------------------------------------------
# Desenho a partir dos arrays salvos (.npz), para a revisao de takes
# --------------------------------------------------------------------------
_CONEXOES: dict | None = None


def _conexoes() -> dict:
    global _CONEXOES
    if _CONEXOES is None:
        from mediapipe.python.solutions import hands_connections, pose_connections

        _CONEXOES = {
            "pose": sorted(pose_connections.POSE_CONNECTIONS),
            "mao": sorted(hands_connections.HAND_CONNECTIONS),
        }
    return _CONEXOES


def desenhar_esqueleto(
    frame_bgr: np.ndarray,
    pose: np.ndarray,
    mao_esq: np.ndarray,
    mao_dir: np.ndarray,
    presenca: np.ndarray,
    cores: dict | None = None,
) -> np.ndarray:
    """Desenha pose e maos (coordenadas normalizadas do MediaPipe) sobre uma COPIA do frame."""
    import cv2

    cores = cores or {"pose": (235, 235, 235), "mao_esq": (246, 130, 59), "mao_dir": (237, 58, 124)}
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    con = _conexoes()

    def pts(arr):
        return [(int(x * w), int(y * h)) for x, y in arr[:, :2]]

    for arr, k, nome, conexoes in (
        (pose, 0, "pose", con["pose"]),
        (mao_esq, 1, "mao_esq", con["mao"]),
        (mao_dir, 2, "mao_dir", con["mao"]),
    ):
        if not presenca[k] or np.isnan(arr[:, :2]).any():
            continue
        p = pts(arr)
        for a, b in conexoes:
            cv2.line(img, p[a], p[b], cores[nome], 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, q, 3, cores[nome], -1, cv2.LINE_AA)
    return img
