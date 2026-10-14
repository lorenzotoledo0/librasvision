"""Acesso a webcam com tratamento especifico para Windows e captura em thread.

A thread de captura le frames continuamente e guarda: o ultimo frame (para a
pre-visualizacao) e, durante a gravacao, todos os frames com o instante de chegada
(para medir o FPS real). Os frames guardados NUNCA sao espelhados.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .erros import ErroCamera
from .utils import eh_windows


@dataclass
class InfoCamera:
    indice: int
    backend: str
    largura: int
    altura: int
    fps_declarado: float


def _backends(cfg_camera: dict) -> list[tuple[str, int]]:
    import cv2

    nomes = cfg_camera.get("backends_windows", ["DSHOW", "MSMF", "ANY"]) if eh_windows() else ["ANY"]
    out = []
    for n in nomes:
        const = getattr(cv2, f"CAP_{n}", None)
        if const is not None:
            out.append((n, const))
    return out or [("ANY", cv2.CAP_ANY)]


def abrir_camera(cfg_camera: dict):
    """Abre a webcam tentando os backends configurados. Retorna (cap, InfoCamera)."""
    import cv2

    indice = int(cfg_camera.get("indice", 0))
    tentativas = []
    for nome, const in _backends(cfg_camera):
        cap = cv2.VideoCapture(indice, const)
        if not cap.isOpened():
            cap.release()
            tentativas.append(f"{nome}: nao abriu")
            continue
        if cfg_camera.get("forcar_mjpg", False):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg_camera.get("largura", 640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg_camera.get("altura", 480)))
        cap.set(cv2.CAP_PROP_FPS, float(cfg_camera.get("fps", 30)))
        ok, frame = False, None
        for _ in range(10):  # algumas webcams entregam frames vazios logo apos abrir
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            cap.release()
            tentativas.append(f"{nome}: abriu mas nao entregou imagem")
            continue
        alt, larg = frame.shape[:2]
        return cap, InfoCamera(indice, nome, larg, alt, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
    raise ErroCamera(
        f"Nao foi possivel usar a camera de indice {indice} ({'; '.join(tentativas)}).\n"
        "Verifique: (1) se a webcam esta conectada; (2) se outro programa (Teams, Zoom, "
        "navegador) esta usando a camera; (3) no Windows, Configuracoes > Privacidade > Camera > "
        "permitir que aplicativos da area de trabalho acessem a camera; (4) se ha mais de uma "
        "camera, troque coleta.camera.indice no config (0, 1, 2...)."
    )


class Captura:
    """Le frames em uma thread. Thread-safe para leitura do ultimo frame e da gravacao."""

    def __init__(self, cap, info: InfoCamera):
        self.cap = cap
        self.info = info
        self._trava = threading.Lock()
        self._ultimo: np.ndarray | None = None
        self._t_ultimo = 0.0
        self._n = 0
        self._ts_recentes: deque[float] = deque(maxlen=45)
        self._gravando = False
        self._buf_frames: list[np.ndarray] = []
        self._buf_ts: list[float] = []
        self._max_frames = 0
        self._duracao_s = 0.0
        self._rodando = False
        self._thread: threading.Thread | None = None
        self.erro: str | None = None

    # ------------------------------------------------------------------
    def iniciar(self) -> "Captura":
        self._rodando = True
        self._thread = threading.Thread(target=self._laco, name="captura", daemon=True)
        self._thread.start()
        return self

    def _laco(self) -> None:
        falhas = 0
        while self._rodando:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok or frame is None:
                falhas += 1
                if falhas > 60:
                    self.erro = "A camera parou de enviar imagens (desconectada ou em uso por outro programa)."
                    self._rodando = False
                time.sleep(0.01)
                continue
            falhas = 0
            with self._trava:
                self._ultimo = frame
                self._t_ultimo = t
                self._n += 1
                self._ts_recentes.append(t)
                if self._gravando:
                    self._buf_frames.append(frame)
                    self._buf_ts.append(t)
                    passou = t - self._buf_ts[0]
                    if len(self._buf_frames) >= self._max_frames or passou >= self._duracao_s:
                        self._gravando = False

    def parar(self) -> None:
        self._rodando = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()

    # ------------------------------------------------------------------
    def ultimo(self) -> tuple[np.ndarray | None, int]:
        with self._trava:
            return self._ultimo, self._n

    def ultimo_com_tempo(self) -> tuple[np.ndarray | None, int, float]:
        """Ultimo frame, contador e instante de chegada (time.perf_counter)."""
        with self._trava:
            return self._ultimo, self._n, self._t_ultimo

    @property
    def fps_atual(self) -> float:
        with self._trava:
            ts = list(self._ts_recentes)
        if len(ts) < 2 or ts[-1] == ts[0]:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    # ------------------------------------------------------------------
    def iniciar_gravacao(self, duracao_s: float, fps_ref: float = 30.0) -> None:
        """Grava ate duracao_s segundos (medidos pelos instantes reais dos frames).

        No modo manual, passe a duracao maxima e chame parar_gravacao() antes.
        """
        with self._trava:
            self._buf_frames, self._buf_ts = [], []
            self._duracao_s = float(duracao_s)
            # trava extra por numero de frames (protege a memoria se o relogio falhar)
            self._max_frames = max(1, int(duracao_s * max(fps_ref, 30.0) * 1.5))
            self._gravando = True

    def parar_gravacao(self) -> tuple[list[np.ndarray], list[float]]:
        with self._trava:
            self._gravando = False
            frames, ts = self._buf_frames, self._buf_ts
            self._buf_frames, self._buf_ts = [], []
        return frames, ts

    @property
    def gravando(self) -> bool:
        with self._trava:
            return self._gravando

    def tempo_gravado(self) -> float:
        with self._trava:
            if len(self._buf_ts) < 1:
                return 0.0
            return time.perf_counter() - self._buf_ts[0]


def frame_para_exibicao(frame: np.ndarray, espelhar: bool = True) -> np.ndarray:
    """Copia espelhada APENAS para mostrar na tela. Nunca usar o resultado para salvar ou extrair."""
    import cv2

    return cv2.flip(frame, 1) if espelhar else frame.copy()


class CameraArquivo:
    """Usa um arquivo de video como se fosse a webcam (no ritmo do FPS do arquivo, em laco).

    Serve para demonstracoes e testes do tempo real sem camera. Os frames NAO sao espelhados."""

    def __init__(self, caminho: str):
        import cv2

        self.caminho = str(caminho)
        self.cap = cv2.VideoCapture(self.caminho)
        if not self.cap.isOpened():
            raise ErroCamera(f"Nao foi possivel abrir o video {caminho}.")
        self.dt = 1.0 / float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._prox = time.perf_counter()

    def read(self):
        import cv2

        espera = self._prox - time.perf_counter()
        if espera > 0:
            time.sleep(espera)
        self._prox = max(self._prox + self.dt, time.perf_counter() - self.dt)
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self) -> None:
        self.cap.release()


def abrir_arquivo_como_camera(caminho: str):
    cam = CameraArquivo(caminho)
    ok, f = cam.read()
    if not ok:
        raise ErroCamera(f"O video {caminho} nao tem frames.")
    alt, larg = f.shape[:2]
    return cam, InfoCamera(-1, "ARQUIVO", larg, alt, 1.0 / cam.dt)


def abrir_camera_ou_video(cfg_camera: dict):
    """Webcam, ou o video indicado na variavel de ambiente LIBRAS_CAMERA_VIDEO (demonstracao e testes)."""
    import os

    video = os.environ.get("LIBRAS_CAMERA_VIDEO")
    if video:
        return abrir_arquivo_como_camera(video)
    return abrir_camera(cfg_camera)


def explicar_falha(mensagem: str, indice: int | None = None) -> tuple[str, str]:
    """(causa provavel, acao) em portugues simples para quem usa o app real, a partir das mensagens de
    abrir_camera e da Captura (as mesmas da bancada, D9)."""
    m = (mensagem or "").lower()
    if "parou de enviar" in m:
        return (
            "A câmera parou de enviar imagens. Ela pode ter sido desconectada ou outro programa passou a usá-la.",
            "Confira o cabo ou a conexão, feche Teams, Zoom ou o navegador se estiverem com a câmera e toque em "
            "Tentar de novo.",
        )
    if "abriu mas nao entregou imagem" in m:
        return (
            "A câmera abriu, mas não entregou imagem. Em geral outro programa está usando a câmera ou a tampa de "
            "privacidade está fechada.",
            "Feche os outros programas que usam a câmera, abra a tampa da câmera e toque em Tentar de novo.",
        )
    if "nao foi possivel usar a camera" in m or "nao abriu" in m:
        outra = (
            f" Se houver mais de uma câmera, troque o número (hoje {indice}) em Configurações > Câmera."
            if indice is not None
            else " Se houver mais de uma câmera, troque o número em Configurações > Câmera."
        )
        return (
            "A câmera não abriu. Ela pode estar desconectada, em uso por outro programa ou bloqueada pela "
            "privacidade do Windows.",
            "Conecte a câmera, feche Teams, Zoom ou o navegador e, no Windows, permita o acesso em Configurações > "
            "Privacidade > Câmera > aplicativos da área de trabalho. Depois toque em Tentar de novo." + outra,
        )
    return (
        "O reconhecimento parou por um problema inesperado.",
        "Toque em Tentar de novo. Se continuar, use "
        "Configurações > Informações técnicas > Copiar relatório técnico e envie à equipe.",
    )
