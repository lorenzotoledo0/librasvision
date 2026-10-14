"""Landmarks da pre-visualizacao em thread propria (sem interface grafica)."""

from __future__ import annotations

import queue
import threading

import numpy as np

from ..landmarks import ExtratorHolistic


class TrabalhadorPrevia:
    """Roda o Holistic sobre alguns frames da previa sem travar a interface.

    O MediaPipe e criado e fechado DENTRO desta thread (fechar o grafo de outra thread, ou deixar o
    interpretador encerrar com ele aberto, pode derrubar o processo na saida)."""

    def __init__(self, cfg_mediapipe: dict):
        self.cfg = cfg_mediapipe
        self._entrada: queue.Queue = queue.Queue(maxsize=1)
        self.lm = None  # ultimo LandmarksFrame
        self.tem_mao = False
        self.ativo = True
        self._parar = False
        self._t = threading.Thread(target=self._laco, daemon=True, name="previa-landmarks")
        self._t.start()

    def enviar(self, frame: np.ndarray) -> None:
        if not self.ativo:
            return
        try:
            self._entrada.put_nowait(frame)
        except queue.Full:
            pass

    def _laco(self) -> None:
        ext = None
        try:
            while not self._parar:
                try:
                    frame = self._entrada.get(timeout=0.2)
                except queue.Empty:
                    continue
                if ext is None:
                    ext = ExtratorHolistic(self.cfg, modo_estatico=False)
                lm, _ = ext.processar(frame)
                self.lm, self.tem_mao = lm, lm.tem_mao
        finally:
            if ext is not None:
                ext.fechar()

    def parar(self, espera_s: float = 3.0) -> None:
        self._parar = True
        self._t.join(timeout=espera_s)
