"""Captura com camera FALSA (frames sinteticos) e conversao de landmarks com resultados falsos."""

import time
from types import SimpleNamespace

import numpy as np

from conftest import frame_sintetico_assimetrico
from libras.camera import Captura, InfoCamera
from libras.landmarks import N_FACE, indices_face, resultado_para_frame


class CameraFalsa:
    """SINTETICA: entrega frames assimetricos a ~fps quadros por segundo."""

    def __init__(self, fps=30):
        self.dt, self.i = 1 / fps, 0

    def read(self):
        time.sleep(self.dt)
        self.i += 1
        return True, frame_sintetico_assimetrico(i=self.i)

    def release(self):
        pass


def test_gravacao_para_na_duracao_e_mede_fps():
    c = Captura(CameraFalsa(fps=25), InfoCamera(0, "FALSA", 640, 480, 30)).iniciar()
    time.sleep(0.2)
    c.iniciar_gravacao(1.0)
    t0 = time.time()
    while c.gravando and time.time() - t0 < 3:
        time.sleep(0.02)
    frames, ts = c.parar_gravacao()
    c.parar()
    dur = ts[-1] - ts[0]
    assert 0.95 <= dur <= 1.2
    fps = (len(ts) - 1) / dur
    assert 18 < fps < 28  # a camera falsa entrega ~25 (com algum atraso de agendamento)
    assert frames[0][:, :320].mean() > 150  # frames guardados nao sao espelhados


def _lm(n, vis=False):
    pts = [SimpleNamespace(x=0.1 * k, y=0.2, z=0.0, visibility=0.9) for k in range(n)]
    return SimpleNamespace(landmark=pts)


def test_resultado_para_frame_formas_e_presenca():
    res = SimpleNamespace(
        pose_landmarks=_lm(33), left_hand_landmarks=None, right_hand_landmarks=_lm(21), face_landmarks=_lm(478)
    )
    f = resultado_para_frame(res)
    assert f.pose.shape == (33, 4) and f.mao_esq.shape == (21, 3) and f.face.shape == (N_FACE, 3)
    assert f.presenca.tolist() == [True, False, True, True]
    assert np.isnan(f.mao_esq).all() and not np.isnan(f.mao_dir).any()
    assert f.tem_mao


def test_indices_face_subconjunto():
    idx = indices_face(["labios", "sobrancelhas", "contorno"])
    assert len(idx) == 96 and max(idx) < N_FACE
