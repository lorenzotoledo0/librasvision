"""Fixtures de teste.

ATENCAO: todos os videos e landmarks criados aqui sao SINTETICOS (quadros coloridos
gerados por codigo, sem pessoas). Servem apenas para testar o encanamento do codigo.
Nada daqui deve ser usado como dado do TCC nem gerar numeros de desempenho.
"""

import os
import shutil
import sys
from pathlib import Path

# No Windows, a plataforma offscreen do Qt nao acha nenhuma fonte (0 familias): o texto sai com
# metricas de reserva e os testes de layout acusam rolagem horizontal que nao existe no app real.
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from libras.config import carregar_config  # noqa: E402
from libras.landmarks import N_FACE, N_MAO, N_POSE, LandmarksFrame  # noqa: E402


@pytest.fixture
def projeto(tmp_path):
    """Raiz de projeto temporaria com o config real e data/ vazio."""
    (tmp_path / "configs").mkdir()
    shutil.copy(RAIZ / "configs" / "config.yaml", tmp_path / "configs" / "config.yaml")
    return tmp_path


@pytest.fixture
def cfg(projeto):
    return carregar_config(projeto / "configs" / "config.yaml")


def frame_sintetico_assimetrico(alt=480, larg=640, i=0):
    """SINTETICO: metade esquerda clara, direita escura, com uma faixa que varia por frame."""
    f = np.zeros((alt, larg, 3), np.uint8)
    f[:, : larg // 2] = 200
    f[(i * 7) % alt : (i * 7) % alt + 4, :] = 120
    return f


def gerar_video_sintetico(caminho: Path, n_frames=75, fps=30.0, larg=640, alt=480, variante=0):
    """SINTETICO: grava um video de quadros coloridos (sem pessoa) para testar importacao."""
    import cv2

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (larg, alt))
    for i in range(n_frames):
        w.write(frame_sintetico_assimetrico(alt, larg, i + 13 * variante))
    w.release()
    return caminho


class ExtratorFalso:
    """SINTETICO: substitui o MediaPipe nos testes; informa presenca de pose/mao fixa."""

    def __init__(self, pose=True, mao=True):
        self.pose, self.mao = pose, mao

    def processar(self, frame):
        nan = np.nan
        pose = np.zeros((N_POSE, 4), np.float32) if self.pose else np.full((N_POSE, 4), nan, np.float32)
        me = np.zeros((N_MAO, 3), np.float32) if self.mao else np.full((N_MAO, 3), nan, np.float32)
        md = np.full((N_MAO, 3), nan, np.float32)
        face = np.full((N_FACE, 3), nan, np.float32)
        return LandmarksFrame(pose, me, md, face, np.array([self.pose, self.mao, False, False])), None

    def processar_frames(self, frames, fps):
        from libras.landmarks import empilhar

        alt, larg = frames[0].shape[:2]
        return empilhar([self.processar(f)[0] for f in frames], larg, alt, fps)

    def fechar(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture
def com_extrator_falso(monkeypatch):
    """Faz a importacao usar ExtratorFalso no lugar do MediaPipe."""
    estado = {"pose": True, "mao": True}

    def fabrica(cfg_mp, modo_estatico=False):
        return ExtratorFalso(estado["pose"], estado["mao"])

    import libras.landmarks as lm

    monkeypatch.setattr(lm, "ExtratorHolistic", fabrica)
    return estado


def cadastrar(cfg, n=1, tcle=True, mao="direita"):
    from libras.coleta.sinalizadores import CadastroSinalizadores

    cad = CadastroSinalizadores(cfg)
    return [cad.cadastrar(mao, tcle, "2026-10-01" if tcle else "").signer_id for _ in range(n)]


# ----------------------------------------------------------------------
# Esqueletos SINTETICOS para testar normalizacao, janelas e splits.
# Sao figuras geometricas simples, nao movimentos de Libras.
# ----------------------------------------------------------------------
def partes_sinteticas(n=90, largura=640, altura=480, ombro_px=160, centro_px=(320, 260), mover_mao=(30, 60), seed=0):
    """SINTETICO: pose com ombros fixos e mao direita oscilando entre os frames mover_mao."""
    from libras.normalizacao import Partes

    rng = np.random.default_rng(seed)
    p = Partes.vazias(n)
    cx, cy = centro_px
    base = rng.normal(0, 40, size=(N_POSE, 2)) + [cx, cy]
    base[11] = [cx + ombro_px / 2, cy]
    base[12] = [cx - ombro_px / 2, cy]
    for t in range(n):
        p.pose[t, :, 0] = base[:, 0] / largura
        p.pose[t, :, 1] = base[:, 1] / altura
        p.pose[t, :, 2] = 0.0
        p.pose[t, :, 3] = 0.9
        dx = 40 * np.sin(t / 3.0) if mover_mao[0] <= t < mover_mao[1] else 0.0
        mao = np.stack([cx - 60 + dx + np.arange(N_MAO) * 2.0, cy - 80 + np.arange(N_MAO) * 1.5], 1)
        p.mao_dir[t, :, 0] = mao[:, 0] / largura
        p.mao_dir[t, :, 1] = mao[:, 1] / altura
        p.mao_dir[t, :, 2] = 0.0
        p.face[t, :, 0] = (cx + rng.normal(0, 20, N_FACE)) / largura
        p.face[t, :, 1] = (cy - 120 + rng.normal(0, 20, N_FACE)) / altura
        p.face[t, :, 2] = 0.0
    p.presenca[:, 0] = True
    p.presenca[:, 2] = True
    p.presenca[:, 3] = True
    return p


def criar_videos_com_landmarks(cfg, n_sinalizadores=4, por_classe=3, seed=0, classes=None):
    """SINTETICO: registra no manifesto 'videos' + npz de landmarks sinteticos (sem MediaPipe).

    Cria arquivos de video sinteticos para o manifesto ficar integro, e os npz com a meta
    esperada para o cache ser considerado valido.
    """
    from libras.coleta.manifesto import Manifesto
    from libras.extracao import caminho_npz, meta_esperada, salvar_npz
    from libras.landmarks import SequenciaLandmarks
    from libras.utils import sha256_arquivo

    ids = cadastrar(cfg, n_sinalizadores)
    m = Manifesto(cfg)
    k = 0
    for s in ids:
        for c in classes if classes is not None else cfg.classes:
            for t in range(1, por_classe + 1):
                k += 1
                vid = f"{s}_{c}_{t:03d}"
                arq = cfg.caminho("videos") / c / f"{s}_{t:03d}.mp4"
                arq.parent.mkdir(parents=True, exist_ok=True)
                arq.write_bytes(f"SINTETICO {vid}".encode())  # conteudo unico -> hash unico
                linha = {
                    "video_id": vid,
                    "sha256": sha256_arquivo(arq),
                    "classe": c,
                    "signer_id": s,
                    "take": t,
                    "arquivo": m.relativo(arq),
                    "status": "ok",
                    "origem": "sintetico",
                }
                m.adicionar(linha)
                p = partes_sinteticas(n=75, seed=seed + k)
                seq = SequenciaLandmarks(p.pose, p.mao_esq, p.mao_dir, p.face, p.presenca, 640, 480, 30.0)
                salvar_npz(caminho_npz(cfg, linha), seq, np.arange(75) / 30.0, meta_esperada(cfg, linha))
    return ids
