"""Familia D (nivel de video). Frames e landmarks SINTETICOS; MediaPipe substituido."""

import numpy as np
import pytest

from conftest import criar_videos_com_landmarks, frame_sintetico_assimetrico, partes_sinteticas
from libras.augmentation import video as av
from libras.dataset import montar_conjunto
from libras.erros import ErroDadosInsuficientes
from libras.splits import gerar_splits, videos_elegiveis


def test_degradacoes_deterministicas_e_com_efeito():
    f = frame_sintetico_assimetrico()
    for v in av.VARIANTES:
        a = av.Degradador(v, 7)(f)
        b = av.Degradador(v, 7)(f)
        assert a.shape == f.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, f), v
    assert av.Degradador("escuro", 1)(f).mean() < f.mean() < av.Degradador("claro", 1)(f).mean()


@pytest.fixture
def variantes_falsas(monkeypatch):
    from libras.extracao import salvar_npz
    from libras.landmarks import SequenciaLandmarks

    def falso(cfg_mp, src, dst, meta):
        p = partes_sinteticas(75, seed=meta["seed"] % 1000)
        seq = SequenciaLandmarks(p.pose, p.mao_esq, p.mao_dir, p.face, p.presenca, 640, 480, 30.0)
        salvar_npz(dst, seq, np.arange(75) / 30.0, meta)
        return {"ok": True, "n_frames": 75, "taxa_mao": 1.0, "taxa_pose": 1.0, "segundos": 0}

    monkeypatch.setattr(av, "extrair_variante", falso)


def test_variantes_so_entram_pelo_treino(cfg, variantes_falsas):
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    r = av.aumentar_lote(cfg, ["escuro", "desfoque"])
    assert r["processados"] == 24 and r["em_cache"] == 0
    assert av.aumentar_lote(cfg, ["escuro"])["em_cache"] == 12
    linhas, _ = videos_elegiveis(cfg)
    conj = montar_conjunto(cfg, linhas, variantes=["escuro", "desfoque"])
    f = gerar_splits(cfg, linhas)["folds"][0]
    teste = conj.indices(f["teste"])
    assert (conj.variante[teste] == "").all()
    treino = conj.indices(f["treino"], variantes=("", "escuro", "desfoque"))
    assert set(conj.variante[treino]) == {"", "escuro", "desfoque"}
    assert not set(conj.video[treino]) & set(f["teste"])
    # robustez (Fase 5): videos de teste degradados ficam acessiveis so quando pedidos
    rob = conj.indices(f["teste"], variantes=("escuro",))
    assert len(rob) and (conj.variante[rob] == "escuro").all()


def test_variante_faltando_mensagem_clara(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    with pytest.raises(ErroDadosInsuficientes, match="aumentar_videos"):
        montar_conjunto(cfg, linhas, variantes=["escuro"])
