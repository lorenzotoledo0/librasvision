"""Previa de data augmentation. Landmarks SINTETICOS (sem pessoas)."""

import os

import numpy as np
import pytest

from conftest import criar_videos_com_landmarks

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def linha(cfg):
    from libras.coleta.manifesto import Manifesto

    criar_videos_com_landmarks(cfg, n_sinalizadores=1, por_classe=1)
    return next(l for l in Manifesto(cfg).linhas if l["classe"] != cfg.nome_nenhum_sinal)


def test_previa_gera_cinco_variacoes_reprodutiveis(cfg, linha):
    from libras.augmentation.previa import gerar_previa

    p = gerar_previa(cfg, linha, n=5, seed=3)
    assert len(p.variacoes) == 5 and p.n_frames == cfg["janela"]["comprimento"]
    q = gerar_previa(cfg, linha, n=5, seed=3)
    for a, b in zip(p.variacoes, q.variacoes):
        np.testing.assert_array_equal(a["mao_dir"], b["mao_dir"])  # mesma seed, mesmas variacoes
    # a familia espacial sempre muda a geometria na previa (probabilidade 1)
    e = gerar_previa(cfg, linha, n=3, seed=1, familias=["espacial"])
    assert all(not np.allclose(v["pose"][..., :2], e.original["pose"][..., :2]) for v in e.variacoes)


def test_previa_sem_familias_repete_o_original(cfg, linha):
    from libras.augmentation.previa import gerar_previa

    p = gerar_previa(cfg, linha, n=2, familias=[])
    for v in p.variacoes:
        np.testing.assert_array_equal(v["pose"], p.original["pose"])


def test_previa_nao_altera_o_config(cfg, linha):
    from libras.augmentation.previa import gerar_previa

    antes = {f: cfg["augmentation"][f]["prob"] for f in ("espacial", "temporal", "ruido")}
    gerar_previa(cfg, linha, n=2)
    assert {f: cfg["augmentation"][f]["prob"] for f in antes} == antes


def test_desenho_e_mosaico(cfg, linha):
    from libras.augmentation.previa import desenhar, gerar_previa, mosaico

    p = gerar_previa(cfg, linha, n=5)
    img = desenhar(p.original, 0, 320, 300, "original")
    assert img.shape == (300, 320, 3) and img.std() > 0  # algo foi desenhado
    assert mosaico(p, 0).shape == (600, 960, 3)  # original + 5 variacoes em 2 linhas de 3


def test_previa_sem_landmarks_explica(cfg, linha):
    from libras.augmentation.previa import gerar_previa
    from libras.erros import ErroLibras
    from libras.extracao import caminho_npz

    caminho_npz(cfg, linha).unlink()
    with pytest.raises(ErroLibras, match="Extrair"):
        gerar_previa(cfg, linha)


def test_dialogo_da_previa(cfg, linha, monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    import libras.app.previa_augmentation as mod

    avisos = []
    monkeypatch.setattr(mod, "info", lambda *a, **k: avisos.append(a[1]))
    d = mod.DialogoPreviaAugmentation(None, cfg, linha)
    assert d.previa is not None and len(d.previa.variacoes) == mod.N_VARIACOES
    d._passo()
    d.ck["espacial"].setChecked(False)  # muda as familias e sorteia de novo
    assert "espacial" not in d.previa.familias
    d.salvar_figura()
    figs = list((cfg.caminho("runs") / "augmentation_previa").glob("*.png"))
    assert len(figs) == 1 and avisos and "Figura salva" in avisos[0]
    d.done(0)
    assert not d.timer.isActive()
