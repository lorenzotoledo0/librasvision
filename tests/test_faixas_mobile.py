"""Item 12.5/12.6: reflow nas tres faixas de largura, alvos de toque, telas so de desktop e reduzir
movimento. Qt offscreen; dados SINTETICOS."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from conftest import cadastrar, criar_videos_com_landmarks  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def esperar(app, s):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()
        time.sleep(0.02)


@pytest.fixture
def janela(qapp, cfg):
    from libras.app.janela import JanelaPrincipal

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    cadastrar(cfg)
    w = JanelaPrincipal(cfg, transicao_ms=0)
    w.show()
    yield w
    w.close()
    w.deleteLater()
    from libras.ui import tema

    tema.configurar(faixa="expandida", reduzir_movimento=False, escala_fonte=1.0)


@pytest.mark.parametrize("largura,faixa", [(1366, "expandida"), (900, "media"), (390, "compacta")])
def test_reflow_nas_tres_faixas(qapp, janela, largura, faixa):
    from libras.app.janela import PAGINAS

    janela.resize(largura, 844)
    esperar(qapp, 0.3)
    assert janela.faixa == faixa
    if faixa == "compacta":
        assert not janela.barra.isVisible() and janela.barra_inferior.isVisible()
        assert (
            len([b for b in janela.botoes_inf.values() if b.isVisible()]) == 5
        )  # Inicio, Gravar, Tempo real, Resultados, Mais
    else:
        assert janela.barra.isVisible() and not janela.barra_inferior.isVisible()
    estouros = {}
    for chave, nome, _ in PAGINAS:
        if chave == "tempo_real":
            continue  # processo de inferencia; coberto pelas capturas
        janela.ir_para(chave)
        esperar(qapp, 0.2)
        p = janela.paginas[chave]
        if p.area.isVisible() and p.area.horizontalScrollBar().maximum() > 0:
            estouros[nome] = p.area.horizontalScrollBar().maximum()
    assert not estouros, f"Rolagem horizontal em {largura} px: {estouros}"


def test_telas_so_de_desktop_avisam_na_faixa_compacta(qapp, janela):
    janela.resize(390, 844)
    esperar(qapp, 0.3)
    for chave in ("dataset", "treino"):
        janela.ir_para(chave)
        esperar(qapp, 0.2)
        p = janela.paginas[chave]
        assert not p.area.isVisible() and p.aviso_desktop is not None and p.aviso_desktop.isVisible()
    janela.resize(1366, 844)
    esperar(qapp, 0.3)
    assert janela.paginas["treino"].area.isVisible() and not janela.paginas["treino"].aviso_desktop.isVisible()


def test_alvos_de_toque_de_44_px(qapp, janela):
    janela.resize(390, 844)
    esperar(qapp, 0.3)
    janela.ir_para("gravar")
    esperar(qapp, 0.3)
    p = janela.paginas["gravar"]
    pequenos = [b.text() for b in p.findChildren(QtWidgets.QPushButton) if b.isVisible() and b.height() < 44]
    assert not pequenos, pequenos
    assert all(b.height() >= 44 for b in janela.botoes_inf.values())


def test_reduzir_movimento(qapp, janela):
    from libras.app.janela import JanelaPrincipal
    from libras.ui import tema

    janela.cfg.setdefault("interface", {})["reduzir_movimento"] = True  # como gravado por Configuracoes
    w2 = JanelaPrincipal(janela.cfg)
    try:
        assert tema.duracao_ms() == 0 and w2.transicao_ms == 0
    finally:
        w2.deleteLater()
        janela.cfg["interface"]["reduzir_movimento"] = False
        tema.configurar(reduzir_movimento=False)
    assert tema.duracao_ms() == tema.TOKENS["movimento"]["duracao_ms"]
