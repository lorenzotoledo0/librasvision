"""Item 8: interface dinamica (Qt offscreen). Dados SINTETICOS."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from conftest import cadastrar, criar_videos_com_landmarks  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    return app


def esperar(app, cond, limite=10.0):
    t = time.time()
    while not cond() and time.time() - t < limite:
        app.processEvents()
        time.sleep(0.02)
    return cond()


@pytest.fixture
def janela(qapp, cfg):
    from libras.app.janela import JanelaPrincipal

    w = JanelaPrincipal(cfg)
    w.show()
    yield w
    w.close()
    w.deleteLater()


def test_barra_lateral_recolhe_e_lembra(qapp, cfg, janela):
    from libras.app.janela import LARGURA_BARRA, LARGURA_BARRA_RECOLHIDA, JanelaPrincipal

    janela.resize(900, 600)
    assert esperar(qapp, lambda: janela.barra.width() == LARGURA_BARRA_RECOLHIDA)
    assert (
        janela.botoes["inicio"].text() == "Início" and janela.botoes["sinalizadores"].text() == "Pessoas"
    )  # trilho com rotulo curto (nada so no hover)
    janela.resize(1400, 900)
    assert esperar(qapp, lambda: janela.barra.width() == LARGURA_BARRA)
    janela.alternar_barra()  # manual: fica recolhida mesmo larga
    assert janela.barra.width() == LARGURA_BARRA_RECOLHIDA
    janela.resize(1300, 820)
    qapp.processEvents()
    geo = janela.saveGeometry()
    janela.close()
    from PySide6.QtCore import QSettings

    ini = QSettings(str(cfg.arquivo.with_name("interface.ini")), QSettings.IniFormat)
    assert ini.value("geometria") == geo  # tamanho e posicao lembrados
    w2 = JanelaPrincipal(cfg)
    try:
        assert w2.barra_recolhida()
    finally:
        w2.deleteLater()


def test_colunas_empilham_em_janela_estreita(qapp, cfg, janela):
    from PySide6.QtCore import Qt

    janela.ir_para("treino")
    janela.resize(1500, 900)
    qapp.processEvents()
    col = janela.paginas["treino"].findChild(QtWidgets.QSplitter)
    assert esperar(qapp, lambda: col.orientation() == Qt.Horizontal)
    janela.resize(870, 560)
    assert esperar(qapp, lambda: col.orientation() == Qt.Vertical)
    area = janela.paginas["treino"].area
    assert esperar(qapp, lambda: area.verticalScrollBar().maximum() > 0)  # rola em vez de cortar


def test_aviso_simples_nao_bloqueia(qapp, cfg, janela, monkeypatch):
    from libras.app.base import info

    def proibido(*a, **k):
        raise AssertionError("aviso simples nao deve abrir dialogo")

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", proibido)
    info(janela.paginas["inicio"], "Configurações salvas.")
    assert getattr(janela, "_toasts", []) and janela._toasts[-1].rotulo.text() == "Configurações salvas."
    assert janela._toasts[-1].isVisible()


def test_indicador_so_acima_de_200_ms(qapp, janela):
    from libras.app.base import iniciar_tarefa

    pag = janela.paginas["inicio"]
    esperar(qapp, lambda: not janela.indicador._n, 10)
    iniciar_tarefa(pag, lambda p, c: time.sleep(0.05), descricao="rapida")
    esperar(qapp, lambda: False, 0.3)
    assert not janela.indicador.visivel
    from libras.ui.dinamica import ATIVIDADE

    iniciar_tarefa(pag, lambda p, c: time.sleep(0.6), descricao="Contando...")
    assert esperar(qapp, lambda: janela.indicador.visivel, 2)
    assert "Contando..." in ATIVIDADE.ativas.values() and janela.indicador.texto.text()
    assert esperar(qapp, lambda: not janela.indicador.visivel, 3)


def test_dados_ao_vivo_sem_reabrir(qapp, cfg, janela):
    from libras.coleta.manifesto import Manifesto

    cadastrar(cfg)
    janela.ir_para("gravar")
    g = janela.paginas["gravar"]
    assert esperar(qapp, lambda: "S01" in [g.cb_sig.itemText(i) for i in range(g.cb_sig.count())], 5)
    g.cb_sig.setCurrentText("S01")
    i = g.cb_cls.findData("prova")
    assert "(0/" in g.cb_cls.itemText(i)
    m = Manifesto(cfg)  # outro processo (ou a linha de comando) grava um take
    arq = cfg.raiz / "data/raw_videos/prova/S01_001.mp4"
    arq.parent.mkdir(parents=True, exist_ok=True)
    arq.write_bytes(b"SINTETICO")
    m.adicionar(
        {
            "video_id": "S01_prova_001",
            "sha256": "x",
            "classe": "prova",
            "signer_id": "S01",
            "take": 1,
            "arquivo": "data/raw_videos/prova/S01_001.mp4",
            "status": "ok",
            "origem": "sintetico",
        }
    )
    assert esperar(qapp, lambda: "(1/" in g.cb_cls.itemText(g.cb_cls.findData("prova")), 5)
    janela.ir_para("inicio")
    assert esperar(qapp, lambda: janela.paginas["inicio"].c_vid[1].text() == "1", 10)
    criar_videos_com_landmarks(cfg, n_sinalizadores=1, por_classe=1, classes=["biblioteca"], seed=9)
    assert esperar(qapp, lambda: janela.paginas["inicio"].c_vid[1].text() == "2", 10)


def test_atalhos_de_aba(qapp, janela):
    from PySide6.QtGui import QKeySequence, QShortcut

    atalhos = {s.key().toString(): s for s in janela.findChildren(QShortcut)}
    atalhos["Ctrl+3"].activated.emit()
    assert janela.atual == "vocabulario"
    atalhos["Ctrl+Tab"].activated.emit()
    assert janela.atual == "gravar"
    atalhos["Ctrl+B"].activated.emit()
    assert janela.barra_manual is not None
    assert QKeySequence("Ctrl+0").toString() in atalhos  # 10a aba


def test_estados_visuais_no_tema():
    from libras.ui.tema import folha_de_estilo

    qss = folha_de_estilo()
    for seletor in (
        "QPushButton:hover",
        "QPushButton:pressed",
        "QPushButton:focus",
        "QPushButton:disabled",
        "QLineEdit:focus",
        "QLineEdit:disabled",
        "QCheckBox::indicator:hover",
        "QCheckBox:disabled",
        "QTabBar::tab:hover",
        "QSlider::handle:horizontal:pressed",
    ):
        assert seletor in qss, seletor
