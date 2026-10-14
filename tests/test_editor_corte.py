"""Item 9: editor de corte. Logica separada da interface (SessaoCorte, leitor sob demanda, curva de
movimento, aplicar_edicao) e um teste de interface offscreen. Videos e landmarks SINTETICOS."""

import os
import time

import cv2
import numpy as np
import pytest

from conftest import cadastrar, gerar_video_sintetico, partes_sinteticas
from libras.coleta.edicao_corte import SessaoCorte, formatar_tempo
from libras.coleta.manifesto import Manifesto

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------- sessao (sem interface)
def test_limites_de_30_quadros():
    s = SessaoCorte(100, 30.0)
    assert s.definir_inicio(90) == 70  # a alca para no limite: fim (100) - 30
    assert s.definir_fim(10) == 100  # fim nao pode ficar a menos de 30 do inicio (70)
    s.definir_inicio(0)
    assert s.definir_fim(5) == 30 and s.trechos() == [(0, 30)]
    with pytest.raises(ValueError):
        SessaoCorte(29, 30.0)
    exata = SessaoCorte(30, 30.0)
    assert exata.definir_inicio(1) == 0 and exata.definir_fim(29) == 30 and not exata.alterado


def test_divisao_respeita_o_minimo_e_restringe_alcas():
    s = SessaoCorte(100, 30.0)
    assert not s.dividir(20) and not s.dividir(80) and s.divisao is None
    assert s.dividir(50) and s.trechos() == [(0, 50), (50, 100)]
    assert s.definir_inicio(40) == 20  # inicio <= divisao - 30
    assert s.definir_fim(60) == 80  # fim >= divisao + 30
    assert all(b - a >= 30 for a, b in s.trechos())


def test_definir_aqui_usa_o_quadro_do_playhead():
    s = SessaoCorte(120, 30.0)
    assert s.definir_inicio_no_quadro(10) == 10 and s.inicio == 10
    assert s.definir_fim_no_quadro(99) == 99 and s.fim == 100  # quadro 99 e o ultimo mantido
    assert "quadros 11 a 100" in s.descricao()


def test_desfazer_refazer_e_arrasto_como_um_passo():
    s = SessaoCorte(100, 30.0)
    s.iniciar_arrasto()
    for q in (5, 10, 15, 20):
        s.definir_inicio(q)
    s.concluir_arrasto()
    s.definir_fim(90)
    assert (s.inicio, s.fim) == (20, 90)
    assert s.desfazer() and (s.inicio, s.fim) == (20, 100)
    assert s.desfazer() and (s.inicio, s.fim) == (0, 100) and not s.pode_desfazer  # arrasto = 1 passo
    assert s.refazer() and s.refazer() and (s.inicio, s.fim) == (20, 90) and not s.pode_refazer
    s.cancelar()
    assert not s.alterado and s.desfazer() and s.fim == 90  # cancelar tambem pode ser desfeito


def test_tempo_formatado_e_instantes_reais():
    assert formatar_tempo(0) == "00:00.00" and formatar_tempo(61.257) == "01:01.26"
    tempos = 100 + np.cumsum(np.r_[0, np.full(59, 0.04)])  # 25 FPS reais
    s = SessaoCorte(60, 30.0, tempos)
    assert s.tempo(25) == pytest.approx(1.0)


# ---------------------------------------------------------------------- leitura e curva
def test_leitor_sob_demanda_devolve_o_quadro_pedido(tmp_path):
    from libras.coleta.leitor_video import LeitorVideo

    arq = gerar_video_sintetico(tmp_path / "v.mp4", n_frames=80)
    cap = cv2.VideoCapture(str(arq))
    ref = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ref.append(f)
    lv = LeitorVideo(arq, tamanho_cache=8)
    try:
        for i in (0, 1, 2, 50, 3, 79, 40, 41, 45, 10):
            assert np.abs(lv.quadro(i).astype(int) - ref[i].astype(int)).mean() < 1, i
        assert len(lv._cache) <= 8  # o video nao e carregado inteiro
        assert lv.quadro(80) is None
    finally:
        lv.fechar()


def test_miniaturas_em_cache(tmp_path):
    from libras.coleta.leitor_video import limpar_cache_miniaturas, miniaturas

    limpar_cache_miniaturas()
    arq = gerar_video_sintetico(tmp_path / "v.mp4", n_frames=60)
    a = miniaturas(arq, 12, altura=40)
    assert len(a) == 12 and a[0][0] == 0 and a[-1][0] == 59 and a[0][1].shape[0] == 40
    assert miniaturas(arq, 12, altura=40) is a


def test_curva_de_movimento_das_maos():
    from libras.normalizacao import movimento_maos_por_frame

    p = partes_sinteticas(n=60)
    c = movimento_maos_por_frame(p, 640, 480)
    assert c.shape == (60,) and np.isfinite(c).all() and c.max() > 0
    p.presenca[:, 1:3] = False
    assert movimento_maos_por_frame(p, 640, 480).max() == 0


# ---------------------------------------------------------------------- aplicar no disco
def _take_real(cfg, n=90):
    from libras.extracao import caminho_npz, meta_esperada, salvar_npz
    from libras.landmarks import SequenciaLandmarks
    from libras.utils import sha256_arquivo

    (sid,) = cadastrar(cfg)
    m = Manifesto(cfg)
    arq = gerar_video_sintetico(cfg.caminho("videos") / "prova" / f"{sid}_001.mp4", n_frames=n)
    l = {
        "video_id": f"{sid}_prova_001",
        "sha256": sha256_arquivo(arq),
        "classe": "prova",
        "signer_id": sid,
        "take": 1,
        "arquivo": m.relativo(arq),
        "status": "ok",
        "origem": "sintetico",
        "n_frames": n,
        "fps_medido": 30,
    }
    m.adicionar(l)
    p = partes_sinteticas(n=n)
    seq = SequenciaLandmarks(p.pose, p.mao_esq, p.mao_dir, p.face, p.presenca, 640, 480, 30.0)
    npz = caminho_npz(cfg, l)
    salvar_npz(npz, seq, 50 + np.arange(n) / 30.0, meta_esperada(cfg, l))
    m.atualizar(l["video_id"], landmarks_path=m.relativo(npz), sha256_npz=sha256_arquivo(npz))
    return l["video_id"]


def test_aplicar_edicao_corta_e_divide_de_uma_vez(cfg):
    from libras.coleta.cortes import aplicar_edicao
    from libras.extracao import caminho_npz, carregar_npz

    vid = _take_real(cfg, 120)
    assert aplicar_edicao(cfg, vid, 0, 120)[0]["video_id"] == vid  # sem alteracao: nada gravado
    assert not (cfg.caminho("lixeira") / "cortes").exists()
    a, b = aplicar_edicao(cfg, vid, 10, 110, 60)
    assert (a["n_frames"], b["n_frames"]) == ("50", "50")
    assert "frames 10-60 de 120" in a["corte"] and "frames 60-110 de 120" in b["corte"]
    for x in (a, b):
        assert carregar_npz(caminho_npz(cfg, x)).tempos[0] == 0.0
    assert len(list((cfg.caminho("lixeira") / "cortes").iterdir())) == 1
    assert Manifesto(cfg).verificar_integridade().ok


# ---------------------------------------------------------------------- interface
@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
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


def test_interface_arrastar_alca_atualiza_inicio_e_fim(qapp, cfg):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from libras.app.janela import JanelaPrincipal

    vid = _take_real(cfg, 90)
    w = JanelaPrincipal(cfg, "revisao", transicao_ms=0)
    w.resize(1400, 900)
    w.show()
    try:
        r = w.paginas["revisao"]
        r.f_status.setCurrentText("todos")
        r.tabela.selecionar_valor(vid)
        assert esperar(qapp, lambda: r.editor.sessao is not None and r.editor.video_id == vid)
        e, lt = r.editor, r.editor.linha_tempo
        assert esperar(qapp, lambda: len(lt.miniaturas) > 0)  # miniaturas em segundo plano
        assert esperar(qapp, lambda: r.editor.tela.pixmap() is not None and not r.editor.tela.pixmap().isNull())
        y = lt.Y_FAIXA + lt.H_FAIXA // 2
        x0 = int(lt.x_fronteira(0))
        QTest.mousePress(lt, Qt.LeftButton, Qt.NoModifier, QPoint(x0, y))
        QTest.mouseMove(lt, QPoint(int(lt.x_fronteira(10)), y))
        QTest.mouseMove(lt, QPoint(int(lt.x_fronteira(15)), y))
        QTest.mouseRelease(lt, Qt.LeftButton, Qt.NoModifier, QPoint(int(lt.x_fronteira(15)), y))
        assert e.sessao.inicio == 15 and e.atual == 15
        xf = int(lt.x_fronteira(90))
        QTest.mousePress(lt, Qt.LeftButton, Qt.NoModifier, QPoint(xf, y))
        QTest.mouseMove(lt, QPoint(int(lt.x_fronteira(20)), y))  # alem do limite: para em 15 + 30
        QTest.mouseRelease(lt, Qt.LeftButton, Qt.NoModifier, QPoint(int(lt.x_fronteira(20)), y))
        assert e.sessao.fim == 45 and "mínimo de 30" in e.lb_info.text()
        assert e.sessao.desfazer() and e.sessao.fim == 90  # o arrasto inteiro e um passo
        # clique na faixa move o playhead; atalhos I, O e S
        QTest.mouseClick(lt, Qt.LeftButton, Qt.NoModifier, QPoint(int(lt.x_quadro(80)), y))
        assert e.atual == 80
        e.atalhos["O"].activated.emit()
        assert e.sessao.fim == 81
        e.ir(40)
        e.atalhos["S"].activated.emit()  # 40 - 15 < 30: recusado, com explicacao
        assert e.sessao.divisao is None and "precisa de 30" in e.lb_info.text()
        e.ir(48)
        e.atalhos["S"].activated.emit()
        assert e.sessao.divisao == 48
        e.atalhos["Ctrl+Z"].activated.emit()
        assert e.sessao.divisao is None
        e.atalhos["Shift+Right"].activated.emit()
        assert e.atual == 78 and e.bt_salvar.isEnabled()
        assert Manifesto(cfg).obter(vid)["corte"] == ""  # nada gravado ate salvar
        e.ir(0)
        e.tocar()  # reproducao no FPS real (30): cerca de 15 quadros em 0,5 s
        esperar(qapp, lambda: False, 0.5)
        e.pausar()
        assert 8 <= e.atual <= 22, e.atual
    finally:
        w.close()
        w.deleteLater()
