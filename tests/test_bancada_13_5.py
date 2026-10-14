"""13.5: a bancada usa o MESMO componente de legenda do app real, pre-visualiza com as preferencias
dele e mostra no Inicio o modelo publicado. Modelos SINTETICOS (bytes escritos pelo teste)."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_pacote import _modelo_final  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mesmo_componente_e_previa_como_no_app_real(qapp, cfg):
    from libras.app.janela import JanelaPrincipal
    from libras.app_real import janela as jr
    from libras.ui.legenda import CamadaLegenda, TelaLegendada

    (cfg.raiz / "configs" / "app_real_preferencias.json").write_text(
        json.dumps({"legenda": {"posicao": "topo", "tamanho": 4, "linhas": 3, "predefinicao": "amarelo"}}),
        encoding="utf-8",
    )
    w = JanelaPrincipal(cfg, transicao_ms=0)
    try:
        w.ir_para("tempo_real")
        tr = w.paginas["tempo_real"]
        assert isinstance(tr.tela, TelaLegendada) and type(tr.tela.camada) is CamadaLegenda
        assert jr.CamadaLegenda is CamadaLegenda  # um so codigo de legenda
        assert tr.tela.camada.estilo.posicao == "embaixo"
        tr.ck_como_real.setChecked(True)
        e = tr.tela.camada.estilo
        assert e.posicao == "topo" and e.tamanho_px == 52 and e.linhas == 3 and e.texto == "#FFE14D"
        tr.ck_como_real.setChecked(False)
        assert tr.tela.camada.estilo.posicao == "embaixo"
    finally:
        w.close()
        w.deleteLater()


def test_inicio_mostra_o_modelo_publicado(qapp, cfg):
    from libras.app.paginas.inicio import coletar_resumo
    from libras.publicacao import publicar

    assert coletar_resumo(cfg)["publicado"] is None
    destino = publicar(cfg, _modelo_final(cfg))
    r = coletar_resumo(cfg)
    assert r["publicado"]["id"] == destino.name
    from libras.app.janela import JanelaPrincipal

    w = JanelaPrincipal(cfg, transicao_ms=0)
    try:
        w.ir_para("inicio")
        p = w.paginas["inicio"]
        import time

        t0 = time.time()
        while destino.name not in p.lb_publicado.text() and time.time() - t0 < 10:
            qapp.processEvents()
        assert destino.name in p.lb_publicado.text() and "biblioteca" in p.lb_publicado.text()
    finally:
        w.close()
        w.deleteLater()
