"""Item 6: selecao de classes no treino. Landmarks SINTETICOS; numeros sem significado.

Causa do problema relatado: sem selecao salva, o padrao era o vocabulario inteiro, entao classes
sem nenhum video entravam marcadas e geravam erro. Agora o padrao sao as classes com videos que
atingem o minimo, e classes sem videos nunca entram na validacao.
"""

import os
import time

import pytest
import yaml

from conftest import criar_videos_com_landmarks
from libras.selecao import (
    avaliar_selecao,
    classes_treino,
    elegiveis,
    limpar_selecao,
    mensagem_minimo_classes,
    minimos,
    pode_marcar,
    situacao,
    tabela_classes,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_so_uma_classe_com_videos(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1, classes=["prova"])
    assert classes_treino(cfg) == ["prova"]
    av = avaliar_selecao(cfg)
    texto = " ".join(av.erros)
    assert "Hoje só 'prova' tem vídeos" in texto and "nenhum_sinal" in texto
    # nenhuma classe sem videos aparece como erro
    for c in ("biblioteca", "secretaria"):
        assert c not in texto
    t = {l.classe: l for l in av.linhas}
    assert not t["biblioteca"].selecionada and t["biblioteca"].bloqueio == ""


def test_nenhuma_classe_com_videos(cfg):
    assert classes_treino(cfg) == []
    av = avaliar_selecao(cfg)
    assert len(av.erros) == 1 and "Nenhuma classe tem vídeos" in av.erros[0]
    assert all(situacao(l, minimos(cfg)) == ("sem vídeos", False) for l in av.linhas)


def test_todas_as_classes_com_videos(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    assert classes_treino(cfg) == cfg.classes
    av = avaliar_selecao(cfg)
    assert av.pode_treinar and not av.erros and not av.avisos


def test_selecao_salva_obsoleta_e_ignorada(cfg):
    from libras.vocabulario import Vocabulario

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1, classes=["prova", "biblioteca"])
    Vocabulario(cfg).arquivar("biblioteca")
    from libras.vocabulario import _invalidar_cache

    _invalidar_cache(cfg)
    cfg["treino"]["classes"] = ["biblioteca", "cantina", "secretaria", "prova"]  # arquivada, inexistente, sem videos
    assert classes_treino(cfg) == ["prova"]
    assert limpar_selecao(cfg, cfg["treino"]["classes"]) == ["prova"]


def test_mensagens_de_minimo():
    assert "Hoje só 'prova' tem vídeos" in mensagem_minimo_classes(2, ["prova"], ["prova"])
    assert "Também têm vídeos: 'b'" in mensagem_minimo_classes(2, ["a"], ["a", "b"])
    assert "Nenhuma classe" in mensagem_minimo_classes(2, [], [])


def test_desmarcar_classe_bloqueada_e_sempre_possivel(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1, classes=["prova", "biblioteca"])
    criar_videos_com_landmarks(cfg, n_sinalizadores=1, por_classe=1, classes=["secretaria"], seed=50)
    tab = tabela_classes(cfg, contar_janelas=False)
    sec = next(l for l in tab if l.classe == "secretaria")
    mins = minimos(cfg)
    assert sec.sinalizadores == 1
    assert pode_marcar(sec, mins, ensaio=False)[0] is False
    assert pode_marcar(sec, mins, ensaio=True)[0] is True
    assert elegiveis(tab, mins) == ["biblioteca", "prova"]
    assert "secretaria" in elegiveis(tab, mins, ensaio=True)


# ---------------------------------------------------------------------- interface
@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    return app


def _abrir_treino(qapp, cfg):
    from libras.app.janela import JanelaPrincipal

    w = JanelaPrincipal(cfg)
    w.ir_para("treino")
    p = w.paginas["treino"]
    t = time.time()
    while not p.tabela_cls and time.time() - t < 30:
        qapp.processEvents()
        time.sleep(0.02)
    return w, p


def test_interface_so_uma_classe(qapp, cfg):
    from PySide6.QtCore import Qt

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1, classes=["prova"])
    w, p = _abrir_treino(qapp, cfg)
    try:
        assert p._selecionadas() == ["prova"]
        assert "Hoje só 'prova' tem vídeos" in p.lb_sel.text()
        assert "biblioteca" not in p.lb_sel.text().split("Hoje")[0].split("marcada(s):")[1]
        for i in range(p.t_cls.rowCount()):
            it = p.t_cls.item(i, 0)
            if it.text() != "prova":
                assert not (it.flags() & Qt.ItemIsEnabled) and it.checkState() == Qt.Unchecked
                assert p.t_cls.item(i, 5).text() == "sem vídeos"
        # clicar numa classe sem videos nao marca
        p.alternar_classe("biblioteca")
        assert p._selecionadas() == ["prova"]
        assert not p.bt_iniciar.isEnabled()
        p.limpar_selecao()
        assert p._selecionadas() == []
        p.marcar_elegiveis()
        assert p._selecionadas() == ["prova"]
    finally:
        w.deleteLater()


def test_interface_limpa_selecao_salva_e_desmarca_bloqueada(qapp, cfg):
    from libras.config import carregar_config, salvar_local

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1, classes=["prova", "biblioteca"])
    criar_videos_com_landmarks(cfg, n_sinalizadores=1, por_classe=1, classes=["secretaria"], seed=50)
    salvar_local(cfg, {"treino": {"classes": ["cantina", "secretaria", "prova", "nenhum_sinal"]}})
    cfg2 = carregar_config(cfg.arquivo)
    w, p = _abrir_treino(qapp, cfg2)
    try:
        local = yaml.safe_load(cfg2.arquivo.with_name("local.yaml").read_text(encoding="utf-8"))
        assert local["treino"]["classes"] == ["prova", "secretaria"]  # cantina e nenhum_sinal sairam
        assert p._selecionadas() == ["prova", "secretaria"]
        assert "abaixo do mínimo" in p.lb_sel.text()
        p.alternar_classe("secretaria")  # desmarcar a bloqueada: sempre permitido
        assert p._selecionadas() == ["prova"]
        p.alternar_classe("secretaria")  # marcar de novo: recusado sem ENSAIO
        assert p._selecionadas() == ["prova"] and "ENSAIO" in p.lb_sel.text()
        p.alternar_classe("biblioteca")
        assert p._selecionadas() == ["biblioteca", "prova"] and p.bt_iniciar.isEnabled()
        local = yaml.safe_load(cfg2.arquivo.with_name("local.yaml").read_text(encoding="utf-8"))
        assert local["treino"]["classes"] == ["biblioteca", "prova"]
    finally:
        w.deleteLater()
