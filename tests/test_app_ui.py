"""Interface (PySide6) em modo offscreen. Dados e videos SINTETICOS; MediaPipe substituido por
ExtratorFalso nos fluxos de gravacao (testa o encanamento da interface, nao a deteccao)."""

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from conftest import ExtratorFalso, cadastrar, criar_videos_com_landmarks, gerar_video_sintetico  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    return app


def esperar(app, cond, limite=30.0):
    t = time.time()
    while not cond() and time.time() - t < limite:
        app.processEvents()
        time.sleep(0.02)
    return cond()


@pytest.fixture
def janela(qapp, cfg, monkeypatch):
    import libras.app.base as base
    from libras.app.janela import JanelaPrincipal

    # a confirmacao de "sair com gravacao em andamento" e modal: num teste que falhou, travaria o fechamento
    monkeypatch.setattr(base, "perguntar", lambda *a, **k: True)
    w = JanelaPrincipal(cfg)
    yield w
    w.close()  # closeEvent: encerra paginas, subprocessos e espera as tarefas em segundo plano
    w.deleteLater()


def test_todas_as_paginas_abrem(qapp, cfg, janela):
    from libras.app.janela import PAGINAS

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    for chave, _, _ in PAGINAS:
        if chave == "tempo_real":
            continue
        janela.ir_para(chave)
        qapp.processEvents()
        assert janela.pilha.currentWidget() is janela.paginas[chave]
    janela.ir_para("inicio")
    assert esperar(qapp, lambda: janela.paginas["inicio"].c_sig[1].text() == "2")


def test_cadastro_e_vocabulario_pela_interface(qapp, cfg, janela):
    from libras.coleta.sinalizadores import CadastroSinalizadores
    from libras.vocabulario import Vocabulario

    janela.ir_para("sinalizadores")
    p = janela.paginas["sinalizadores"]
    p.tcle.setChecked(True)
    QtWidgets.QMessageBox.information = lambda *a, **k: None
    p.cadastrar()
    assert CadastroSinalizadores(cfg).ids_com_tcle() == ["S01"]
    janela.ir_para("vocabulario")
    v = janela.paginas["vocabulario"]
    v.novo.setText("Sala de Aula")
    v.adicionar()
    assert "sala_de_aula" in Vocabulario(cfg).slugs()
    assert v.tabela.rowCount() == 5


def test_gravar_aceitar_e_cortar(qapp, cfg, janela, tmp_path, monkeypatch):
    import libras.coleta.previa as previa
    import libras.landmarks as lm
    from libras.coleta.manifesto import Manifesto

    monkeypatch.setattr(lm, "ExtratorHolistic", lambda *a, **k: ExtratorFalso())
    monkeypatch.setattr(previa, "ExtratorHolistic", lambda *a, **k: ExtratorFalso())
    video = gerar_video_sintetico(tmp_path / "cam.mp4", n_frames=40)
    monkeypatch.setenv("LIBRAS_CAMERA_VIDEO", str(video))
    cfg["coleta"]["gravacao"]["contagem_regressiva_s"] = 0
    cfg["coleta"]["gravacao"]["duracao_s"] = 2.0
    cadastrar(cfg)
    janela.ir_para("gravar")
    g = janela.paginas["gravar"]
    g.ligar()
    assert esperar(qapp, lambda: g.estado == "previa", 20)
    g.iniciar_take()
    assert esperar(qapp, lambda: g.estado == "revisao", 40), g.status.text()
    g.aceitar()
    linhas = Manifesto(cfg).linhas
    assert len(linhas) == 1 and linhas[0]["landmarks_path"] and linhas[0]["sha256_npz"]
    g.desligar()
    n = int(linhas[0]["n_frames"])
    janela.ir_para("revisao")
    r = janela.paginas["revisao"]
    import libras.app.paginas.revisao as rev

    monkeypatch.setattr(rev, "perguntar", lambda *a, **k: True)
    r.carregar(linhas[0]["video_id"])
    r.editor.ir(5)
    r.editor.definir_inicio_aqui()
    assert not Manifesto(cfg).linhas[0]["corte"]  # nada gravado antes de salvar
    r.editor.salvar()
    assert esperar(qapp, lambda: Manifesto(cfg).linhas[0]["corte"] != "", 20)
    assert int(Manifesto(cfg).linhas[0]["n_frames"]) == n - 5


def _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch):
    """Camera sintetica e um contador de extratores criados (cada take da rajada cria um grafo novo)."""
    import libras.coleta.previa as previa
    import libras.landmarks as lm

    criados = []
    monkeypatch.setattr(lm, "ExtratorHolistic", lambda *a, **k: criados.append(1) or ExtratorFalso())
    monkeypatch.setattr(previa, "ExtratorHolistic", lambda *a, **k: ExtratorFalso())
    video = gerar_video_sintetico(tmp_path / "cam.mp4", n_frames=40)
    monkeypatch.setenv("LIBRAS_CAMERA_VIDEO", str(video))
    g_cfg = cfg["coleta"]["gravacao"]
    g_cfg["contagem_regressiva_s"], g_cfg["intervalo_rajada_s"], g_cfg["duracao_s"] = 0, 0.3, 0.5
    cadastrar(cfg)
    janela.ir_para("gravar")
    g = janela.paginas["gravar"]
    g.ligar()
    assert esperar(qapp, lambda: g.estado == "previa", 20)
    return g, criados


def test_rajada_extrai_so_no_fim_e_revisa_antes_de_salvar(qapp, cfg, janela, tmp_path, monkeypatch):
    from libras.coleta.manifesto import Manifesto

    g, criados = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    g.sp_rajada.setValue(3)
    g.iniciar_rajada()
    # enquanto a rajada grava, nenhum take e extraido nem salvo; a legenda mostra o progresso da meta
    assert esperar(qapp, lambda: len(g._fila) == 2 and g.estado == "intervalo", 30)
    assert criados == [] and len(Manifesto(cfg).linhas) == 0
    assert "S01/biblioteca: 2 de 10" in g._texto_meta_rajada()
    # depois da extracao vem a revisao rapida: nada salvo ate a pessoa decidir
    assert esperar(qapp, lambda: g.estado == "revisao_rajada", 60), g.status.text()
    assert len(criados) == 3 and len(Manifesto(cfg).linhas) == 0
    assert not g.lista_rajada.isHidden() and g.lista_rajada.count() == 3
    g.aceitar()  # take 1: manter
    g.descartar()  # take 2: descartar
    g.aceitar()  # take 3: manter
    assert [t["manter"] for t in g._revisar] == [True, False, True]
    g._salvar_revisados()
    assert esperar(qapp, lambda: g.estado == "previa" and not g._em_rajada, 30), g.status.text()
    linhas = Manifesto(cfg).linhas
    assert len(linhas) == 2 and all(l["landmarks_path"] and l["sha256_npz"] for l in linhas)
    assert [int(l["take"]) for l in linhas] == [1, 2]
    assert "2 de 3 takes salvos" in g.status.text() and "1 descartados" in g.status.text()


def test_rajada_marcar_e_desmarcar_todos(qapp, cfg, janela, tmp_path, monkeypatch):
    from libras.coleta.manifesto import Manifesto

    g, _ = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    g.sp_rajada.setValue(2)
    g.iniciar_rajada()
    assert esperar(qapp, lambda: g.estado == "revisao_rajada", 60), g.status.text()
    g._marcar_rajada(False)
    assert not any(t["manter"] for t in g._revisar)
    g._salvar_revisados()
    assert esperar(qapp, lambda: g.estado == "previa", 30)
    assert len(Manifesto(cfg).linhas) == 0 and "2 descartados" in g.status.text()


def test_parar_rajada_processa_o_que_foi_gravado(qapp, cfg, janela, tmp_path, monkeypatch):
    from libras.coleta.manifesto import Manifesto

    g, criados = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    g.sp_rajada.setValue(10)
    g.iniciar_rajada()
    assert esperar(qapp, lambda: len(g._fila) == 1 and g.estado == "intervalo", 30)
    g.parar_rajada()
    assert esperar(qapp, lambda: g.estado == "revisao_rajada", 60), g.status.text()
    g._salvar_revisados()
    assert esperar(qapp, lambda: g.estado == "previa" and not g._em_rajada, 30), g.status.text()
    assert len(criados) == 1 and len(Manifesto(cfg).linhas) == 1


def test_parar_e_descartar_o_take_em_gravacao(qapp, cfg, janela, tmp_path, monkeypatch):
    g, criados = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    cfg["coleta"]["gravacao"]["duracao_s"] = 3.0  # tempo para pegar o segundo take no meio
    g.sp_rajada.setValue(5)
    g.iniciar_rajada()
    assert esperar(qapp, lambda: len(g._fila) == 1 and g.estado == "gravando", 30)
    assert g.bt_parar_desc.isEnabled()
    g._tecla_esc()  # Esc durante a gravacao da rajada: para e descarta o take atual
    assert esperar(qapp, lambda: g.estado == "revisao_rajada", 60), g.status.text()
    assert len(g._revisar) == 1 and len(criados) == 1


def test_campo_da_rajada_segue_a_memoria(qapp, cfg, janela, tmp_path, monkeypatch):
    g, _ = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    assert g.sp_rajada.maximum() == min(50, g._takes_que_cabem())
    cfg["coleta"]["gravacao"]["memoria_max_rajada_mb"] = 60  # 640x480 por 0,5 s: cerca de 13 MB por take
    g._ajustar_limite_rajada()
    assert g.sp_rajada.maximum() == g._takes_que_cabem() == 4


def test_rajada_respeita_limite_de_memoria(qapp, cfg, janela, tmp_path, monkeypatch):
    import libras.app.paginas.gravar as gravar

    g, _ = _preparar_rajada(qapp, cfg, janela, tmp_path, monkeypatch)
    cfg["coleta"]["gravacao"]["memoria_max_rajada_mb"] = 1  # 640x480 por 0,5 s nao cabe nem 2 takes
    avisos = []
    monkeypatch.setattr(gravar, "erro", lambda *a, **k: avisos.append(a[1]))
    g.sp_rajada.setValue(5)
    g.iniciar_rajada()
    assert avisos and "cabem até 1 takes" in avisos[0]
    assert g.estado == "previa" and not g._em_rajada


def test_revisao_em_lote(qapp, cfg, janela, monkeypatch):
    import libras.app.paginas.revisao as rev
    from libras.coleta.manifesto import Manifesto

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=2)
    man = Manifesto(cfg)
    ids = [l["video_id"] for l in man.linhas]
    man.definir_status_lote(ids[:3], "pendente")
    monkeypatch.setattr(rev, "perguntar", lambda *a, **k: True)
    janela.ir_para("revisao")
    r = janela.paginas["revisao"]
    r.f_status.setCurrentText("todos")
    r._marcar("pendente")
    assert r.marcados == set(ids[:3]) and "3 marcado" in r.lb_marcados.text()
    r._status_lote("ok")
    assert all(l["status"] == "ok" for l in Manifesto(cfg).linhas)
    r._marcar("todos")
    assert len(r.marcados) == len(ids)
    r._status_lote("rejeitado")
    assert all(l["status"] == "rejeitado" for l in Manifesto(cfg).linhas)
    r._marcar(None)
    assert not r.marcados and not r.bt_aprovar_lote.isEnabled()


def test_curvas_ao_vivo_sem_teste_por_epoca(qapp, cfg, janela, tmp_path):
    janela.ir_para("treino")
    t = janela.paginas["treino"]
    arq = tmp_path / "prog.jsonl"
    eventos = [
        {
            "tipo": "inicio",
            "seeds": [1],
            "folds": [0, 1],
            "classes": ["a", "b"],
            "protocolo": "x",
            "execucao": "20261006_101010_teste",
        }
    ]
    for ep in range(1, 6):
        eventos.append(
            {
                "tipo": "epoca",
                "seed": 1,
                "fold": 0,
                "epoca": ep,
                "loss": 1 / ep,
                "accuracy": ep / 10,
                "val_loss": abs(ep - 3) + 0.5,
                "val_accuracy": 0.5,
                "execucao": "20261006_101010_teste",
            }
        )
    t.arq_prog, t._offset = arq, 0
    arq.write_text("\n".join(json.dumps(e) for e in eventos) + "\n", encoding="utf-8")
    t._ler_progresso()
    k = (1, 0, "20261006_101010_teste")
    assert t.cb_curva.count() == 1 and "teste | seed 1" in t.cb_curva.itemText(0)
    assert t.curvas.melhor_epoca(k) == 3  # menor val_loss
    assert all("teste" not in c for c in t.curvas.series[k])  # a curva so tem treino e validacao
    assert t.t_teste.rowCount() == 0
    fim = {
        "tipo": "fim_particao",
        "seed": 1,
        "fold": 0,
        "melhor_epoca": 3,
        "epocas": 5,
        "classes": ["a", "b"],
        "sinalizadores_teste": ["S02"],
        "teste_janela": {"acuracia": 0.5, "f1_macro": 0.4},
        "teste_video": {"acuracia": 0.6, "f1_macro": 0.5},
        "matriz_video": [[1, 0], [1, 1]],
        "matriz_janela": [[2, 1], [1, 2]],
        "execucao": "20261006_101010_teste",
    }
    with open(arq, "a", encoding="utf-8") as f:
        f.write(json.dumps(fim) + "\n" + '{"tipo": "epoca", "incomplet')  # linha incompleta: ignorada
    t._ler_progresso()
    assert t.t_teste.rowCount() == 1 and t.t_teste.item(0, 2).text() == "S02"
    t.t_teste.selectRow(0)


def test_treino_bloqueia_classe_abaixo_do_minimo(qapp, cfg, janela, monkeypatch):
    from libras.coleta.manifesto import Manifesto

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    m = Manifesto(cfg)
    for l in m.linhas:
        if l["classe"] == "prova" and l["signer_id"] == "S02":
            m.definir_status(l["video_id"], "rejeitado")
    janela.ir_para("treino")
    t = janela.paginas["treino"]
    assert esperar(qapp, lambda: bool(t.tabela_cls), 30)
    erros = []
    import libras.app.paginas.treino as tr

    monkeypatch.setattr(tr, "erro", lambda _p, e: erros.append(str(e)))
    assert "prova" not in t._selecionadas()  # abaixo do minimo: fora por padrao (item 6)
    t.alternar_classe("prova")
    assert "prova" not in t._selecionadas()  # marcar e recusado sem ENSAIO
    t.marcadas.add("prova")  # selecao forcada (ex.: salva antes): o Iniciar recusa
    t._preencher_classes()
    t.iniciar()
    assert erros and "prova" in erros[0] and t.proc is None


def test_tempo_real_sem_modelo_informa(qapp, cfg, janela):
    janela.ir_para("tempo_real")
    tr = janela.paginas["tempo_real"]
    assert esperar(qapp, lambda: tr.sinal.text() == "Sem modelo", 60), tr.tela.text()
    assert "--final" in tr.tela.text()
    janela.ir_para("inicio")
    assert tr.proc is None
