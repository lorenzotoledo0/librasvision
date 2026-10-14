"""Item 10: teste ao vivo do modelo. Videos, landmarks e modelos SINTETICOS (o modelo minimo e
treinado com dados sem significado; serve so para testar o encanamento)."""

import io
import json
import os
import time

import numpy as np
import pytest

from conftest import ExtratorFalso, criar_videos_com_landmarks, gerar_video_sintetico, partes_sinteticas
from libras.teste_ao_vivo import (
    AVISO_TESTE,
    SEM_PREDICAO,
    SessaoTeste,
    aviso_sinalizador_visto,
    classe_prevista,
    etapa_decisao,
    listar_modelos,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
RAPIDO = {"treino": {"epocas": 1, "paciencia": 1}, "modelo": {"camadas_lstm": [4]}}


def _modelo_ficticio(cfg, nome, ensaio=False, sinalizadores=("S01",)):
    p = cfg.caminho("runs") / nome
    p.mkdir(parents=True)
    (p / "modelo.keras").write_bytes(b"SINTETICO")
    (p / "metadados.json").write_text(
        json.dumps(
            {
                "classes": ["a", "b"],
                "ensaio": ensaio,
                "validacao_por": "sinalizadores",
                "sinalizadores": list(sinalizadores),
                "videos": 4,
            }
        )
    )
    return p


# ---------------------------------------------------------------------- logica
def test_listar_modelos(cfg):
    assert listar_modelos(cfg) == []
    _modelo_ficticio(cfg, "20261001_100000_final")
    _modelo_ficticio(cfg, "20261005_090000_final", ensaio=True)
    (cfg.caminho("runs") / "20261006_000000_incompleto").mkdir()  # sem modelo: fora da lista
    ms = listar_modelos(cfg)
    assert [m.nome for m in ms] == ["20261005_090000_final", "20261001_100000_final"]
    assert ms[0].ensaio and "ENSAIO" in ms[0].rotulo() and "ENSAIO" not in ms[1].rotulo()
    assert ms[0].data == "05/10/2026 09:00" and ms[0].classes == ["a", "b"] and "sinalizadores" in ms[0].protocolo


def test_etapas_da_decisao():
    assert etapa_decisao("sem_maos", None, 0, 3, None)["texto"] == "Nenhuma mão visível"
    assert etapa_decisao("baixa_confianca", None, 0, 3, None)["etapa"] == "baixa_confianca"
    e = etapa_decisao("", "prova", 2, 3, None)
    assert e["etapa"] == "aguardando" and e["contador"] == 2 and "(2 de 3)" in e["texto"]
    assert etapa_decisao("", "prova", 3, 3, "prova")["etapa"] == "confirmado"


def test_sessao_acerto_erro_nenhum_sinal_e_matriz(cfg):
    s = SessaoTeste(["a", "b", "nenhum_sinal"], "nenhum_sinal", "m", ensaio=True)
    assert s.registrar("a", [0.5, 0.3, 0.2], "baixa_confianca")["acerto"]  # ignora o limiar de exibicao
    assert not s.registrar("b", [0.6, 0.3, 0.1], "")["acerto"]
    assert s.registrar("nenhum_sinal", None, "sem_maos")["previsto"] == "nenhum_sinal"  # sem sinalizar vale
    assert s.registrar("a", None, "sem_pose")["previsto"] == SEM_PREDICAO
    m = s.matriz()
    assert s.colunas == ["a", "b", "nenhum_sinal", SEM_PREDICAO]
    assert m.tolist() == [[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]] and m.sum() == len(s.tentativas)
    assert s.acuracia() == 0.5 and s.por_classe()["a"] == {"acertos": 1, "erros": 1}
    p = s.salvar(cfg, {"sinalizador_testado": "S09"})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert p.name.startswith("teste_") and p.parent.name == "tempo_real"
    assert d["aviso"] == AVISO_TESTE and "ENSAIO" in d["etiqueta"] and d["matriz_esperado_x_previsto"] == m.tolist()
    assert d["hardware"] and d["sinalizador_testado"] == "S09"
    s.zerar()
    assert s.acuracia() is None and s.matriz().sum() == 0


def test_aviso_de_sinalizador_visto_no_treino():
    meta = {"sinalizadores": ["S01", "S02"]}
    assert "otimista" in aviso_sinalizador_visto(meta, "S02") and "D6" in aviso_sinalizador_visto(meta, "S02")
    assert aviso_sinalizador_visto(meta, "S07") is None and aviso_sinalizador_visto(meta, None) is None


def test_classe_prevista_sem_nenhum_sinal_no_modelo():
    assert classe_prevista(None, "sem_maos", ["a", "b"], None) == SEM_PREDICAO


# ---------------------------------------------------------------------- pipeline real com modelo minimo
@pytest.fixture(scope="module")
def modelo_minimo(tmp_path_factory):
    import shutil

    from conftest import RAIZ
    from libras.config import carregar_config
    from libras.treino import OpcoesTreino, treinar_final

    raiz = tmp_path_factory.mktemp("proj_modelo")
    (raiz / "configs").mkdir()
    shutil.copy(RAIZ / "configs" / "config.yaml", raiz / "configs" / "config.yaml")
    cfg = carregar_config(raiz / "configs" / "config.yaml")
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    pasta = treinar_final(cfg, OpcoesTreino(nome="final", verbose=0, sobrescrever=RAPIDO))
    return cfg, pasta


class ExtratorPlausivel(ExtratorFalso):
    """SINTETICO: devolve, em sequencia, os quadros de partes_sinteticas (corpo e maos em posicoes
    plausiveis, com movimento), para a janela ser valida e o modelo rodar."""

    def __init__(self):
        super().__init__()
        self.p = partes_sinteticas(n=90)
        self.i = 0

    def processar(self, frame):
        from libras.landmarks import LandmarksFrame

        k = self.i % self.p.n
        self.i += 1
        p = self.p
        return LandmarksFrame(p.pose[k], p.mao_esq[k], p.mao_dir[k], p.face[k], p.presenca[k].copy()), None


def _rodar_servico(cfg, video, comandos=(), duracao=6.0):
    from libras.camera import abrir_arquivo_como_camera
    from libras.tempo_real_servico import Decodificador, executar_servico

    saida = io.BytesIO()

    def entrada():
        for atraso, cmd in comandos:
            time.sleep(atraso)
            yield cmd + "\n"
        time.sleep(duracao)
        yield "sair\n"

    executar_servico(cfg, saida, entrada(), abrir_camera=lambda _c: abrir_arquivo_como_camera(video))
    return Decodificador().alimentar(saida.getvalue())


def test_tentativas_com_video_e_modelo_minimo(modelo_minimo, tmp_path, monkeypatch):
    import libras.landmarks as lm

    cfg, pasta = modelo_minimo
    video = gerar_video_sintetico(tmp_path / "teste.mp4", n_frames=90)
    monkeypatch.setattr(lm, "ExtratorHolistic", lambda *a, **k: ExtratorPlausivel())  # maos "visiveis"
    msgs = _rodar_servico(cfg, video, [(2.5, "parametros limiar=0.95 suavizacao=2 confirmacoes=4")])
    info = next(m for m, _ in msgs if m["tipo"] == "info")
    assert info["classes"] == cfg.classes and info["parametros"]["confirmacoes"] == 3
    quadros = [m for m, _ in msgs if m["tipo"] == "quadro" and m.get("probs")]
    assert quadros and all(len(q["probs"]) == len(info["classes"]) for q in quadros)
    assert all("texto" in q["etapa"] for q in quadros)
    ult = quadros[-1]
    s = SessaoTeste(info["classes"], info["nome_nenhum_sinal"], info["modelo"])
    previsto = info["classes"][int(np.argmax(ult["probs"]))]
    assert s.registrar(previsto, ult["probs"], ult["motivo"])["acerto"]
    outra = next(c for c in info["classes"] if c != previsto)
    assert not s.registrar(outra, ult["probs"], ult["motivo"])["acerto"]
    # o ajuste de parametros vale so no processo: o config nao muda
    assert cfg["tempo_real"]["limiar_confianca"] != 0.95
    etapas = {q["etapa"]["etapa"] for q in quadros[-5:]}
    assert etapas <= {"baixa_confianca", "nenhum_sinal", "aguardando", "confirmado"}
    # pessoa sem sinalizar (sem maos): a tentativa de nenhum_sinal vale
    monkeypatch.setattr(lm, "ExtratorHolistic", lambda *a, **k: ExtratorFalso(mao=False))
    msgs = _rodar_servico(cfg, video, duracao=3.0)
    q = [m for m, _ in msgs if m["tipo"] == "quadro"][-1]
    assert q["motivo"] == "sem_maos" and q["etapa"]["texto"] == "Nenhuma mão visível"
    assert s.registrar("nenhum_sinal", q.get("probs"), q["motivo"])["acerto"]


# ---------------------------------------------------------------------- interface
@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    return app


def esperar(app, cond, limite=15.0):
    t = time.time()
    while not cond() and time.time() - t < limite:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_interface_sem_modelo(qapp, cfg):
    from libras.app.janela import JanelaPrincipal

    w = JanelaPrincipal(cfg, transicao_ms=0)
    try:
        w.ir_para("tempo_real")
        tr = w.paginas["tempo_real"]
        assert tr.cb_modelo.count() == 0 and not tr.bt_aplicar.isEnabled() and tr.bt_ir_treino.isVisibleTo(tr)
        assert esperar(qapp, lambda: tr.sinal.text() == "Sem modelo", 60)
        assert "Treine o modelo final" in tr.tela.text()
        tr._ir_treino_final()
        assert w.atual == "treino" and w.paginas["treino"].cb_modo.currentData() == "final"
    finally:
        w.close()
        w.deleteLater()


def test_interface_testar_acertos(qapp, cfg, tmp_path, monkeypatch):
    import libras.app.paginas.tempo_real as trm
    from libras.app.janela import JanelaPrincipal
    from libras.coleta.sinalizadores import CadastroSinalizadores

    CadastroSinalizadores(cfg).cadastrar("direita", True, "2026-10-01")
    _modelo_ficticio(cfg, "20261005_090000_final", ensaio=True)
    cmds = tmp_path / "comandos.txt"
    monkeypatch.setenv("LIBRAS_TESTE_COMANDOS", str(cmds))
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(trm, "MODULO_SERVICO", "servico_falso_teste")
    w = JanelaPrincipal(cfg, transicao_ms=0)
    w.show()
    try:
        w.ir_para("tempo_real")
        tr = w.paginas["tempo_real"]
        assert tr.cb_modelo.count() == 1 and "ENSAIO" in tr.cb_modelo.itemText(0)
        assert esperar(qapp, lambda: bool(tr.ultimo) and tr.sessao is not None)
        assert not tr.et_ensaio.isHidden() and "ENSAIO" in tr.et_ensaio.text() and "ENSAIO" in tr.lb_aviso_teste.text()
        assert tr.barras.classes == ["a", "b", "nenhum_sinal"] and tr.barras.probs == [0.7, 0.2, 0.1]
        assert "Aguardando predições iguais (1 de 3)" in tr.etapa.text()
        # vista principal: so a camera com a legenda; diagnostico e teste ficam ocultos ate pedir
        assert tr.col_teste.isHidden() and not tr.modo_teste
        assert tr.tela.parcial_legenda == "Em análise: a" and tr.tela.texto_legenda == ""
        tr.atalho_t.activated.emit()  # fora do modo teste o atalho T nao registra nada
        assert not tr.sessao.tentativas
        tr.atalho_d.activated.emit()
        assert tr.modo_teste and tr.col_teste.isVisible() and tr.bt_modo.isChecked()
        tr.cb_quem.setCurrentText("S01")
        assert not tr.lb_visto.isHidden() and "otimista" in tr.lb_visto.text()
        tr.cb_esperado.setCurrentText("a")
        tr.atalho_t.activated.emit()
        tr.cb_esperado.setCurrentText("b")
        tr.registrar_tentativa()
        assert tr.t_matriz.item(0, 1).text() == "1" and tr.t_matriz.item(1, 1).text() == "1"
        assert "50%" in tr.lb_acc.text()
        tr.sp_limiar.setValue(0.8)
        assert esperar(qapp, lambda: cmds.exists() and "parametros limiar=0.80" in cmds.read_text())
        p = tr.salvar_sessao()
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["aviso"] == AVISO_TESTE and d["ensaio"] and d["sinalizador_testado"] == "S01"
        assert d["parametros"]["limiar"] == 0.8 and "otimista" in d["aviso_sinalizador_visto"]
        tr.alternar_modo_teste(False)
        assert tr.col_teste.isHidden() and tr.bt_modo.text() == "Diagnóstico e teste"
        w.ir_para("inicio")  # sair da aba encerra o processo
        assert tr.proc is None
    finally:
        w.close()
        w.deleteLater()


def test_aba_treino_treina_o_modelo_final(qapp, cfg):
    """10.1: a aba Treino treina o modelo final (subprocesso com progresso) e ele aparece no Tempo real."""
    from libras.app.janela import JanelaPrincipal

    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    w = JanelaPrincipal(cfg, transicao_ms=0)
    w.show()
    try:
        w.ir_para("treino")
        t = w.paginas["treino"]
        assert esperar(qapp, lambda: bool(t.tabela_cls), 30)
        t.preparar_modelo_final()
        t.epocas.setValue(2)
        t.iniciar()
        assert t.proc is not None and t.proc.rodando()
        assert esperar(qapp, lambda: not t.proc.rodando(), 300), t.log.toPlainText()[-2000:]
        assert t.curvas.terminou and t.curvas.inicio.get("protocolo") == "final"
        assert t.t_teste.rowCount() == 1 and "modelo final" in t.t_teste.item(0, 2).text()
        modelos = listar_modelos(cfg)
        assert len(modelos) == 1 and modelos[0].classes == cfg.classes and not modelos[0].ensaio
        assert "todos os sinalizadores" in modelos[0].protocolo
        w.ir_para("tempo_real")
        tr = w.paginas["tempo_real"]
        assert tr.cb_modelo.count() == 1 and tr.bt_aplicar.isEnabled() and tr.bt_ir_treino.isHidden()
    finally:
        w.close()
        w.deleteLater()


# ---------------------------------------------------------------------- legenda sobre o video
def test_legenda_do_quadro():
    from libras.app.paginas.tempo_real import legenda_do_quadro

    assert legenda_do_quadro({"texto": ["oi", "tudo"], "etapa": {"etapa": "confirmado"}}) == ("oi tudo", "")
    assert legenda_do_quadro({"texto": [], "etapa": {"etapa": "sem_maos"}}) == ("", "Mostre as mãos para a câmera")
    assert legenda_do_quadro(
        {"etapa": {"etapa": "aguardando", "texto": "Aguardando predições iguais (1 de 3): a"}}
    ) == ("", "Em análise: a")
    assert legenda_do_quadro({"motivo": "aquecendo"}) == ("", "Preparando...")
    assert legenda_do_quadro({"texto": ["oi"], "etapa": {"etapa": "baixa_confianca"}}) == ("oi", "")


def test_cauda_que_cabe():
    from libras.app.paginas.tempo_real import cauda_que_cabe

    assert cauda_que_cabe(["a", "b"], lambda s: True) == "a b"
    assert cauda_que_cabe(["um", "dois", "tres", "quatro"], lambda s: len(s) <= 15) == "... tres quatro"
    assert cauda_que_cabe(["palavralonga"], lambda s: False) == "palavralonga"
    assert cauda_que_cabe([], lambda s: False) == ""


def test_legenda_desenhada_sobre_a_imagem(qapp):
    """SINTETICO: imagem branca; a legenda (o mesmo componente do app real, roll-up de 2 linhas) escurece a
    faixa de baixo e mostra so as ultimas linhas."""
    from PySide6.QtGui import QColor, QPixmap

    from libras.app.paginas.tempo_real import TelaLegendada

    tela = TelaLegendada("", proporcao=4 / 3)
    tela.resize(640, 480)
    pm = QPixmap(640, 480)
    pm.fill(QColor(255, 255, 255))
    tela.setPixmap(pm)
    for i in range(40):
        tela.camada.rollup.adicionar(f"palavra{i}")
    tela.camada.parcial = "Em análise: oi"
    img = tela.grab().toImage()
    linhas = tela.legenda_visivel.split("\n")
    assert len(linhas) == 2 and linhas[-1].endswith("palavra39") and "palavra0 " not in tela.legenda_visivel
    assert tela.texto_legenda.startswith("palavra0 palavra1")  # o texto completo continua guardado
    assert QColor(img.pixel(320, 30)).lightness() == 255  # topo da imagem intacto (sobre a pessoa nao)
    escuros = sum(QColor(img.pixel(x, y)).lightness() < 100 for x in range(0, 640, 8) for y in range(330, 480, 8))
    assert escuros > 0.2 * (80 * 19)  # caixa escura translucida na parte de baixo
    tela.limpar()
    tela.grab()
    assert tela.legenda_visivel == "" and tela.camada.ultimo_retangulo is None
