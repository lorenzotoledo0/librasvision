"""13.3, 13.4, 13.8 e 13.9: tela do app real (offscreen) com um servico de tempo real FALSO (SINTETICO:
imagem cinza e palavras ficticias, nao e predicao de modelo). Pacotes de modelo SINTETICOS."""

import itertools
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from libras import pacote as pk  # noqa: E402
from libras.utils import sha256_arquivo  # noqa: E402

TESTES = Path(__file__).resolve().parent
CLASSES = ["biblioteca", "prova", "secretaria", "nenhum_sinal"]


def pacote_sintetico(base: Path, ident: str, publicado_em: str) -> Path:
    p = base / ident
    p.mkdir(parents=True)
    (p / "modelo.keras").write_bytes(b"SINTETICO " + ident.encode())
    (p / "metadados.json").write_text(json.dumps({"classes": CLASSES, "nome_nenhum_sinal": "nenhum_sinal"}))
    cartao = {
        "formato": pk.FORMATO,
        "id": ident,
        "classes": CLASSES,
        "nome_nenhum_sinal": "nenhum_sinal",
        "rotulos": {"biblioteca": "Biblioteca", "prova": "Prova", "secretaria": "Secretaria"},
        "publicado_em": publicado_em,
        "criado_em": "2026-10-06T10:00:00",
        "protocolo": "final",
        "ensaio": False,
        "decisao": {},
        "versoes": {"mediapipe": "0.10.21"},
        "hash": {"algoritmo": "sha256", **{n: sha256_arquivo(p / n) for n in pk.ARQUIVOS_COM_HASH}},
    }
    (p / "cartao.json").write_text(json.dumps(cartao))
    return p


@pytest.fixture(scope="module")
def qapp():
    from libras.app_real.janela import folha_extra
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo() + folha_extra())
    return app


def esperar(app, cond=lambda: False, limite=10.0):
    t = time.time()
    while not cond() and time.time() - t < limite:
        app.processEvents()
        time.sleep(0.02)
    return cond()


class Dialogos:
    """Substitui os dialogos (que bloqueiam) e guarda o que seria mostrado."""

    def __init__(self):
        self.textos = []

    def __call__(self, parent, titulo, texto, tipo="info", *a, **k):
        reg = self

        class D:
            def exec(self_):
                reg.textos.append((titulo, texto))

        return D()


@pytest.fixture
def raiz_app(tmp_path):
    """Pasta do app real SEM data/ nem runs/ (so configs/ e publicados/)."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "app_real.yaml").write_text("pasta_modelos: publicados\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def abrir(qapp, raiz_app, monkeypatch, tmp_path):
    import libras.app_real.janela as jm
    from libras import config_real
    from libras.app_real.preferencias import Preferencias

    monkeypatch.setattr(jm, "MODULO_SERVICO", "servico_falso_real")
    monkeypatch.setenv("PYTHONPATH", str(TESTES) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("LIBRAS_TESTE_COMANDOS", str(tmp_path / "comandos.txt"))
    dlg = Dialogos()
    monkeypatch.setattr(jm, "Dialogo", dlg)
    janelas = []

    def _abrir(largura=1366, altura=800, guia_visto=True, iniciar=True, perguntar=lambda *a, **k: True):
        prefs = Preferencias.da_raiz(raiz_app)
        prefs.dados["guia_visto"] = guia_visto
        prefs.salvar()
        w = jm.JanelaReal(config_real.carregar(raiz_app / "configs" / "app_real.yaml"), prefs)
        w.dialogos = dlg
        w.resize(largura, altura)
        w.show()
        janelas.append(w)
        if iniciar:
            w.iniciar(perguntar)
        return w

    yield _abrir
    for w in janelas:
        w.close()
        w.deleteLater()
    qapp.processEvents()


def ativo(app, w, limite=15.0):
    return esperar(app, lambda: w.estado == "ativo" and w.vista.quadro is not None, limite)


# ---------------------------------------------------------------------- modelo (13.2 no app real)
def test_sem_modelo_explica_e_importa_pacote(qapp, abrir, raiz_app, tmp_path):
    w = abrir()
    assert w.estado == "sem_modelo" and w.vista.cartao.isVisible()
    assert "Importar pacote" in [b.text() for b in w.vista.cartao.botoes][0]
    assert "Nada é reconhecido" in w.vista.cartao.texto.text()
    origem = pacote_sintetico(tmp_path / "origem", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    zp = pk.compactar(origem, tmp_path / "pacote.zip")
    assert w.importar_pacote(str(zp))
    assert ativo(qapp, w), w.estado
    assert (raiz_app / "publicados" / "20261007_100000_aaaa1111").is_dir()
    assert not (raiz_app / "data").exists() and not (raiz_app / "runs").exists()  # funciona sem data/ e runs/


def test_pacote_alterado_mensagem_clara(qapp, abrir, raiz_app):
    p = pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    from libras.app_real.preferencias import Preferencias

    pr = Preferencias.da_raiz(raiz_app)
    pr.dados["modelo"]["ativo"] = p.name
    pr.salvar()
    with open(p / "modelo.keras", "ab") as f:
        f.write(b"x")
    w = abrir()
    assert w.estado == "sem_modelo" and "hash diferente" in w.vista.cartao.texto.text()


def test_pacote_escolhido_alterado_nao_troca_calado(qapp, abrir, raiz_app):
    """Com outro pacote valido na pasta, o escolhido alterado mostra o problema e deixa o usuario escolher."""
    from libras.app_real.preferencias import Preferencias

    ruim = pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    pacote_sintetico(raiz_app / "publicados", "20261001_100000_bbbb2222", "2026-10-01T10:00:00")
    pr = Preferencias.da_raiz(raiz_app)
    pr.dados["modelo"]["ativo"] = ruim.name
    pr.salvar()
    with open(ruim / "modelo.keras", "ab") as f:
        f.write(b"x")
    w = abrir()
    assert w.estado == "sem_modelo" and "hash diferente" in w.vista.cartao.texto.text()
    assert w.prefs["modelo"]["ativo"] == ruim.name  # nada foi trocado sem perguntar
    usar = [b for b in w.vista.cartao.botoes if "20261001_100000_bbbb2222" in b.text()]
    assert usar
    usar[0].click()
    assert w.prefs["modelo"]["ativo"] == "20261001_100000_bbbb2222" and w.prefs["modelo"]["anterior"] == ruim.name
    assert ativo(qapp, w)


def test_tentar_de_novo_encerra_o_processo_anterior(qapp, abrir, raiz_app):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    assert ativo(qapp, w)
    antigo = w.proc
    w.painel._tentar_camera()  # Configuracoes > Camera > Tentar de novo com a camera ligada
    assert esperar(qapp, lambda: not antigo.rodando(), 10)  # a camera nao fica com dois processos
    assert esperar(qapp, lambda: w.proc is not None and w.proc is not antigo, 10)  # o novo so abre depois
    assert ativo(qapp, w)


def test_oferece_pacote_novo_e_volta_ao_anterior(qapp, abrir, raiz_app):
    from libras.app_real.preferencias import Preferencias

    pacote_sintetico(raiz_app / "publicados", "20261001_100000_aaaa1111", "2026-10-01T10:00:00")
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_bbbb2222", "2026-10-07T10:00:00")
    pr = Preferencias.da_raiz(raiz_app)
    pr.dados["modelo"]["ativo"] = "20261001_100000_aaaa1111"
    pr.salvar()
    perguntas = []
    w = abrir(perguntar=lambda *a, **k: perguntas.append(a[1]) or True)
    assert perguntas and "modelo novo" in perguntas[0] and "Biblioteca" in perguntas[0]
    assert w.cartao_ativo["id"] == "20261007_100000_bbbb2222" and w.modelos.pode_voltar()
    assert ativo(qapp, w)
    w.voltar_modelo_anterior()
    assert w.cartao_ativo["id"] == "20261001_100000_aaaa1111"
    # recusar: nao pergunta de novo
    pr2 = Preferencias.da_raiz(raiz_app)
    w.close()
    w2 = abrir(perguntar=lambda *a, **k: False)
    assert w2.cartao_ativo["id"] == "20261001_100000_aaaa1111"
    assert "20261007_100000_bbbb2222" in w2.prefs["modelo"]["recusados"]
    assert pr2  # preferencias em arquivo proprio
    assert (raiz_app / "configs" / "app_real_preferencias.json").exists()


# ---------------------------------------------------------------------- legenda, pausa e copiar
def test_legenda_rotulos_pausa_buffer_e_parar(qapp, abrir, raiz_app, monkeypatch):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    monkeypatch.setenv("LIBRAS_FALSO_PALAVRAS", "biblioteca,prova,secretaria,prova")
    w = abrir()
    assert ativo(qapp, w)
    assert esperar(qapp, lambda: "Biblioteca" in w.camada.rollup.texto_completo(), 6)  # rotulo do cartao
    w.alternar_pausa()
    congelado = w.camada.rollup.texto_visivel()
    assert w.camada.rollup.pausado and "Retomar" in w.barra.bt_pausa.toolTip()
    assert esperar(qapp, lambda: len(w.camada.rollup.buffer) >= 2, 6)
    assert w.camada.parcial_efetivo().startswith("Legenda pausada") and "novas" in w.camada.parcial_efetivo()
    assert w.copiar_texto() == congelado == QtWidgets.QApplication.clipboard().text()
    w.alternar_pausa()
    assert (
        w.camada.rollup.texto_completo().split()
        == ["Biblioteca", "Prova", "Secretaria", "Prova"][: len(w.camada.rollup.texto_completo().split())]
    )
    w.alternar_pausa()
    w.parar_camera()  # parar a camera encerra a pausa
    assert not w.camada.rollup.pausado and w.estado == "parado"


def test_limpar_legenda_avisa_o_servico(qapp, abrir, raiz_app, tmp_path):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    assert ativo(qapp, w)
    w.camada.rollup.adicionar("oi")
    w.limpar_legenda()
    assert w.camada.rollup.texto_completo() == ""
    assert esperar(
        qapp, lambda: (tmp_path / "comandos.txt").exists() and "limpar" in (tmp_path / "comandos.txt").read_text()
    )


def test_legenda_nao_e_espelhada(qapp, abrir, raiz_app, monkeypatch):
    from libras.ui.legenda import CamadaLegenda

    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    assert ativo(qapp, w)
    escalas = []
    original = CamadaLegenda.desenhar

    def espiao(self, p, caixa, t=None):
        escalas.append(p.transform().m11())
        return original(self, p, caixa, t)

    monkeypatch.setattr(CamadaLegenda, "desenhar", espiao)
    w.camada.rollup.adicionar("Prova")
    w.vista.grab()
    assert escalas and all(s > 0 for s in escalas)  # nenhuma inversao horizontal no texto


# ---------------------------------------------------------------------- faixas, selo e barra minima
@pytest.mark.parametrize("largura,altura", [(1366, 800), (900, 700), (390, 844)])
def test_tela_nas_tres_faixas_alvos_e_selo(qapp, abrir, raiz_app, largura, altura):
    from libras.app_real.janela import SELO, SELO_DETALHE

    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir(largura, altura)
    assert ativo(qapp, w)
    esperar(qapp, limite=0.3)
    b = w.barra
    for bt in (b.bt_camera, b.bt_pausa, b.bt_tela, b.bt_sinais, b.selo, w.vista.bt_config):
        assert bt.isVisible() and bt.height() >= 44 and bt.width() >= 44, bt.accessibleName()
        g = bt.mapTo(w, bt.rect().topLeft())
        assert 0 <= g.x() and g.x() + bt.width() <= w.width() + 1, (bt.accessibleName(), g.x(), bt.width())
    assert b.aviso.isVisible() and "não substitui" in b.aviso.text()
    if largura < 600:
        assert b.selo.text() == "" and b.selo.accessibleName() == SELO  # so o icone; o texto abre ao tocar
        b.selo.click()  # folha pequena (nao e dialogo modal), com o texto completo
        assert w.vista.folha.isVisible() and w.vista.folha.texto.text() == SELO_DETALHE
        g = w.vista.folha.geometry()
        assert g.right() <= w.vista.width() and g.bottom() <= w.barra.y()
        w.vista.folha.bt_fechar.click()
        assert not w.vista.folha.isVisible()
    else:
        assert b.selo.text() == SELO
    if largura >= 600:  # na faixa compacta os rotulos sao curtos, embaixo do icone
        assert b.bt_camera.text() in ("Parar câmera", "Parar")


def test_sinais_reconhecidos_lista_os_rotulos(qapp, abrir, raiz_app):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    w.barra.bt_sinais.click()
    titulo, texto = w.dialogos.textos[-1]
    assert titulo == "Sinais reconhecidos" and "Biblioteca" in texto and "nenhum_sinal" not in texto
    assert pk.AVISO_USO in texto


# ---------------------------------------------------------------------- Configuracoes (13.8)
@pytest.mark.parametrize("largura,altura", [(1366, 800), (900, 700), (390, 844)])
def test_configuracoes_f2_toque_e_previa(qapp, abrir, raiz_app, largura, altura):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir(largura, altura)
    assert ativo(qapp, w)
    w.activateWindow()
    w.vista.bt_config.click()  # toque
    assert w.painel.isVisible() and w.camada.exemplo is not None
    g = w.painel.geometry()
    if largura < 600:
        assert g.width() == w.centralWidget().width() and g.y() > 0  # folha que sobe de baixo
    else:
        assert g.right() == w.centralWidget().width() - 1 and g.x() > 0  # painel lateral
    antes = w.camada.medir(500)
    w.painel.cb_tam.setCurrentIndex(4)  # a legenda de exemplo reage na hora
    assert w.camada.estilo.tamanho_px == 52 and w.camada.medir(500) != antes
    salvo = json.loads((raiz_app / "configs" / "app_real_preferencias.json").read_text(encoding="utf-8"))
    assert salvo["legenda"]["tamanho"] == 4
    assert w.painel.area.horizontalScrollBar().maximum() == 0  # nada estoura na horizontal
    QTest.keyClick(w, Qt.Key_F2)
    assert not w.painel.isVisible() and w.camada.exemplo is None
    QTest.keyClick(w, Qt.Key_F2)
    assert w.painel.isVisible()
    w.painel.bt_fechar.click()
    assert not w.painel.isVisible()


def test_cores_personalizadas_ficam_nas_preferencias_e_avisam_contraste(qapp, abrir, raiz_app):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir(iniciar=False)
    w.alternar_configuracoes()
    w.painel.definir_cor("cor_texto", "#808080")
    w.painel.definir_cor("cor_fundo", "#909090")
    assert w.painel.cb_pred.currentData() == "personalizada" and w.camada.estilo.texto == "#808080"
    assert w.painel.lb_contraste.isVisibleTo(w.painel) and "4,5:1" in w.painel.lb_contraste.text()
    salvo = json.loads((raiz_app / "configs" / "app_real_preferencias.json").read_text(encoding="utf-8"))
    assert salvo["legenda"]["cor_texto"] == "#808080"
    w.painel.restaurar()
    assert w.camada.estilo.texto == "#FFFFFF" and not w.painel.lb_contraste.isVisibleTo(w.painel)


def test_barra_tecnica_desligada_por_padrao_e_cada_grupo(qapp, abrir, raiz_app):
    from libras.app_real.preferencias import GRUPOS_TECNICOS

    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    assert ativo(qapp, w)
    w.camada.rollup.adicionar("Prova")
    w.atualizar_barra_tecnica()
    w.vista.grab()
    assert w.vista.linhas_tecnicas == [] and w.vista.ultimo_layout["barra_tecnica"] is None
    legenda_antes = w.vista.ultimo_layout["legenda"]
    rotulos = {"taxa": "FPS cam", "quadros": "Pulados", "tempo": "Lat", "recursos": "RAM", "sistema": "Câmera"}
    for g in GRUPOS_TECNICOS:
        w.painel.ck_tec[g].setChecked(True)
        w.atualizar_barra_tecnica()
        assert rotulos[g] in " ".join(w.vista.linhas_tecnicas), (g, w.vista.linhas_tecnicas)
        w.painel.ck_tec[g].setChecked(False)
    w.atualizar_barra_tecnica()
    w.vista.grab()
    assert w.vista.linhas_tecnicas == [] and w.vista.ultimo_layout["legenda"] == legenda_antes
    # na faixa fixa no alto a barra empurra a legenda; desligada, a legenda volta ao lugar e tamanho originais
    w.prefs["legenda"]["posicao"] = "topo"
    w.aplicar_preferencias_legenda()
    w.vista.grab()
    topo_sem = w.vista.ultimo_layout["legenda"]
    w.painel.ck_tec["taxa"].setChecked(True)
    w.atualizar_barra_tecnica()
    w.vista.grab()
    topo_com = w.vista.ultimo_layout["legenda"]
    assert topo_com.y >= w.vista.ultimo_layout["barra_tecnica"].base + 8  # comeca abaixo da barra
    w.painel.ck_tec["taxa"].setChecked(False)
    w.atualizar_barra_tecnica()
    w.vista.grab()
    assert w.vista.ultimo_layout["legenda"] == topo_sem and w.vista.ultimo_layout["barra_tecnica"] is None
    texto = w.relatorio_tecnico()
    assert "relatório técnico" in texto and str(raiz_app) not in texto and "S01" not in texto


@pytest.mark.parametrize("largura,altura", [(1366, 800), (900, 700), (390, 844)])
def test_nada_se_cruza_com_tudo_ligado_na_tela(qapp, abrir, raiz_app, monkeypatch, largura, altura):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    enq = {
        "pessoa": True,
        "ombros": "parcial",
        "mao_cortada": ["esquerda"],
        "pouca_luz": True,
        "brilho": 30,
        "cabeca": [0.4, 0.02, 0.6, 0.2],
        "maos_suficientes": True,
    }  # cabeca simulada no topo
    monkeypatch.setenv("LIBRAS_FALSO_ENQ", json.dumps(enq))
    w = abrir(largura, altura)
    for g in w.prefs["tecnicas"]:
        w.prefs["tecnicas"][g] = True
    w.prefs["legenda"]["linhas"] = 3
    assert ativo(qapp, w)
    for i in range(12):
        w.camada.rollup.adicionar(["Biblioteca", "Prova", "Secretaria"][i % 3])
    assert esperar(qapp, lambda: w.vista.aviso is not None, 5)  # aviso de enquadramento (histerese de 1 s)
    for pos in ("embaixo", "pessoa", "topo"):
        w.prefs["legenda"]["posicao"] = pos
        w.aplicar_preferencias_legenda()
        w.atualizar_barra_tecnica()
        w.vista.grab()
        L = w.vista.ultimo_layout
        rets = {k: L[k] for k in ("legenda", "barra_tecnica", "etiqueta", "botao_config", "barra_minima")}
        assert all(rets.values()), (pos, {k: v for k, v in rets.items() if not v})
        for (a, ra), (b, rb) in itertools.combinations(rets.items(), 2):
            assert not ra.cruza(rb), (largura, pos, a, ra, b, rb)
        assert L["barra_tecnica"].direita <= L["botao_config"].x
        sel = w.barra.selo.geometry()  # o selo (na barra minima) tambem nao e coberto
        from libras.ui.sobreposicoes import Ret

        r_selo = Ret(sel.x(), w.barra.y() + sel.y(), sel.width(), sel.height())
        for k in ("legenda", "barra_tecnica", "etiqueta", "botao_config"):
            assert not rets[k].cruza(r_selo), (pos, k)


# ---------------------------------------------------------------------- guia e falha de camera (13.9)
def test_guia_aparece_uma_vez_e_continuar_mesmo_assim(qapp, abrir, raiz_app, monkeypatch):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    monkeypatch.setenv(
        "LIBRAS_FALSO_ENQ",
        json.dumps(
            {"pessoa": True, "ombros": "fora", "mao_cortada": [], "pouca_luz": False, "maos_suficientes": False}
        ),
    )
    w = abrir(guia_visto=False)
    assert w.vista.guia.isVisible() and w.vista.contorno_guia
    assert ativo(qapp, w)
    assert esperar(qapp, lambda: w.vista.guia.bt.text() == "Continuar mesmo assim", 5)
    w.vista.guia.bt.click()
    assert not w.vista.guia.isVisible() and w.prefs["guia_visto"]
    w.close()
    w2 = abrir(guia_visto=True)
    assert not w2.vista.guia.isVisible()
    w2.painel.secoes  # "Ver guia de posicionamento" reabre
    w2.painel._ver_guia()
    assert w2.vista.guia.isVisible()


def test_avisos_de_enquadramento_e_chave(qapp, abrir, raiz_app, monkeypatch):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    monkeypatch.setenv(
        "LIBRAS_FALSO_ENQ",
        json.dumps({"pessoa": True, "ombros": "ok", "mao_cortada": ["esquerda"], "pouca_luz": False}),
    )
    w = abrir()
    assert ativo(qapp, w)
    assert esperar(qapp, lambda: w.vista.aviso is not None, 5)
    assert "mãos sai da imagem" in w.vista.aviso and w.vista.bordas == ["esquerda"]
    w.vista.grab()
    bordas = w.vista.ultimo_layout["bordas"]
    img = w.vista.ultimo_layout["imagem"]
    assert list(bordas) == ["direita"] and bordas["direita"].direita == pytest.approx(img.direita)  # previa espelhada
    w.painel.ck_avisos.setChecked(False)
    assert esperar(qapp, lambda: w.vista.aviso is None, 3)


def test_falha_de_camera_tem_causa_acao_e_prioridade(qapp, abrir, raiz_app, monkeypatch):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    monkeypatch.setenv("LIBRAS_FALSO_ERRO", "Nao foi possivel usar a camera de indice 0 (ANY: nao abriu).")
    w = abrir()
    assert esperar(qapp, lambda: w.estado == "falha", 10)
    c = w.vista.cartao
    assert c.isVisible() and "não abriu" in c.texto.text() and "Tentar de novo" in [b.text() for b in c.botoes]
    assert w.vista.aviso is None and w.vista.bordas == []


def test_tela_cheia_esconde_e_mostra_controles(qapp, abrir, raiz_app):
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    w = abrir()
    assert ativo(qapp, w)
    w.prefs["tecnicas"]["taxa"] = True
    w.atualizar_barra_tecnica()
    w.alternar_tela_cheia()
    assert w.isFullScreen()
    w._esconder_controles()
    w.atualizar_barra_tecnica()
    w.vista.grab()
    L = w.vista.ultimo_layout
    assert not w.barra.isVisible() and not w.vista.bt_config.isVisible() and L["barra_tecnica"] is None
    assert w.prefs["tecnicas"]["taxa"]  # esconder por inatividade nao muda a preferencia
    w.atividade()
    w.vista.grab()
    assert w.barra.isVisible() and w.vista.bt_config.isVisible() and w.vista.ultimo_layout["barra_tecnica"] is not None
    w.alternar_tela_cheia()
    assert not w.isFullScreen()


def test_sessao_da_interface_nao_cria_arquivos(qapp, abrir, raiz_app, tmp_path, monkeypatch):
    """Selo (13.9-B), lado da interface: uma sessao com camera, palavras, pausa, copiar e relatorio nao cria
    nenhum arquivo alem das preferencias (servico FALSO; o real e testado em test_servico_app_real)."""
    pacote_sintetico(raiz_app / "publicados", "20261007_100000_aaaa1111", "2026-10-07T10:00:00")
    monkeypatch.setenv("LIBRAS_FALSO_PALAVRAS", "biblioteca,prova")
    raiz_repo = Path(__file__).resolve().parents[1]

    def arquivos():
        return {
            p
            for r in (raiz_app, raiz_repo, Path.cwd())
            for p in r.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        }

    w = abrir(iniciar=False)
    antes = arquivos()
    w.iniciar(lambda *a, **k: True)
    assert ativo(qapp, w)
    esperar(qapp, lambda: len(w.camada.rollup.historico) >= 2, 5)
    w.alternar_pausa()
    w.copiar_texto()
    w.relatorio_tecnico()
    w.alternar_pausa()
    w.parar_camera()
    esperar(qapp, limite=0.5)
    novos = arquivos() - antes - {raiz_app / "configs" / "app_real_preferencias.json", tmp_path / "comandos.txt"}
    assert not novos, sorted(map(str, novos))
