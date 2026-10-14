"""Painel de Configuracoes do app real (13.8): painel lateral (folha que sobe de baixo na faixa
compacta), aberto pela engrenagem ou por F2. A camera continua rodando e a legenda de exemplo aparece
sobre o video ao vivo, reagindo a cada mudanca.

Secoes: Legenda, Informacoes tecnicas, Camera, Modelo e Sobre. Nao ha ajuste de decisao (limiar,
media movel, confirmacoes): isso fica so na bancada.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from .. import pacote as pk
from ..ui import tema
from ..ui.componentes import botao, dica, formulario, linha, rotulo
from ..ui.icones import icone
from ..ui.legenda import CONTORNOS, NOMES_POSICAO, POSICOES, TOK, estilo_de
from .preferencias import GRUPOS_TECNICOS, NOMES_GRUPOS


def _form() -> QFormLayout:
    """Rotulo acima do campo: o painel e estreito (e vira folha de 390 px na faixa compacta)."""
    f = formulario()
    f.setRowWrapPolicy(QFormLayout.WrapAllRows)
    return f


def _combo() -> QComboBox:
    c = QComboBox()
    c.setMinimumContentsLength(12)
    c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    return c


NOMES_TAMANHO = ["Pequeno", "Médio", "Grande", "Maior", "Máximo"]
SECOES = ("legenda", "tecnicas", "camera", "modelo", "sobre")


class PainelConfiguracoes(QFrame):
    def __init__(self, janela):
        super().__init__()
        self.j = janela
        self.prefs = janela.prefs
        self.setObjectName("painel_config")
        self._carregando = False
        ext = QVBoxLayout(self)
        ext.setContentsMargins(0, 0, 0, 0)
        ext.setSpacing(0)
        topo = QHBoxLayout()
        topo.setContentsMargins(tema.ESPACO[4], tema.ESPACO[3], tema.ESPACO[3], tema.ESPACO[2])
        topo.addWidget(rotulo("Configurações", "titulo_secao"), 1)
        self.bt_fechar = QToolButton()
        self.bt_fechar.setObjectName("bt_fechar")
        self.bt_fechar.setIcon(icone("fechar", tema.CORES["texto"]))
        self.bt_fechar.setIconSize(QSize(22, 22))
        self.bt_fechar.setFixedSize(tema.ALVO_TOQUE, tema.ALVO_TOQUE)
        self.bt_fechar.setAccessibleName("Fechar Configurações (F2)")
        self.bt_fechar.setToolTip("Fechar (F2)")
        self.bt_fechar.clicked.connect(lambda: janela.alternar_configuracoes())
        topo.addWidget(self.bt_fechar)
        ext.addLayout(topo)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        conteudo = QWidget()
        self.lay = QVBoxLayout(conteudo)
        self.lay.setContentsMargins(tema.ESPACO[4], tema.ESPACO[1], tema.ESPACO[4], tema.ESPACO[5])
        self.lay.setSpacing(tema.ESPACO[3])
        self.area.setWidget(conteudo)
        ext.addWidget(self.area, 1)
        self.secoes: dict[str, QWidget] = {}
        self._secao_legenda()
        self._secao_tecnicas()
        self._secao_camera()
        self._secao_modelo()
        self._secao_sobre()
        self.lay.addStretch(1)
        self.carregar()

    # ------------------------------------------------------------------ estrutura
    def _cabecalho(self, chave: str, titulo: str) -> None:
        r = rotulo(titulo, "titulo_painel")
        self.secoes[chave] = r
        if self.lay.count():
            sep = QFrame()
            sep.setObjectName("separador")
            sep.setFrameShape(QFrame.HLine)
            self.lay.addWidget(sep)
        self.lay.addWidget(r)

    def aplicar_faixa(self, faixa: str) -> None:
        self.setProperty("modo", "folha" if faixa == "compacta" else "lateral")
        self.style().unpolish(self)
        self.style().polish(self)

    def mostrar_secao(self, chave: str | None) -> None:
        if chave in self.secoes:
            self.area.ensureWidgetVisible(self.secoes[chave], 0, 0)
            self.area.verticalScrollBar().setValue(self.secoes[chave].y())

    # ------------------------------------------------------------------ 1. Legenda
    def _secao_legenda(self) -> None:
        self._cabecalho("legenda", "Legenda")
        self.lay.addWidget(dica("As mudanças aparecem na hora na legenda de exemplo sobre a câmera."))
        f = _form()
        self.cb_pred = _combo()
        for chave, d in TOK["predefinicoes"].items():
            self.cb_pred.addItem(d["nome"], chave)
        self.cb_pred.addItem("Personalizada", "personalizada")
        self.cb_pred.currentIndexChanged.connect(self._mudou_pred)
        f.addRow("Cores", self.cb_pred)
        self.cb_tam = _combo()
        for i, n in enumerate(NOMES_TAMANHO):
            self.cb_tam.addItem(f"{n} ({TOK['tamanhos_px'][i]} px)", i)
        self.cb_tam.currentIndexChanged.connect(lambda _: self._mudou("tamanho", self.cb_tam.currentData()))
        f.addRow("Tamanho", self.cb_tam)
        self.cb_fonte = _combo()
        for chave, d in TOK["fontes"].items():
            self.cb_fonte.addItem(d["nome"], chave)
        self.cb_fonte.currentIndexChanged.connect(lambda _: self._mudou("fonte", self.cb_fonte.currentData()))
        f.addRow("Fonte", self.cb_fonte)
        self.bt_cor_texto = botao("Escolher cor do texto", acao=lambda: self._escolher_cor("cor_texto"))
        self.bt_cor_fundo = botao("Escolher cor do fundo", acao=lambda: self._escolher_cor("cor_fundo"))
        f.addRow("Cor do texto", self.bt_cor_texto)
        f.addRow("Cor do fundo", self.bt_cor_fundo)
        self.sl_opac = QSlider(Qt.Horizontal)
        self.sl_opac.setRange(0, 100)
        self.sl_opac.setMinimumHeight(tema.ALVO_TOQUE)
        self.sl_opac.valueChanged.connect(self._mudou_opac)
        self.lb_opac = rotulo("", "secundario")
        f.addRow("Opacidade do fundo", self._com_valor(self.sl_opac, self.lb_opac))
        self.cb_contorno = _combo()
        for chave, nome in CONTORNOS.items():
            self.cb_contorno.addItem(nome, chave)
        self.cb_contorno.currentIndexChanged.connect(lambda _: self._mudou("contorno", self.cb_contorno.currentData()))
        f.addRow("Contorno ou sombra", self.cb_contorno)
        self.sl_larg = QSlider(Qt.Horizontal)
        self.sl_larg.setRange(TOK["largura_pct"]["minimo"], TOK["largura_pct"]["maximo"])
        self.sl_larg.setMinimumHeight(tema.ALVO_TOQUE)
        self.sl_larg.valueChanged.connect(self._mudou_larg)
        self.lb_larg = rotulo("", "secundario")
        f.addRow("Largura (% da imagem)", self._com_valor(self.sl_larg, self.lb_larg))
        self.cb_linhas = _combo()
        self.cb_linhas.addItem("2 linhas", 2)
        self.cb_linhas.addItem("3 linhas", 3)
        self.cb_linhas.currentIndexChanged.connect(lambda _: self._mudou("linhas", self.cb_linhas.currentData()))
        f.addRow("Linhas", self.cb_linhas)
        self.sp_sumir = QSpinBox()
        self.sp_sumir.setRange(2, 30)
        self.sp_sumir.setSuffix(" s")
        self.sp_sumir.valueChanged.connect(lambda v: self._mudou("sumir_s", float(v)))
        f.addRow("Tempo até sumir", self.sp_sumir)
        self.cb_pos = _combo()
        for chave in POSICOES:
            self.cb_pos.addItem(NOMES_POSICAO[chave], chave)
        self.cb_pos.currentIndexChanged.connect(lambda _: self._mudou("posicao", self.cb_pos.currentData()))
        f.addRow("Posição", self.cb_pos)
        self.lay.addLayout(f)
        self.ck_destacar = self._chave("Destacar palavra confirmada", "a palavra nova ganha destaque por um instante")
        self.ck_destacar.toggled.connect(lambda v: self._mudou("destacar", bool(v)))
        self.ck_reduzir = self._chave("Reduzir movimento", "sem rolagem, sem esmaecer aos poucos e sem pulso")
        self.ck_reduzir.toggled.connect(self._mudou_reduzir)
        self.lb_contraste = rotulo("", "alerta_aviso", quebra=True)
        self.lay.addWidget(self.lb_contraste)
        self.lay.addLayout(
            linha(
                botao("Restaurar padrão", acao=self.restaurar),
                botao("Limpar legenda", acao=self.j.limpar_legenda),
                botao("Copiar texto", acao=self.j.copiar_texto),
            )
        )

    def _chave(self, nome: str, descricao: str = "") -> QCheckBox:
        """Chave liga/desliga com nome curto e a descricao embaixo (texto que quebra no painel estreito)."""
        ck = QCheckBox(nome)
        ck.setMinimumHeight(tema.ALVO_TOQUE)
        self.lay.addWidget(ck)
        if descricao:
            d = dica(descricao[:1].upper() + descricao[1:] + ".")
            d.setContentsMargins(28, 0, 0, 0)
            self.lay.addWidget(d)
        return ck

    @staticmethod
    def _com_valor(w, lb) -> QWidget:
        c = QWidget()
        h = QHBoxLayout(c)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        lb.setMinimumWidth(44)
        h.addWidget(lb)
        return c

    # ------------------------------------------------------------------ 2. Informacoes tecnicas
    def _secao_tecnicas(self) -> None:
        self._cabecalho("tecnicas", "Informações técnicas")
        self.lay.addWidget(
            dica(
                "Ligue um grupo para mostrar uma barra fina no alto da câmera. Tudo desligado: a barra "
                "não aparece. Quadros pulados são normais (o quadro que chega enquanto o anterior é "
                "processado é descartado para manter o tempo real)."
            )
        )
        self.ck_tec: dict[str, QCheckBox] = {}
        for g in GRUPOS_TECNICOS:
            nome, desc = NOMES_GRUPOS[g].split(" (", 1)
            ck = self._chave(nome, desc.rstrip(")"))
            ck.toggled.connect(lambda v, g=g: self._mudou_tecnica(g, v))
            self.ck_tec[g] = ck
        self.lay.addLayout(linha(botao("Copiar relatório técnico", acao=self.j.relatorio_tecnico)))

    # ------------------------------------------------------------------ 3. Camera
    def _secao_camera(self) -> None:
        self._cabecalho("camera", "Câmera")
        f = _form()
        self.sp_camera = QSpinBox()
        self.sp_camera.setRange(0, 9)
        f.addRow("Número da câmera", self.sp_camera)
        self.lay.addLayout(f)
        self.lay.addWidget(dica("Com mais de uma câmera, troque o número e toque em Tentar de novo."))
        self.lay.addLayout(linha(botao("Tentar de novo", primario=True, acao=self._tentar_camera)))
        self.ck_avisos = self._chave("Avisos de enquadramento", "ombros, mão cortada na borda e pouca luz")
        self.ck_avisos.toggled.connect(self._mudou_avisos)

    # ------------------------------------------------------------------ 4. Modelo
    def _secao_modelo(self) -> None:
        self._cabecalho("modelo", "Modelo")
        self.lb_modelo = rotulo("", quebra=True)
        self.lb_modelo.setTextFormat(Qt.PlainText)
        self.lay.addWidget(self.lb_modelo)
        self.bt_voltar = botao("Voltar ao modelo anterior", acao=self.j.voltar_modelo_anterior)
        self.lay.addLayout(
            linha(botao("Importar pacote (.zip)", acao=lambda: self.j.importar_pacote()), self.bt_voltar)
        )

    def atualizar_modelo(self) -> None:
        c = self.j.cartao_ativo
        if not c:
            self.lb_modelo.setText("Nenhum modelo instalado.")
        else:
            linhas = [f"{r}: {v}" for r, v in pk.resumo(c)]
            av = c.get("avaliacao")
            if av:
                linhas.append(av.get("ressalva", pk.RESSALVA_AVALIACAO))
            self.lb_modelo.setText("\n".join(linhas))
        self.bt_voltar.setEnabled(self.j.modelos.pode_voltar())
        if hasattr(self, "lb_versao"):
            self.lb_versao.setText(self._texto_versao())

    # ------------------------------------------------------------------ 5. Sobre
    def _secao_sobre(self) -> None:
        self._cabecalho("sobre", "Sobre")
        self.lay.addWidget(rotulo(pk.AVISO_USO, quebra=True))
        self.lb_versao = rotulo(self._texto_versao(), "secundario", quebra=True)
        self.lay.addWidget(self.lb_versao)
        from .janela import SELO_DETALHE

        self.lb_privacidade = rotulo(SELO_DETALHE, quebra=True)
        self.lay.addWidget(self.lb_privacidade)
        self.lay.addLayout(linha(botao("Ver guia de posicionamento", acao=self._ver_guia)))

    def _texto_versao(self) -> str:
        c = self.j.cartao_ativo or {}
        return f"LibrasVision {__version__} | modelo {c.get('id', 'nenhum')}"

    # ------------------------------------------------------------------ carregar e salvar
    def carregar(self) -> None:
        self._carregando = True
        L = self.prefs["legenda"]
        self.cb_pred.setCurrentIndex(max(0, self.cb_pred.findData(L["predefinicao"])))
        self.cb_tam.setCurrentIndex(max(0, self.cb_tam.findData(int(L["tamanho"]))))
        self.cb_fonte.setCurrentIndex(max(0, self.cb_fonte.findData(L["fonte"])))
        est = estilo_de(L)
        self.sl_opac.setValue(int(round(est.opacidade * 100)))
        self.lb_opac.setText(f"{self.sl_opac.value()}%")
        self.cb_contorno.setCurrentIndex(max(0, self.cb_contorno.findData(L["contorno"])))
        self.sl_larg.setValue(int(L["largura_pct"]))
        self.lb_larg.setText(f"{self.sl_larg.value()}%")
        self.cb_linhas.setCurrentIndex(0 if int(L["linhas"]) < 3 else 1)
        self.sp_sumir.setValue(int(round(float(L["sumir_s"]))))
        self.cb_pos.setCurrentIndex(max(0, self.cb_pos.findData(L["posicao"])))
        self.ck_destacar.setChecked(bool(L["destacar"]))
        self.ck_reduzir.setChecked(bool(self.prefs["reduzir_movimento"]))
        for g, ck in self.ck_tec.items():
            ck.setChecked(bool(self.prefs["tecnicas"].get(g)))
        idx = self.prefs["camera"]["indice"]
        self.sp_camera.setValue(int(idx if idx is not None else self.j.cfg["camera"]["indice"]))
        self.ck_avisos.setChecked(bool(self.prefs["camera"]["avisos_enquadramento"]))
        self._carregando = False
        self._atualizar_aviso()

    def _salvar_legenda(self) -> None:
        self.prefs.salvar()
        self.j.aplicar_preferencias_legenda()
        self._atualizar_aviso()

    def _mudou(self, chave: str, valor) -> None:
        if self._carregando:
            return
        self.prefs["legenda"][chave] = valor
        self._salvar_legenda()

    def _mudou_pred(self, _i: int) -> None:
        if self._carregando:
            return
        chave = self.cb_pred.currentData()
        L = self.prefs["legenda"]
        if chave == "personalizada" and L["predefinicao"] != "personalizada":
            base = estilo_de(L)  # parte das cores que estavam na tela
            L["cor_texto"], L["cor_fundo"], L["opacidade"] = base.texto, base.fundo, base.opacidade
        L["predefinicao"] = chave
        self._carregando = True
        self.sl_opac.setValue(int(round(estilo_de(L).opacidade * 100)))
        self.lb_opac.setText(f"{self.sl_opac.value()}%")
        self._carregando = False
        self._salvar_legenda()

    def _virar_personalizada(self) -> None:
        L = self.prefs["legenda"]
        if L["predefinicao"] != "personalizada":
            base = estilo_de(L)
            L["cor_texto"], L["cor_fundo"], L["opacidade"] = base.texto, base.fundo, base.opacidade
            L["predefinicao"] = "personalizada"
            self._carregando = True
            self.cb_pred.setCurrentIndex(self.cb_pred.findData("personalizada"))
            self._carregando = False

    def _mudou_opac(self, v: int) -> None:
        self.lb_opac.setText(f"{v}%")
        if self._carregando:
            return
        self._virar_personalizada()
        self.prefs["legenda"]["opacidade"] = v / 100
        self._salvar_legenda()

    def _mudou_larg(self, v: int) -> None:
        self.lb_larg.setText(f"{v}%")
        self._mudou("largura_pct", int(v))

    def definir_cor(self, chave: str, cor_hex: str) -> None:
        """Usado pelo seletor de cor e pelos testes."""
        self._virar_personalizada()
        self.prefs["legenda"][chave] = cor_hex
        self._salvar_legenda()

    def _escolher_cor(self, chave: str) -> None:
        atual = estilo_de(self.prefs["legenda"])
        cor = QColorDialog.getColor(
            QColor(atual.texto if chave == "cor_texto" else atual.fundo),
            self,
            "Cor do texto" if chave == "cor_texto" else "Cor do fundo",
        )
        if cor.isValid():
            self.definir_cor(chave, cor.name().upper())

    def _atualizar_aviso(self) -> None:
        est = estilo_de(self.prefs["legenda"])
        self.lb_contraste.setText(" ".join(est.avisos))
        self.lb_contraste.setVisible(bool(est.avisos))

    def _mudou_reduzir(self, v: bool) -> None:
        if self._carregando:
            return
        self.prefs.dados["reduzir_movimento"] = bool(v)
        tema.configurar(reduzir_movimento=bool(v))
        self.prefs.salvar()
        self.j.aplicar_preferencias_legenda()

    def restaurar(self) -> None:
        self.prefs.restaurar_legenda()
        self.carregar()
        self.j.aplicar_preferencias_legenda()

    def _mudou_tecnica(self, grupo: str, v: bool) -> None:
        if self._carregando:
            return
        self.prefs["tecnicas"][grupo] = bool(v)
        self.prefs.salvar()
        self.j.atualizar_barra_tecnica()

    def _tentar_camera(self) -> None:
        self.prefs["camera"]["indice"] = int(self.sp_camera.value())
        self.prefs.salvar()
        self.j.alternar_configuracoes()
        self.j.tentar_de_novo()

    def _mudou_avisos(self, v: bool) -> None:
        if self._carregando:
            return
        self.prefs["camera"]["avisos_enquadramento"] = bool(v)
        self.j.guia_avisos.habilitado = bool(v)
        self.prefs.salvar()

    def _ver_guia(self) -> None:
        self.j.alternar_configuracoes()
        self.j.mostrar_guia()
