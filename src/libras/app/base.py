"""Pagina da bancada e reexportacao dos componentes compartilhados (libras.ui.componentes).

Os componentes moraram aqui ate o item 13; foram movidos (nao copiados) para libras.ui, que o app real
tambem usa. As paginas da bancada continuam importando de ..base.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QToolButton, QVBoxLayout, QWidget

from ..ui.componentes import (  # noqa: F401 - reexportados para as paginas da bancada
    RAIZ_CODIGO,
    Dialogo,
    DialogoDigitar,
    EstadoVazio,
    Secao,
    Subprocesso,
    Tabela,
    Tarefa,
    TelaVideo,
    aguardar_encerramentos,
    aguardar_tarefas,
    avisar,
    botao,
    cartao_numero,
    dica,
    encerrar_em_segundo_plano,
    erro,
    etiqueta,
    formulario,
    ha_encerramento_pendente,
    info,
    iniciar_tarefa,
    linha,
    log,
    mudar_etiqueta,
    painel,
    pedir_texto,
    perguntar,
    rotulo,
    separador,
    widget_com,
)


class Pagina(QWidget):
    """Base das paginas: cabecalho fixo no topo (titulo, subtitulo e Atualizar), conteudo numa area com
    rolagem (nunca cortado em janela pequena ou com escala de 125%/150%) e, se a pagina usar
    adicionar_acao, uma barra de acoes fixa embaixo, sempre visivel (as acoes principais nunca ficam
    abaixo da dobra). ao_entrar/ao_sair sao chamados na troca de aba; atualizar_dados quando manifesto,
    cadastro, vocabulario, splits ou runs mudam."""

    titulo = ""
    subtitulo = ""

    def __init__(self, app):
        super().__init__()
        self.app = app
        from ..ui import tema
        from ..ui.icones import icone

        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)
        self.cabecalho = QFrame()
        self.cabecalho.setObjectName("cabecalho_pagina")
        topo = QHBoxLayout(self.cabecalho)
        topo.setContentsMargins(24, 10, 16, 10)
        topo.setSpacing(12)
        textos = QVBoxLayout()
        textos.setSpacing(2)
        self.lb_titulo = rotulo(self.titulo, "titulo_pagina", quebra=True)
        textos.addWidget(self.lb_titulo)
        if self.subtitulo:
            self.lb_subtitulo = rotulo(self.subtitulo, "subtitulo", quebra=True)
            textos.addWidget(self.lb_subtitulo)
        topo.addLayout(textos, 1)
        self.bt_atualizar = QToolButton()
        self.bt_atualizar.setObjectName("bt_icone")
        self.bt_atualizar.setIcon(icone("atualizar", tema.CORES["texto2"]))
        self.bt_atualizar.setIconSize(QSize(20, 20))
        self.bt_atualizar.setFixedSize(tema.ALVO_TOQUE, tema.ALVO_TOQUE)
        self.bt_atualizar.setCursor(Qt.PointingHandCursor)
        self.bt_atualizar.setToolTip("Atualizar os dados desta tela (F5). Em geral ela já se atualiza sozinha.")
        self.bt_atualizar.setAccessibleName("Atualizar")
        self.bt_atualizar.clicked.connect(lambda: self.app.atualizar())
        topo.addWidget(self.bt_atualizar, 0, Qt.AlignTop)
        externo.addWidget(self.cabecalho)
        self.aviso_desktop: QFrame | None = None
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        conteudo = QWidget()
        conteudo.setObjectName("conteudo")
        self.raiz = QVBoxLayout(conteudo)
        self.raiz.setContentsMargins(24, 16, 24, 20)
        self.raiz.setSpacing(16)
        self.area.setWidget(conteudo)
        self.area.viewport().installEventFilter(self)
        externo.addWidget(self.area, 1)
        from ..ui.dinamica import FluxoLayout

        self.barra_acoes = QFrame()
        self.barra_acoes.setObjectName("barra_acoes")
        self.acoes = FluxoLayout(8)  # quebra a linha em janela estreita, nunca estoura na horizontal
        self.acoes.setContentsMargins(24, 10, 24, 10)
        self.barra_acoes.setLayout(self.acoes)
        self.barra_acoes.setVisible(False)
        externo.addWidget(self.barra_acoes)

    def adicionar_acao(self, *widgets) -> None:
        """Poe botoes (ou campos) na barra de acoes fixa do rodape, sempre visivel, fora da rolagem."""
        for w in widgets:
            self.acoes.addWidget(w)
        self.barra_acoes.setVisible(True)

    @property
    def cfg(self):
        return self.app.cfg

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent

        if ev.type() == QEvent.Resize and obj is self.area.viewport():
            from ..ui.dinamica import Colunas

            larg = ev.size().width() - self.raiz.contentsMargins().left() - self.raiz.contentsMargins().right()
            for c in self.findChildren(Colunas):
                c.ajustar(larg)
        return False

    def ao_entrar(self) -> None:
        pass

    def ao_sair(self) -> None:
        pass

    def atualizar_dados(self, tipos: set) -> None:
        """Chamado na pagina aberta quando os dados mudam (ver dinamica.MonitorDados)."""

    # Telas que so fazem sentido no computador (D18): na faixa compacta mostram um aviso claro em vez
    # de um layout quebrado.
    somente_desktop = False
    texto_somente_desktop = (
        "Esta tela precisa de uma janela mais larga. Use o computador ou aumente a janela "
        "para pelo menos 600 px de largura."
    )

    def aplicar_faixa(self, faixa: str) -> None:
        margem = 12 if faixa == "compacta" else 24
        self.raiz.setContentsMargins(margem, 16, margem, 20)
        self.cabecalho.layout().setContentsMargins(margem, 10, max(8, margem - 8), 10)
        self.acoes.setContentsMargins(margem, 10, margem, 10)
        if hasattr(self, "lb_subtitulo"):
            self.lb_subtitulo.setVisible(faixa != "compacta")  # no celular o titulo basta
        if not self.somente_desktop:
            return
        compacta = faixa == "compacta"
        if compacta and self.aviso_desktop is None:
            self.aviso_desktop = EstadoVazio(
                "Disponível só no computador",
                self.texto_somente_desktop,
                "Ir para o Início",
                lambda: self.app.ir_para("inicio"),
            )
            self.layout().addWidget(self.aviso_desktop)
        if self.aviso_desktop is not None:
            self.aviso_desktop.setVisible(compacta)
        self.area.setVisible(not compacta)

    def encerrar(self) -> None:
        self.ao_sair()
