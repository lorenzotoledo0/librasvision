"""Pagina da bancada e reexportacao dos componentes compartilhados (libras.ui.componentes).

Os componentes moraram aqui ate o item 13; foram movidos (nao copiados) para libras.ui, que o app real
tambem usa. As paginas da bancada continuam importando de ..base.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..ui.componentes import (  # noqa: F401 - reexportados para as paginas da bancada
    RAIZ_CODIGO,
    Dialogo,
    DialogoDigitar,
    EstadoVazio,
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
    """Base das paginas: titulo e subtitulo fixos no topo e conteudo em uma area com rolagem (o conteudo
    nunca e cortado em janela pequena ou com escala de 125%/150%). ao_entrar/ao_sair sao chamados na
    troca de aba; atualizar_dados quando manifesto, cadastro, vocabulario, splits ou runs mudam."""

    titulo = ""
    subtitulo = ""

    def __init__(self, app):
        super().__init__()
        self.app = app
        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)
        cab = QVBoxLayout()
        cab.setContentsMargins(24, 16, 24, 8)
        cab.setSpacing(4)
        topo = QHBoxLayout()
        topo.setSpacing(8)
        self.lb_titulo = rotulo(self.titulo, "titulo_pagina", quebra=True)
        topo.addWidget(self.lb_titulo, 1)
        self.bt_atualizar = botao(
            "Atualizar",
            variante="fantasma",
            acao=lambda: self.app.atualizar(),
            dica="Atualiza os dados desta tela (F5). Em geral ela já se atualiza sozinha.",
        )
        topo.addWidget(self.bt_atualizar, 0, Qt.AlignTop)
        cab.addLayout(topo)
        if self.subtitulo:
            self.lb_subtitulo = rotulo(self.subtitulo, "subtitulo", quebra=True)
            cab.addWidget(self.lb_subtitulo)
        externo.addLayout(cab)
        self.aviso_desktop: QFrame | None = None
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        conteudo = QWidget()
        conteudo.setObjectName("conteudo")
        self.raiz = QVBoxLayout(conteudo)
        self.raiz.setContentsMargins(24, 8, 24, 20)
        self.raiz.setSpacing(16)
        self.area.setWidget(conteudo)
        self.area.viewport().installEventFilter(self)
        externo.addWidget(self.area, 1)

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
        self.raiz.setContentsMargins(margem, 8, margem, 20)
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
