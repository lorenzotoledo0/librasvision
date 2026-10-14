"""Galeria de componentes (Ctrl+Shift+G, ou Configuracoes > Galeria): mostra cada componente do
sistema de design em todos os estados (normal, hover, pressionado, foco, desabilitado), para
conferir os tokens visualmente. Os estados sao forcados pela propriedade "estado" (a folha de
estilo gera a mesma regra para o pseudo-estado e para a propriedade)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..ui import tema
from .base import EstadoVazio, Tabela, avisar, botao, dica, etiqueta, painel, rotulo

ESTADOS = ["normal", "hover", "pressionado", "foco", "desabilitado"]
VARIANTES = [
    ("primario", "Primário"),
    ("secundario", "Secundário"),
    ("fantasma", "Fantasma"),
    ("destrutivo", "Destrutivo"),
]


def _forcar(w: QWidget, estado: str) -> QWidget:
    if estado == "desabilitado":
        w.setEnabled(False)
    elif estado != "normal":
        w.setProperty("estado", estado)
    w.setFocusPolicy(Qt.NoFocus)
    return w


class GaleriaComponentes(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Galeria de componentes")
        self.resize(1100, 820)
        externo = QVBoxLayout(self)
        area = QScrollArea()
        area.setWidgetResizable(True)
        conteudo = QWidget()
        conteudo.setObjectName("conteudo")
        lay = QVBoxLayout(conteudo)
        lay.setSpacing(tema.ESPACO[5])
        area.setWidget(conteudo)
        externo.addWidget(area)
        self.secoes: dict[str, QWidget] = {}

        # botoes x estados
        p, l = self._secao("Botões", lay)
        g = QGridLayout()
        for j, est in enumerate(ESTADOS):
            g.addWidget(rotulo(est, "secundario"), 0, j + 1)
        for i, (var, nome) in enumerate(VARIANTES, start=1):
            g.addWidget(rotulo(nome), i, 0)
            for j, est in enumerate(ESTADOS):
                g.addWidget(_forcar(botao(nome, variante=var), est), i, j + 1)
        l.addLayout(g)

        # campos
        p, l = self._secao("Campos", lay)
        g = QGridLayout()
        for j, est in enumerate(["normal", "hover", "foco", "desabilitado"]):
            g.addWidget(rotulo(est, "secundario"), 0, j)
            ed = QLineEdit("Texto digitado")
            cb = QComboBox()
            cb.addItems(["biblioteca", "prova"])
            sp = QSpinBox()
            sp.setValue(10)
            ck = QCheckBox("Opção marcada")
            ck.setChecked(True)
            for k, w in enumerate((ed, cb, sp, ck), start=1):
                g.addWidget(_forcar(w, est), k, j)
        l.addLayout(g)

        # etiquetas, dica, progresso
        p, l = self._secao("Etiquetas, dica e progresso", lay)
        fila = QGridLayout()
        for k, (tipo, txt) in enumerate(
            [
                ("ok", "ok"),
                ("pendente", "pendente"),
                ("ensaio", "ENSAIO"),
                ("erro", "erro"),
                ("info", "informação"),
                ("captando", "mãos detectadas"),
            ]
        ):
            fila.addWidget(etiqueta(txt, tipo), 0, k)
        l.addLayout(fila)
        l.addWidget(dica("Dica: explicação curta sempre visível, sem depender do mouse."))
        pb = QProgressBar()
        pb.setValue(62)
        l.addWidget(pb)
        oc = QProgressBar()
        oc.setObjectName("ocupado")
        oc.setRange(0, 0)
        l.addWidget(oc)

        # cartao, estado vazio, tabela
        p, l = self._secao("Cartão, estado vazio e tabela", lay)
        c, lc = painel()
        c.setObjectName("cartao")
        lc.addWidget(rotulo("Vídeos 'ok'", "secundario"))
        lc.addWidget(rotulo("128", "numero"))
        lc.addWidget(rotulo("4 pendentes", "secundario"))
        l.addWidget(c)
        l.addWidget(
            EstadoVazio(
                "Ainda não dá para treinar", "Grave 'nenhum_sinal' para ter pelo menos 2 classes.", "Ir para Gravar"
            )
        )
        t = Tabela(["Classe", "Vídeos", "Situação"])
        t.preencher([["biblioteca", 12, "elegível"], ["prova", 3, "abaixo do mínimo"], ["secretaria", 0, "sem vídeos"]])
        l.addWidget(t)

        # toasts e dialogos
        p, l = self._secao("Avisos e diálogos", lay)
        fila = QGridLayout()
        for k, tipo in enumerate(["info", "ok", "aviso", "erro"]):
            fila.addWidget(
                botao(f"Aviso {tipo}", acao=lambda _=False, tp=tipo: avisar(self, f"Exemplo de aviso ({tp}).", tp)),
                0,
                k,
            )
        fila.addWidget(botao("Diálogo de confirmação", acao=self._dialogo), 0, 4)
        l.addLayout(fila)

        # tipografia e cores
        p, l = self._secao("Tipografia", lay)
        for nome in ["legenda", "corpo", "corpo_grande", "titulo_secao", "titulo_pagina", "numero", "texto_ao_vivo"]:
            r = QLabel(f"{nome}: {tema.px(nome)} px  Biblioteca, prova, secretaria")
            r.setStyleSheet(f"font-size: {tema.px(nome)}px;")
            l.addWidget(r)
        p, l = self._secao("Rampa (só dados e assinatura) e cores semânticas, com contraste sobre o painel", lay)
        g = QGridLayout()
        cores = [(f"rampa {i + 1}", c_) for i, c_ in enumerate(tema.RAMPA)] + [
            (n, tema.COR[n])
            for n in (
                "sucesso",
                "alerta",
                "erro",
                "info",
                "texto",
                "texto_secundario",
                "texto_terciario",
                "borda_controle",
                "foco",
            )
        ]
        for k, (nome, cor) in enumerate(cores):
            amostra = QFrame()
            amostra.setFixedSize(56, 28)
            amostra.setStyleSheet(f"background: {cor}; border-radius: 6px;")
            g.addWidget(amostra, k // 4 * 2, k % 4)
            g.addWidget(
                rotulo(f"{nome} {cor}  {tema.contraste(cor, tema.CORES['painel']):.1f}:1", "secundario"),
                k // 4 * 2 + 1,
                k % 4,
            )
        l.addLayout(g)
        lay.addStretch(1)

    def _secao(self, titulo: str, lay):
        p, l = painel()
        l.addWidget(rotulo(titulo, "titulo_painel"))
        lay.addWidget(p)
        self.secoes[titulo] = p
        return p, l

    def _dialogo(self) -> None:
        from .base import perguntar

        perguntar(self, "Exemplo de confirmação destrutiva: mover para a lixeira?", destrutivo=True)
