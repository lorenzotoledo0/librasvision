"""Comportamento dinamico da interface (item 8): colunas responsivas, notificacoes nao bloqueantes,
indicador de atividade, dados ao vivo e transicao entre abas.

Nada aqui conhece o dominio (manifesto, treino...): so apresentacao. A logica continua nos modulos
de libras.*, e as paginas so chamam funcoes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QObject, QPoint, QPropertyAnimation, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QSplitter,
    QWidget,
)

log = logging.getLogger("libras.app")


# ----------------------------------------------------------------------
# Colunas responsivas
# ----------------------------------------------------------------------
class Colunas(QSplitter):
    """Duas ou mais colunas com divisor arrastavel. Em largura estreita, as colunas passam a ficar
    uma embaixo da outra (a pagina ganha rolagem vertical), sem cortar nem sobrepor nada.

    Imita a API do QHBoxLayout usada nas paginas: addWidget(w, peso) e addLayout(layout, peso).
    """

    def __init__(self, largura_minima: int = 900):
        super().__init__(Qt.Horizontal)
        self.largura_minima = largura_minima
        self.setChildrenCollapsible(False)
        self.setHandleWidth(10)
        self._pesos: list[int] = []

    def addWidget(self, w: QWidget, peso: int = 1) -> None:  # noqa: N802 - mesma API do Qt
        super().addWidget(w)
        self._pesos.append(max(1, int(peso)))
        self.setStretchFactor(self.count() - 1, self._pesos[-1])

    def addLayout(self, lay, peso: int = 1) -> None:  # noqa: N802
        w = QWidget()
        lay.setContentsMargins(0, 0, 0, 0)
        w.setLayout(lay)
        self.addWidget(w, peso)

    def setSpacing(self, _v: int) -> None:  # noqa: N802 - compatibilidade com QHBoxLayout
        pass

    def ajustar(self, largura_disponivel: int) -> None:
        """Decide a orientacao pela largura VISIVEL (a da area de rolagem da pagina), nao pela propria
        largura: em horizontal o minimo das colunas somadas pode ser maior que a janela, e entao a
        propria largura nunca ficaria abaixo do limite."""
        self.gerenciada = True
        alvo = Qt.Vertical if largura_disponivel < self.largura_minima else Qt.Horizontal
        if self.orientation() != alvo:
            self.setOrientation(alvo)
            total = sum(self._pesos) or 1
            base = max(self.height(), 400) if alvo == Qt.Vertical else max(largura_disponivel, 400)
            self.setSizes([int(base * p / total) for p in self._pesos])

    def resizeEvent(self, ev) -> None:  # noqa: N802
        if not getattr(self, "gerenciada", False):  # fora de uma Pagina: usa a propria largura
            self.ajustar(ev.size().width())
            self.gerenciada = False
        super().resizeEvent(ev)
        if ev.oldSize().width() != ev.size().width():
            QTimer.singleShot(0, self.updateGeometry)  # a altura dos filhos depende da largura

    def _altura_filho(self, w: QWidget, largura: int) -> int:
        """Altura que o filho precisa NA LARGURA QUE ELE TEM (texto que quebra, filas de botoes que
        descem). O QSplitter normalmente ignora isso e entao o conteudo se sobrepunha."""
        base = w.minimumSizeHint().height()
        if w.hasHeightForWidth() and largura > 0:
            return max(base, w.heightForWidth(largura))
        return base

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        s = super().minimumSizeHint()
        filhos = [self.widget(i) for i in range(self.count()) if not self.widget(i).isHidden()]
        if not filhos:
            return s
        if self.orientation() == Qt.Horizontal:
            h = max(self._altura_filho(w, w.width()) for w in filhos)
        else:
            h = sum(self._altura_filho(w, self.width()) for w in filhos) + self.handleWidth() * (len(filhos) - 1)
        return QSize(s.width(), max(s.height(), h))

    def sizeHint(self) -> QSize:  # noqa: N802
        s = super().sizeHint()
        return QSize(s.width(), max(s.height(), self.minimumSizeHint().height()))


class FluxoLayout(QLayout):
    """Linha que quebra: os itens ficam lado a lado e descem para a linha seguinte quando nao cabem
    (botoes e campos em janela estreita, 390 px). Baseado no exemplo FlowLayout da documentacao do Qt."""

    def __init__(self, espaco: int = 8):
        super().__init__()
        self._itens = []
        self.setSpacing(espaco)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802
        self._itens.append(item)

    def addSpacing(self, largura: int) -> None:  # noqa: N802 - compatibilidade com QHBoxLayout
        w = QWidget()
        w.setFixedSize(largura, 1)
        self.addWidget(w)

    def addStretch(self, _f: int = 0) -> None:  # noqa: N802 - o fluxo ja alinha a esquerda
        pass

    def count(self) -> int:
        return len(self._itens)

    def itemAt(self, i):  # noqa: N802
        return self._itens[i] if 0 <= i < len(self._itens) else None

    def takeAt(self, i):  # noqa: N802
        return self._itens.pop(i) if 0 <= i < len(self._itens) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, largura: int) -> int:  # noqa: N802
        return self._arrumar(QRect(0, 0, largura, 0), so_medir=True)

    def setGeometry(self, r) -> None:  # noqa: N802
        super().setGeometry(r)
        self._arrumar(r, so_medir=False)
        if r.width() != getattr(self, "_largura", 0):
            self._largura = r.width()
            h = self.heightForWidth(r.width())
            if h != getattr(self, "_altura", None):
                self._altura = h
                QTimer.singleShot(0, self._propagar)  # os pais reservam a nova altura (nada se sobrepoe)

    def _propagar(self) -> None:
        self.invalidate()
        w = self.parentWidget()
        while w is not None:
            w.updateGeometry()
            w = w.parentWidget()

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        """Largura: a do maior item. Altura: a necessaria na largura atual (com as linhas quebradas)."""
        s = QSize()
        for it in self._itens:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        s = s + QSize(m.left() + m.right(), m.top() + m.bottom())
        larg = getattr(self, "_largura", 0)
        if larg > 0:
            s.setHeight(max(s.height(), self.heightForWidth(larg)))
        return s

    def _arrumar(self, r, so_medir: bool) -> int:
        m = self.contentsMargins()
        area = r.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, alt_linha = area.x(), area.y(), 0
        esp = self.spacing()
        linhas: list[list] = [[]]
        for it in self._itens:
            w = it.widget()
            if w is not None and w.isHidden():
                continue
            hint = it.sizeHint()
            prox = x + hint.width()
            if prox > area.right() + 1 and alt_linha > 0:
                x = area.x()
                y += alt_linha + esp
                prox = x + hint.width()
                alt_linha = 0
                linhas.append([])
            linhas[-1].append((it, x, y, hint))
            x = prox + esp
            alt_linha = max(alt_linha, hint.height())
        if not so_medir:
            for linha_ in linhas:  # centraliza verticalmente os itens de cada linha
                if not linha_:
                    continue
                h = max(hh.height() for _, _, _, hh in linha_)
                for it, xx, yy, hint in linha_:
                    larg = min(hint.width(), area.width())
                    it.setGeometry(QRect(QPoint(xx, yy + (h - hint.height()) // 2), QSize(larg, hint.height())))
        return y + alt_linha - r.y() + m.bottom()


# ----------------------------------------------------------------------
# Notificacoes nao bloqueantes (toasts)
# ----------------------------------------------------------------------
class _Toast(QFrame):
    def __init__(self, pai: QWidget, texto: str, tipo: str, duracao_ms: int):
        super().__init__(pai)
        from . import tema
        from .componentes import etiqueta

        self.setObjectName("toast")
        self.setProperty("tipo", tipo)  # cor da borda vem da folha de estilo (tokens)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        # o tipo tambem aparece em texto (nada so por cor)
        lay.addWidget(
            etiqueta(
                {"ok": "Concluído", "aviso": "Atenção", "erro": "Erro"}.get(tipo, "Aviso"),
                {"ok": "ok", "aviso": "pendente", "erro": "erro"}.get(tipo, "info"),
            ),
            0,
            Qt.AlignTop,
        )
        self.rotulo = QLabel(texto)
        self.rotulo.setWordWrap(True)
        self.rotulo.setMaximumWidth(420)
        lay.addWidget(self.rotulo)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.mousePressEvent = lambda _e: self.fechar()  # clique fecha
        self._efeito = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._efeito)
        self._anim = QPropertyAnimation(self._efeito, b"opacity", self)
        self._anim.setDuration(0 if tema.ESTADO["reduzir_movimento"] else int(tema.TOKENS["movimento"]["toast_ms"]))
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        QTimer.singleShot(duracao_ms, self.fechar)

    def fechar(self) -> None:
        if getattr(self, "_fechando", False):
            return
        self._fechando = True
        self._anim.stop()
        self._anim.setStartValue(self._efeito.opacity())
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._remover)
        self._anim.start()

    def _remover(self) -> None:
        pai = self.parentWidget()
        self.hide()
        self.deleteLater()
        if pai is not None and hasattr(pai, "_toasts"):
            pai._toasts = [t for t in pai._toasts if t is not self]
            posicionar_toasts(pai)


def posicionar_toasts(janela: QWidget) -> None:
    y = janela.height() - 18
    for t in reversed(getattr(janela, "_toasts", [])):
        t.adjustSize()
        y -= t.height()
        t.move(janela.width() - t.width() - 22, y)
        y -= 8


def mostrar_toast(origem: QWidget | None, texto: str, tipo: str = "info", duracao_ms: int | None = None) -> None:
    """Aviso simples que some sozinho (clique para fechar). Diálogos ficam para erros e confirmacoes."""
    janela = origem.window() if origem is not None else None
    if janela is None:
        log.info("Aviso: %s", texto)
        return
    duracao_ms = duracao_ms or min(9000, 3500 + 35 * len(texto))
    t = _Toast(janela, texto, tipo, duracao_ms)
    janela._toasts = (getattr(janela, "_toasts", []) + [t])[-4:]
    t.show()
    t.raise_()
    posicionar_toasts(janela)
    log.info("Aviso exibido: %s", texto.replace("\n", " "))


# ----------------------------------------------------------------------
# Indicador de atividade (operacoes acima de 200 ms)
# ----------------------------------------------------------------------
class Atividade(QObject):
    """Conta as tarefas em andamento. A janela mostra o indicador so se algo passar de 200 ms."""

    mudou = Signal(int, str)
    ATRASO_MS = 200

    def __init__(self):
        super().__init__()
        self.ativas: dict[int, str] = {}
        self._seq = 0

    def iniciar(self, descricao: str = "") -> int:
        self._seq += 1
        self.ativas[self._seq] = descricao
        self.mudou.emit(len(self.ativas), descricao)
        return self._seq

    def terminar(self, ident: int) -> None:
        self.ativas.pop(ident, None)
        desc = next(reversed(self.ativas.values()), "") if self.ativas else ""
        self.mudou.emit(len(self.ativas), desc)


ATIVIDADE = Atividade()


class IndicadorAtividade(QWidget):
    """Barra fina indeterminada + texto, exibida apos 200 ms de atividade."""

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 0, 28, 0)
        lay.setSpacing(10)
        self.barra = QProgressBar()
        self.barra.setObjectName("ocupado")
        self.barra.setRange(0, 0)
        self.barra.setTextVisible(False)
        self.barra.setFixedHeight(4)
        self.texto = QLabel("")
        self.texto.setObjectName("secundario")
        lay.addWidget(self.barra, 1)
        lay.addWidget(self.texto)
        self.setFixedHeight(18)
        self.barra.hide()
        self.texto.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._mostrar)
        self._n = 0
        ATIVIDADE.mudou.connect(self._mudou)

    def _mudou(self, n: int, desc: str) -> None:
        self._n = n
        self.texto.setText(desc or "Processando...")
        if n and not self.barra.isVisible() and not self._timer.isActive():
            self._timer.start(Atividade.ATRASO_MS)
        if not n:
            self._timer.stop()
            self.barra.hide()
            self.texto.hide()

    def _mostrar(self) -> None:
        if self._n:
            self.barra.show()
            self.texto.show()

    @property
    def visivel(self) -> bool:
        return self.barra.isVisible()


# ----------------------------------------------------------------------
# Dados ao vivo
# ----------------------------------------------------------------------
class MonitorDados(QObject):
    """Confere a cada segundo se manifesto, cadastro, vocabulario, splits ou runs mudaram (data e tamanho
    do arquivo; custo desprezivel) e avisa a pagina aberta. Pega tambem mudancas feitas por scripts de
    linha de comando ou por outro processo (treino). Polling em vez de QFileSystemWatcher porque a
    escrita atomica (substituir o arquivo) faz o watcher perder o arquivo no Windows."""

    mudou = Signal(set)

    def __init__(self, cfg, intervalo_ms: int = 1000):
        super().__init__()
        self.cfg = cfg
        self._assinaturas: dict[str, tuple] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.verificar)
        self._timer.start(intervalo_ms)
        self.verificar(emitir=False)

    def alvos(self) -> dict[str, list[Path]]:
        c = self.cfg
        runs = c.caminho("runs")
        return {
            "manifesto": [c.caminho("manifesto")],
            "sinalizadores": [c.caminho("sinalizadores")],
            "vocabulario": [c.caminho("vocabulario")],
            "splits": [c.caminho("splits")],
            "runs": [runs] + ([p for p in runs.iterdir() if p.is_dir()] if runs.exists() else []),
        }

    @staticmethod
    def _assinatura(caminhos: list[Path]) -> tuple:
        out = []
        for p in caminhos:
            try:
                s = p.stat()
                out.append((str(p), s.st_mtime_ns, s.st_size))
            except OSError:
                out.append((str(p), None, None))
        return tuple(out)

    def verificar(self, emitir: bool = True) -> set:
        mudaram = set()
        try:
            for nome, caminhos in self.alvos().items():
                a = self._assinatura(caminhos)
                if self._assinaturas.get(nome) != a:
                    if nome in self._assinaturas:
                        mudaram.add(nome)
                    self._assinaturas[nome] = a
        except Exception:  # noqa: BLE001 - monitor nunca derruba a interface
            log.debug("Falha no monitor de dados", exc_info=True)
        if mudaram and emitir:
            self.mudou.emit(mudaram)
        return mudaram


# ----------------------------------------------------------------------
# Transicao entre abas
# ----------------------------------------------------------------------
def animar_entrada(w: QWidget, duracao_ms: int = 150) -> None:
    """Fade-in curto da pagina. O efeito e removido ao fim (efeitos de opacidade deixam a pintura de
    video mais cara se ficarem ligados)."""
    if duracao_ms <= 0:
        return
    ef = QGraphicsOpacityEffect(w)
    ef.setOpacity(0.25)
    w.setGraphicsEffect(ef)
    anim = QPropertyAnimation(ef, b"opacity", w)
    anim.setDuration(duracao_ms)
    anim.setStartValue(0.25)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)

    def fim():
        if w.graphicsEffect() is ef:
            w.setGraphicsEffect(None)

    anim.finished.connect(fim)
    anim.start(QPropertyAnimation.DeleteWhenStopped)
