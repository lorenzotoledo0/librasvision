"""Editor de corte da aba Revisao (item 9), no estilo de editores de video, com uma unica funcao:
cortar. Previa grande com esqueleto, barra de controles e linha do tempo com regua, miniaturas,
curva de movimento das maos, cabeca de reproducao e alcas de inicio e fim.

A interface so apresenta e chama funcoes: o estado e as regras ficam em coleta.edicao_corte
(SessaoCorte), a leitura de quadros em coleta.leitor_video e a gravacao em coleta.cortes.
Nada e gravado em disco ate "Salvar corte".
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import QCheckBox, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ..coleta.edicao_corte import SessaoCorte, formatar_tempo
from ..coleta.leitor_video import LeitorVideo
from ..ui import tema
from ..ui.icones import icone
from .base import TelaVideo, botao, iniciar_tarefa, linha, rotulo

log = logging.getLogger("libras.app")


def _pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


# ----------------------------------------------------------------------
# Decodificacao em segundo plano (so o quadro mais recente pedido)
# ----------------------------------------------------------------------
class Decodificador(QThread):
    pronto = Signal(int, object)

    def __init__(self, caminho: str):
        super().__init__()
        self.setObjectName("decodificador de video")
        self.leitor = LeitorVideo(caminho)
        self._pedido: int | None = None
        self._ev = threading.Event()
        self._parar = False

    def pedir(self, i: int) -> None:
        self._pedido = int(i)
        self._ev.set()

    def run(self) -> None:
        while not self._parar:
            self._ev.wait(0.2)
            self._ev.clear()
            i, self._pedido = self._pedido, None
            if i is None or self._parar:
                continue
            try:
                f = self.leitor.quadro(i)
            except Exception:  # noqa: BLE001
                log.exception("Falha ao decodificar o quadro %s", i)
                f = None
            self.pronto.emit(i, f)

    def encerrar(self) -> None:
        """Fecha o arquivo (no Windows, arquivo aberto nao pode ir para a lixeira)."""
        self._parar = True
        self._ev.set()
        self.wait(3000)
        self.leitor.fechar()


# ----------------------------------------------------------------------
# Linha do tempo
# ----------------------------------------------------------------------
class LinhaDoTempo(QWidget):
    """Regua, miniaturas, curva de movimento, trecho mantido com alcas, divisao e playhead."""

    playhead_movido = Signal(int)
    inicio_arrastado = Signal(int)  # fronteira (primeiro quadro mantido)
    fim_arrastado = Signal(int)  # fronteira exclusiva (ultimo mantido + 1)
    arraste_iniciado = Signal(str)
    arraste_concluido = Signal(str)

    ESQ = DIR = 14
    Y_REGUA, H_REGUA = 0, 22
    Y_FAIXA, H_FAIXA = 26, 58
    Y_CURVA, H_CURVA = 88, 30
    ALCA = 10

    def __init__(self):
        super().__init__()
        self.sessao: SessaoCorte | None = None
        self.playhead = 0
        self.miniaturas: list[tuple[int, QPixmap]] = []
        self.curva: np.ndarray | None = None
        self._arraste: str | None = None
        self._sobre: str | None = None
        self.setMinimumHeight(self.Y_CURVA + self.H_CURVA + 8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------------ geometria
    @property
    def n(self) -> int:
        return self.sessao.n if self.sessao else 0

    def _largura(self) -> float:
        return max(1.0, self.width() - self.ESQ - self.DIR)

    def x_fronteira(self, q: float) -> float:
        """x da fronteira antes do quadro q (q em [0, n])."""
        return self.ESQ + (q / max(1, self.n)) * self._largura()

    def x_quadro(self, i: int) -> float:
        return self.x_fronteira(i + 0.5)

    def quadro_em(self, x: float) -> int:
        return int(np.clip(np.floor((x - self.ESQ) / self._largura() * self.n), 0, max(0, self.n - 1)))

    def fronteira_em(self, x: float) -> int:
        """Encaixa no quadro mais proximo."""
        return int(np.clip(round((x - self.ESQ) / self._largura() * self.n), 0, self.n))

    def _alvo(self, x: float, y: float) -> str | None:
        if not self.sessao:
            return None
        if abs(x - self.x_fronteira(self.sessao.inicio)) <= self.ALCA:
            return "inicio"
        if abs(x - self.x_fronteira(self.sessao.fim)) <= self.ALCA:
            return "fim"
        if y < self.Y_FAIXA or abs(x - self.x_quadro(self.playhead)) <= 5:
            return "playhead"
        return "faixa"

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if not self.sessao or ev.button() != Qt.LeftButton:
            return
        x, y = ev.position().x(), ev.position().y()
        alvo = self._alvo(x, y)
        if alvo in ("inicio", "fim"):
            self._arraste = alvo
            self.arraste_iniciado.emit(alvo)
        else:  # clique na faixa ou na regua move o playhead, e arrastar continua movendo
            self._arraste = "playhead"
            self.playhead_movido.emit(self.quadro_em(x))

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if not self.sessao:
            return
        x, y = ev.position().x(), ev.position().y()
        if self._arraste == "inicio":
            self.inicio_arrastado.emit(self.fronteira_em(x))
        elif self._arraste == "fim":
            self.fim_arrastado.emit(self.fronteira_em(x))
        elif self._arraste == "playhead":
            self.playhead_movido.emit(self.quadro_em(x))
        else:
            alvo = self._alvo(x, y)
            if alvo != self._sobre:
                self._sobre = alvo
                self.setCursor(Qt.SizeHorCursor if alvo in ("inicio", "fim") else Qt.PointingHandCursor)
                self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._arraste in ("inicio", "fim"):
            self.arraste_concluido.emit(self._arraste)
        self._arraste = None

    def leaveEvent(self, _ev) -> None:  # noqa: N802
        self._sobre = None
        self.update()

    # ------------------------------------------------------------------ pintura
    def _passo_regua(self) -> float:
        dur = self.sessao.duracao() if self.sessao else 1.0
        px_por_s = self._largura() / max(dur, 1e-6)
        for passo in (0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 30, 60):
            if passo * px_por_s >= 70:
                return passo
        return 60.0

    def paintEvent(self, _ev) -> None:  # noqa: N802
        c = tema.CORES
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        esq, larg = self.ESQ, self._largura()
        p.fillRect(self.rect(), QColor(c["painel"]))
        faixa = QRectF(esq, self.Y_FAIXA, larg, self.H_FAIXA)
        banda = QRectF(esq, self.Y_CURVA, larg, self.H_CURVA)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["fundo"]))
        p.drawRoundedRect(faixa, 6, 6)
        p.drawRoundedRect(banda, 6, 6)
        if not self.sessao:
            p.setPen(QColor(c["texto3"]))
            p.drawText(faixa, Qt.AlignCenter, "Selecione um take")
            return
        s = self.sessao
        # regua
        p.setPen(QPen(QColor(c["texto3"]), 1))
        passo = self._passo_regua()
        fonte = p.font()
        fonte.setPointSizeF(max(7.0, fonte.pointSizeF() - 2))
        p.setFont(fonte)
        t, dur = 0.0, s.duracao()
        while t <= dur + 1e-9:
            x = esq + t / dur * larg
            p.drawLine(QPointF(x, self.H_REGUA - 7), QPointF(x, self.H_REGUA))
            p.drawText(
                QRectF(x + 3, 2, 80, self.H_REGUA - 6),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{t:.2f} s" if passo < 1 else f"{t:.0f} s",
            )
            t += passo
        # miniaturas
        p.save()
        p.setClipRect(faixa)
        if self.miniaturas:
            k = len(self.miniaturas)
            for j, (_i, pm) in enumerate(self.miniaturas):
                alvo = QRectF(esq + j * larg / k, self.Y_FAIXA, larg / k + 1, self.H_FAIXA)
                fonte_r = QRectF(pm.rect())
                proporcao = alvo.width() / alvo.height()
                if fonte_r.width() / fonte_r.height() > proporcao:  # recorta no centro
                    nova = fonte_r.height() * proporcao
                    fonte_r = QRectF(fonte_r.x() + (fonte_r.width() - nova) / 2, fonte_r.y(), nova, fonte_r.height())
                p.drawPixmap(alvo, pm, fonte_r)
        else:
            p.setPen(QColor(c["texto3"]))
            p.drawText(faixa, Qt.AlignCenter, "Gerando miniaturas...")
        p.restore()
        # curva de movimento das maos
        if self.curva is not None and len(self.curva) == s.n and s.n > 1:
            mx = float(np.max(self.curva)) or 1.0
            pts = [
                QPointF(self.x_quadro(i), self.Y_CURVA + self.H_CURVA - 3 - (v / mx) * (self.H_CURVA - 8))
                for i, v in enumerate(self.curva)
            ]
            area = QPainterPath(QPointF(pts[0].x(), self.Y_CURVA + self.H_CURVA - 2))
            for q in pts:
                area.lineTo(q)
            area.lineTo(QPointF(pts[-1].x(), self.Y_CURVA + self.H_CURVA - 2))
            cor = QColor(c["acento1"])
            cor.setAlpha(55)
            p.setPen(Qt.NoPen)
            p.setBrush(cor)
            p.drawPath(area)
            p.setPen(QPen(QColor(c["acento1"]), 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF(pts))
        p.setPen(QColor(c["texto3"]))
        p.drawText(banda.adjusted(6, 0, 0, 0), Qt.AlignLeft | Qt.AlignTop, "movimento das mãos")
        # fora do trecho: escurecido
        topo, base = self.Y_FAIXA, self.Y_CURVA + self.H_CURVA
        xi, xf = self.x_fronteira(s.inicio), self.x_fronteira(s.fim)
        sombra = tema.qcolor("sobreposicao_fora_do_corte")
        p.setPen(Qt.NoPen)
        p.setBrush(sombra)
        if xi > esq:
            p.drawRect(QRectF(esq, topo, xi - esq, base - topo))
        if xf < esq + larg:
            p.drawRect(QRectF(xf, topo, esq + larg - xf, base - topo))
        # contorno do trecho mantido e alcas
        grad = QLinearGradient(xi, 0, xf, 0)
        grad.setColorAt(0, QColor(c["acento1"]))
        grad.setColorAt(1, QColor(c["acento2"]))
        p.setPen(QPen(QBrush(grad), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(xi, topo - 1, xf - xi, base - topo + 2), 6, 6)
        for nome, x, cor in (("inicio", xi, c["acento1"]), ("fim", xf, c["acento2"])):
            q = QColor(cor).lighter(130) if (self._sobre == nome or self._arraste == nome) else QColor(cor)
            r = QRectF(x - self.ALCA / 2, topo - 1, self.ALCA, base - topo + 2)
            p.setPen(Qt.NoPen)
            p.setBrush(q)
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QPen(tema.qcolor("texto_sobre_acento"), 1.2))
            for dy in (-6, 0, 6):
                y = (topo + base) / 2 + dy
                p.drawLine(QPointF(x - 1.5, y), QPointF(x + 1.5, y))
        # divisao
        if s.divisao is not None:
            xd = self.x_fronteira(s.divisao)
            p.setPen(QPen(tema.qcolor("texto_sobre_acento"), 1.5, Qt.DashLine))
            p.drawLine(QPointF(xd, topo), QPointF(xd, base))
            p.setPen(tema.qcolor("texto_sobre_acento"))
            p.drawText(QRectF(xd + 4, topo + 2, 80, 16), Qt.AlignLeft, "divisão")
        # playhead
        xp = self.x_quadro(self.playhead)
        p.setPen(QPen(QColor(c["texto"]), 2))
        p.drawLine(QPointF(xp, 6), QPointF(xp, base + 2))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["texto"]))
        p.drawPolygon(QPolygonF([QPointF(xp - 6, 2), QPointF(xp + 6, 2), QPointF(xp, 10)]))
        p.end()


# ----------------------------------------------------------------------
# Editor
# ----------------------------------------------------------------------
class EditorCorte(QWidget):
    """Previa + controles + linha do tempo + acoes. Emite salvar_pedido(inicio, fim, divisao)."""

    salvar_pedido = Signal(int, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sessao: SessaoCorte | None = None
        self.dados = None
        self.dec: Decodificador | None = None
        self.video_id: str | None = None
        self.caminho: str | None = None
        self.atual = 0
        self._ultimo_quadro: np.ndarray | None = None
        self._tocando = False
        self._t0 = 0.0
        self._q0 = 0
        self._aviso_limite = ""
        self._ger_mini = 0
        self.encerrado = False
        self.setFocusPolicy(Qt.StrongFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.tela = TelaVideo("Selecione um take na lista")
        self.tela.setMinimumSize(320, 220)
        lay.addWidget(self.tela, 1)

        def ic(nome, texto, dica, acao):
            """Botao com icone E rotulo curto (nada so no hover; alvo de toque com texto)."""
            b = QToolButton()
            b.setObjectName("acao")
            b.setIcon(icone(nome, tema.CORES["texto"], tema.CORES["sobre_acento"]))
            b.setText(texto)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setToolTip(dica)
            b.setAccessibleName(texto)
            b.setFocusPolicy(Qt.NoFocus)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(acao)
            return b

        self.bt_inicio = ic("inicio_video", "Início", "Ir ao início (Home)", lambda: self.ir(0))
        self.bt_menos1s = ic(
            "quadro_ant",
            "-1 s",
            "Voltar 1 segundo (Shift + seta para a esquerda)",
            lambda: self.passo(-self._um_segundo()),
        )
        self.bt_ant = ic("quadro_ant", "Quadro", "Quadro anterior (seta para a esquerda)", lambda: self.passo(-1))
        self.bt_play = ic("play", "Reproduzir", "Reproduzir / pausar (Espaço)", self.alternar)
        self.bt_prox = ic("quadro_prox", "Quadro", "Próximo quadro (seta para a direita)", lambda: self.passo(1))
        self.bt_mais1s = ic(
            "quadro_prox",
            "+1 s",
            "Avançar 1 segundo (Shift + seta para a direita)",
            lambda: self.passo(self._um_segundo()),
        )
        self.bt_fim = ic(
            "fim_video", "Fim", "Ir ao fim (End)", lambda: self.ir(self.sessao.n - 1 if self.sessao else 0)
        )
        self.lb_tempo = rotulo("00:00.00 / 00:00.00", "titulo_painel")
        self.lb_quadro = rotulo("", "secundario")
        self.ck_esq = QCheckBox("Esqueleto")
        self.ck_esq.setChecked(True)
        self.ck_esq.setFocusPolicy(Qt.NoFocus)
        self.ck_esq.toggled.connect(lambda _: self._mostrar_ultimo())
        lay.addLayout(
            linha(self.bt_inicio, self.bt_menos1s, self.bt_ant, self.bt_play, self.bt_prox, self.bt_mais1s, self.bt_fim)
        )
        lay.addLayout(linha(self.lb_tempo, self.lb_quadro, self.ck_esq))
        self.linha_tempo = LinhaDoTempo()
        self.linha_tempo.playhead_movido.connect(self.ir)
        self.linha_tempo.arraste_iniciado.connect(self._arraste_inicio)
        self.linha_tempo.arraste_concluido.connect(self._arraste_fim)
        self.linha_tempo.inicio_arrastado.connect(self._arrastar_inicio)
        self.linha_tempo.fim_arrastado.connect(self._arrastar_fim)
        lay.addWidget(self.linha_tempo)

        def acao(texto, f, dica="", primario=False):
            b = botao(texto, primario=primario, acao=f, dica=dica)
            b.setFocusPolicy(Qt.NoFocus)
            return b

        self.bt_def_ini = acao("Definir início aqui", self.definir_inicio_aqui, "Atalho: I")
        self.bt_def_fim = acao("Definir fim aqui", self.definir_fim_aqui, "Atalho: O")
        self.bt_dividir = acao("Dividir aqui", self.dividir_aqui, "Divide em dois takes no playhead. Atalho: S")
        self.bt_desfazer = ic("desfazer", "Desfazer", "Desfazer (Ctrl+Z)", self.desfazer)
        self.bt_refazer = ic("refazer", "Refazer", "Refazer (Ctrl+Y)", self.refazer)
        self.bt_cancelar = acao("Cancelar", self.cancelar, "Volta ao take sem edição (nada foi gravado)")
        self.bt_salvar = acao(
            "Salvar corte", self.salvar, "Grava o corte: o original vai para a lixeira", primario=True
        )
        lay.addLayout(
            linha(
                self.bt_def_ini, self.bt_def_fim, self.bt_dividir, self.bt_desfazer, self.bt_refazer, esticar_fim=True
            )
        )
        self.lb_info = rotulo("", "secundario", quebra=True)
        lay.addWidget(self.lb_info)
        lay.addLayout(linha(self.bt_cancelar, self.bt_salvar))
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tique)
        self._atalhos()
        self._atualizar()

    # ------------------------------------------------------------------
    def _atalhos(self) -> None:
        mapa = [
            ("Space", self.alternar),
            ("Left", lambda: self.passo(-1)),
            ("Right", lambda: self.passo(1)),
            ("Shift+Left", lambda: self.passo(-self._um_segundo())),
            ("Shift+Right", lambda: self.passo(self._um_segundo())),
            ("Home", lambda: self.ir(0)),
            ("End", lambda: self.ir(self.sessao.n - 1 if self.sessao else 0)),
            ("I", self.definir_inicio_aqui),
            ("O", self.definir_fim_aqui),
            ("S", self.dividir_aqui),
            ("Ctrl+Z", self.desfazer),
            ("Ctrl+Y", self.refazer),
            ("Ctrl+Shift+Z", self.refazer),
        ]
        self.atalhos = {}
        for tecla, f in mapa:
            s = QShortcut(QKeySequence(tecla), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(f)
            self.atalhos[tecla] = s

    def _um_segundo(self) -> int:
        return self.sessao.quadros_por_segundo() if self.sessao else 30

    # ------------------------------------------------------------------ carregar
    def carregar(self, video_id: str, caminho: str, dados=None, fps: float = 30.0) -> None:
        from ..normalizacao import movimento_maos_por_frame

        self.liberar()
        if self.encerrado:  # aplicativo fechando: um retorno atrasado de tarefa nao reabre o video
            return
        dec = Decodificador(caminho)
        n = dados.partes.n if dados is not None else dec.leitor.n
        tempos = dados.tempos if dados is not None else None
        self.sessao = SessaoCorte(n, fps or dec.leitor.fps, tempos)
        self.dados, self.dec, self.video_id, self.caminho = dados, dec, video_id, caminho
        dec.pronto.connect(self._quadro_pronto)
        dec.start()
        self.linha_tempo.sessao = self.sessao
        self.linha_tempo.miniaturas = []
        self.linha_tempo.curva = (
            movimento_maos_por_frame(dados.partes, dados.largura, dados.altura) if dados is not None else None
        )
        self.atual = 0
        self._aviso_limite = ""
        self.ir(0)
        self._gerar_miniaturas()
        self._atualizar()
        self.setFocus()

    def _gerar_miniaturas(self) -> None:
        from ..coleta.leitor_video import miniaturas

        if not self.caminho:
            return
        self._ger_mini += 1
        ger, caminho = self._ger_mini, self.caminho
        qtd = int(np.clip(self.linha_tempo.width() // 64, 8, 40))

        def pronto(lista):
            if ger == self._ger_mini and self.caminho == caminho:
                self.linha_tempo.miniaturas = [(i, _pixmap(img)) for i, img in lista]
                self.linha_tempo.update()

        iniciar_tarefa(
            self,
            lambda p, c: miniaturas(caminho, qtd),
            pronto,
            lambda e: log.warning("Falha nas miniaturas: %s", e),
            descricao="Gerando miniaturas...",
        )

    def liberar(self) -> None:
        """Para a reproducao e fecha o arquivo de video (antes de salvar, excluir ou trocar de take)."""
        self.pausar()
        if self.dec is not None:
            self.dec.encerrar()
            self.dec = None
        self._ger_mini += 1

    def limpar(self) -> None:
        self.liberar()
        self.sessao = self.dados = None
        self.video_id = self.caminho = None
        self.linha_tempo.sessao = None
        self.linha_tempo.miniaturas = []
        self.linha_tempo.curva = None
        self.tela.clear()
        self.tela.setText("Selecione um take na lista")
        self._atualizar()

    @property
    def tem_alteracoes(self) -> bool:
        return bool(self.sessao and self.sessao.alterado)

    # ------------------------------------------------------------------ reproducao
    def ir(self, i: int) -> None:
        if not self.sessao:
            return
        self.atual = int(np.clip(i, 0, self.sessao.n - 1))
        self.linha_tempo.playhead = self.atual
        self.linha_tempo.update()
        if self.dec is not None:
            self.dec.pedir(self.atual)
        self._atualizar_tempo()

    def passo(self, d: int) -> None:
        self.pausar()
        self.ir(self.atual + int(d))

    def alternar(self) -> None:
        if self._tocando:
            self.pausar()
        else:
            self.tocar()

    def tocar(self) -> None:
        if not self.sessao:
            return
        if self.atual >= self.sessao.n - 1:
            self.ir(self.sessao.inicio)
        self._tocando = True
        self._t0, self._q0 = time.perf_counter(), self.atual
        self.bt_play.setIcon(icone("pausa", tema.CORES["texto"], tema.CORES["sobre_acento"]))
        self.bt_play.setText("Pausar")
        self.timer.start(10)

    def pausar(self) -> None:
        self._tocando = False
        self.timer.stop()
        self.bt_play.setIcon(icone("play", tema.CORES["texto"], tema.CORES["sobre_acento"]))
        self.bt_play.setText("Reproduzir")

    def _tique(self) -> None:
        """Reproducao no FPS real do take: o quadro mostrado vem do tempo decorrido."""
        if not self.sessao:
            return
        alvo = self._q0 + int((time.perf_counter() - self._t0) * self.sessao.fps)
        if alvo >= self.sessao.n:
            self.ir(self.sessao.n - 1)
            self.pausar()
        elif alvo != self.atual:
            self.ir(alvo)

    def _quadro_pronto(self, i: int, f) -> None:
        if f is None or self.sessao is None or self.sender() is not self.dec:
            return
        self._ultimo_quadro = f
        self._ultimo_indice = i
        self._mostrar_ultimo()

    def _mostrar_ultimo(self) -> None:
        f = self._ultimo_quadro
        if f is None or self.sessao is None:
            return
        i = getattr(self, "_ultimo_indice", 0)
        if self.ck_esq.isChecked() and self.dados is not None and i < self.dados.partes.n:
            from ..landmarks import desenhar_esqueleto

            pt = self.dados.partes
            f = desenhar_esqueleto(f, pt.pose[i], pt.mao_esq[i], pt.mao_dir[i], pt.presenca[i])
        img = cv2.flip(f, 1)  # espelhado so na tela, como na gravacao
        if not (self.sessao.inicio <= i < self.sessao.fim):
            img = (img * 0.35).astype(img.dtype)
        self.tela.mostrar(img)

    # ------------------------------------------------------------------ edicao
    def _arraste_inicio(self, _alca: str) -> None:
        self.pausar()
        if self.sessao:
            self.sessao.iniciar_arrasto()

    def _arraste_fim(self, _alca: str) -> None:
        if self.sessao:
            self.sessao.concluir_arrasto()
        self._atualizar()

    def _arrastar_inicio(self, q: int) -> None:
        r = self.sessao.definir_inicio(q)
        self._aviso_limite = (
            f"A alça para no limite: mínimo de {self.sessao.minimo} quadros por trecho." if r != q else ""
        )
        self.ir(r)
        self._atualizar()

    def _arrastar_fim(self, q: int) -> None:
        r = self.sessao.definir_fim(q)
        self._aviso_limite = (
            f"A alça para no limite: mínimo de {self.sessao.minimo} quadros por trecho." if r != q else ""
        )
        self.ir(r - 1)
        self._atualizar()

    def definir_inicio_aqui(self) -> None:
        if self.sessao:
            r = self.sessao.definir_inicio_no_quadro(self.atual)
            self._aviso_limite = (
                ""
                if r == self.atual
                else (f"Início ajustado para o quadro {r + 1}: mínimo de {self.sessao.minimo} quadros por trecho.")
            )
            self._atualizar()

    def definir_fim_aqui(self) -> None:
        if self.sessao:
            r = self.sessao.definir_fim_no_quadro(self.atual)
            self._aviso_limite = (
                ""
                if r == self.atual
                else (f"Fim ajustado para o quadro {r + 1}: mínimo de {self.sessao.minimo} quadros por trecho.")
            )
            self._atualizar()

    def dividir_aqui(self) -> None:
        if not self.sessao:
            return
        if not self.sessao.dividir(self.atual):
            a, b = self.sessao.limites_divisao()
            self._aviso_limite = (
                f"Não dá para dividir no quadro {self.atual + 1}: cada take precisa de "
                f"{self.sessao.minimo} quadros (divida entre os quadros {a + 1} e {b + 1})."
            )
        else:
            self._aviso_limite = ""
        self._atualizar()

    def desfazer(self) -> None:
        if self.sessao and self.sessao.desfazer():
            self._aviso_limite = ""
            self._atualizar()

    def refazer(self) -> None:
        if self.sessao and self.sessao.refazer():
            self._aviso_limite = ""
            self._atualizar()

    def cancelar(self) -> None:
        if self.sessao:
            self.sessao.cancelar()
            self._aviso_limite = ""
            self._atualizar()

    def salvar(self) -> None:
        if self.sessao and self.sessao.alterado:
            self.pausar()
            self.salvar_pedido.emit(self.sessao.inicio, self.sessao.fim, self.sessao.divisao)

    # ------------------------------------------------------------------
    def _atualizar_tempo(self) -> None:
        s = self.sessao
        if not s:
            self.lb_tempo.setText("00:00.00 / 00:00.00")
            self.lb_quadro.setText("")
            return
        self.lb_tempo.setText(f"{formatar_tempo(s.tempo(self.atual))} / {formatar_tempo(s.duracao())}")
        self.lb_quadro.setText(f"quadro {self.atual + 1} de {s.n}")

    def _atualizar(self) -> None:
        s = self.sessao
        tem = s is not None
        for b in (
            self.bt_inicio,
            self.bt_ant,
            self.bt_play,
            self.bt_prox,
            self.bt_fim,
            self.bt_def_ini,
            self.bt_def_fim,
            self.bt_dividir,
        ):
            b.setEnabled(tem)
        self.bt_desfazer.setEnabled(tem and s.pode_desfazer)
        self.bt_refazer.setEnabled(tem and s.pode_refazer)
        self.bt_cancelar.setEnabled(tem and s.alterado)
        self.bt_salvar.setEnabled(tem and s.alterado)
        if tem:
            txt = s.descricao()
            if s.alterado:
                txt += " Alterações ainda não gravadas."
            if self._aviso_limite:
                txt += "\n" + self._aviso_limite
            self.lb_info.setText(txt)
        else:
            self.lb_info.setText("")
        self.linha_tempo.update()
        self._atualizar_tempo()
        self._mostrar_ultimo()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._mostrar_ultimo()
