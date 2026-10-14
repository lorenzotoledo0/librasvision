"""Tela unica do app real (13.3): camera ocupando a janela, legenda sobre o video, botao de
Configuracoes no canto superior direito e barra minima embaixo. Sem abas nem barra lateral.

O reconhecimento roda no subprocesso de tempo real (D15), iniciado ao ligar a camera e encerrado ao
parar ou fechar; esta interface nunca importa TensorFlow nem MediaPipe. A decisao (limiar, media movel,
confirmacoes, minimo de maos) vem do pacote do modelo; o usuario nao a ajusta.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QFont, QFontMetricsF, QGuiApplication, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, config_real
from .. import pacote as pk
from ..camera import explicar_falha
from ..tempo_real_servico import Decodificador
from ..ui import tema
from ..ui.componentes import Dialogo, Subprocesso, avisar, botao, etiqueta, mudar_etiqueta, perguntar, rotulo
from ..ui.icones import icone, svg
from ..ui.legenda import CamadaLegenda
from ..ui.sobreposicoes import Ret, SuavizadorCabeca, cabeca_na_tela, lado_na_tela, posicionar
from ..utils import hardware, memoria_processo_mb
from . import tecnicas
from .guia import GuiaAvisos, itens_primeira_abertura
from .modelos import ControleModelos
from .preferencias import Preferencias

log = logging.getLogger("libras.app_real")

MODULO_SERVICO = "libras.tempo_real_servico"  # substituido nos testes por um servico falso
NOME_APP = "LibrasVision"
SELO = "Processamento neste computador. Nenhuma imagem é gravada nem enviada."
SELO_DETALHE = SELO + " Só o texto reconhecido aparece na tela; ele não é guardado em disco."
EXEMPLO_LEGENDA = ["Exemplo de legenda", "com as suas escolhas"]
ESCONDER_CONTROLES_MS = 3000
SEM_QUADRO_ALERTA_S = 3.0


# ======================================================================
# Barra minima
# ======================================================================
class BarraMinima(QFrame):
    """Iniciar/parar camera, pausar legenda, tela cheia, Sinais reconhecidos, indicador de maos e selo
    de privacidade; abaixo, o aviso permanente. Na faixa compacta os botoes mostram o rotulo curto
    embaixo do icone (o nome nunca fica so na dica, D18) e o selo vira so o icone, que abre o texto."""

    def __init__(self, janela: "JanelaReal"):
        super().__init__()
        self.setObjectName("barra_minima")
        self.j = janela
        lay = QVBoxLayout(self)
        lay.setContentsMargins(tema.ESPACO[3], tema.ESPACO[2], tema.ESPACO[3], tema.ESPACO[2])
        lay.setSpacing(tema.ESPACO[1])
        fila = QHBoxLayout()
        fila.setSpacing(tema.ESPACO[2])
        self.bt_camera = self._bt("camera", "Iniciar câmera", "Câmera", janela.alternar_camera)
        self.bt_pausa = self._bt("pausa", "Pausar legenda (P)", "Pausar", janela.alternar_pausa)
        self.bt_tela = self._bt("tela_cheia", "Tela cheia (F11)", "Tela cheia", janela.alternar_tela_cheia)
        self.bt_sinais = self._bt("lista", "Sinais reconhecidos", "Sinais", janela.mostrar_sinais)
        for b in (self.bt_camera, self.bt_pausa, self.bt_tela, self.bt_sinais):
            fila.addWidget(b)
        fila.addStretch(1)
        self.selo = self._bt("cadeado", SELO, "Privacidade", janela.mostrar_privacidade)
        self.selo.setObjectName("selo")
        fila.addWidget(self.selo)
        lay.addLayout(fila)
        fila2 = QHBoxLayout()  # indicador discreto de maos e aviso permanente
        fila2.setSpacing(tema.ESPACO[2])
        self.et_maos = etiqueta("Sem mãos", "pendente")
        self.et_maos.setAccessibleName("Indicador de mãos")
        fila2.addWidget(self.et_maos, 0, Qt.AlignTop)
        self.aviso = rotulo(pk.AVISO_USO, "secundario", quebra=True)
        self.aviso.setObjectName("aviso_uso")
        fila2.addWidget(self.aviso, 1)
        lay.addLayout(fila2)
        self.faixa = ""

    def _bt(self, ic: str, texto: str, curto: str, acao) -> QToolButton:
        b = QToolButton()
        b.setIcon(icone(ic, tema.CORES["texto"]))
        b.setIconSize(QSize(22, 22))
        b.setText(texto)
        b.setProperty("curto", curto)
        b.setProperty("longo", texto)
        b.setAccessibleName(texto)
        b.setToolTip(texto)
        b.setMinimumSize(tema.ALVO_TOQUE, tema.ALVO_TOQUE)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(acao)
        b.setObjectName("bt_barra")
        return b

    def definir_texto(self, b: QToolButton, longo: str, curto: str | None = None) -> None:
        b.setProperty("longo", longo)
        if curto is not None:
            b.setProperty("curto", curto)
        b.setAccessibleName(longo)
        b.setToolTip(longo)
        self.aplicar_faixa(self.faixa, forcar=True)

    def aplicar_faixa(self, faixa: str, forcar: bool = False) -> None:
        if faixa == self.faixa and not forcar:
            return
        self.faixa = faixa
        for b in (self.bt_camera, self.bt_pausa, self.bt_tela, self.bt_sinais, self.selo):
            if faixa == "expandida":
                b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                b.setText(b.property("longo"))
            else:
                b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                b.setText(b.property("curto"))
        if faixa == "compacta":  # selo: so o icone; tocar abre o texto
            self.selo.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.selo.setText("")
        elif faixa == "media":
            self.selo.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.selo.setText(SELO)


# ======================================================================
# Cartao de estado (sem modelo, falha de camera, parado)
# ======================================================================
class CartaoEstado(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("cartao_estado")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(tema.ESPACO[5], tema.ESPACO[4], tema.ESPACO[5], tema.ESPACO[4])
        self.lay.setSpacing(tema.ESPACO[3])
        self.titulo = rotulo("", "titulo_painel", quebra=True)
        self.texto = rotulo("", quebra=True)
        self.acao = rotulo("", "secundario", quebra=True)
        for w in (self.titulo, self.texto, self.acao):
            self.lay.addWidget(w)
        self.fila = QHBoxLayout()
        self.fila.setSpacing(tema.ESPACO[2])
        self.lay.addLayout(self.fila)
        self.botoes: list = []
        self.chave = ""

    def definir(self, chave: str, titulo: str, texto: str, acao: str, botoes: list[tuple[str, object, bool]]) -> None:
        self.chave = chave
        self.titulo.setText(titulo)
        self.texto.setText(texto)
        self.acao.setText(acao)
        self.acao.setVisible(bool(acao))
        for b in self.botoes:
            b.deleteLater()
        self.botoes = []
        for txt, fn, prim in botoes:
            b = botao(txt, primario=prim, acao=fn)
            self.fila.addWidget(b)
            self.botoes.append(b)
        self.adjustSize()


# ======================================================================
# Folha pequena com texto (selo de privacidade na faixa compacta: nada fica so no hover, D18)
# ======================================================================
class FolhaTexto(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("folha_texto")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(tema.ESPACO[4], tema.ESPACO[3], tema.ESPACO[3], tema.ESPACO[3])
        lay.setSpacing(tema.ESPACO[2])
        topo = QHBoxLayout()
        self.titulo = rotulo("", "titulo_painel")
        topo.addWidget(self.titulo, 1)
        self.bt_fechar = QToolButton()
        self.bt_fechar.setObjectName("bt_fechar")
        self.bt_fechar.setIcon(icone("fechar", tema.CORES["texto"]))
        self.bt_fechar.setFixedSize(tema.ALVO_TOQUE, tema.ALVO_TOQUE)
        self.bt_fechar.setAccessibleName("Fechar")
        self.bt_fechar.clicked.connect(self.hide)
        topo.addWidget(self.bt_fechar)
        lay.addLayout(topo)
        self.texto = rotulo("", quebra=True)
        lay.addWidget(self.texto)

    def mostrar(self, titulo: str, texto: str, area: QRectF) -> None:
        self.titulo.setText(titulo)
        self.texto.setText(texto)
        w = int(min(area.width() - 2 * tema.ESPACO[2], 460))
        self.setFixedWidth(w)
        lay = self.layout()
        lay.invalidate()
        lay.activate()
        h = lay.totalHeightForWidth(w) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        self.setFixedHeight(h)
        self.move(int(area.x() + (area.width() - w) / 2), int(area.bottom() - h - tema.ESPACO[2]))
        self.show()
        self.raise_()


# ======================================================================
# Guia de posicionamento (primeira abertura)
# ======================================================================
class PainelGuia(QFrame):
    def __init__(self, ao_comecar):
        super().__init__()
        self.setObjectName("painel_guia")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(tema.ESPACO[4], tema.ESPACO[3], tema.ESPACO[4], tema.ESPACO[3])
        lay.setSpacing(tema.ESPACO[2])
        lay.addWidget(rotulo("Antes de começar", "titulo_painel"))
        self.descricao = rotulo(
            "Fique no contorno: cabeça e ombros visíveis, e espaço para as mãos.", "secundario", quebra=True
        )
        lay.addWidget(self.descricao)
        self.itens: list[tuple[QLabel, QLabel]] = []
        for _ in range(3):
            h = QHBoxLayout()
            et = etiqueta("ajuste", "pendente")
            tx = rotulo("", quebra=True)
            h.addWidget(et, 0, Qt.AlignVCenter)
            h.addWidget(tx, 1)
            lay.addLayout(h)
            self.itens.append((et, tx))
        self.bt = botao("Começar", primario=True, acao=ao_comecar)
        lay.addWidget(self.bt)

    def atualizar(self, enq: dict | None) -> None:
        itens = itens_primeira_abertura(enq)
        for (et, tx), (texto, ok) in zip(self.itens, itens):
            tx.setText(texto)
            mudar_etiqueta(et, "ok" if ok else "ajuste", "ok" if ok else "pendente")
        self.bt.setText("Começar" if all(ok for _, ok in itens) else "Continuar mesmo assim")


# ======================================================================
# Vista da camera (desenha video, legenda, barra tecnica, avisos e bordas)
# ======================================================================
class VistaCamera(QWidget):
    def __init__(self, janela: "JanelaReal"):
        super().__init__()
        self.j = janela
        self.setObjectName("vista_camera")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 200)
        self.quadro: QPixmap | None = None
        self.proporcao = 4 / 3
        self.mensagem_centro = ""
        self.linhas_tecnicas: list[str] = []
        self.aviso: str | None = None
        self.bordas: list[str] = []
        self.cabeca_norm = None
        self.contorno_guia = False
        self.controles_visiveis = True
        self.ultimo_layout: dict = {}
        self.bt_config = QToolButton(self)
        self.bt_config.setObjectName("bt_config")
        self.bt_config.setIcon(icone("engrenagem", tema.CORES["texto"]))
        self.bt_config.setIconSize(QSize(24, 24))
        self.bt_config.setFixedSize(tema.ALVO_TOQUE, tema.ALVO_TOQUE)
        self.bt_config.setAccessibleName("Configurações (F2)")
        self.bt_config.setToolTip("Configurações (F2)")
        self.bt_config.setCursor(Qt.PointingHandCursor)
        self.bt_config.clicked.connect(janela.alternar_configuracoes)
        self.cartao = CartaoEstado()
        self.cartao.setParent(self)
        self.cartao.hide()
        self.guia = PainelGuia(janela.fechar_guia)
        self.guia.setParent(self)
        self.guia.hide()
        self.folha = FolhaTexto()
        self.folha.setParent(self)
        self.folha.hide()

    # ------------------------------------------------------------------ geometria
    def rect_imagem(self) -> Ret:
        """Imagem inteira na janela (sem cortar: a pessoa precisa ver as proprias maos), centralizada."""
        W, H = self.width(), self.height()
        bm = self.j.altura_barra_minima() if self.j.barra.isVisible() else 0
        h_disp = H - (bm if self.j.faixa == "compacta" else 0)
        w = min(W, h_disp * self.proporcao)
        h = w / self.proporcao
        return Ret((W - w) / 2, (h_disp - h) / 2, w, h)

    def font_barra(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(max(14, tema.px("corpo")))
        try:  # numeros de largura fixa (nao "pulam" a cada atualizacao)
            f.setFeature(QFont.Tag("tnum"), 1)
        except (AttributeError, TypeError):
            f.setFamilies(tema.TIPO["mono"])
        return f

    def calcular_layout(self) -> dict:
        img = self.rect_imagem()
        jan = Ret(0, 0, self.width(), self.height())
        fm = QFontMetricsF(self.font_barra())
        alt_barra = 0.0
        if self.linhas_tecnicas and self.controles_visiveis:
            alt_barra = max(float(tema.ALVO_TOQUE), len(self.linhas_tecnicas) * fm.lineSpacing() + 2 * tema.ESPACO[2])
        cam = self.j.camada
        tam = cam.medir(min(img.w - 2 * tema.ESPACO[2], img.w * cam.estilo.largura_frac))
        etq = None
        if self.aviso:
            fm2 = QFontMetricsF(self.font_etiqueta())
            extra = 3 * tema.ESPACO[3] + 22
            disp = img.w - 3 * 8 - (tema.ALVO_TOQUE if self.controles_visiveis else 0) - extra
            larg_txt = fm2.horizontalAdvance(self.aviso)
            n = 1 if larg_txt <= disp else len(self._quebrar(self.aviso, fm2, disp))
            etq = (min(larg_txt, disp) + extra, n * fm2.lineSpacing() + 2 * tema.ESPACO[2] + 4)
        cab = cabeca_na_tela(self.cabeca_norm, img) if cam.estilo.posicao == "pessoa" else None
        out = posicionar(
            jan,
            img,
            legenda=tam,
            posicao=cam.estilo.posicao,
            cabeca=cab,
            botao_config=self.controles_visiveis,
            barra_tecnica=alt_barra,
            etiqueta=etq,
            barra_minima=self.j.altura_barra_minima() if self.controles_visiveis else 0.0,
            bordas=[lado_na_tela(l) for l in self.bordas],
        )
        out["imagem"] = img
        return out

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self.posicionar_filhos()

    def posicionar_filhos(self) -> None:
        lay = self.calcular_layout()
        self.ultimo_layout = lay
        b = lay["botao_config"]
        self.bt_config.setVisible(b is not None)
        if b is not None:
            self.bt_config.move(int(b.x), int(b.y))
        img = lay["imagem"]
        if self.cartao.isVisible() or self.cartao.chave:
            w = int(min(img.w - 2 * tema.ESPACO[4], 520))
            self.cartao.setFixedWidth(max(220, w))
            self.cartao.adjustSize()
            h = self.cartao.sizeHint().height()
            self.cartao.setFixedHeight(h)
            self.cartao.move(
                int(img.x + (img.w - self.cartao.width()) / 2), int(max(img.y + 56, img.y + (img.h - h) / 2))
            )
        if self.guia.isVisible():
            self.guia.descricao.setVisible(self.j.faixa != "compacta")  # na faixa compacta o painel cabe sob a imagem
            gw = int(min(340, img.w - 2 * tema.ESPACO[3]))
            self.guia.setFixedWidth(max(220, gw))
            lay_g = self.guia.layout()
            lay_g.invalidate()
            lay_g.activate()
            gh = lay_g.totalHeightForWidth(gw) if lay_g.hasHeightForWidth() else lay_g.totalSizeHint().height()
            self.guia.setFixedHeight(gh)
            bm = self.j.altura_barra_minima() if self.j.barra.isVisible() else 0
            limite = self.height() - bm - tema.ESPACO[1]
            m = tema.ESPACO[2]
            ocupados = [lay[k] for k in ("legenda", "etiqueta", "botao_config", "barra_tecnica") if lay.get(k)]
            meio = (img.y + min(img.base, limite) - gh) / 2
            candidatos = [
                (img.x + (img.w - gw) / 2, img.base + tema.ESPACO[1]),  # faixa preta abaixo da imagem
                (img.direita - gw - m, meio),
                (img.x + m, meio),
            ]  # ao lado do contorno
            candidatos += [
                (x0, y0)
                for x0 in (img.direita - gw - m, img.x + m)
                for y0 in (img.y + tema.ALVO_TOQUE + 2 * m, limite - gh)
            ]
            x, y = candidatos[1]
            for cx, cy in candidatos:  # o guia nao cobre legenda, etiqueta, botao nem barra tecnica
                r = Ret(cx, cy, gw, gh)
                if cy >= 0 and r.base <= limite and not any(r.cruza(o) for o in ocupados):
                    x, y = cx, cy
                    break
            self.guia.move(int(x), int(max(0, y)))
            self.guia.raise_()

    # ------------------------------------------------------------------ desenho
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.fillRect(self.rect(), tema.qcolor("video_fundo"))
        lay = self.calcular_layout()
        self.ultimo_layout = lay
        img = lay["imagem"]
        r_img = QRectF(img.x, img.y, img.w, img.h)
        if self.quadro is not None and not self.quadro.isNull():
            p.drawPixmap(r_img, self.quadro, QRectF(self.quadro.rect()))
        elif self.mensagem_centro and not self.cartao.isVisible():
            p.setPen(tema.qcolor("texto_secundario"))
            p.drawText(r_img, Qt.AlignCenter | Qt.TextWordWrap, self.mensagem_centro)
        if self.contorno_guia:
            self._desenhar_contorno(p, img)
        for lado, r in lay["bordas"].items():
            p.fillRect(QRectF(r.x, r.y, r.w, r.h), tema.qcolor("alerta"))
        if lay["barra_tecnica"] is not None:
            self._desenhar_barra(p, lay["barra_tecnica"])
        if lay["legenda"] is not None:
            L = lay["legenda"]
            self.j.camada.desenhar(p, QRectF(L.x, L.y, L.w, L.h))
        if lay["etiqueta"] is not None:
            self._desenhar_etiqueta(p, lay["etiqueta"])
        p.end()

    def _desenhar_barra(self, p: QPainter, r: Ret) -> None:
        fundo = tema.qcolor("legenda_fundo")
        p.setPen(Qt.NoPen)
        p.setBrush(fundo)
        p.drawRoundedRect(QRectF(r.x, r.y, r.w, r.h), tema.RAIO["p"], tema.RAIO["p"])
        f = self.font_barra()
        p.setFont(f)
        fm = QFontMetricsF(f)
        p.setPen(tema.qcolor("legenda_texto"))  # cor neutra: pulados nao sao erro
        y = r.y + (r.h - len(self.linhas_tecnicas) * fm.lineSpacing()) / 2
        for linha in self.linhas_tecnicas:
            p.drawText(
                QRectF(r.x + tema.ESPACO[3], y, r.w - 2 * tema.ESPACO[3], fm.lineSpacing()),
                Qt.AlignLeft | Qt.AlignVCenter,
                linha,
            )
            y += fm.lineSpacing()

    def font_etiqueta(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(tema.px("corpo_grande"))
        return f

    @staticmethod
    def _quebrar(texto: str, fm: QFontMetricsF, largura: float) -> list[str]:
        linhas, atual = [], ""
        for w in texto.split():
            tent = (atual + " " + w).strip()
            if atual and fm.horizontalAdvance(tent) > largura:
                linhas.append(atual)
                atual = w
            else:
                atual = tent
        return linhas + ([atual] if atual else [])

    def _desenhar_etiqueta(self, p: QPainter, r: Ret) -> None:
        from PySide6.QtSvg import QSvgRenderer

        rr = QRectF(r.x, r.y, r.w, r.h)
        p.setPen(QPen(tema.qcolor("alerta"), 2))
        p.setBrush(tema.qcolor("legenda_fundo"))
        p.drawRoundedRect(rr.adjusted(1, 1, -1, -1), tema.RAIO["m"], tema.RAIO["m"])
        ic = QRectF(r.x + tema.ESPACO[3], r.y + (r.h - 22) / 2, 22, 22)
        QSvgRenderer(QByteArray(svg("alerta", tema.COR["alerta"], 22))).render(p, ic)
        p.setFont(self.font_etiqueta())
        p.setPen(tema.qcolor("legenda_texto"))
        p.drawText(
            QRectF(ic.right() + tema.ESPACO[2], r.y, r.w - ic.width() - 2 * tema.ESPACO[3] - 2, r.h),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            self.aviso or "",
        )

    def _desenhar_contorno(self, p: QPainter, img: Ret) -> None:
        """Contorno de cabeca e ombros e a area das maos (guia da primeira abertura)."""
        cx, s = img.x + img.w / 2, img.h
        pen = QPen(tema.qcolor("legenda_texto"), 2, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(cx - 0.09 * s, img.y + 0.12 * s, 0.18 * s, 0.24 * s))
        p.drawArc(QRectF(cx - 0.3 * s, img.y + 0.38 * s, 0.6 * s, 0.5 * s), 30 * 16, 120 * 16)
        p.setPen(QPen(tema.qcolor("foco"), 2, Qt.DotLine))
        p.drawRoundedRect(QRectF(cx - 0.42 * s, img.y + 0.3 * s, 0.84 * s, 0.6 * s), 12, 12)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        self.j.atividade()
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        self.j.atividade()
        super().mousePressEvent(ev)


# ======================================================================
# Janela
# ======================================================================
class JanelaReal(QMainWindow):
    def __init__(self, cfg_real=None, prefs: Preferencias | None = None, relogio=time.monotonic):
        super().__init__()
        self.cfg = cfg_real or config_real.carregar()
        self.prefs = prefs or Preferencias.da_raiz(self.cfg.raiz)
        self.modelos = ControleModelos(self.cfg, self.prefs)
        self.relogio = relogio
        tema.configurar(reduzir_movimento=bool(self.prefs["reduzir_movimento"]))
        self.setWindowTitle(NOME_APP)
        self.camada = CamadaLegenda(self.prefs["legenda"], relogio)
        self.guia_avisos = GuiaAvisos(float(self.cfg["enquadramento"]["histerese_s"]))
        self.guia_avisos.habilitado = bool(self.prefs["camera"]["avisos_enquadramento"])
        self.suav = SuavizadorCabeca()
        self.proc: Subprocesso | None = None
        self.dec = Decodificador()
        self.estado = "parado"
        self.cartao_ativo: dict | None = None
        self.info_modelo: dict = {}
        self.info_camera: dict = {}
        self.ultimo: dict = {}
        self.t_ultimo_quadro: float | None = None
        self.exibidos: deque = deque(maxlen=90)
        self.medidas: dict = {}
        self.faixa = ""
        self.erro_recebido = False
        self.parando_por_usuario = False

        central = QWidget()
        central.setObjectName("central_real")
        self.setCentralWidget(central)
        self.vista = VistaCamera(self)
        self.vista.setParent(central)
        self.barra = BarraMinima(self)
        self.barra.setParent(central)
        from .configuracoes import PainelConfiguracoes

        self.painel = PainelConfiguracoes(self)
        self.painel.setParent(central)
        self.painel.hide()
        central.installEventFilter(self)
        self.barra.installEventFilter(self)  # texto que quebra muda a altura da barra: reposiciona
        self._posicionando = False

        self.t_tecnico = QTimer(self)
        self.t_tecnico.setInterval(int(self.cfg["informacoes_tecnicas"]["intervalo_ms"]))
        self.t_tecnico.timeout.connect(self.atualizar_barra_tecnica)
        self.t_tecnico.start()
        self.t_anim = QTimer(self)
        self.t_anim.setInterval(33)
        self.t_anim.timeout.connect(self._tique)
        self.t_anim.start()
        self.t_esconder = QTimer(self)
        self.t_esconder.setSingleShot(True)
        self.t_esconder.setInterval(ESCONDER_CONTROLES_MS)
        self.t_esconder.timeout.connect(self._esconder_controles)
        self._atalhos()
        self.resize(1366, 800)  # tamanho ao sair do modo maximizado
        self._maximizada_antes = True

    # ------------------------------------------------------------------ geometria e faixas
    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent

        if (obj is self.centralWidget() and ev.type() == QEvent.Resize) or (
            obj is self.barra and ev.type() == QEvent.LayoutRequest
        ):
            self._posicionar()
        return False

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        QTimer.singleShot(0, self._posicionar)

    def altura_barra_minima(self) -> int:
        if not self.barra.isVisible() and self.isFullScreen():
            return 0
        larg = max(1, self.vista.width() if self.vista.width() > 1 else self.width())
        lay = self.barra.layout()
        alt = lay.totalHeightForWidth(larg) if lay.hasHeightForWidth() else self.barra.sizeHint().height()
        return max(alt, self.barra.sizeHint().height(), tema.ALVO_TOQUE + 2 * tema.ESPACO[2])

    def _posicionar(self) -> None:
        if self._posicionando:
            return
        self._posicionando = True
        try:
            self._posicionar_agora()
        finally:
            self._posicionando = False

    def _posicionar_agora(self) -> None:
        c = self.centralWidget()
        W, H = c.width(), c.height()
        faixa = tema.faixa_para_largura(self.width())
        if faixa != self.faixa:
            self.faixa = faixa
            tema.configurar(faixa=faixa)
            self.barra.aplicar_faixa(faixa)
            self.painel.aplicar_faixa(faixa)
        x_v, y_v, w_v, h_v = 0, 0, W, H
        if self.painel.isVisible():
            if faixa == "compacta":  # folha que sobe de baixo; a camera fica acima
                hp = int(H * 0.6)
                self.painel.setGeometry(0, H - hp, W, hp)
                h_v = H - hp
            else:  # painel lateral; a camera continua ao lado, com a legenda de exemplo
                wp = min(400, int(W * 0.42))
                self.painel.setGeometry(W - wp, 0, wp, H)
                w_v = W - wp
        self.vista.setGeometry(x_v, y_v, w_v, h_v)
        self.barra.aplicar_faixa(tema.faixa_para_largura(w_v))  # a barra segue a largura que ela tem
        bm = self.altura_barra_minima()
        mostrar_barra = self.vista.controles_visiveis and not (self.painel.isVisible() and faixa == "compacta")
        self.barra.setVisible(mostrar_barra)
        if mostrar_barra:
            self.barra.setGeometry(x_v, y_v + h_v - bm, w_v, bm)
            self.barra.raise_()
        self.vista.posicionar_filhos()

    # ------------------------------------------------------------------ atalhos
    def _atalhos(self) -> None:
        for tecla, fn in (
            ("F2", self.alternar_configuracoes),
            ("P", self.alternar_pausa),
            ("F11", self.alternar_tela_cheia),
            ("Ctrl+C", self.copiar_texto),
            ("Ctrl+L", self.limpar_legenda),
        ):
            QShortcut(QKeySequence(tecla), self, activated=fn)
        QShortcut(QKeySequence("Esc"), self, activated=self._esc)

    def _esc(self) -> None:
        if self.painel.isVisible():
            self.alternar_configuracoes()
        elif self.isFullScreen():
            self.alternar_tela_cheia()

    # ------------------------------------------------------------------ modelo
    def preparar_modelo(self, perguntar_troca=perguntar) -> dict | None:
        """Escolhe o pacote ativo e, se houver um publicado depois dele, oferece trocar (13.2)."""
        novo = self.modelos.novo_disponivel()
        if novo is not None:
            c = novo["cartao"]
            sinais = ", ".join(pk.rotulo(c, x) for x in c["classes"] if x != c.get("nome_nenhum_sinal"))
            if perguntar_troca(
                self,
                f"Há um modelo novo publicado em {c.get('publicado_em', '-')}, com os sinais: "
                f"{sinais}.\n\nTrocar para o modelo novo? O atual fica guardado para voltar.",
                "Modelo novo",
            ):
                self.modelos.trocar(novo["id"])
            else:
                self.modelos.recusar(novo["id"])
        it = self.modelos.ativo()
        self.cartao_ativo = it["cartao"] if it else None
        self.painel.atualizar_modelo()
        return it

    def importar_pacote(self, arquivo: str | None = None) -> bool:
        if arquivo is None:
            from PySide6.QtWidgets import QFileDialog

            arquivo, _ = QFileDialog.getOpenFileName(
                self, "Importar pacote do modelo", str(Path.home()), "Pacote (*.zip)"
            )
            if not arquivo:
                return False
        try:
            it = self.modelos.importar(Path(arquivo))
        except pk.ErroPacote as e:
            Dialogo(self, "Pacote inválido", str(e), "erro").exec()
            return False
        self.cartao_ativo = it["cartao"]
        self.painel.atualizar_modelo()
        avisar(self, "Modelo importado e ativado.", "ok")
        self.reiniciar_camera()
        return True

    def voltar_modelo_anterior(self) -> None:
        if self.modelos.voltar_anterior():
            self.cartao_ativo = self.modelos.ativo()["cartao"]
            self.painel.atualizar_modelo()
            self.reiniciar_camera()

    def rotulo(self, classe: str) -> str:
        return pk.rotulo(self.cartao_ativo or {}, classe)

    # ------------------------------------------------------------------ camera e subprocesso
    def alternar_camera(self) -> None:
        if self.proc is not None:
            self.parar_camera()
        else:
            self.iniciar_camera()

    def reiniciar_camera(self) -> None:
        if self.proc is not None:
            self.parar_camera()
        self.iniciar_camera()

    def iniciar_camera(self) -> None:
        if self.proc is not None and self.proc.rodando():
            return
        if Subprocesso.ha_encerrando():  # a camera ainda esta com o processo anterior
            self.vista.mensagem_centro = "Aguardando a câmera ser liberada..."
            self.vista.update()
            QTimer.singleShot(150, self.iniciar_camera)
            return
        it = self.modelos.ativo()
        if it is None:
            self._estado_sem_modelo(self.modelos.problema_do_ativo())
            return
        self.cartao_ativo = it["cartao"]
        self.dec = Decodificador()
        self.erro_recebido = False
        self.parando_por_usuario = False
        self.ultimo, self.info_camera = {}, {}
        self.vista.quadro = None
        self.vista.cartao.chave = ""
        self.vista.cartao.hide()
        self.vista.mensagem_centro = "Ligando a câmera e preparando o reconhecimento..."
        self.estado = "aquecendo"
        self.medidas = {"t_clique": time.perf_counter()}
        indice = self.prefs["camera"]["indice"]
        args = [
            "--pacote",
            str(it["pasta"]),
            "--sem-disco",
            "--config-real",
            str(self.cfg.arquivo or config_real.ARQUIVO_PADRAO),
        ]
        if indice is not None:
            args += ["--camera", str(int(indice))]
        self.proc = Subprocesso(self, binario=True)
        self.proc.p.readyReadStandardOutput.connect(self._ler)
        self.proc.p.readyReadStandardError.connect(lambda: self.proc and self.proc.p.readAllStandardError())
        self.proc.terminou.connect(self._terminou)
        self.proc.iniciar_modulo(MODULO_SERVICO, args)
        self.barra.definir_texto(self.barra.bt_camera, "Parar câmera", "Parar")
        self.barra.bt_camera.setIcon(icone("parar", tema.CORES["texto"]))
        if not self.prefs["guia_visto"]:
            self.mostrar_guia()
        self.vista.update()

    def parar_camera(self) -> None:
        """Encerra o subprocesso sem esperar (D15) e encerra a pausa (13.9-D)."""
        self.parando_por_usuario = True
        if self.proc is not None and self.proc.rodando():
            self.proc.encerrar_sem_esperar("sair")
        self.proc = None
        self.estado = "parado"
        if self.camada.rollup.pausado:
            self.camada.rollup.retomar()
            self._atualizar_botao_pausa()
        self.vista.quadro = None
        self.vista.mensagem_centro = "Câmera parada. Toque em Iniciar câmera."
        self.vista.aviso, self.vista.bordas = None, []
        self.barra.definir_texto(self.barra.bt_camera, "Iniciar câmera", "Câmera")
        self.barra.bt_camera.setIcon(icone("camera", tema.CORES["texto"]))
        mudar_etiqueta(self.barra.et_maos, "Sem mãos", "pendente")
        self.vista.update()

    def _terminou(self, codigo: int) -> None:
        if self.parando_por_usuario or self.erro_recebido:
            return
        if codigo not in (0, 15, -15):
            self._estado_falha("O reconhecimento terminou inesperadamente.")

    def _ler(self) -> None:
        if not self.proc:
            return
        for meta, jpg in self.dec.alimentar(bytes(self.proc.p.readAllStandardOutput())):
            t = meta.get("tipo")
            if t == "erro":
                self.erro_recebido = True
                if meta.get("sem_modelo"):
                    self.proc = None
                    self._estado_sem_modelo(meta.get("mensagem"))
                else:
                    self._estado_falha(meta.get("mensagem", ""))
            elif t == "info":
                if "camera" in meta and "modelo" not in meta:
                    self.info_camera = meta["camera"]
                else:
                    self.info_modelo = meta
            elif t == "quadro":
                self._quadro(meta, jpg)

    def _quadro(self, meta: dict, jpg: bytes) -> None:
        agora = self.relogio()
        if jpg:
            pm = QPixmap()
            if pm.loadFromData(jpg, "JPG"):
                self.vista.quadro = pm
                self.vista.proporcao = pm.width() / max(1, pm.height())
                self.exibidos.append(time.perf_counter())
                if "t_primeiro_quadro" not in self.medidas and "t_clique" in self.medidas:
                    self.medidas["t_primeiro_quadro"] = time.perf_counter()
                    self.medidas["camera_ligou_s"] = self.medidas["t_primeiro_quadro"] - self.medidas["t_clique"]
        if self.estado != "falha":
            self.estado = "aquecendo" if (meta.get("motivo") == "aquecendo") else "ativo"
        self.ultimo = meta
        self.t_ultimo_quadro = agora
        self.vista.mensagem_centro = ""
        self.camada.receber_quadro(meta, self.rotulo, agora)
        enq = meta.get("enquadramento") or {}
        self.vista.cabeca_norm = self.suav.atualizar(enq.get("cabeca"), agora)
        chave = self.guia_avisos.atualizar(enq or None, agora)
        self.vista.aviso = self.guia_avisos.texto(chave) if chave else None
        self.vista.bordas = list(self.guia_avisos.lados) if chave == "mao_cortada" else []
        sem_mao = meta.get("motivo") in ("sem_maos", "aquecendo", "sem_pose")
        mudar_etiqueta(
            self.barra.et_maos, "Sem mãos" if sem_mao else "Mãos detectadas", "pendente" if sem_mao else "captando"
        )
        if self.vista.guia.isVisible():
            self.vista.guia.atualizar(enq)
        self.vista.update()

    def _estado_sem_modelo(self, problema: str | None = None) -> None:
        self.estado = "sem_modelo"
        texto = (
            "O LibrasVision precisa de um pacote de modelo, publicado pela equipe na bancada, para reconhecer "
            "sinais. Nada é reconhecido sem ele."
        )
        if problema:
            texto = problema + "\n\n" + texto
        botoes = [("Importar pacote (.zip)", self.importar_pacote, True)]
        outro = self.modelos.mais_recente_valido() if problema else None
        titulo = "Nenhum modelo instalado"
        if problema:
            titulo = "O modelo escolhido não pode ser usado"
            if outro is not None:
                botoes.append((f"Usar o modelo {outro['id']}", lambda: self._usar_modelo(outro["id"]), False))
        self.vista.cartao.definir(
            "sem_modelo",
            titulo,
            texto,
            f"Peça o arquivo .zip à equipe e toque em Importar pacote. Pasta de modelos: {self.modelos.base.name}/",
            botoes,
        )
        self._mostrar_cartao()

    def _estado_falha(self, mensagem: str) -> None:
        self.estado = "falha"
        if self.proc is not None:
            self.proc.encerrar_sem_esperar("sair")
            self.proc = None
        causa, acao = explicar_falha(
            mensagem,
            self.prefs["camera"]["indice"]
            if self.prefs["camera"]["indice"] is not None
            else self.cfg["camera"]["indice"],
        )
        self.vista.aviso, self.vista.bordas = None, []  # falha de camera tem prioridade sobre qualquer aviso
        self.vista.quadro = None
        self.vista.cartao.definir(
            "falha",
            "Não foi possível usar a câmera",
            causa,
            acao,
            [
                ("Tentar de novo", self.tentar_de_novo, True),
                ("Escolher outra câmera", lambda: self.alternar_configuracoes("camera"), False),
            ],
        )
        self.barra.definir_texto(self.barra.bt_camera, "Iniciar câmera", "Câmera")
        self._mostrar_cartao()

    def _mostrar_cartao(self) -> None:
        self.vista.cartao.show()
        self.vista.cartao.raise_()
        self.vista.posicionar_filhos()
        self.vista.update()

    def _usar_modelo(self, ident: str) -> None:
        self.modelos.trocar(ident)  # o anterior fica guardado para voltar
        self.cartao_ativo = self.modelos.ativo()["cartao"]
        self.painel.atualizar_modelo()
        self.tentar_de_novo()

    def tentar_de_novo(self) -> None:
        """Fecha o cartao e liga a camera de novo; se o reconhecimento estiver rodando, encerra antes
        (a camera so pode estar com um processo)."""
        self.vista.cartao.hide()
        self.vista.cartao.chave = ""
        if self.proc is not None:
            self.parar_camera()
        self.iniciar_camera()

    # ------------------------------------------------------------------ legenda
    def alternar_pausa(self) -> None:
        self.camada.rollup.alternar_pausa(self.relogio())
        self._atualizar_botao_pausa()
        self.vista.update()

    def _atualizar_botao_pausa(self) -> None:
        pausado = self.camada.rollup.pausado
        self.barra.definir_texto(
            self.barra.bt_pausa,
            "Retomar legenda (P)" if pausado else "Pausar legenda (P)",
            "Retomar" if pausado else "Pausar",
        )
        self.barra.bt_pausa.setIcon(icone("play" if pausado else "pausa", tema.CORES["texto"]))

    def limpar_legenda(self) -> None:
        """Limpa a tela e o buffer da pausa e retoma; o texto do servico tambem e limpo."""
        self.camada.rollup.limpar()
        if self.proc is not None:
            self.proc.escrever("limpar\n")
        self._atualizar_botao_pausa()
        self.vista.update()

    def copiar_texto(self) -> str:
        """Na pausa, exatamente o texto congelado na tela; fora dela, todo o texto desde a ultima limpeza."""
        r = self.camada.rollup
        texto = r.texto_visivel() if r.pausado else r.texto_completo()
        QGuiApplication.clipboard().setText(texto)
        avisar(self, "Texto copiado." if texto else "Ainda não há texto para copiar.", "ok" if texto else "info")
        return texto

    def aplicar_preferencias_legenda(self) -> None:
        self.camada.aplicar_preferencias(self.prefs["legenda"])
        self.vista.update()

    # ------------------------------------------------------------------ informacoes tecnicas
    def extra_tecnico(self) -> dict:
        agora = time.perf_counter()
        exib = [t for t in self.exibidos if agora - t <= 1.0]
        return {
            "fps_exibicao": float(len(exib)),
            "memoria_interface_mb": memoria_processo_mb(),
            "modelo": (self.cartao_ativo or {}).get("id"),
            "camera": self.info_camera,
        }

    def atualizar_barra_tecnica(self) -> None:
        ligados = self.prefs["tecnicas"]
        if not any(ligados.values()):
            novas = []
        else:
            fm = QFontMetricsF(self.vista.font_barra())
            larg = self.vista.rect_imagem().w - 2 * 8 - tema.ALVO_TOQUE - 3 * 8 - 2 * tema.ESPACO[3]
            novas = tecnicas.texto_barra(
                self.ultimo,
                self.extra_tecnico(),
                ligados,
                self.faixa,
                lambda s: fm.horizontalAdvance(s) <= larg,
                2 if self.faixa == "compacta" else 1,
            )
            if not novas:
                novas = [
                    tecnicas.texto_barra(
                        self.ultimo, self.extra_tecnico(), {"taxa": True}, self.faixa, lambda s: True, 1
                    )[0][:60]
                ]
        if novas != self.vista.linhas_tecnicas:
            self.vista.linhas_tecnicas = novas
            self.vista.update()
        if (
            self.estado == "ativo"
            and self.t_ultimo_quadro is not None
            and self.relogio() - self.t_ultimo_quadro > SEM_QUADRO_ALERTA_S
        ):
            self.vista.aviso = "A câmera parou de enviar imagens"  # unico caso em que quadros viram alerta
            self.vista.update()

    def relatorio_tecnico(self) -> str:
        texto = tecnicas.relatorio(self.ultimo, self.extra_tecnico(), self.cartao_ativo, __version__, hardware())
        QGuiApplication.clipboard().setText(texto)  # nada e gravado em disco
        avisar(self, "Relatório técnico copiado. Cole no e-mail ou na mensagem para a equipe.", "ok")
        return texto

    # ------------------------------------------------------------------ tela cheia e inatividade
    def alternar_tela_cheia(self) -> None:
        if self.isFullScreen():
            if self._maximizada_antes:
                self.showMaximized()
            else:
                self.showNormal()
            self.t_esconder.stop()
            self._mostrar_controles()
            self.barra.definir_texto(self.barra.bt_tela, "Tela cheia (F11)", "Tela cheia")
            self.barra.bt_tela.setIcon(icone("tela_cheia", tema.CORES["texto"]))
        else:
            self._maximizada_antes = self.isMaximized()
            self.showFullScreen()
            self.t_esconder.start()
            self.barra.definir_texto(self.barra.bt_tela, "Sair da tela cheia (F11)", "Sair")
            self.barra.bt_tela.setIcon(icone("sair_tela_cheia", tema.CORES["texto"]))

    def atividade(self) -> None:
        if self.isFullScreen():
            self._mostrar_controles()
            self.t_esconder.start()

    def _esconder_controles(self) -> None:
        """So na tela cheia: barra minima, botao de Configuracoes e barra tecnica somem juntos (a
        preferencia das informacoes tecnicas nao muda)."""
        if self.isFullScreen() and not self.painel.isVisible():
            self.vista.controles_visiveis = False
            self.barra.hide()
            self._posicionar()
            self.vista.update()

    def _mostrar_controles(self) -> None:
        if not self.vista.controles_visiveis:
            self.vista.controles_visiveis = True
            self.barra.show()
            self._posicionar()
            self.vista.update()

    # ------------------------------------------------------------------ paineis
    def alternar_configuracoes(self, secao: str | None = None) -> None:
        abrir = not self.painel.isVisible() or isinstance(secao, str)
        self.painel.setVisible(abrir)
        if abrir:
            self.painel.raise_()
            self.painel.mostrar_secao(secao if isinstance(secao, str) else None)
            self.camada.exemplo = list(EXEMPLO_LEGENDA)  # legenda de exemplo sobre o video ao vivo
        else:
            self.camada.exemplo = None
        self._posicionar()
        self.vista.update()

    def mostrar_sinais(self) -> None:
        c = self.cartao_ativo
        if not c:
            Dialogo(self, "Sinais reconhecidos", "Nenhum modelo instalado.", "info").exec()
            return
        sinais = [pk.rotulo(c, x) for x in c["classes"] if x != c.get("nome_nenhum_sinal")]
        Dialogo(
            self,
            "Sinais reconhecidos",
            "Este modelo reconhece só estes sinais:\n\n" + "\n".join(f"  {s}" for s in sinais) + "\n\n" + pk.AVISO_USO,
            "info",
        ).exec()

    def mostrar_privacidade(self) -> None:
        """Toque no selo: folha pequena que sobe acima da barra minima (nao e dialogo modal)."""
        h = self.vista.height() - (self.altura_barra_minima() if self.barra.isVisible() else 0)
        self.vista.folha.mostrar("Privacidade", SELO_DETALHE, QRectF(0, 0, self.vista.width(), h))

    def mostrar_guia(self) -> None:
        self.vista.contorno_guia = True
        self.vista.guia.atualizar(self.ultimo.get("enquadramento"))
        self.vista.guia.show()
        self.vista.guia.raise_()
        self.vista.posicionar_filhos()
        self.vista.update()

    def fechar_guia(self) -> None:
        """'Comecar' ou 'Continuar mesmo assim': mostra-se uma vez (gravado nas preferencias)."""
        self.vista.guia.hide()
        self.vista.contorno_guia = False
        self.prefs.dados["guia_visto"] = True
        self.prefs.salvar()
        self.vista.update()

    # ------------------------------------------------------------------ ciclo
    def _tique(self) -> None:
        if self.camada.rollup.precisa_animar(self.relogio()):
            self.vista.update()

    def iniciar(self, perguntar_troca=perguntar) -> None:
        """Chamado ao abrir: confere o modelo (oferece pacote novo) e liga a camera."""
        if self.preparar_modelo(perguntar_troca) is None:
            self._estado_sem_modelo(self.modelos.problema_do_ativo())
            return
        self.iniciar_camera()

    def closeEvent(self, ev) -> None:  # noqa: N802
        if self.proc is not None and self.proc.rodando():
            self.proc.escrever("sair\n")
            if not self.proc.p.waitForFinished(5000):
                self.proc.parar()
        self.proc = None
        Subprocesso.aguardar_todos(5000)
        super().closeEvent(ev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m libras.app_real", description="LibrasVision: câmera e legenda.")
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if "tensorflow" not in sys.modules:  # esta interface nunca usa TensorFlow (D15)
        sys.modules["tensorflow"] = None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(NOME_APP)
    app.setStyle("Fusion")
    fonte = QFont()
    fonte.setFamilies(tema.FONTES_ALTERNATIVAS)
    fonte.setPixelSize(tema.px("corpo"))
    app.setFont(fonte)
    app.setStyleSheet(tema.folha_de_estilo() + folha_extra())
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)  # nada vai para arquivo (13.9-B)
    try:
        cfg = config_real.carregar(a.config)
    except Exception as e:  # noqa: BLE001
        Dialogo(None, NOME_APP, f"Não foi possível ler a configuração:\n\n{e}", "erro").exec()
        return 2
    janela = JanelaReal(cfg)
    janela.showMaximized()  # abre ocupando a tela, em modo janela; F11 continua alternando a tela cheia
    QTimer.singleShot(0, janela.iniciar)
    return app.exec()


def folha_extra() -> str:
    """Estilos das pecas do app real (sobre o video), a partir dos tokens."""
    c = tema.COR
    e = tema.ESPACO
    return f"""
    QFrame#barra_minima {{ background: {c["superficie_1"]}; border-top: 1px solid {c["borda"]}; }}
    QToolButton#bt_barra, QToolButton#selo {{ color: {c["texto"]}; background: transparent; border: 1px solid transparent;
        border-radius: {tema.RAIO["m"]}px; padding: {e[1]}px {e[2]}px; min-height: {tema.ALVO_TOQUE - 2}px; }}
    QToolButton#bt_barra:hover, QToolButton#selo:hover {{ background: {c["superficie_3"]}; }}
    QToolButton#bt_barra:focus, QToolButton#selo:focus, QToolButton#bt_config:focus {{ border: 2px solid {c["foco"]}; }}
    QToolButton#selo {{ color: {c["texto_secundario"]}; }}
    QToolButton#bt_config, QToolButton#bt_fechar {{ background: {c["superficie_2"]};
        border: 1px solid {c["borda_controle"]}; border-radius: {tema.ALVO_TOQUE // 2}px; }}
    QToolButton#bt_fechar:focus {{ border: 2px solid {c["foco"]}; }}
    QFrame#cartao_estado, QFrame#painel_guia, QFrame#folha_texto {{ background: {c["superficie_2"]}; border: 1px solid {c["borda_forte"]};
        border-radius: {tema.RAIO["g"]}px; }}
    QFrame#painel_config {{ background: {c["superficie_1"]}; border-left: 1px solid {c["borda"]}; }}
    QFrame#painel_config[modo="folha"] {{ border-left: none; border-top: 1px solid {c["borda_forte"]};
        border-top-left-radius: {tema.RAIO["g"]}px; border-top-right-radius: {tema.RAIO["g"]}px; }}
    """
