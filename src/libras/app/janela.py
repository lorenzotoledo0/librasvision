"""Janela principal: barra lateral + paginas. Ponto de entrada: `python -m libras.app`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

from PySide6.QtCore import QObject, QSettings, QSize, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QFont, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__  # noqa: E402
from ..config import carregar_config  # noqa: E402
from ..ui import tema  # noqa: E402
from ..ui.dinamica import IndicadorAtividade, MonitorDados, animar_entrada, posicionar_toasts  # noqa: E402
from ..ui.icones import icone  # noqa: E402
from .base import Pagina, botao, rotulo  # noqa: E402

log = logging.getLogger("libras.app")
LIMITE_TROCA_MS = 100  # meta do item 7
LARGURA_BARRA, LARGURA_BARRA_RECOLHIDA = 240, 96
RECOLHER_ABAIXO_DE = tema.FAIXAS["expandida_min_px"]
DURACAO_TRANSICAO_MS = int(tema.TOKENS["movimento"]["duracao_ms"])

ATALHOS = [
    ("Ctrl+1 ... Ctrl+9, Ctrl+0", "Ir para a aba 1 a 10"),
    ("Ctrl+Tab / Ctrl+Shift+Tab", "Próxima aba / aba anterior"),
    ("Ctrl+B", "Recolher ou expandir a barra lateral (botão de menu)"),
    ("F5", "Atualizar os dados da aba (botão Atualizar no topo)"),
    ("F1", "Esta lista de atalhos (Configurações > Atalhos)"),
    ("Ctrl+Shift+G", "Galeria de componentes (Configurações > Galeria)"),
    ("Gravar: C", "Ligar ou desligar a câmera"),
    ("Gravar: Espaço", "Gravar take (no modo manual, Espaço também para)"),
    ("Gravar: Enter / Esc / R", "Aceitar / descartar / regravar o take"),
    ("Revisão: Espaço, setas, Shift+setas", "Reproduzir/pausar, quadro a quadro, 1 s (botões do editor)"),
    ("Revisão: I / O / S / Ctrl+Z / Ctrl+Y", "Início, fim, dividir, desfazer, refazer"),
    ("Tempo real: D", "Abrir ou fechar Diagnóstico e teste"),
    ("Tempo real: T", "Registrar tentativa (com Diagnóstico e teste aberto)"),
]

PAGINAS = [
    ("inicio", "Início", "inicio"),
    ("sinalizadores", "Sinalizadores", "sinalizadores"),
    ("vocabulario", "Vocabulário", "vocabulario"),
    ("gravar", "Gravar", "gravar"),
    ("revisao", "Revisão", "revisao"),
    ("dataset", "Dataset", "dataset"),
    ("treino", "Treino", "treino"),
    ("resultados", "Resultados", "resultados"),
    ("tempo_real", "Tempo real", "tempo_real"),
    ("configuracoes", "Configurações", "configuracoes"),
]
ROTULO_CURTO = {
    "inicio": "Início",
    "sinalizadores": "Pessoas",
    "vocabulario": "Palavras",
    "gravar": "Gravar",
    "revisao": "Revisão",
    "dataset": "Dataset",
    "treino": "Treino",
    "resultados": "Resultados",
    "tempo_real": "Ao vivo",
    "configuracoes": "Ajustes",
}
# faixa compacta: barra inferior com 5 destinos (Material: 3 a 5; ver docs/analise_interface.md)
DESTINOS_INFERIORES = ["inicio", "gravar", "tempo_real", "resultados", "mais"]


def linha_atalho(a, b) -> QHBoxLayout:
    h = QHBoxLayout()
    a.setMinimumWidth(240)
    h.addWidget(a)
    h.addWidget(b, 1)
    return h


class _PonteErros(QObject):
    """Leva notificacoes de erro de qualquer thread para a thread da interface."""

    erro = Signal(str, str)


class PaginaMais(Pagina):
    """Faixa compacta: os destinos que nao cabem na barra inferior."""

    titulo = "Mais"
    subtitulo = "Outras telas do aplicativo."

    def __init__(self, app):
        super().__init__(app)
        for chave, texto, ic in PAGINAS:
            if chave in DESTINOS_INFERIORES:
                continue
            b = botao(texto, variante="secundario", acao=lambda _=False, k=chave: self.app.ir_para(k))
            b.setIcon(icone(ic, tema.CORES["texto"], tema.CORES["sobre_acento"]))
            b.setMinimumHeight(tema.ALVO_TOQUE + 8)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.raiz.addWidget(b)
        self.raiz.addStretch(1)


def _botao_nav(chave: str, texto: str, ic: str, dica: str) -> QToolButton:
    b = QToolButton()
    b.setObjectName("navitem")
    b.setCheckable(True)
    b.setText(texto)
    b.setIcon(icone(ic, tema.CORES["texto2"], tema.CORES["sobre_acento"]))
    b.setIconSize(QSize(22, 22))
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip(dica)
    b.setAccessibleName(texto)
    b.setFocusPolicy(Qt.TabFocus)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return b


class JanelaPrincipal(QMainWindow):
    def __init__(self, cfg, aba_inicial: str = "inicio", transicao_ms: int | None = None):
        super().__init__()
        self.cfg = cfg
        self._transicao_fixa = transicao_ms
        self.setWindowTitle(f"LibrasVision Lab {__version__}")
        self.setMinimumSize(360, 520)  # faixa compacta (celular deitado/retrato, 390 px)
        self.ajustes = QSettings(str(Path(cfg.arquivo).with_name("interface.ini")), QSettings.IniFormat)
        self.faixa: str | None = None
        raiz = QWidget()
        raiz.setObjectName("raiz")
        self.setCentralWidget(raiz)
        externo = QVBoxLayout(raiz)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        externo.addLayout(h, 1)

        # ---- barra lateral (expandida) / trilho de icones (media)
        self.barra = QFrame()
        self.barra.setObjectName("barra_lateral")
        bl = QVBoxLayout(self.barra)
        bl.setContentsMargins(0, 8, 0, 12)
        bl.setSpacing(0)
        self.bt_menu = _botao_nav("menu", "Menu", "menu", "Recolher ou expandir a barra lateral (Ctrl+B)")
        self.bt_menu.setCheckable(False)
        self.bt_menu.setAccessibleName("Recolher ou expandir a barra lateral")
        self.bt_menu.clicked.connect(self.alternar_barra)
        bl.addWidget(self.bt_menu)
        self.lb_marca = rotulo("LibrasVision Lab", "marca")
        self.lb_marca_sub = rotulo("Reconhecimento de sinais de Libras", "marca_sub", quebra=True)
        bl.addWidget(self.lb_marca)
        bl.addWidget(self.lb_marca_sub)
        self.assinatura = QFrame()
        self.assinatura.setObjectName("assinatura")
        bl.addWidget(self.assinatura)
        bl.addSpacing(8)
        self.grupo = QButtonGroup(self)
        self.grupo.setExclusive(True)
        self.botoes: dict[str, QToolButton] = {}
        for i, (chave, texto, ic) in enumerate(PAGINAS):
            b = _botao_nav(chave, texto, ic, f"{texto} (Ctrl+{(i + 1) % 10})")
            b.clicked.connect(lambda _=False, k=chave: self.ir_para(k))
            self.grupo.addButton(b)
            bl.addWidget(b)
            self.botoes[chave] = b
        bl.addStretch(1)
        self.lb_versao = rotulo(f"  versão {__version__}", "secundario")
        bl.addWidget(self.lb_versao)
        h.addWidget(self.barra)

        direita = QVBoxLayout()
        direita.setContentsMargins(0, 0, 0, 0)
        direita.setSpacing(0)
        self.indicador = IndicadorAtividade()
        direita.addWidget(self.indicador)
        self.pilha = QStackedWidget()
        direita.addWidget(self.pilha, 1)
        h.addLayout(direita, 1)

        # ---- barra inferior (compacta)
        self.barra_inferior = QFrame()
        self.barra_inferior.setObjectName("barra_inferior")
        bi = QHBoxLayout(self.barra_inferior)
        bi.setContentsMargins(4, 0, 4, 0)
        bi.setSpacing(0)
        self.grupo_inf = QButtonGroup(self)
        self.grupo_inf.setExclusive(True)
        self.botoes_inf: dict[str, QToolButton] = {}
        nomes = {k: (t_, ic) for k, t_, ic in PAGINAS}
        nomes["mais"] = ("Mais", "menu")
        for chave in DESTINOS_INFERIORES:
            texto, ic = nomes[chave]
            b = _botao_nav(chave, ROTULO_CURTO.get(chave, texto), ic, texto)
            b.setProperty("modo", "inferior")
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.clicked.connect(lambda _=False, k=chave: self.ir_para(k))
            self.grupo_inf.addButton(b)
            bi.addWidget(b)
            self.botoes_inf[chave] = b
        externo.addWidget(self.barra_inferior)

        self.barra_manual: bool | None = None  # None = automatico pela largura
        manual = self.ajustes.value("barra_recolhida", "")
        if manual in ("true", "false"):
            self.barra_manual = manual == "true"
        self.aplicar_preferencias(reaplicar_estilo=False)
        geo = self.ajustes.value("geometria")
        if geo is None or not self.restoreGeometry(geo):
            tela = QApplication.primaryScreen()
            disp = tela.availableGeometry() if tela else None
            larg, alt = (
                (1360, 860)
                if disp is None
                else (min(1360, int(disp.width() * 0.92)), min(860, int(disp.height() * 0.9)))
            )
            self.resize(max(larg, 860), max(alt, 540))

        self.paginas: dict = {}
        self.atual: str | None = None
        self.ultima_troca: dict = {}
        self._atalhos()
        self.monitor = MonitorDados(cfg)
        self.monitor.mudou.connect(self._dados_mudaram)
        self._aplicar_faixa()
        self.ir_para(aba_inicial if aba_inicial in self.botoes else "inicio")
        self.pilha.setFocusPolicy(Qt.StrongFocus)
        QTimer.singleShot(0, self.pilha.setFocus)  # foco inicial no conteudo, nao no botao de menu
        QTimer.singleShot(400, self.precarregar)

    @property
    def transicao_ms(self) -> int:
        if self._transicao_fixa is not None:
            return self._transicao_fixa
        return tema.duracao_ms()

    # ------------------------------------------------------------------
    def aplicar_preferencias(self, reaplicar_estilo: bool = True) -> None:
        """Escala de fonte (100% a 150%) e reduzir movimento, de Configuracoes (configs/local.yaml)."""
        pref = self.cfg.get("interface") or {}
        tema.configurar(
            escala_fonte=float(pref.get("escala_fonte", 1.0)),
            reduzir_movimento=bool(pref.get("reduzir_movimento", False)),
        )
        if reaplicar_estilo:
            self._reaplicar_estilo()

    def _reaplicar_estilo(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        fonte = app.font()
        fonte.setFamilies(tema.FONTES_ALTERNATIVAS)
        fonte.setPixelSize(tema.px("corpo"))
        app.setFont(fonte)
        app.setStyleSheet(tema.folha_de_estilo())

    def _atalhos(self) -> None:
        for i, (chave, _, _) in enumerate(PAGINAS):
            QShortcut(QKeySequence(f"Ctrl+{(i + 1) % 10}"), self, activated=lambda k=chave: self.ir_para(k))
        QShortcut(QKeySequence("Ctrl+Tab"), self, activated=lambda: self._vizinha(1))
        QShortcut(QKeySequence("Ctrl+Shift+Tab"), self, activated=lambda: self._vizinha(-1))
        QShortcut(QKeySequence("Ctrl+B"), self, activated=self.alternar_barra)
        QShortcut(QKeySequence("F5"), self, activated=self.atualizar)
        QShortcut(QKeySequence("F1"), self, activated=self.mostrar_atalhos)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, activated=self.abrir_galeria)

    def atualizar(self) -> None:
        self._dados_mudaram({"manual"})

    def abrir_galeria(self):
        from .galeria import GaleriaComponentes

        g = GaleriaComponentes(self)
        g.show()
        self._galeria = g
        return g

    def _vizinha(self, passo: int) -> None:
        chaves = [k for k, _, _ in PAGINAS]
        i = chaves.index(self.atual) if self.atual in chaves else 0
        self.ir_para(chaves[(i + passo) % len(chaves)])

    def mostrar_atalhos(self) -> None:
        from .base import rotulo as rot

        d = QDialog(self)
        d.setWindowTitle("Atalhos de teclado")
        lay = QVBoxLayout(d)
        lay.addWidget(rot("Todo atalho tem um botão equivalente na tela (para toque).", "dica", quebra=True))
        for tecla, desc in ATALHOS:
            lay.addLayout(linha_atalho(rot(tecla, "titulo_painel"), rot(desc, "secundario", quebra=True)))
        d.setMinimumWidth(min(560, self.width() - 40))
        d.exec()

    # ------------------------------------------------------------------ faixas de largura
    def barra_recolhida(self) -> bool:
        if self.faixa == "compacta":
            return True
        if self.barra_manual is not None:
            return self.barra_manual
        return self.width() < RECOLHER_ABAIXO_DE

    def _aplicar_barra(self) -> None:
        compacta = self.faixa == "compacta"
        rec = self.barra_recolhida()
        self.barra.setVisible(not compacta)
        self.barra_inferior.setVisible(compacta)
        self.barra.setFixedWidth(LARGURA_BARRA_RECOLHIDA if rec else LARGURA_BARRA)
        for w in (self.lb_marca, self.lb_marca_sub, self.lb_versao):
            w.setVisible(not rec)
        estilo = Qt.ToolButtonTextUnderIcon if rec else Qt.ToolButtonTextBesideIcon
        for chave, texto, _ in PAGINAS:
            b = self.botoes[chave]
            b.setToolButtonStyle(estilo)
            b.setText(ROTULO_CURTO[chave] if rec else texto)  # o trilho mostra rotulo curto, nunca so icone
            b.setProperty("modo", "trilho" if rec else "lateral")
            b.style().unpolish(b)
            b.style().polish(b)
        self.bt_menu.setToolButtonStyle(Qt.ToolButtonIconOnly if not rec else Qt.ToolButtonTextUnderIcon)
        self.bt_menu.setText("Menu")
        self.bt_menu.setProperty("modo", "trilho" if rec else "lateral")

    def _aplicar_faixa(self) -> None:
        faixa = tema.faixa_para_largura(self.width())
        if faixa != self.faixa:
            self.faixa = faixa
            tema.configurar(faixa=faixa)
            self._reaplicar_estilo()
            for p in getattr(self, "paginas", {}).values():
                p.aplicar_faixa(faixa)
        self._aplicar_barra()

    def alternar_barra(self) -> None:
        if self.faixa == "compacta":
            self.ir_para("mais")
            return
        self.barra_manual = not self.barra_recolhida()
        self.ajustes.setValue("barra_recolhida", "true" if self.barra_manual else "false")
        self._aplicar_barra()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._aplicar_faixa()
        posicionar_toasts(self)

    # ------------------------------------------------------------------
    def _dados_mudaram(self, tipos: set) -> None:
        """Dados ao vivo: a pagina aberta se atualiza sozinha; as outras se atualizam ao entrar."""
        if self.atual and self.atual in self.paginas:
            try:
                self.paginas[self.atual].atualizar_dados(set(tipos))
            except Exception:  # noqa: BLE001
                log.exception("Falha ao atualizar a pagina %s", self.atual)

    # ------------------------------------------------------------------
    def pagina(self, chave: str):
        if chave not in self.paginas:
            if chave == "mais":
                p = PaginaMais(self)
            else:
                from importlib import import_module

                p = import_module(f"libras.app.paginas.{chave}").PaginaClasse(self)
            self.paginas[chave] = p
            self.pilha.addWidget(p)
            if self.faixa:
                p.aplicar_faixa(self.faixa)
        return self.paginas[chave]

    def ir_para(self, chave: str) -> None:
        """Troca de aba. Cada troca e medida e registrada no log (item 7): sair da aba anterior,
        criar a pagina (so na primeira visita), entrar e ate a tela ser pintada."""
        if chave == self.atual or (chave not in self.botoes and chave != "mais"):
            return
        t0 = time.perf_counter()
        anterior = self.atual
        if self.atual:
            try:
                self.paginas[self.atual].ao_sair()
            except Exception:  # noqa: BLE001
                log.exception("Falha ao sair da pagina %s", self.atual)
        t1 = time.perf_counter()
        criada = chave not in self.paginas
        p = self.pagina(chave)
        t2 = time.perf_counter()
        self.pilha.setCurrentWidget(p)
        if chave in self.botoes:
            self.botoes[chave].setChecked(True)
        inf = self.botoes_inf.get(chave) or self.botoes_inf["mais"]
        inf.setChecked(True)
        self.atual = chave
        if anterior is not None:
            animar_entrada(p, self.transicao_ms)
        try:
            p.ao_entrar()
        except Exception as e:  # noqa: BLE001
            from .base import erro

            erro(self, e)
        t3 = time.perf_counter()
        tempos = {"sair": (t1 - t0) * 1000, "criar": (t2 - t1) * 1000, "entrar": (t3 - t2) * 1000}

        def pintado():
            total = (time.perf_counter() - t0) * 1000
            self.ultima_troca = {"de": anterior, "para": chave, "total_ms": total, "criada": criada, **tempos}
            nivel = logging.WARNING if total > LIMITE_TROCA_MS else logging.INFO
            log.log(
                nivel,
                "Troca de aba %s -> %s: %.0f ms (sair %.0f, criar %.0f, entrar %.0f)%s",
                anterior,
                chave,
                total,
                tempos["sair"],
                tempos["criar"],
                tempos["entrar"],
                " [primeira visita]" if criada else "",
            )

        QTimer.singleShot(0, pintado)

    def precarregar(self) -> None:
        """Cria as paginas restantes uma a uma, quando a interface esta ociosa: a primeira visita a cada
        aba fica tao rapida quanto as seguintes. Nenhuma pagina faz trabalho pesado no construtor."""
        faltando = [k for k, _, _ in PAGINAS if k not in self.paginas]
        if not faltando:
            return
        t = time.perf_counter()
        self.pagina(faltando[0])
        log.debug("Pagina %s pre-carregada em %.0f ms", faltando[0], (time.perf_counter() - t) * 1000)
        QTimer.singleShot(30, self.precarregar)

    def recarregar_config(self) -> None:
        self.cfg = carregar_config(self.cfg.arquivo)

    def closeEvent(self, ev) -> None:  # noqa: N802
        for chave, p in self.paginas.items():
            if hasattr(p, "pode_fechar") and not p.pode_fechar():
                ev.ignore()
                return
        for chave, p in self.paginas.items():
            try:
                p.encerrar()
            except Exception:  # noqa: BLE001
                log.exception("Falha ao encerrar %s", chave)
        self.ajustes.setValue("geometria", self.saveGeometry())
        self.ajustes.sync()
        from .base import Subprocesso, aguardar_encerramentos, aguardar_tarefas

        Subprocesso.aguardar_todos(5000)
        if not aguardar_tarefas(8000):
            log.warning("Tarefas em segundo plano nao terminaram em 8 s")
        if not aguardar_encerramentos(10.0):
            log.warning("Encerramento de camera/MediaPipe nao terminou em 10 s")
        log.info("Aplicativo fechado")
        ev.accept()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m libras.app")
    ap.add_argument("--config", default=None)
    ap.add_argument("--aba", default="inicio", help="pagina inicial: " + ", ".join(k for k, _, _ in PAGINAS))
    a = ap.parse_args(argv)
    # Este processo nunca usa TensorFlow (treino e tempo real rodam em processos separados).
    # O MediaPipe 0.10.21 importaria o TensorFlow so por um utilitario opcional: bloquear economiza ~500 MB.
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("LibrasVision Lab")
    app.setStyle("Fusion")
    fonte = QFont()
    fonte.setFamilies(tema.FONTES_ALTERNATIVAS)
    fonte.setPixelSize(tema.px("corpo"))
    app.setFont(fonte)
    app.setStyleSheet(tema.folha_de_estilo())
    from .base import Dialogo

    try:
        cfg = carregar_config(a.config)
    except Exception as e:  # noqa: BLE001
        Dialogo(None, "LibrasVision Lab", f"Não foi possível ler a configuração:\n\n{e}", "erro").exec()
        return 2
    from ..coleta.manutencao import limpar_temporarios
    from ..registro import configurar_log, definir_notificador

    arq_log = configurar_log(cfg.caminho("logs"), "app")
    ponte = _PonteErros()
    definir_notificador(lambda t, m: ponte.erro.emit(t, m))
    try:
        limpar_temporarios(cfg)
        janela = JanelaPrincipal(cfg, a.aba)
    except Exception as e:  # noqa: BLE001
        log.exception("Falha ao abrir o aplicativo")
        Dialogo(
            None, "LibrasVision Lab", f"Não foi possível abrir o aplicativo:\n\n{e}\n\nDetalhes em: {arq_log}", "erro"
        ).exec()
        return 2
    ponte.erro.connect(lambda t, m: Dialogo(janela, t, m, "erro").exec())
    janela.show()
    return app.exec()
