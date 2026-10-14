"""Componentes reutilizaveis da interface (bancada e app real): paineis, botoes, tabelas, video, dialogos,
tarefas e subprocessos em segundo plano. Nao importa nada da bancada (libras.app) nem da coleta."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..erros import ErroLibras

log = logging.getLogger("libras.app")
RAIZ_CODIGO = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def painel(nome: str = "painel", margem: int = 16) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName(nome)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(margem, margem, margem, margem)
    lay.setSpacing(10)
    return f, lay


class Secao(QFrame):
    """Painel com titulo que recolhe: fechado, mostra so o titulo e um resumo de uma linha (o que esta
    escolhido), para formularios longos caberem na tela sem rolar. Usa o mesmo estilo de painel."""

    def __init__(self, titulo: str, aberta: bool = True, recolhivel: bool = True):
        super().__init__()
        self.setObjectName("painel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)
        self.recolhivel = recolhivel
        if recolhivel:
            self.cabeca = QToolButton()
            self.cabeca.setObjectName("cabeca_secao")
            self.cabeca.setText(titulo)
            self.cabeca.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.cabeca.setCursor(Qt.PointingHandCursor)
            self.cabeca.setAccessibleName(f"{titulo} (mostrar ou recolher)")
            self.cabeca.clicked.connect(lambda: self.abrir(not self.aberta))
            lay.addWidget(self.cabeca, 0, Qt.AlignLeft)
        else:
            self.cabeca = rotulo(titulo, "titulo_painel")
            lay.addWidget(self.cabeca)
        self.resumo = rotulo("", "secundario", quebra=True)
        lay.addWidget(self.resumo)
        self.corpo = QWidget()
        self.conteudo = QVBoxLayout(self.corpo)
        self.conteudo.setContentsMargins(0, 0, 0, 0)
        self.conteudo.setSpacing(8)
        lay.addWidget(self.corpo)
        self.aberta = True
        self.abrir(aberta)

    def abrir(self, aberta: bool) -> None:
        self.aberta = bool(aberta) or not self.recolhivel
        self.corpo.setVisible(self.aberta)
        self.resumo.setVisible(not self.aberta and bool(self.resumo.text()))
        if self.recolhivel:
            self.cabeca.setArrowType(Qt.DownArrow if self.aberta else Qt.RightArrow)

    def definir_resumo(self, texto: str) -> None:
        self.resumo.setText(texto)
        self.resumo.setVisible(not self.aberta and bool(texto))


def rotulo(texto: str = "", nome: str | None = None, quebra: bool = False) -> QLabel:
    l = QLabel(texto)
    if nome:
        l.setObjectName(nome)
    l.setWordWrap(quebra)
    l.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return l


def botao(
    texto: str, primario: bool = False, perigo: bool = False, acao=None, dica: str = "", variante: str | None = None
) -> QPushButton:
    """Botao padronizado (D18): variante "primario", "secundario" (padrao), "fantasma" ou "destrutivo".
    A dica e complementar: a acao tem de estar clara no proprio texto (nada so no hover)."""
    variante = variante or ("primario" if primario else "destrutivo" if perigo else "secundario")
    b = QPushButton(texto)
    if variante != "secundario":
        b.setObjectName({"destrutivo": "perigo"}.get(variante, variante))
    b.setCursor(Qt.PointingHandCursor)
    if acao:
        b.clicked.connect(acao)
    if dica:
        b.setToolTip(dica)
    return b


def etiqueta(texto: str, tipo: str = "info") -> QLabel:
    """Etiqueta de estado (ok, pendente, ensaio, erro, info, captando): cor, borda e TEXTO."""
    l = QLabel(texto)
    l.setObjectName("etiqueta")
    l.setProperty("tipo", tipo)
    l.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return l


def mudar_etiqueta(l: QLabel, texto: str, tipo: str) -> None:
    l.setText(texto)
    if l.property("tipo") != tipo:
        l.setProperty("tipo", tipo)
        l.style().unpolish(l)
        l.style().polish(l)


def dica(texto: str) -> QLabel:
    """Explicacao curta sempre visivel (substitui informacao que antes so aparecia no hover)."""
    l = rotulo(texto, "dica", quebra=True)
    return l


def formulario():
    """Formulario que poe o rotulo acima do campo quando a linha nao cabe (janela estreita)."""
    from PySide6.QtWidgets import QFormLayout

    f = QFormLayout()
    f.setRowWrapPolicy(QFormLayout.WrapLongRows)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    f.setVerticalSpacing(10)
    return f


def separador() -> QFrame:
    f = QFrame()
    f.setObjectName("separador")
    return f


class EstadoVazio(QFrame):
    """Estado vazio com orientacao: o que falta e o proximo passo (com acao opcional)."""

    def __init__(self, titulo: str, texto: str, acao_texto: str = "", acao=None, icone_nome: str = "alerta"):
        super().__init__()
        from .icones import icone

        self.setObjectName("estado_vazio")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(8)
        topo = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icone(icone_nome, _cor("texto2")).pixmap(28, 28))
        topo.addWidget(ic)
        self.titulo = rotulo(titulo, "titulo_painel", quebra=True)
        topo.addWidget(self.titulo, 1)
        lay.addLayout(topo)
        self.texto = rotulo(texto, "secundario", quebra=True)
        lay.addWidget(self.texto)
        self.botao = None
        if acao_texto:
            self.botao = botao(acao_texto, primario=True, acao=acao)
            lay.addLayout(linha(self.botao))

    def definir(self, titulo: str, texto: str) -> None:
        self.titulo.setText(titulo)
        self.texto.setText(texto)


def _cor(nome: str) -> str:
    from . import tema

    return tema.CORES[nome]


def linha(*widgets, esticar_fim: bool = True):
    """Fila de controles. Com esticar_fim (padrao) a fila QUEBRA em janela estreita (FluxoLayout);
    sem esticar_fim e uma QHBoxLayout comum, para quando um item precisa esticar."""
    if esticar_fim:
        from .dinamica import FluxoLayout

        f = FluxoLayout(8)
        for w in widgets:
            if isinstance(w, int):
                f.addSpacing(w)
            else:
                f.addWidget(w)
        return f
    h = QHBoxLayout()
    h.setSpacing(8)
    for w in widgets:
        if isinstance(w, int):
            h.addSpacing(w)
        else:
            h.addWidget(w)
    if esticar_fim:
        h.addStretch(1)
    return h


# ----------------------------------------------------------------------
# Tabela
# ----------------------------------------------------------------------
class Tabela(QTableWidget):
    def __init__(self, colunas: list[str], selecao_unica: bool = True):
        super().__init__(0, len(colunas))
        self.setHorizontalHeaderLabels(colunas)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(
            QAbstractItemView.SingleSelection if selecao_unica else QAbstractItemView.ExtendedSelection
        )
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)
        self.setMinimumHeight(150)

    def preencher(self, linhas: list[list], cores: list[str | None] | None = None) -> None:
        """Preenche e preserva a linha selecionada (pela 1a coluna), para os dados ao vivo nao tirarem a
        selecao do usuario."""
        chave = self.valor_selecionado(0)
        self.blockSignals(True)
        self.setRowCount(0)
        for i, l in enumerate(linhas):
            self.insertRow(i)
            for j, v in enumerate(l):
                it = QTableWidgetItem("" if v is None else str(v))
                if cores and cores[i]:
                    from PySide6.QtGui import QColor

                    it.setForeground(QColor(cores[i]))
                self.setItem(i, j, it)
        self.resizeColumnsToContents()
        if chave is not None:
            self.selecionar_valor(chave)
        self.blockSignals(False)

    def selecionar_valor(self, valor: str, coluna: int = 0) -> bool:
        for r in range(self.rowCount()):
            it = self.item(r, coluna)
            if it is not None and it.text() == valor:
                self.selectRow(r)
                return True
        return False

    def valor_selecionado(self, coluna: int = 0) -> str | None:
        r = self.currentRow()
        if r < 0 or self.item(r, coluna) is None:
            return None
        return self.item(r, coluna).text()


# ----------------------------------------------------------------------
# Imagem
# ----------------------------------------------------------------------
class TelaVideo(QLabel):
    """Mostra frames BGR (numpy) mantendo a proporcao."""

    def __init__(self, texto_vazio: str = "", proporcao: float | None = None):
        super().__init__(texto_vazio)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("tela_video")
        self.setWordWrap(True)
        self.setMinimumSize(240, 180)
        self.proporcao = proporcao  # largura/altura; com ela a altura acompanha a largura (4:3)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred if proporcao else QSizePolicy.Expanding)
        sp.setHeightForWidth(bool(proporcao))
        self.setSizePolicy(sp)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return bool(self.proporcao)

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return int(w / self.proporcao) if self.proporcao else super().heightForWidth(w)

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        return QSize(640, 480) if self.proporcao else super().sizeHint()

    def mostrar(self, frame_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mostrar_jpeg(self, dados: bytes) -> None:
        pm = QPixmap()
        if pm.loadFromData(dados, "JPG"):
            self.setPixmap(pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ----------------------------------------------------------------------
# Dialogos
# ----------------------------------------------------------------------
class Dialogo(QDialog):
    """Dialogo padronizado. Nao usa QMessageBox, que toca o som do sistema no Windows: o publico e surdo
    e nenhum retorno pode depender de som (o aplicativo nao emite som)."""

    def __init__(self, parent, titulo: str, texto: str, tipo: str = "info", botoes=(("OK", "primario"),)):
        super().__init__(parent)
        from .icones import icone

        self.setWindowTitle(titulo)
        self.escolha: int | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)
        topo = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(
            icone(
                {"erro": "alerta", "pergunta": "alerta"}.get(tipo, "ok"),
                _cor({"erro": "erro", "pergunta": "aviso"}.get(tipo, "texto2")),
            ).pixmap(32, 32)
        )
        topo.addWidget(ic, 0, Qt.AlignTop)
        topo.addWidget(
            etiqueta(
                {"erro": "Erro", "pergunta": "Confirmar"}.get(tipo, "Aviso"),
                {"erro": "erro", "pergunta": "pendente"}.get(tipo, "info"),
            ),
            0,
            Qt.AlignTop,
        )
        topo.addStretch(1)
        lay.addLayout(topo)
        self.rotulo = rotulo(texto, quebra=True)
        self.rotulo.setMinimumWidth(380)
        lay.addWidget(self.rotulo)
        bs = []
        for i, (txt, var) in enumerate(botoes):
            b = botao(txt, variante=var, acao=lambda _=False, i=i: self._escolher(i))
            bs.append(b)
        self.botoes = bs
        lay.addLayout(linha(*bs))
        if bs:
            bs[-1].setDefault(True)

    def _escolher(self, i: int) -> None:
        self.escolha = i
        self.accept()


def erro(parent, e: Exception | str, titulo: str = "Erro") -> None:
    if isinstance(e, Exception) and not isinstance(e, ErroLibras):
        log.error("Erro na interface", exc_info=e)
        texto = f"Erro inesperado: {type(e).__name__}: {e}\n\nDetalhes no log (runs/logs/app.log)."
    else:
        texto = str(e)
    log.warning("Erro exibido: %s", texto.replace("\n", " "))
    Dialogo(parent, titulo, texto, "erro").exec()


def info(parent, texto: str, titulo: str = "LibrasVision") -> None:
    """Aviso simples: notificacao nao bloqueante (item 8). Dialogos ficam para erros e confirmacoes."""
    from .dinamica import mostrar_toast

    mostrar_toast(parent, texto, "ok")


def avisar(parent, texto: str, tipo: str = "info") -> None:
    from .dinamica import mostrar_toast

    mostrar_toast(parent, texto, tipo)


def perguntar(parent, texto: str, titulo: str = "Confirmar", destrutivo: bool = False) -> bool:
    d = Dialogo(
        parent, titulo, texto, "pergunta", (("Não", "secundario"), ("Sim", "destrutivo" if destrutivo else "primario"))
    )
    d.exec()
    return d.escolha == 1


def pedir_texto(parent, titulo: str, rotulo_: str, padrao: str = "") -> str | None:
    t, ok = QInputDialog.getText(parent, titulo, rotulo_, QLineEdit.Normal, padrao)
    return t if ok else None


class DialogoDigitar(QDialog):
    """Confirmacao que exige digitar um texto exato (ex.: o ID do sinalizador)."""

    def __init__(self, parent, titulo: str, explicacao: str, esperado: str):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.esperado = esperado
        lay = QVBoxLayout(self)
        lay.addWidget(rotulo(explicacao, quebra=True))
        lay.addWidget(rotulo(f"Digite {esperado} para confirmar:", "secundario"))
        self.campo = QLineEdit()
        lay.addWidget(self.campo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setEnabled(False)
        self.campo.textChanged.connect(lambda t: self.ok.setEnabled(t.strip() == esperado))
        lay.addWidget(bb)
        self.setMinimumWidth(420)

    def texto(self) -> str:
        return self.campo.text()


# ----------------------------------------------------------------------
# Tarefas em segundo plano
# ----------------------------------------------------------------------
class Tarefa(QThread):
    """Executa funcao(progresso, cancelado) fora da thread da interface.

    progresso(i, n, texto) atualiza a barra; cancelado() informa se o usuario pediu para parar.
    """

    progresso = Signal(int, int, str)
    concluido = Signal(object)
    falhou = Signal(object)

    def __init__(self, funcao, parent=None):
        super().__init__(parent)
        self.funcao = funcao
        self._cancelar = False

    def cancelar(self) -> None:
        self._cancelar = True

    def run(self) -> None:
        try:
            r = self.funcao(lambda i, n, t="": self.progresso.emit(int(i), int(n), str(t)), lambda: self._cancelar)
            self.concluido.emit(r)
        except Exception as e:  # noqa: BLE001 - repassado a interface
            if not isinstance(e, ErroLibras):
                log.error("Falha em tarefa de segundo plano\n%s", traceback.format_exc())
            self.falhou.emit(e)


def iniciar_tarefa(
    dono, funcao, ao_concluir=None, ao_falhar=None, ao_progresso=None, descricao: str = "Processando..."
) -> Tarefa:
    """Roda funcao fora da thread da interface. Se passar de 200 ms, o indicador de atividade aparece."""
    from .dinamica import ATIVIDADE

    t = Tarefa(funcao, dono)
    t.setObjectName(f"tarefa: {descricao}")
    ident = ATIVIDADE.iniciar(descricao)
    t.finished.connect(lambda: ATIVIDADE.terminar(ident))
    if ao_concluir:
        t.concluido.connect(ao_concluir)
    t.falhou.connect(ao_falhar or (lambda e: erro(dono, e)))
    if ao_progresso:
        t.progresso.connect(ao_progresso)
    dono._tarefas = [x for x in getattr(dono, "_tarefas", []) if x.isRunning()] + [t]  # mantem referencia
    _TAREFAS[:] = [x for x in _TAREFAS if x.isRunning()] + [t]
    t.start()
    return t


_TAREFAS: list = []


def aguardar_tarefas(espera_ms: int = 8000) -> bool:
    """Ao fechar o aplicativo: pede cancelamento e espera as tarefas em andamento (destruir uma
    QThread ainda rodando pode derrubar o processo na saida)."""
    for x in list(_TAREFAS):
        if x.isRunning():
            x.cancelar()
    ok = all(x.wait(espera_ms) for x in list(_TAREFAS))
    _TAREFAS.clear()
    return ok


class Subprocesso(QObject):
    """Roda um script do projeto em outro processo (isola TensorFlow, permite cancelar)."""

    linha = Signal(str)
    progresso = Signal(int, int)
    terminou = Signal(int)

    _RX_PROG = re.compile(r"^\[(\d+)/(\d+)\]")

    def __init__(self, parent=None, binario: bool = False):
        super().__init__(parent)
        self.p = QProcess(self)
        self.binario = binario
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(RAIZ_CODIGO / "src") + os.pathsep + env.value("PYTHONPATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("TF_CPP_MIN_LOG_LEVEL", "2")
        self.p.setProcessEnvironment(env)
        self.p.setWorkingDirectory(str(RAIZ_CODIGO))
        if not binario:
            self.p.setProcessChannelMode(QProcess.MergedChannels)
            self.p.readyReadStandardOutput.connect(self._ler)
        self.p.finished.connect(lambda codigo, _s: self.terminou.emit(int(codigo)))
        self._resto = ""

    @staticmethod
    def interpretador() -> str:
        """python.exe ao lado do pythonw.exe (o aplicativo abre sem console, mas os subprocessos
        precisam de stdout); no Windows o console do subprocesso fica oculto (CREATE_NO_WINDOW)."""
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
            return str(exe.parent / "python.exe")
        return str(exe)

    def _preparar_windows(self) -> None:
        if sys.platform.startswith("win") and hasattr(self.p, "setCreateProcessArgumentsModifier"):

            def sem_janela(args):
                args.flags |= 0x08000000  # CREATE_NO_WINDOW

            self.p.setCreateProcessArgumentsModifier(sem_janela)

    def iniciar_script(self, script: str, args: list[str]) -> None:
        self._preparar_windows()
        self.p.start(self.interpretador(), [str(RAIZ_CODIGO / "scripts" / script), *args])

    def iniciar_modulo(self, modulo: str, args: list[str]) -> None:
        self._preparar_windows()
        self.p.start(self.interpretador(), ["-m", modulo, *args])

    def rodando(self) -> bool:
        return self.p.state() != QProcess.NotRunning

    def _ler(self) -> None:
        txt = self._resto + bytes(self.p.readAllStandardOutput()).decode("utf-8", "replace")
        partes = txt.split("\n")
        self._resto = partes.pop()
        for l in partes:
            l = l.rstrip("\r")
            m = self._RX_PROG.match(l)
            if m:
                self.progresso.emit(int(m.group(1)), int(m.group(2)))
            self.linha.emit(l)

    def escrever(self, texto: str) -> None:
        if self.rodando():
            self.p.write(texto.encode("utf-8"))

    def parar(self, espera_ms: int = 4000) -> None:
        """Encerramento SINCRONO (so ao fechar o aplicativo). Na troca de aba use encerrar_sem_esperar."""
        if not self.rodando():
            return
        self.p.terminate()
        if not self.p.waitForFinished(espera_ms):
            self.p.kill()
            self.p.waitForFinished(2000)

    def encerrar_sem_esperar(self, comando: str | None = None, prazo_ms: int = 8000) -> None:
        """Pede o encerramento e volta na hora (item 7: a troca de aba esperava o processo terminar,
        cerca de 1,5 s no ambiente de desenvolvimento). Se nao terminar no prazo, e encerrado a forca."""
        if not self.rodando():
            return
        for sinal in (self.p.readyReadStandardOutput, self.p.readyReadStandardError):
            try:
                sinal.disconnect()
            except (RuntimeError, TypeError):
                pass
        _ENCERRANDO.add(self)
        self.p.finished.connect(lambda *_: _ENCERRANDO.discard(self))
        if comando:
            self.escrever(comando + "\n")
        else:
            self.p.terminate()

        def forcar():
            if self.rodando():
                log.warning("Subprocesso nao terminou em %d ms; encerrando a forca", prazo_ms)
                self.p.kill()

        QTimer.singleShot(prazo_ms, forcar)

    @staticmethod
    def ha_encerrando() -> bool:
        return any(s.rodando() for s in list(_ENCERRANDO))

    @staticmethod
    def aguardar_todos(espera_ms: int = 8000) -> None:
        for s in list(_ENCERRANDO):
            if s.rodando() and not s.p.waitForFinished(espera_ms):
                s.p.kill()
                s.p.waitForFinished(2000)
        _ENCERRANDO.clear()


_ENCERRANDO: set = set()  # subprocessos em encerramento (referencia ate terminarem)

# ----------------------------------------------------------------------
# Encerramentos em segundo plano (camera, MediaPipe da previa): a troca de aba nao espera por eles.
_PENDENTES: list[threading.Thread] = []


def encerrar_em_segundo_plano(nome: str, *funcoes) -> threading.Thread:
    def rodar():
        for f in funcoes:
            try:
                f()
            except Exception:  # noqa: BLE001
                log.exception("Falha ao encerrar %s", nome)

    th = threading.Thread(target=rodar, name=f"encerrar-{nome}", daemon=True)
    _PENDENTES[:] = [x for x in _PENDENTES if x.is_alive()] + [th]
    th.start()
    return th


def aguardar_encerramentos(espera_s: float = 10.0) -> bool:
    """Espera os encerramentos pendentes (antes de reabrir a camera e ao fechar o aplicativo)."""
    import time

    fim = time.monotonic() + espera_s
    for th in list(_PENDENTES):
        th.join(timeout=max(0.0, fim - time.monotonic()))
    vivos = [x for x in _PENDENTES if x.is_alive()]
    _PENDENTES[:] = vivos
    return not vivos


def ha_encerramento_pendente() -> bool:
    return any(x.is_alive() for x in _PENDENTES)


def cartao_numero(titulo: str) -> tuple[QFrame, QLabel, QLabel]:
    f, lay = painel("painel", 14)
    lay.addWidget(rotulo(titulo, "secundario"))
    n = rotulo("-", "numero")
    lay.addWidget(n)
    d = rotulo("", "secundario", quebra=True)
    lay.addWidget(d)
    return f, n, d


def widget_com(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
