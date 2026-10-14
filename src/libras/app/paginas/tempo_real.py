"""Tempo real: reconhecimento pela webcam e teste ao vivo do modelo (item 10).

A inferencia roda em OUTRO PROCESSO (libras.tempo_real_servico), iniciado ao abrir esta aba e
encerrado ao sair dela (D15, D16): o TensorFlow so ocupa memoria aqui. O processo usa o mesmo
pipeline do treino (normalizacao, janelas e decisao de inferencia.py); esta pagina so mostra o que
ele envia e chama a logica de libras.teste_ao_vivo. Sem modelo treinado nada e predito.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...tempo_real_servico import Decodificador
from ...teste_ao_vivo import AVISO_TESTE, SessaoTeste, aviso_sinalizador_visto, listar_modelos
from ...ui import tema
from ...ui.dinamica import Colunas
from ...ui.legenda import TelaLegendada, cauda_que_cabe, legenda_do_quadro  # noqa: F401 - reexportados
from ..base import Pagina, Subprocesso, Tabela, botao, dica, etiqueta, formulario, info, linha, mudar_etiqueta, rotulo

MODULO_SERVICO = "libras.tempo_real_servico"  # substituido nos testes por um servico falso
OUTRA_PESSOA = "outra pessoa (não cadastrada)"


class BarrasProbabilidade(QWidget):
    """Uma barra por classe do modelo (inclusive nenhum_sinal), com a marca do limiar."""

    ALTURA_LINHA = 30

    def __init__(self):
        super().__init__()
        self.classes: list[str] = []
        self.probs: list[float] | None = None
        self.limiar = 0.6
        self.nada: str | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def definir(self, classes: list[str], nada: str | None) -> None:
        self.classes, self.nada, self.probs = list(classes), nada, None
        self.setMinimumHeight(max(1, len(self.classes)) * self.ALTURA_LINHA + 26)
        self.update()

    def atualizar(self, probs: list[float] | None, limiar: float) -> None:
        self.probs, self.limiar = probs, limiar
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        """Barras na rampa (uma cor por classe, so para distinguir; nao indica estado). A maior
        probabilidade e marcada por contorno e pela palavra "maior", nao so pela cor; nenhum_sinal e
        neutra e rotulada."""
        c = tema.CORES
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        larg_rot, larg_pct = 128, 92
        x0 = larg_rot
        larg = max(10, self.width() - larg_rot - larg_pct)
        n = len(self.classes)
        topo = int(max(range(len(self.probs)), key=lambda i: self.probs[i])) if self.probs else -1
        alt = self.ALTURA_LINHA
        for i, cl in enumerate(self.classes):
            y = 3 + i * alt
            v = float(self.probs[i]) if self.probs and i < len(self.probs) else 0.0
            p.setPen(tema.qcolor("texto" if i == topo else "texto_secundario"))
            p.drawText(QRectF(0, y, larg_rot - 10, alt - 6), Qt.AlignVCenter | Qt.AlignRight, cl)
            trilho = QRectF(x0, y + 5, larg, alt - 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c["borda"]))
            p.drawRoundedRect(trilho, 5, 5)
            if v > 0:
                r = QRectF(x0, y + 5, larg * min(1.0, v), alt - 16)
                cor = tema.COR["curva_dados"] if cl == self.nada else tema.cor_rampa(i / max(1, n - 1))
                p.setBrush(QColor(cor))
                p.drawRoundedRect(r, 5, 5)
            if i == topo:
                p.setPen(QPen(tema.qcolor("texto"), 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(trilho.adjusted(-2, -2, 2, 2), 6, 6)
            p.setPen(tema.qcolor("texto" if i == topo else "texto_secundario"))
            sufixo = "  maior" if i == topo else ""
            p.drawText(QRectF(x0 + larg + 8, y, larg_pct, alt - 6), Qt.AlignVCenter, f"{100 * v:3.0f}%{sufixo}")
        if self.classes:
            xl = x0 + larg * self.limiar
            fim = 3 + n * alt - 8
            p.setPen(QPen(tema.qcolor("foco"), 1.5, Qt.DashLine))
            p.drawLine(int(xl), 2, int(xl), fim)
            p.setPen(tema.qcolor("foco"))
            p.drawText(QRectF(xl - 60, fim + 2, 120, 20), Qt.AlignHCenter, f"limiar {100 * self.limiar:.0f}%")
        p.end()


class IndicadorDecisao(QWidget):
    """Etapas da decisao (D15) em sequencia, com forma e texto (nada so por cor):
    mao visivel -> confianca acima do limiar -> predicoes iguais (contador) -> sinal confirmado.
    Etapa concluida: circulo cheio com visto; etapa que bloqueou: X e o motivo; pendente: anel."""

    ETAPAS = ["Mãos visíveis", "Confiança acima do limiar", "Predições iguais", "Sinal confirmado"]

    def __init__(self):
        super().__init__()
        self.estado = {"etapa": "aquecendo", "texto": "Aguardando 1 s de imagem"}
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def definir(self, etapa: dict | None) -> None:
        self.estado = etapa or {"etapa": "aquecendo", "texto": ""}
        self.update()

    def situacao(self) -> list[str]:
        """'ok', 'falha', 'andamento' ou 'pendente' para cada etapa."""
        e = self.estado.get("etapa", "aquecendo")
        if e in ("aquecendo", "sem_maos", "sem_pose"):
            return ["falha", "pendente", "pendente", "pendente"]
        if e in ("baixa_confianca", "nenhum_sinal"):
            return ["ok", "falha", "pendente", "pendente"]
        if e == "aguardando":
            return ["ok", "ok", "andamento", "pendente"]
        return ["ok", "ok", "ok", "ok"]

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        sit = self.situacao()
        n = len(self.ETAPAS)
        larg = self.width() / n
        cy = 16
        for i, (rot, s) in enumerate(zip(self.ETAPAS, sit)):
            cx = larg * i + larg / 2
            if i:
                cor_l = "sucesso" if s in ("ok", "andamento") else "borda_controle"
                p.setPen(QPen(tema.qcolor(cor_l), 2))
                p.drawLine(int(cx - larg + 14), cy, int(cx - 14), cy)
            cor = {"ok": "sucesso", "falha": "alerta", "andamento": "foco"}.get(s, "borda_controle")
            p.setPen(QPen(tema.qcolor(cor), 2))
            p.setBrush(tema.qcolor(cor) if s == "ok" else Qt.NoBrush)
            p.drawEllipse(QRectF(cx - 10, cy - 10, 20, 20))
            p.setPen(QPen(tema.qcolor("fundo" if s == "ok" else cor), 2))
            if s == "ok":
                p.drawPolyline([QPointF(cx - 5, cy), QPointF(cx - 1, cy + 4), QPointF(cx + 5, cy - 4)])
            elif s == "falha":
                p.drawLine(QPointF(cx - 4, cy - 4), QPointF(cx + 4, cy + 4))
                p.drawLine(QPointF(cx + 4, cy - 4), QPointF(cx - 4, cy + 4))
            elif s == "andamento":
                p.setBrush(tema.qcolor(cor))
                p.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))
            texto = rot
            if s == "andamento" and "contador" in self.estado:
                texto = f"{rot} ({self.estado['contador']} de {self.estado['necessarias']})"
            p.setPen(tema.qcolor("texto" if s in ("falha", "andamento") else "texto_secundario"))
            f = p.font()
            f.setBold(s in ("falha", "andamento"))
            p.setFont(f)
            p.drawText(
                QRectF(cx - larg / 2 + 2, cy + 14, larg - 4, self.height() - cy - 14),
                Qt.AlignHCenter | Qt.TextWordWrap,
                texto,
            )
        p.end()


class PaginaClasse(Pagina):
    titulo = "Tempo real"
    subtitulo = (
        "A câmera com a palavra reconhecida como legenda. Ajustes, diagnóstico e teste de acertos ficam em "
        '"Diagnóstico e teste". O modelo carrega em outro processo ao abrir esta aba e é liberado ao sair.'
    )

    def __init__(self, app):
        super().__init__(app)
        self.proc: Subprocesso | None = None
        self.dec = Decodificador()
        self.modelos = []
        self.meta_modelo: dict = {}
        self.ultimo: dict = {}
        self.video_teste: str | None = None
        self.sessao: SessaoTeste | None = None
        self._params_do_modelo = True
        self._erro_recebido = False
        self.modo_teste = False

        # vista principal: so a camera, com a legenda por cima; uma linha discreta de controles
        self.lb_ativo = rotulo("Nenhum modelo ativo", "secundario")
        self.et_ensaio = etiqueta("ENSAIO: não usar no TCC", "ensaio")
        self.et_ensaio.hide()
        self.bt_limpar = botao("Limpar legenda", acao=self.limpar_legenda)
        self.bt_modo = botao(
            "Diagnóstico e teste",
            acao=self.alternar_modo_teste,
            dica="Mostra probabilidades, etapas da decisão, ajustes, tempos e o teste de acertos (D).",
        )
        self.bt_modo.setCheckable(True)
        # 13.5: a mesma legenda do app real, com as preferencias dele (o que se testa e o que o usuario ve)
        self.ck_como_real = QCheckBox("Pré-visualizar como no app real")
        self.ck_como_real.setToolTip(
            "Aplica as preferências de legenda do app LibrasVision (posição, tamanho, cores, "
            "linhas) e os rótulos do vocabulário."
        )
        self.ck_como_real.toggled.connect(self.previa_como_real)
        self.raiz.addLayout(linha(self.et_ensaio, self.lb_ativo, self.ck_como_real, self.bt_limpar, self.bt_modo))

        h = Colunas(1050)
        esq = QVBoxLayout()
        esq.setSpacing(tema.ESPACO[3])
        self.tela = TelaLegendada("Iniciando...", proporcao=4 / 3)
        esq.addWidget(self.tela)
        self.bt_ir_treino = botao("Treinar o modelo final", primario=True, acao=self._ir_treino_final)
        self.bt_ir_treino.hide()
        esq.addLayout(linha(self.bt_ir_treino))
        esq.addWidget(dica("Protótipo de pesquisa: não substitui um intérprete de Libras."))
        esq.addStretch(1)
        h.addLayout(esq, 3)
        # modo diagnostico e teste: coluna a direita (embaixo em janela estreita), oculta por padrao
        dir_ = QVBoxLayout()
        dir_.setSpacing(tema.ESPACO[3])
        self.abas = QTabWidget()
        # cada aba rola por dentro: o video continua inteiro na tela enquanto se le o diagnostico
        for wdg, nome, dica_aba in (
            (self._aba_diagnostico(), "Decisão", "Etapas, probabilidades, ajustes e tempos"),
            (self._aba_teste(), "Testar acertos", "Registre tentativas com o botão ou a tecla T"),
            (self._aba_modelo(), "Modelo", "Modelo, câmera ou arquivo de vídeo"),
        ):
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            area.setWidget(wdg)
            self.abas.setTabToolTip(self.abas.addTab(area, nome), dica_aba)
        self.abas.setMinimumHeight(440)
        dir_.addWidget(self.abas, 1)
        h.addLayout(dir_, 2)
        self.col_teste = h.widget(1)
        self.col_teste.setMinimumWidth(340)
        self.col_teste.hide()
        self.colunas = h
        self.raiz.addWidget(h, 1)

        self.atalho_t = QShortcut(QKeySequence("T"), self)
        self.atalho_t.setContext(Qt.WidgetWithChildrenShortcut)
        self.atalho_t.activated.connect(self._atalho_registrar)
        self.atalho_d = QShortcut(QKeySequence("D"), self)
        self.atalho_d.setContext(Qt.WidgetWithChildrenShortcut)
        self.atalho_d.activated.connect(self.alternar_modo_teste)
        self.atualizar_modelos()
        self._atualizar_botoes()

    # ------------------------------------------------------------------ modo
    def alternar_modo_teste(self, ligado: bool | None = None) -> None:
        """Vista principal (so camera e legenda) ou modo diagnostico e teste (coluna extra)."""
        self.modo_teste = (not self.modo_teste) if ligado is None or not isinstance(ligado, bool) else ligado
        self.bt_modo.blockSignals(True)
        self.bt_modo.setChecked(self.modo_teste)
        self.bt_modo.blockSignals(False)
        self.bt_modo.setText("Fechar diagnóstico" if self.modo_teste else "Diagnóstico e teste")
        self.col_teste.setVisible(self.modo_teste)
        self._limitar_video()
        if self.modo_teste:  # coluna que estava oculta: o QSplitter a mostraria quase sem espaco
            h = self.colunas
            if h.orientation() == Qt.Horizontal:
                total = max(h.width(), 800)
                h.setSizes([int(total * 0.58), int(total * 0.42)])
            else:
                h.setSizes([self.tela.heightForWidth(max(1, h.width())) + 70, self.abas.minimumHeight()])
                QTimer.singleShot(
                    60, lambda: self.area.ensureWidgetVisible(self.abas, 0, 0)
                )  # tela estreita: rola ate as abas
        self.colunas.updateGeometry()

    def _atalho_registrar(self) -> None:
        if self.modo_teste:
            self.registrar_tentativa()

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        r = super().eventFilter(obj, ev)
        if ev.type() == QEvent.Resize and obj is self.area.viewport():
            self._limitar_video()
        return r

    def _limitar_video(self) -> None:
        """O video ocupa a altura visivel (sem rolar para ver a legenda), com a linha de controles acima."""
        alt = self.area.viewport().height() - 120
        self.tela.setMaximumHeight(max(240, alt))

    # ------------------------------------------------------------------ construcao
    def _aba_modelo(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        f = formulario()
        self.cb_modelo = QComboBox()
        self.cb_modelo.setMinimumContentsLength(14)
        self.cb_modelo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sp_camera = QSpinBox()
        self.sp_camera.setRange(0, 9)
        f.addRow("Modelo", self.cb_modelo)
        f.addRow("Câmera", self.sp_camera)
        lay.addLayout(f)
        self.bt_aplicar = botao("Usar este modelo e câmera", primario=True, acao=self._reiniciar)
        lay.addLayout(linha(self.bt_aplicar, botao("Atualizar lista", acao=self.atualizar_modelos)))
        lay.addWidget(rotulo("Por padrão é usado o modelo final mais recente.", "secundario", quebra=True))
        self.bt_video = botao("Testar com arquivo de vídeo...", acao=self.escolher_video)
        self.bt_voltar_camera = botao("Voltar à câmera", acao=self.voltar_camera)
        lay.addLayout(linha(self.bt_video, self.bt_voltar_camera))
        lay.addWidget(
            rotulo(
                "Com um arquivo, o mesmo pipeline roda sobre o vídeo: teste reprodutível.", "secundario", quebra=True
            )
        )
        lay.addStretch(1)
        return w

    def _aba_diagnostico(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.et_captando = etiqueta("Aguardando imagem", "info")
        self.lb_fps = rotulo("", "secundario")
        lay.addLayout(linha(self.et_captando, self.lb_fps))
        self.sinal = rotulo("", "parcial")
        lay.addWidget(self.sinal)
        self.indicador = IndicadorDecisao()
        lay.addWidget(self.indicador)
        self.etapa = rotulo("", "secundario", quebra=True)
        lay.addWidget(self.etapa)
        lay.addWidget(rotulo("Texto reconhecido até agora", "secundario"))
        self.texto = rotulo("", quebra=True)
        self.texto.setMinimumHeight(tema.px("corpo") + 8)
        lay.addWidget(self.texto)
        lay.addWidget(
            rotulo(
                "Probabilidade de cada classe (média móvel). A linha tracejada é o limiar de confiança: abaixo dele "
                "nada vira texto.",
                "secundario",
                quebra=True,
            )
        )
        self.barras = BarrasProbabilidade()
        lay.addWidget(self.barras)
        f = formulario()
        self.sp_limiar = QDoubleSpinBox()
        self.sp_limiar.setRange(0.0, 1.0)
        self.sp_limiar.setSingleStep(0.05)
        self.sp_suav = QSpinBox()
        self.sp_suav.setRange(1, 30)
        self.sp_conf = QSpinBox()
        self.sp_conf.setRange(1, 30)
        for wdg in (self.sp_limiar, self.sp_suav, self.sp_conf):
            wdg.valueChanged.connect(self._enviar_parametros)
        f.addRow("Limiar de confiança", self.sp_limiar)
        f.addRow("Média móvel (predições)", self.sp_suav)
        f.addRow("Predições iguais para confirmar", self.sp_conf)
        lay.addLayout(f)
        lay.addWidget(
            rotulo("Ajustes valem só durante este teste: não alteram o config nem o modelo.", "secundario", quebra=True)
        )
        self.ck_lm = QCheckBox("Esqueleto na imagem")
        self.ck_lm.toggled.connect(lambda v: self._cmd(f"landmarks {int(v)}"))
        lay.addWidget(self.ck_lm)
        self.tempos = rotulo("", "mono")
        lay.addWidget(self.tempos)
        lay.addLayout(
            linha(botao("Salvar medição", acao=lambda: self._cmd("salvar")), botao("Reiniciar", acao=self._reiniciar))
        )
        lay.addStretch(1)
        return w

    def _aba_teste(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lb_aviso_teste = rotulo(AVISO_TESTE, "alerta_aviso", quebra=True)
        lay.addWidget(self.lb_aviso_teste)
        f = formulario()
        self.cb_quem = QComboBox()
        self.cb_quem.currentIndexChanged.connect(self._atualizar_aviso_visto)
        self.cb_esperado = QComboBox()
        f.addRow("Quem está testando", self.cb_quem)
        f.addRow("Sinal que vai fazer", self.cb_esperado)
        lay.addLayout(f)
        self.lb_visto = rotulo("", "alerta_aviso", quebra=True)
        lay.addWidget(self.lb_visto)
        self.bt_registrar = botao("Registrar tentativa (T)", primario=True, acao=self.registrar_tentativa)
        lay.addLayout(linha(self.bt_registrar))
        lay.addWidget(
            rotulo(
                "Faça o sinal e registre logo depois: vale a classe de maior probabilidade média nas "
                "últimas predições, sem o limiar. Para nenhum_sinal, fique sem sinalizar.",
                "secundario",
                quebra=True,
            )
        )
        self.lb_ultima = rotulo("", quebra=True)
        lay.addWidget(self.lb_ultima)
        self.lb_acc = rotulo("Acurácia da sessão: -", "titulo_painel")
        lay.addWidget(self.lb_acc)
        self.t_classes = Tabela(["Classe", "Acertos", "Erros"])
        self.t_classes.setMinimumHeight(120)
        lay.addWidget(self.t_classes)
        lay.addWidget(rotulo("Matriz da sessão (linhas: esperado; colunas: previsto)", "secundario", quebra=True))
        self.t_matriz = Tabela(["Esperado"])
        self.t_matriz.setMinimumHeight(120)
        lay.addWidget(self.t_matriz)
        lay.addLayout(
            linha(botao("Zerar sessão", acao=self.zerar_sessao), botao("Salvar sessão", acao=self.salvar_sessao))
        )
        return w

    # ------------------------------------------------------------------ ciclo da aba
    def ao_entrar(self) -> None:
        self._atualizar_quem()
        self.atualizar_modelos()
        self._iniciar()

    def ao_sair(self) -> None:
        """Pede ao processo para sair e NAO espera (item 7: a espera travava a troca de aba)."""
        if self.proc and self.proc.rodando():
            self.proc.encerrar_sem_esperar("sair")
        self.proc = None
        self.tela.clear()
        self.tela.setText("Reconhecimento parado")
        self._atualizar_botoes()

    def encerrar(self) -> None:
        if self.proc and self.proc.rodando():
            self._cmd("sair")
            if not self.proc.p.waitForFinished(5000):
                self.proc.parar()
        self.proc = None
        Subprocesso.aguardar_todos(5000)

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"runs", "manual"}:
            self.atualizar_modelos()
        if tipos & {"sinalizadores", "manual"}:
            self._atualizar_quem()

    # ------------------------------------------------------------------ modelos e camera
    def atualizar_modelos(self) -> None:
        atual = self.cb_modelo.currentData()
        self.modelos = listar_modelos(self.cfg)
        self.cb_modelo.blockSignals(True)
        self.cb_modelo.clear()
        for m in self.modelos:
            self.cb_modelo.addItem(m.rotulo(), str(m.pasta))
        i = self.cb_modelo.findData(atual) if atual else -1
        self.cb_modelo.setCurrentIndex(i if i >= 0 else 0)
        self.cb_modelo.blockSignals(False)
        if not self.sp_camera.hasFocus():
            self.sp_camera.setValue(int(self.cfg["coleta"]["camera"].get("indice", 0)))
        self._atualizar_botoes()

    def _atualizar_quem(self) -> None:
        from ...coleta.sinalizadores import CadastroSinalizadores

        atual = self.cb_quem.currentText()
        self.cb_quem.blockSignals(True)
        self.cb_quem.clear()
        self.cb_quem.addItems([OUTRA_PESSOA] + CadastroSinalizadores(self.cfg).ids())
        self.cb_quem.setCurrentText(atual or OUTRA_PESSOA)
        self.cb_quem.blockSignals(False)
        self._atualizar_aviso_visto()

    def _atualizar_botoes(self) -> None:
        tem = bool(self.modelos)
        self.bt_aplicar.setEnabled(tem)
        self.bt_video.setEnabled(tem)
        self.bt_voltar_camera.setVisible(self.video_teste is not None)
        self.bt_ir_treino.setVisible(not tem)
        self.bt_registrar.setEnabled(bool(self.sessao) and self.proc is not None)

    def escolher_video(self) -> None:
        arq, _ = QFileDialog.getOpenFileName(
            self, "Vídeo para testar", str(self.cfg.caminho("videos")), "Vídeos (*.mp4 *.avi *.mov *.mkv)"
        )
        if arq:
            self.testar_video(arq)

    def testar_video(self, caminho: str) -> None:
        """Mesmo painel e mesmo pipeline, com um arquivo no lugar da camera (teste reprodutivel)."""
        self.video_teste = str(caminho)
        self._reiniciar()

    def voltar_camera(self) -> None:
        self.video_teste = None
        self._reiniciar()

    def _ir_treino_final(self) -> None:
        self.app.ir_para("treino")
        t = self.app.paginas["treino"]
        if hasattr(t, "preparar_modelo_final"):
            t.preparar_modelo_final()

    # ------------------------------------------------------------------ processo
    def _reiniciar(self) -> None:
        self.ao_sair()
        self._iniciar()

    def _iniciar(self) -> None:
        if self.proc and self.proc.rodando():
            return
        if Subprocesso.ha_encerrando():  # a camera ainda esta com o processo anterior
            self.tela.setText("Aguardando o encerramento anterior da câmera...")
            if self.app.atual == "tempo_real":
                QTimer.singleShot(150, lambda: self.app.atual == "tempo_real" and self._iniciar())
            return
        self.dec = Decodificador()
        self.sinal.setText("")
        self.etapa.setText("")
        self.tela.limpar()
        self._erro_recebido = False
        self.ultimo = {}
        args = ["--config", str(self.cfg.arquivo)]
        if self.modelos:
            args += ["--modelo", self.cb_modelo.currentData() or str(self.modelos[0].pasta)]
            self.tela.setText("Carregando o modelo (TensorFlow) em outro processo...")
        if self.video_teste:
            args += ["--video", self.video_teste]
        else:
            args += ["--camera", str(self.sp_camera.value())]
        self.proc = Subprocesso(self, binario=True)
        self.proc.p.readyReadStandardOutput.connect(self._ler)
        self.proc.p.readyReadStandardError.connect(lambda: self.proc and self.proc.p.readAllStandardError())
        self.proc.terminou.connect(self._terminou)
        self.proc.iniciar_modulo(MODULO_SERVICO, args)
        self._atualizar_botoes()

    def previa_como_real(self, ligado: bool) -> None:
        prefs = None
        if ligado:
            from ...app_real.preferencias import Preferencias

            prefs = Preferencias.da_raiz(self.cfg.raiz)["legenda"]
        self.tela.camada.aplicar_preferencias(prefs)
        self.tela.update()

    def _rotulo(self, classe: str) -> str:
        if not self.ck_como_real.isChecked():
            return classe
        try:
            from ...vocabulario import vocabulario_em_cache

            return vocabulario_em_cache(self.cfg).obter(classe).rotulo
        except Exception:  # noqa: BLE001 - classe fora do vocabulario atual
            return classe

    def limpar_legenda(self) -> None:
        self._cmd("limpar")
        self.tela.limpar()

    def _cmd(self, texto: str) -> None:
        if self.proc:
            self.proc.escrever(texto + "\n")

    def _enviar_parametros(self) -> None:
        if self._params_do_modelo:
            return
        self._cmd(
            f"parametros limiar={self.sp_limiar.value():.2f} suavizacao={self.sp_suav.value()} "
            f"confirmacoes={self.sp_conf.value()}"
        )
        self.barras.limiar = self.sp_limiar.value()
        self.barras.update()

    def _terminou(self, codigo: int) -> None:
        # a mensagem de erro enviada pelo servico (camera, modelo) e mais util que o codigo de saida
        if (
            codigo not in (0, 15, -15)
            and not self._erro_recebido
            and (self.tela.pixmap() is None or self.tela.pixmap().isNull())
        ):
            self.tela.setText(f"O reconhecimento terminou (código {codigo}). Veja runs/logs/tempo_real.log.")

    def _ler(self) -> None:
        if not self.proc:
            return
        for meta, jpg in self.dec.alimentar(bytes(self.proc.p.readAllStandardOutput())):
            t = meta.get("tipo")
            if t == "erro":
                self._erro_recebido = True
                self.tela.clear()
                self.tela.setText(
                    meta["mensagem"]
                    + (
                        "\n\nTreine o modelo final na aba Treino (modo 'Modelo final')."
                        if meta.get("sem_modelo")
                        else ""
                    )
                )
                self.sinal.setText("Sem modelo" if meta.get("sem_modelo") else "Erro")
                if meta.get("sem_modelo"):
                    self.bt_ir_treino.show()
            elif t == "info":
                if "medicao_salva" in meta:
                    info(self, f"Medição salva em {meta['medicao_salva']}")
                elif "camera" in meta and "modelo" not in meta:
                    self.info_camera = meta["camera"]
                else:
                    self._modelo_carregado(meta)
            elif t == "quadro":
                self._quadro(meta, jpg)

    def _modelo_carregado(self, meta: dict) -> None:
        self.meta_modelo = meta
        origem = f"vídeo {Path(self.video_teste).name}" if self.video_teste else f"câmera {self.sp_camera.value()}"
        self.lb_ativo.setText(f"Modelo {meta['modelo']} · {len(meta['classes'])} classes · {origem}")
        self.et_ensaio.setVisible(bool(meta.get("ensaio")))
        self.barras.definir(meta["classes"], meta.get("nome_nenhum_sinal"))
        par = meta.get("parametros") or {}
        self._params_do_modelo = True
        self.sp_limiar.setValue(float(par.get("limiar", 0.6)))
        self.sp_suav.setValue(int(par.get("suavizacao", 5)))
        self.sp_conf.setValue(int(par.get("confirmacoes", 3)))
        self._params_do_modelo = False
        self.barras.limiar = self.sp_limiar.value()
        if not self.sessao or self.sessao.modelo != meta["modelo"] or self.sessao.classes != meta["classes"]:
            self.sessao = SessaoTeste(
                list(meta["classes"]), meta.get("nome_nenhum_sinal"), meta["modelo"], bool(meta.get("ensaio"))
            )
            self.cb_esperado.clear()
            self.cb_esperado.addItems(meta["classes"])
            self.lb_ultima.setText("")
            self._mostrar_sessao()
        self.lb_aviso_teste.setText(
            AVISO_TESTE + ("\nENSAIO: modelo treinado abaixo dos mínimos." if meta.get("ensaio") else "")
        )
        self._atualizar_aviso_visto()
        self._atualizar_botoes()

    def _quadro(self, meta: dict, jpg: bytes) -> None:
        self.ultimo = meta
        if jpg:
            self.tela.mostrar_jpeg(jpg)
        etapa = meta.get("etapa") or {}
        cand = meta.get("classe")
        if etapa.get("etapa") == "aguardando":
            self.sinal.setText(f"Em análise: {etapa.get('texto', '').split(': ')[-1]}")
        else:
            self.sinal.setText(f"Agora: {cand}" if cand else "")
        self.etapa.setText(etapa.get("texto", ""))
        self.indicador.definir(etapa)
        sem_mao = meta.get("motivo") in ("sem_maos", "aquecendo", "sem_pose")
        mudar_etiqueta(
            self.et_captando,
            "Sem mãos visíveis" if sem_mao else "Mãos detectadas",
            "pendente" if sem_mao else "captando",
        )
        self.barras.atualizar(meta.get("probs"), self.sp_limiar.value())
        self.texto.setText(" ".join(meta.get("texto", [])))
        self.tela.receber_quadro(meta, self._rotulo)
        tp = meta.get("tempos", {})

        def ms(k):
            return f"{tp[k]:.0f} ms" if tp.get(k) is not None else "-"

        self.lb_fps.setText(f"{tp.get('fps_processado', 0):.0f} quadros/s processados")
        self.tempos.setText(
            f"FPS câmera     {tp.get('fps_camera', 0):.1f}\n"
            f"FPS processado {tp.get('fps_processado', 0):.1f}\n"
            f"MediaPipe      {ms('mediapipe')}   Modelo {ms('modelo')}\n"
            f"Ponta a ponta  {ms('ponta_a_ponta')} (mediana)\n"
            f"Memória        {tp.get('memoria_mb') or 0:.0f} MB"
        )

    # ------------------------------------------------------------------ testar acertos
    def _atualizar_aviso_visto(self) -> None:
        quem = self.cb_quem.currentText()
        meta = {"sinalizadores": self.meta_modelo.get("lista_sinalizadores", [])}
        aviso = aviso_sinalizador_visto(meta, None if quem == OUTRA_PESSOA else quem)
        self.lb_visto.setText(aviso or "")
        self.lb_visto.setVisible(bool(aviso))

    def registrar_tentativa(self) -> None:
        if not self.sessao or not self.ultimo:
            self.lb_ultima.setText("Aguardando o modelo e a imagem.")
            return
        esp = self.cb_esperado.currentText()
        t = self.sessao.registrar(
            esp, self.ultimo.get("probs"), self.ultimo.get("motivo", ""), self.ultimo.get("confianca", 0.0)
        )
        cor = tema.CORES["sucesso"] if t["acerto"] else tema.CORES["erro"]
        self.lb_ultima.setText(
            f"<span style='color:{cor}'>Tentativa {t['n']}: "
            f"{'acerto' if t['acerto'] else 'erro'}</span> (esperado {t['esperado']}, previsto "
            f"{t['previsto']})"
        )
        self._mostrar_sessao()

    def _mostrar_sessao(self) -> None:
        s = self.sessao
        if not s:
            return
        acc = s.acuracia()
        self.lb_acc.setText(
            f"Acurácia da sessão: {'-' if acc is None else f'{100 * acc:.0f}%'} ({len(s.tentativas)} tentativa(s))"
        )
        self.t_classes.preencher([[c, v["acertos"], v["erros"]] for c, v in s.por_classe().items()])
        cols = s.colunas
        m = s.matriz()
        self.t_matriz.clear()
        self.t_matriz.setColumnCount(len(cols) + 1)
        self.t_matriz.setHorizontalHeaderLabels(["Esperado"] + cols)
        self.t_matriz.setRowCount(len(s.classes))
        for i, c in enumerate(s.classes):
            self.t_matriz.setItem(i, 0, QTableWidgetItem(c))
            for j in range(len(cols)):
                it = QTableWidgetItem(str(int(m[i, j])))
                it.setTextAlignment(Qt.AlignCenter)
                if m[i, j] and cols[j] == c:
                    it.setForeground(QColor(tema.CORES["sucesso"]))
                elif m[i, j]:
                    it.setForeground(QColor(tema.CORES["erro"]))
                self.t_matriz.setItem(i, j + 1, it)
        self.t_matriz.resizeColumnsToContents()

    def zerar_sessao(self) -> None:
        if self.sessao:
            self.sessao.zerar()
            self.lb_ultima.setText("")
            self._mostrar_sessao()

    def salvar_sessao(self):
        if not self.sessao or not self.sessao.tentativas:
            info(self, "Registre pelo menos uma tentativa antes de salvar.")
            return None
        quem = self.cb_quem.currentText()
        p = self.sessao.salvar(
            self.cfg,
            {
                "sinalizador_testado": None if quem == OUTRA_PESSOA else quem,
                "aviso_sinalizador_visto": self.lb_visto.text() or None,
                "origem": f"video: {Path(self.video_teste).name}"
                if self.video_teste
                else f"camera {self.sp_camera.value()}",
                "parametros": {
                    "limiar": self.sp_limiar.value(),
                    "suavizacao": self.sp_suav.value(),
                    "confirmacoes": self.sp_conf.value(),
                },
            },
        )
        info(self, f"Sessão salva em {p}. {AVISO_TESTE}")
        return p
