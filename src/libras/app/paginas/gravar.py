"""Gravar: webcam, gravacao de takes, extracao completa de landmarks e revisao imediata.

Regras mantidas (decisoes D9): previa espelhada so na tela; arquivo salvo sem espelhar; sem TCLE o
sinalizador nao aparece; FPS medido pelos instantes reais. Ao parar a gravacao, o MediaPipe processa
TODOS os frames (barra de progresso); ao aceitar, .mp4 e .npz sao salvos juntos.

Na rajada, os takes ficam em memoria e a extracao de todos acontece so no fim (ultimo take ou Parar),
para os takes sairem um atras do outro. O numero de takes e limitado por coleta.gravacao.memoria_max_rajada_mb
(o maximo do campo acompanha a camera). Depois da extracao vem a revisao rapida: cada take passa na tela
com o esqueleto e a pessoa marca o que fica (Enter) ou sai (Esc); so os marcados sao salvos.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ...camera import Captura, abrir_camera_ou_video
from ...coleta.gravacao import MetaTake, registrar_take
from ...coleta.manifesto import Manifesto
from ...coleta.qualidade import avaliar_clipe, brilho
from ...coleta.sinalizadores import CadastroSinalizadores
from ...erros import ErroLibras
from ...ui.dinamica import Colunas
from ...vocabulario import Vocabulario
from ..base import (
    Pagina,
    Secao,
    TelaVideo,
    botao,
    encerrar_em_segundo_plano,
    erro,
    formulario,
    iniciar_tarefa,
    linha,
    rotulo,
)

log = logging.getLogger("libras.app.gravar")
AMARELO, VERMELHO, BRANCO = (40, 200, 230), (60, 60, 230), (235, 235, 235)


class PaginaClasse(Pagina):
    titulo = "Gravar"
    subtitulo = "Espaço grava, Enter aceita, Esc descarta, R regrava. A prévia é espelhada só na tela."

    def __init__(self, app):
        super().__init__(app)
        self.captura: Captura | None = None
        self.previa = None
        self.estado = "parado"
        self._fim_contagem = 0.0
        self._modo_manual = False
        self._clipe = None
        self._info = None
        self._meta = None
        self._rev_i = 0
        self._rajada = 0
        self._em_rajada = False
        self._fila: list[tuple[list, list]] = []  # takes da rajada ainda sem extracao: (frames, instantes)
        self._a_salvar: list = []  # takes da rajada ja extraidos, salvos um por vez
        self._res_rajada: dict = {}
        self._revisar: list[dict] = []  # revisao rapida da rajada: {frames, info, manter}
        self._rev_k = 0  # take da rajada em exibicao
        self._base_meta = (0, 0)  # takes ja salvos e meta do sinalizador/palavra no inicio da rajada
        self._ultimo_n = -1
        self._cont = 0
        self._extrator = None

        h = Colunas(760)
        h.setSpacing(16)
        esq = QVBoxLayout()
        esq.setSpacing(8)
        self.tela = TelaVideo("Câmera desligada")  # ocupa a altura que sobrar (a pagina cabe na tela)
        esq.addWidget(self.tela, 1)
        self.barra = QProgressBar()
        self.barra.setVisible(False)
        esq.addWidget(self.barra)
        self.status = rotulo("", "secundario", quebra=True)
        self.qualidade = rotulo("", "alerta_aviso", quebra=True)
        esq.addWidget(self.status)
        esq.addWidget(self.qualidade)
        # revisao rapida da rajada: marcado = sera salvo
        self.lista_rajada = QListWidget()
        self.lista_rajada.setVisible(False)
        self.lista_rajada.setMaximumHeight(170)
        self.lista_rajada.currentRowChanged.connect(self._ver_take_rajada)
        self.lista_rajada.itemChanged.connect(self._marcacao_mudou)
        esq.addWidget(self.lista_rajada)
        h.addLayout(esq, 3)

        op = self.cfg["coleta"]["opcoes"]
        dir_ = QVBoxLayout()
        dir_.setSpacing(12)
        # ---- take: o que muda a cada gravacao
        self.sec_take = Secao("Take", recolhivel=False)
        f = formulario()
        self.cb_sig = QComboBox()
        self.cb_sig.currentIndexChanged.connect(self._atualizar_classes)
        self.cb_cls = QComboBox()
        self.cb_cls.currentIndexChanged.connect(self._mostrar_meta)
        self.nova = QLineEdit()
        self.nova.setPlaceholderText("ex.: sala de aula")
        f.addRow("Sinalizador", self.cb_sig)
        f.addRow("Palavra", self.cb_cls)
        h_nova = linha(self.nova, botao("Criar", acao=self.criar_palavra, dica="Criar palavra"), esticar_fim=False)
        h_nova.setStretch(0, 1)
        f.addRow("Nova palavra", h_nova)
        self.sec_take.conteudo.addLayout(f)
        self.meta = rotulo("", "secundario", quebra=True)
        self.sec_take.conteudo.addWidget(self.meta)
        dir_.addWidget(self.sec_take)
        # ---- condicoes: mudam pouco numa sessao; fechadas mostram o resumo
        self.sec_cond = Secao("Condições", aberta=False)
        f2 = formulario()
        self.cb_ilum, self.cb_dist, self.cb_fundo, self.cb_mao = (QComboBox() for _ in range(4))
        for cb, k in (
            (self.cb_ilum, "iluminacao"),
            (self.cb_dist, "distancia"),
            (self.cb_fundo, "fundo"),
            (self.cb_mao, "mao_dominante"),
        ):
            cb.addItems(op[k])
            cb.currentIndexChanged.connect(self._resumir_secoes)
        self.obs = QLineEdit()
        self.obs.textChanged.connect(self._resumir_secoes)
        f2.addRow("Iluminação", self.cb_ilum)
        f2.addRow("Distância", self.cb_dist)
        f2.addRow("Fundo", self.cb_fundo)
        f2.addRow("Mão dominante", self.cb_mao)
        f2.addRow("Observação", self.obs)
        self.sec_cond.conteudo.addLayout(f2)
        dir_.addWidget(self.sec_cond)
        # ---- opcoes da gravacao
        self.sec_opc = Secao("Opções", aberta=False)
        g = QButtonGroup(self)
        dur = self.cfg["coleta"]["gravacao"]["duracao_s"]
        self.r_fixo = QRadioButton(f"Duração fixa ({dur:g} s)")
        self.r_manual = QRadioButton("Manual (Espaço inicia e para)")
        self.r_fixo.setChecked(True)
        g.addButton(self.r_fixo)
        g.addButton(self.r_manual)
        self.ck_lm = QCheckBox("Mostrar landmarks na prévia")
        self.ck_lm.setChecked(bool(self.cfg["coleta"]["qualidade"]["previa_landmarks"]))
        self.ck_guia = QCheckBox("Guia de enquadramento")
        self.ck_guia.setChecked(True)
        for w in (self.r_fixo, self.r_manual, self.ck_lm, self.ck_guia):
            self.sec_opc.conteudo.addWidget(w)
            w.toggled.connect(self._resumir_secoes)
        dir_.addWidget(self.sec_opc)
        dir_.addStretch(1)
        h.addLayout(dir_, 2)
        self.raiz.addWidget(h, 1)

        # ---- barra de acoes fixa: so os botoes que valem no estado atual (ver _ajustar)
        self.sp_rajada = QSpinBox()
        self.sp_rajada.setRange(2, 50)
        self.sp_rajada.setValue(5)
        self.sp_rajada.setSuffix(" takes")
        self.sp_rajada.setToolTip(
            "Takes na rajada. O máximo acompanha a memória disponível para a resolução da câmera."
        )
        self.sp_rajada.setAccessibleName("Takes na rajada")
        self.bt_cam = botao("Ligar câmera", acao=self.ligar, dica="Atalho: C")
        self.bt_grav = botao("Gravar take (Espaço)", primario=True, acao=self.iniciar_take)
        self.bt_raj = botao("Gravar rajada", acao=self.iniciar_rajada)
        self.bt_parar = botao(
            "Parar rajada", acao=self.parar_rajada, dica="O take em gravação termina e é aproveitado."
        )
        self.bt_parar_desc = botao(
            "Parar e descartar o take atual (Esc)",
            acao=lambda: self.parar_rajada(descartar_atual=True),
            dica="Para a rajada sem aproveitar o take que está sendo gravado.",
        )
        self.bt_aceitar = botao("Aceitar (Enter)", primario=True, acao=self.aceitar)
        self.bt_desc = botao("Descartar (Esc)", acao=self.descartar)
        self.bt_regr = botao("Regravar (R)", acao=self.regravar)
        self.bt_salvar_raj = botao("Salvar marcados", primario=True, acao=self._salvar_revisados)
        self.bt_marcar_raj = botao("Marcar todos", acao=lambda: self._marcar_rajada(True))
        self.bt_desmarcar_raj = botao("Desmarcar todos", acao=lambda: self._marcar_rajada(False))
        self.bt_importar = botao("Importar vídeos...", variante="fantasma", acao=self.importar)
        self.adicionar_acao(
            self.bt_cam,
            self.bt_grav,
            self.bt_raj,
            self.sp_rajada,
            self.bt_parar,
            self.bt_parar_desc,
            self.bt_aceitar,
            self.bt_desc,
            self.bt_regr,
            self.bt_salvar_raj,
            self.bt_marcar_raj,
            self.bt_desmarcar_raj,
            self.bt_importar,
        )
        self._resumir_secoes()

        for tecla, f_ in (
            (Qt.Key_Space, self._tecla_espaco),
            (Qt.Key_Return, self.aceitar),
            (Qt.Key_Enter, self.aceitar),
            (Qt.Key_Escape, self._tecla_esc),
            (Qt.Key_R, self.regravar),
            (Qt.Key_C, self._tecla_camera),
        ):
            s = QShortcut(QKeySequence(tecla), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda f_=f_: self._se_sem_foco_em_texto(f_))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._laco)
        self._ajustar()

    # ------------------------------------------------------------------
    def ao_entrar(self) -> None:
        self._atualizar_listas()
        self.timer.start(30)

    def atualizar_dados(self, tipos: set) -> None:
        """Contagens n/meta ao vivo. Nao mexe na camera nem em take em andamento."""
        if tipos & {"manifesto", "vocabulario", "sinalizadores", "manual"} and self.estado not in (
            "gravando",
            "analisando",
        ):
            self._atualizar_listas()

    def _atualizar_listas(self) -> None:
        cad = CadastroSinalizadores(self.cfg)
        atual = self.cb_sig.currentText()
        self.cb_sig.blockSignals(True)
        self.cb_sig.clear()
        self.cb_sig.addItems(cad.ids_com_tcle())
        if atual in cad.ids_com_tcle():
            self.cb_sig.setCurrentText(atual)
        self.cb_sig.blockSignals(False)
        self._atualizar_classes()
        if not cad.ids_com_tcle():
            self.status.setText("Cadastre um sinalizador com TCLE confirmado (aba Sinalizadores) para gravar.")

    def ao_sair(self) -> None:
        if self.estado in ("contagem", "gravando", "analisando", "intervalo"):
            self._rajada = 0
        self.desligar()
        self.timer.stop()
        if self._em_rajada and self.estado not in ("analisando", "revisao_rajada"):
            self._processar_rajada()  # takes ja gravados nao se perdem; o take em andamento, sim

    def pode_fechar(self) -> bool:
        pendentes = len(self._fila) + len(self._a_salvar) + len(self._revisar)
        if self.estado in ("gravando", "analisando", "revisao_rajada") or pendentes:
            from ..base import perguntar

            extra = f" {pendentes} take(s) da rajada ainda não foram salvos e serão perdidos." if pendentes else ""
            return perguntar(self, f"Há uma gravação em andamento.{extra} Sair mesmo assim?")
        return True

    def _se_sem_foco_em_texto(self, f) -> None:
        if isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox)):
            return
        f()

    def _tecla_espaco(self) -> None:
        self.iniciar_take()

    def _tecla_camera(self) -> None:
        if self.estado in ("parado", "previa"):
            self.ligar()

    def _tecla_esc(self) -> None:
        if self.estado in ("revisao", "revisao_rajada"):
            self.descartar()
        else:
            self.parar_rajada(descartar_atual=self.estado == "gravando")

    # ------------------------------------------------------------------
    def _atualizar_classes(self) -> None:
        voc = Vocabulario(self.cfg)
        sid = self.cb_sig.currentText()
        cont = Manifesto(self.cfg).contagens(("ok", "pendente"))
        atual = self.cb_cls.currentData()
        self.cb_cls.blockSignals(True)
        self.cb_cls.clear()
        for c in voc.ativas():
            n = cont.get((c.slug, sid), 0)
            meta = voc.meta(c.slug, sid) if sid else c.meta_por_sinalizador
            self.cb_cls.addItem(f"{c.rotulo}  ({n}/{meta})", c.slug)
        if atual is not None:
            i = self.cb_cls.findData(atual)
            if i >= 0:
                self.cb_cls.setCurrentIndex(i)
        self.cb_cls.blockSignals(False)
        s = CadastroSinalizadores(self.cfg).obter(sid) if sid else None
        if s and s.mao_dominante:
            self.cb_mao.setCurrentText(s.mao_dominante)
        self._mostrar_meta()

    def _mostrar_meta(self) -> None:
        sid, cls = self.cb_sig.currentText(), self.cb_cls.currentData()
        if not sid or not cls:
            self.meta.setText("")
            return
        man = Manifesto(self.cfg)
        n = man.contagens(("ok", "pendente")).get((cls, sid), 0)
        meta = Vocabulario(self.cfg).meta(cls, sid)
        extra = " Meta atingida; gravar além dela é permitido." if n >= meta else ""
        self.meta.setText(f"{sid} / {cls}: {n} de {meta} takes. Próximo: take {man.proximo_take(sid, cls):03d}.{extra}")

    def criar_palavra(self) -> None:
        try:
            c = Vocabulario(self.cfg).adicionar(self.nova.text())
        except ErroLibras as e:
            erro(self, e)
            return
        self.nova.clear()
        self._atualizar_classes()
        self.cb_cls.setCurrentIndex(self.cb_cls.findData(c.slug))
        self.status.setText(f"Palavra criada: {c.rotulo} ({c.slug}).")

    def _meta_take(self) -> MetaTake:
        return MetaTake(
            self.cb_cls.currentData() or "",
            self.cb_sig.currentText(),
            self.cb_ilum.currentText(),
            self.cb_dist.currentText(),
            self.cb_fundo.currentText(),
            self.cb_mao.currentText(),
            self.obs.text().strip(),
        )

    # ------------------------------------------------------------------
    def _resumir_secoes(self, *_a) -> None:
        cond = [self.cb_ilum.currentText(), self.cb_dist.currentText(), self.cb_fundo.currentText()]
        cond.append(f"mão {self.cb_mao.currentText()}")
        if self.obs.text().strip():
            cond.append("com observação")
        self.sec_cond.definir_resumo(" · ".join(cond))
        opc = [self.r_fixo.text().lower() if self.r_fixo.isChecked() else "manual"]
        opc += [n for ck, n in ((self.ck_lm, "landmarks"), (self.ck_guia, "guia")) if ck.isChecked()]
        self.sec_opc.definir_resumo(" · ".join(opc))

    def _ajustar(self) -> None:
        e = self.estado
        rev, rev_raj = e == "revisao", e == "revisao_rajada"
        em_rajada = self._em_rajada and e in ("contagem", "gravando", "intervalo")
        normal = not (rev or rev_raj or em_rajada)
        for w in (self.bt_cam, self.bt_grav, self.bt_raj, self.sp_rajada):
            w.setVisible(normal)
        for w in (self.bt_parar, self.bt_parar_desc):
            w.setVisible(em_rajada)
        for w in (self.bt_aceitar, self.bt_desc):
            w.setVisible(rev or rev_raj)
        self.bt_regr.setVisible(rev)
        self.bt_importar.setVisible(normal)
        self.bt_cam.setText("Desligar câmera" if e != "parado" else "Ligar câmera")
        self.bt_cam.setEnabled(e in ("parado", "previa"))
        self.bt_grav.setEnabled(e == "previa" or (e == "gravando" and self._modo_manual))
        self.bt_raj.setEnabled(e == "previa")
        self.bt_parar.setEnabled(self._rajada > 0 and e in ("contagem", "gravando", "intervalo"))
        self.bt_parar_desc.setEnabled(self._em_rajada and e == "gravando")
        for b in (self.bt_aceitar, self.bt_desc):
            b.setEnabled(e in ("revisao", "revisao_rajada"))
        self.bt_regr.setEnabled(e == "revisao")
        rev = e == "revisao_rajada"
        self.lista_rajada.setVisible(rev)
        for b in (self.bt_salvar_raj, self.bt_marcar_raj, self.bt_desmarcar_raj):
            b.setVisible(rev)
        self.bt_aceitar.setText("Manter e próximo (Enter)" if rev else "Aceitar (Enter)")
        self.bt_desc.setText("Descartar e próximo (Esc)" if rev else "Descartar (Esc)")

    def ligar(self) -> None:
        """Abre a camera em segundo plano (no Windows pode levar segundos). Antes, espera o encerramento
        da camera anterior, se ainda estiver em andamento (item 7)."""
        if self.estado == "abrindo":
            return
        if self.estado != "parado":
            self.desligar()
            return
        from ..base import aguardar_encerramentos

        cfg_camera = dict(self.cfg["coleta"]["camera"])
        self.estado = "abrindo"
        self.tela.setText("Abrindo a câmera...")
        self._ajustar()

        def trabalho(p, c):
            aguardar_encerramentos(10.0)
            return abrir_camera_ou_video(cfg_camera)

        def pronto(r):
            from ...coleta.previa import TrabalhadorPrevia

            cap, info = r
            if self.estado != "abrindo" or self.app.atual != "gravar":  # saiu da aba enquanto abria
                encerrar_em_segundo_plano("camera", cap.release)
                self.estado = "parado"
                self._ajustar()
                return
            self.captura = Captura(cap, info).iniciar()
            self.previa = TrabalhadorPrevia(self.cfg["mediapipe"])
            self.estado = "previa"
            self._ajustar_limite_rajada()
            self.status.setText(
                f"Câmera {info.indice} via {info.backend}: {info.largura}x{info.altura}, "
                f"{info.fps_declarado:.0f} FPS declarados."
            )
            self._ajustar()

        def falhou(e):
            self.estado = "parado"
            self.tela.setText("Câmera desligada")
            self._ajustar()
            erro(self, e, "Câmera")

        iniciar_tarefa(self, trabalho, pronto, falhou)

    def desligar(self, esperar: bool = False) -> None:
        """Na troca de aba, o MediaPipe da previa e a camera fecham em segundo plano (antes isso
        segurava a interface por cerca de 3 s). Ao fechar o aplicativo, esperar=True."""
        previa, captura = self.previa, self.captura
        self.previa = self.captura = None
        funcs = [f for f in (previa and previa.parar, captura and captura.parar) if f]
        if funcs:
            th = encerrar_em_segundo_plano("camera", *funcs)
            if esperar:
                th.join(timeout=10)
        if self.estado not in ("analisando", "revisao_rajada"):
            self.estado = "parado"
        self.tela.clear()
        self.tela.setText("Câmera desligada")
        self._ajustar()

    def encerrar(self) -> None:
        if self.estado in ("contagem", "gravando", "analisando", "intervalo"):
            self._rajada = 0
        self.desligar(esperar=True)
        self.timer.stop()
        if self._extrator is not None:
            self._extrator.fechar()

    def _validar(self) -> MetaTake | None:
        m = self._meta_take()
        try:
            CadastroSinalizadores(self.cfg).exigir_tcle(m.signer_id)
        except ErroLibras as e:
            erro(self, e, "Consentimento")
            return None
        if not m.classe:
            erro(self, "Escolha a palavra.")
            return None
        return m

    def iniciar_take(self) -> None:
        if self.estado == "gravando" and self._modo_manual:
            self._finalizar()
            return
        if self.estado != "previa":
            return
        m = self._validar()
        if m:
            self._meta, self._modo_manual = m, self.r_manual.isChecked()
            self._contagem()

    def _takes_que_cabem(self) -> int:
        """Takes da rajada que cabem no limite de memoria (frames brutos BGR ate a extracao no fim)."""
        g, info = self.cfg["coleta"]["gravacao"], self.captura.info
        fps = max(float(info.fps_declarado or 0.0), float(self.cfg["coleta"]["camera"]["fps"]))
        por_take = info.largura * info.altura * 3 * fps * float(g["duracao_s"])
        return max(1, int(float(g["memoria_max_rajada_mb"]) * 1024**2 // max(por_take, 1)))

    def _ajustar_limite_rajada(self) -> None:
        """O maximo do campo segue o que cabe na memoria com a resolucao real da camera."""
        cabe = self._takes_que_cabem()
        self.sp_rajada.setMaximum(max(2, min(50, cabe)))
        self.bt_raj.setToolTip(
            f"Até {cabe} takes por rajada com a câmera em {self.captura.info.largura}x{self.captura.info.altura}."
        )

    def iniciar_rajada(self) -> None:
        if self.estado != "previa":
            return
        m = self._validar()
        if not m:
            return
        n, cabe = self.sp_rajada.value(), self._takes_que_cabem()
        if n > cabe:
            erro(
                self,
                f"Com a câmera em {self.captura.info.largura}x{self.captura.info.altura}, cabem até {cabe} "
                f"takes por rajada na memória (limite de {self.cfg['coleta']['gravacao']['memoria_max_rajada_mb']} MB "
                "em coleta.gravacao.memoria_max_rajada_mb). Reduza o número de takes.",
                "Rajada",
            )
            return
        self._meta, self._modo_manual, self._rajada = m, False, n
        self._em_rajada, self._fila = True, []
        n_salvos = Manifesto(self.cfg).contagens(("ok", "pendente")).get((m.classe, m.signer_id), 0)
        self._base_meta = (n_salvos, Vocabulario(self.cfg).meta(m.classe, m.signer_id))
        if self.previa:
            self.previa.ativo = False  # CPU livre para a captura durante toda a rajada
        self._contagem()

    def parar_rajada(self, descartar_atual: bool = False) -> None:
        """Encerra a rajada e processa os takes ja gravados. Sem descartar_atual, o take em gravacao termina
        normalmente e entra na revisao; com descartar_atual, ele e jogado fora na hora."""
        if not self._em_rajada:
            return
        self._rajada = 0
        if self.estado == "gravando" and descartar_atual and self.captura:
            self.captura.parar_gravacao()  # frames descartados: nunca foram salvos
            self._processar_rajada()
            self.status.setText("Take em gravação descartado. " + self.status.text())
        elif self.estado in ("contagem", "intervalo"):
            self._processar_rajada()
        self._ajustar()

    def _contagem(self) -> None:
        self.estado = "contagem"
        self._fim_contagem = time.perf_counter() + float(self.cfg["coleta"]["gravacao"]["contagem_regressiva_s"])
        self.qualidade.setText("")
        self._ajustar()

    def _comecar(self) -> None:
        g = self.cfg["coleta"]["gravacao"]
        if self.previa:
            self.previa.ativo = False  # libera CPU para a captura manter o FPS
        self.captura.iniciar_gravacao(
            g["duracao_max_manual_s"] if self._modo_manual else g["duracao_s"],
            fps_ref=self.cfg["coleta"]["camera"]["fps"],
        )
        self.estado = "gravando"
        self._ajustar()

    def _finalizar(self) -> None:
        frames, ts = self.captura.parar_gravacao()
        if self._em_rajada:
            self._fila.append((frames, ts))
            self._rajada = max(0, self._rajada - 1)
            if self._rajada > 0 and self.captura:
                self.estado = "intervalo"
                self._fim_contagem = time.perf_counter() + float(self.cfg["coleta"]["gravacao"]["intervalo_rajada_s"])
                self.status.setText(f"Rajada: {len(self._fila)} take(s) gravado(s); a extração começa no fim.")
                self._ajustar()
            else:
                self._processar_rajada()
            return
        self._clipe = (frames, ts)
        self.estado = "analisando"
        self.barra.setVisible(True)
        self.barra.setRange(0, len(frames))
        self.barra.setValue(0)
        self.status.setText(f"Extraindo landmarks de {len(frames)} frames...")
        self._ajustar()
        fps_decl = self.captura.info.fps_declarado

        def trabalho(prog, canc):
            from ...landmarks import ExtratorHolistic

            if self._extrator is not None:  # novo grafo a cada take: o rastreamento nao vaza entre clipes
                self._extrator.fechar()
            self._extrator = ExtratorHolistic(self.cfg["mediapipe"], modo_estatico=False)
            return avaliar_clipe(
                frames,
                ts,
                fps_decl,
                self.cfg["coleta"],
                self._extrator,
                progresso=lambda i, n: prog(i, n) if i % 5 == 0 or i == n else None,
            )

        iniciar_tarefa(self, trabalho, self._apos_analise, self._falha_analise, lambda i, n, t: self.barra.setValue(i))

    def _falha_analise(self, e) -> None:
        self.barra.setVisible(False)
        erro(self, e, "Análise")
        self.estado = "previa" if self.captura else "parado"
        self._ajustar()

    def _apos_analise(self, info) -> None:
        self.barra.setVisible(False)
        self._info = info
        if self.previa:
            self.previa.ativo = True
        self.status.setText(
            f"{info.n_frames} frames, {info.duracao_s:.2f} s, FPS medido {info.fps_medido:.1f}, "
            f"mão detectada em {info.taxa_mao:.0%}, corpo em {info.taxa_pose:.0%}, "
            f"brilho {info.brilho_medio:.0f}."
        )
        self.qualidade.setText(
            ("Erro: " + "; ".join(info.erros))
            if info.erros
            else (("Avisos: " + ", ".join(info.avisos)) if info.avisos else "Sem avisos de qualidade.")
        )
        self.estado = "revisao"
        self._rev_i = 0
        self._ajustar()

    def _salvar(self, status: str) -> None:
        frames, _ = self._clipe
        try:
            r = registrar_take(
                self.cfg,
                Manifesto(self.cfg),
                CadastroSinalizadores(self.cfg),
                frames,
                self._info,
                self._meta,
                status=status,
            )
            self.status.setText(f"Salvo: {r['arquivo']} e landmarks ({r['status']}).")
        except ErroLibras as e:
            erro(self, e, "Não foi possível salvar")
        self._clipe = None
        self._atualizar_classes()

    # ------------------------------------------------------------------
    def _processar_rajada(self) -> None:
        """Fim da rajada: extrai os landmarks de todos os takes gravados (segundo plano) e depois salva."""
        fila, self._fila = self._fila, []
        self._rajada = 0
        if not fila:
            self._em_rajada = False
            if self.previa:
                self.previa.ativo = True
            self.estado = "previa" if self.captura else "parado"
            self._ajustar()
            return
        total = sum(len(f) for f, _ in fila)
        self.estado = "analisando"
        self.barra.setVisible(True)
        self.barra.setRange(0, max(total, 1))
        self.barra.setValue(0)
        self.status.setText(f"Extraindo landmarks dos {len(fila)} takes da rajada ({total} frames)...")
        self.qualidade.setText("")
        self._ajustar()
        fps_decl = self.captura.info.fps_declarado if self.captura else 0.0
        cfg_mp, cfg_coleta = self.cfg["mediapipe"], self.cfg["coleta"]

        def trabalho(prog, canc):
            from ...landmarks import ExtratorHolistic

            infos, feitos = [], 0
            for k, (frames, ts) in enumerate(fila):
                if canc():
                    break
                texto = f"take {k + 1} de {len(fila)}"
                # novo grafo a cada take: o rastreamento nao vaza entre clipes
                with ExtratorHolistic(cfg_mp, modo_estatico=False) as ext:
                    infos.append(
                        avaliar_clipe(
                            frames,
                            ts,
                            fps_decl,
                            cfg_coleta,
                            ext,
                            progresso=lambda i, n, b=feitos, t=texto: prog(b + i, total, t)
                            if i % 5 == 0 or i == n
                            else None,
                        )
                    )
                feitos += len(frames)
            return infos

        def pronto(infos):
            self._revisar = [
                {"frames": frames, "info": info, "manter": not info.erros} for (frames, _ts), info in zip(fila, infos)
            ]
            fila.clear()
            self._iniciar_revisao_rajada()

        def falhou(e):
            self._em_rajada = False
            if self.previa:
                self.previa.ativo = True
            self._falha_analise(e)

        def progresso(i, n, t):
            self.barra.setValue(i)
            self.status.setText(f"Extraindo landmarks da rajada: {t}...")

        iniciar_tarefa(self, trabalho, pronto, falhou, progresso, descricao="Extraindo landmarks da rajada")

    def _salvar_proximo_da_rajada(self) -> None:
        """Um take por volta do laco de eventos: a interface continua respondendo entre os salvamentos."""
        if not self._a_salvar:
            self._concluir_rajada()
            return
        frames, info = self._a_salvar.pop(0)  # memoria do take liberada assim que ele e salvo
        status = "pendente" if info.avisos else "ok"
        try:
            registrar_take(
                self.cfg, Manifesto(self.cfg), CadastroSinalizadores(self.cfg), frames, info, self._meta, status=status
            )
            self._res_rajada[status] += 1
        except ErroLibras as e:
            self._res_rajada["falhas"].append(str(e))
        self.barra.setValue(self.barra.value() + 1)
        QTimer.singleShot(0, self._salvar_proximo_da_rajada)

    def _concluir_rajada(self) -> None:
        r = self._res_rajada
        self._em_rajada = False
        self.barra.setVisible(False)
        if self.previa:
            self.previa.ativo = True
        salvos = r["ok"] + r["pendente"]
        self.status.setText(
            f"Rajada concluída: {salvos} de {r['total']} takes salvos com landmarks "
            f"({r['ok']} ok, {r['pendente']} pendentes para revisão, {r['descartados']} descartados na revisão)."
        )
        self.qualidade.setText(
            f"{r['pendente']} take(s) com aviso de qualidade ficaram pendentes; confira na aba Revisão."
            if r["pendente"]
            else ""
        )
        self.estado = "previa" if self.captura else "parado"
        self._atualizar_classes()
        self._ajustar()
        if r["falhas"]:
            erro(self, f"{len(r['falhas'])} take(s) da rajada não foram salvos:\n" + "\n".join(r["falhas"]), "Rajada")

    # ------------------------------------------------------------------
    # Revisao rapida da rajada
    def _iniciar_revisao_rajada(self) -> None:
        self.barra.setVisible(False)
        self.estado = "revisao_rajada"
        self._rev_k, self._rev_i = 0, 0
        self.lista_rajada.blockSignals(True)
        self.lista_rajada.clear()
        for k, t in enumerate(self._revisar):
            it = QListWidgetItem(self._texto_item_rajada(k))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if t["manter"] else Qt.Unchecked)
            if t["info"].erros:
                it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable)  # take com erro nao pode ser salvo
            self.lista_rajada.addItem(it)
        self.lista_rajada.setCurrentRow(0)
        self.lista_rajada.blockSignals(False)
        self.status.setText(
            f"Revisão da rajada: {len(self._revisar)} takes. Enter mantém e passa ao próximo, Esc descarta. "
            "Só os marcados serão salvos."
        )
        self.qualidade.setText("")
        self._ajustar()

    def _texto_item_rajada(self, k: int) -> str:
        info = self._revisar[k]["info"]
        q = ("erro: " + "; ".join(info.erros)) if info.erros else (", ".join(info.avisos) or "sem avisos")
        return f"Take {k + 1}: {info.n_frames} frames, mão em {info.taxa_mao:.0%} | {q}"

    def _ver_take_rajada(self, k: int) -> None:
        if 0 <= k < len(self._revisar):
            self._rev_k, self._rev_i = k, 0

    def _marcacao_mudou(self, it) -> None:
        k = self.lista_rajada.row(it)
        if 0 <= k < len(self._revisar):
            self._revisar[k]["manter"] = it.checkState() == Qt.Checked

    def _marcar_rajada(self, manter: bool) -> None:
        for k, t in enumerate(self._revisar):
            if not t["info"].erros:
                self.lista_rajada.item(k).setCheckState(Qt.Checked if manter else Qt.Unchecked)

    def _decidir_take_rajada(self, manter: bool) -> None:
        k = self._rev_k
        if not (0 <= k < len(self._revisar)):
            return
        if not self._revisar[k]["info"].erros:
            self.lista_rajada.item(k).setCheckState(Qt.Checked if manter else Qt.Unchecked)
        if k + 1 < len(self._revisar):
            self.lista_rajada.setCurrentRow(k + 1)
        else:
            self.status.setText("Último take revisado. Confira as marcações e clique em Salvar marcados.")
            self.bt_salvar_raj.setFocus()

    def _salvar_revisados(self) -> None:
        if self.estado != "revisao_rajada":
            return
        manter = [t for t in self._revisar if t["manter"] and not t["info"].erros]
        self._res_rajada = {
            "ok": 0,
            "pendente": 0,
            "falhas": [],
            "total": len(self._revisar),
            "descartados": len(self._revisar) - len(manter),
        }
        self._a_salvar = [(t["frames"], t["info"]) for t in manter]
        self._revisar = []
        self.lista_rajada.clear()
        self.estado = "analisando"
        self.barra.setVisible(True)
        self.barra.setRange(0, max(len(self._a_salvar), 1))
        self.barra.setValue(0)
        self.status.setText(f"Salvando os {len(self._a_salvar)} takes marcados...")
        self._ajustar()
        QTimer.singleShot(0, self._salvar_proximo_da_rajada)

    def aceitar(self) -> None:
        if self.estado == "revisao_rajada":
            self._decidir_take_rajada(True)
            return
        if self.estado == "revisao":
            self._salvar("ok")
            self.estado = "previa" if self.captura else "parado"
            self._ajustar()

    def descartar(self) -> None:
        if self.estado == "revisao_rajada":
            self._decidir_take_rajada(False)
            return
        if self.estado == "revisao":
            self._clipe = None
            self.status.setText("Take descartado (nada foi salvo).")
            self.estado = "previa" if self.captura else "parado"
            self._ajustar()

    def regravar(self) -> None:
        if self.estado == "revisao":
            self._clipe = None
            self.estado = "previa" if self.captura else "parado"
            self.iniciar_take()

    # ------------------------------------------------------------------
    def _laco(self) -> None:
        if self.captura is not None and self.captura.erro:
            erro(self, self.captura.erro, "Câmera")
            self.desligar()
            return
        agora = time.perf_counter()
        if self.estado in ("contagem", "intervalo") and agora >= self._fim_contagem and self.captura:
            if self.estado == "intervalo":
                self._contagem()
            else:
                self._comecar()
        elif self.estado == "gravando" and not self.captura.gravando:
            self._finalizar()
        if self.estado == "revisao_rajada" and self._revisar:
            t = self._revisar[min(self._rev_k, len(self._revisar) - 1)]
            frames, seq = t["frames"], t["info"].sequencia
            if frames:
                i = self._rev_i % len(frames)
                self._rev_i += 1
                img = frames[i]
                if seq is not None:
                    from ...landmarks import desenhar_esqueleto

                    img = desenhar_esqueleto(img, seq.pose[i], seq.mao_esq[i], seq.mao_dir[i], seq.presenca[i])
                marca = "MANTER" if t["manter"] else "DESCARTAR"
                self._exibir(
                    img,
                    f"RAJADA take {self._rev_k + 1}/{len(self._revisar)} [{marca}]: Enter mantem, Esc descarta",
                    AMARELO,
                )
            return
        if self.estado == "revisao" and self._clipe:
            frames = self._clipe[0]
            i = self._rev_i % len(frames)
            self._rev_i += 1
            img = frames[i]
            seq = self._info.sequencia if self._info else None
            if seq is not None:
                from ...landmarks import desenhar_esqueleto

                img = desenhar_esqueleto(img, seq.pose[i], seq.mao_esq[i], seq.mao_dir[i], seq.presenca[i])
            self._exibir(img, "REVISAO: Enter aceita, Esc descarta, R regrava", AMARELO)
            return
        if self.captura is None:
            return
        frame, n = self.captura.ultimo()
        if frame is None or n == self._ultimo_n:
            return
        self._ultimo_n = n
        legenda, cor, desenho = None, BRANCO, frame
        if self.estado == "previa":
            self._cont += 1
            if (
                self.ck_lm.isChecked()
                and self.previa
                and self._cont % int(self.cfg["coleta"]["qualidade"]["previa_landmarks_cada_n_frames"]) == 0
            ):
                self.previa.enviar(frame)
            legenda = f"FPS camera {self.captura.fps_atual:.0f} | brilho {brilho(frame[::8, ::8]):.0f}"
            if self.ck_lm.isChecked() and self.previa:
                legenda += " | mao: " + ("SIM" if self.previa.tem_mao else "nao")
                lm = self.previa.lm
                if lm is not None:
                    from ...landmarks import desenhar_esqueleto

                    desenho = desenhar_esqueleto(frame, lm.pose, lm.mao_esq, lm.mao_dir, lm.presenca)
        elif self.estado in ("contagem", "intervalo"):
            r = max(0.0, self._fim_contagem - agora)
            legenda = f"{'Proxima em' if self.estado == 'intervalo' else 'Gravando em'} {r:.1f} s" + (
                f" (faltam {self._rajada})" if self._rajada else ""
            )
            legenda += self._texto_meta_rajada()
            cor = AMARELO
        elif self.estado == "gravando":
            legenda, cor = f"GRAVANDO {self.captura.tempo_gravado():.1f} s" + self._texto_meta_rajada(), VERMELHO
        elif self.estado == "analisando":
            legenda, cor = "Extraindo landmarks...", AMARELO
        self._exibir(desenho, legenda, cor, gravando=self.estado == "gravando")

    def _texto_meta_rajada(self) -> str:
        """Progresso do sinalizador na palavra durante a rajada: salvos + gravados nesta rajada / meta."""
        if not self._em_rajada or self._meta is None:
            return ""
        salvos, meta = self._base_meta
        atual = len(self._fila) + (1 if self.estado == "gravando" else 0)
        return f" | {self._meta.signer_id}/{self._meta.classe}: {salvos + atual} de {meta}"

    def _exibir(self, frame, legenda=None, cor=BRANCO, gravando=False) -> None:
        img = cv2.flip(frame, 1)  # espelhado SO na tela
        h, w = img.shape[:2]
        if self.ck_guia.isChecked() and self.estado in ("previa", "contagem", "intervalo"):
            m = int(0.06 * w)
            cv2.rectangle(img, (m, m), (w - m, h - 2), (200, 200, 200), 1)
            y = int(0.42 * h)
            for x in range(m, w - m, 16):
                cv2.line(img, (x, y), (x + 8, y), (200, 200, 200), 1)
        if gravando:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), VERMELHO, 6)
        if legenda:
            cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(
                img,
                legenda.encode("ascii", "ignore").decode(),
                (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor,
                1,
                cv2.LINE_AA,
            )
        if self.estado in ("contagem", "intervalo"):
            n = int(np.ceil(max(0.0, self._fim_contagem - time.perf_counter())))
            cv2.putText(img, str(n), (w // 2 - 30, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, AMARELO, 6)
        self.tela.mostrar(img)

    # ------------------------------------------------------------------
    def importar(self) -> None:
        from .importar import DialogoImportar

        DialogoImportar(self).exec()
        self._atualizar_classes()
