"""Treino: selecao de classes, configuracao, execucao em SUBPROCESSO e curvas ao vivo.

O treino roda em outro processo (isola o TensorFlow, permite cancelar, nao trava a tela) e escreve
eventos em um arquivo .jsonl. Esta pagina le o arquivo e atualiza o grafico.

O grafico por epoca mostra SO treino e validacao. O teste nao entra na curva: ele nao pode orientar
nada durante o treino. Metricas e matriz de confusao de teste aparecem so ao fim de cada particao.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...selecao import POR_QUE_MINIMO_CLASSES
from ...ui import tema
from ...ui.dinamica import Colunas
from ...utils import carimbo
from ..base import (
    Pagina,
    Secao,
    Subprocesso,
    Tabela,
    botao,
    dica,
    erro,
    formulario,
    iniciar_tarefa,
    linha,
    perguntar,
    rotulo,
)
from ..graficos import GraficoCurvas, GraficoMatriz

MODOS = {
    "particoes": "Avaliação por partições (números do TCC)",
    "final": "Modelo final com todos os sinalizadores (para o tempo real)",
    "ablacao": "Ablação completa da augmentation",
}
FAMILIAS = [("espacial", "Espacial"), ("temporal", "Temporal"), ("ruido", "Ruído"), ("video", "Vídeo (família D)")]


def texto_restante(seg: float) -> str:
    """Estimativa legivel do tempo restante (arredondada: e uma estimativa, nao um relogio)."""
    if seg < 60:
        return "menos de 1 min"
    m = int(round(seg / 60))
    if m < 60:
        return f"cerca de {m} min"
    h, m = divmod(m, 60)
    return f"cerca de {h} h {m:02d} min" if m else f"cerca de {h} h"


class Curvas:
    """Estado das curvas por (seed, particao), alimentado pelos eventos do arquivo de progresso."""

    def __init__(self):
        self.series: dict[tuple, dict] = {}
        self.fim: dict[tuple, dict] = {}
        self.inicio: dict = {}
        self.execucoes: list[str] = []  # na ablacao, uma execucao por condicao
        self.plano: dict = {}  # ablacao: condicoes previstas e particoes por condicao
        self.t0: float | None = None  # instante do primeiro inicio (relogio do processo de treino)
        self.t_fins: list[float] = []
        self.terminou: dict | None = None
        self.classes: list[str] = []

    def aplicar(self, ev: dict) -> None:
        t = ev.get("tipo")
        if t == "plano_ablacao":
            self.plano = ev
        if t == "inicio":
            if self.t0 is None:
                self.t0 = ev.get("t")
            self.inicio = ev
            self.classes = ev.get("classes", [])
            if ev.get("execucao") and ev["execucao"] not in self.execucoes:
                self.execucoes.append(ev["execucao"])
        elif t == "epoca":
            k = (ev["seed"], ev["fold"], ev.get("execucao", ""))
            s = self.series.setdefault(k, {"epoca": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []})
            for c in s:
                s[c].append(ev.get(c, np.nan))
        elif t == "fim_particao":
            self.fim[(ev["seed"], ev["fold"], ev.get("execucao", ""))] = ev
            if ev.get("t"):
                self.t_fins.append(ev["t"])
        elif t == "fim":
            self.terminou = ev

    def progresso(self) -> tuple[int, int]:
        """(particoes concluidas, total). Na ablacao com plano: da ablacao inteira. Sem plano: da execucao
        ATUAL (contar as concluidas de todas as condicoes contra o total de uma travava a barra em 100%)."""
        if self.plano.get("condicoes") and self.plano.get("particoes_por_condicao"):
            total = int(self.plano["condicoes"]) * int(self.plano["particoes_por_condicao"])
            return min(len(self.fim), total), total
        if not self.inicio.get("seeds"):
            return 0, 0
        total = len(self.inicio["seeds"]) * len(self.inicio.get("folds", [1]))
        atual = self.inicio.get("execucao", "")
        feitas = sum(1 for k in self.fim if k[2] == atual)
        return min(feitas, total), total

    def restante_s(self, agora: float) -> float | None:
        """Estimativa do tempo restante: tempo medio por particao ja concluida x particoes que faltam.
        None ate a primeira particao terminar (antes disso nao ha base para estimar)."""
        feitas, total = self.progresso()
        if self.t0 is None or not self.t_fins or feitas == 0 or total == 0:
            return None
        por_particao = (self.t_fins[-1] - self.t0) / len(self.t_fins)
        decorrido_na_atual = max(0.0, agora - self.t_fins[-1])
        return max(0.0, (total - feitas) * por_particao - decorrido_na_atual)

    def melhor_epoca(self, k: tuple) -> int | None:
        if k in self.fim:
            return self.fim[k].get("melhor_epoca")
        s = self.series.get(k)
        if not s or not s["epoca"]:
            return None
        v = np.array(s["val_loss"], float)
        base = v if np.isfinite(v).any() else np.array(s["loss"], float)
        return int(s["epoca"][int(np.nanargmin(base))])


class PaginaClasse(Pagina):
    titulo = "Treino"
    somente_desktop = True
    subtitulo = (
        "Escolha as classes e a configuração. O treino roda em outro processo; as curvas de treino e "
        "validação aparecem ao vivo. O teste só é mostrado ao fim de cada partição."
    )

    def __init__(self, app):
        super().__init__(app)
        self.proc: Subprocesso | None = None
        self.arq_prog: Path | None = None
        self._offset = 0
        self.curvas = Curvas()
        self.tabela_cls = []
        self.marcadas: set[str] = set()
        h = Colunas(1000)
        h.setSpacing(16)
        esq = QVBoxLayout()
        esq.setSpacing(12)
        sec_cls = Secao("Classes", recolhivel=False)
        lay = sec_cls.conteudo
        self.t_cls = Tabela(["Classe", "Vídeos", "Landm.", "Sinaliz.", "Janelas", "Situação"])
        for j, txt in enumerate(
            [
                "Clique na linha para marcar ou desmarcar",
                "Vídeos com status 'ok'",
                "Vídeos 'ok' com landmarks válidos",
                "Sinalizadores com vídeos utilizáveis",
                "Janelas de 1 s elegíveis para o treino",
                "",
            ]
        ):
            self.t_cls.horizontalHeaderItem(j).setToolTip(txt)
        self.t_cls.setMinimumHeight(170)
        self.t_cls.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.t_cls.cellClicked.connect(self._clique_classe)
        lay.addWidget(self.t_cls)
        lay.addWidget(
            dica(
                "Toque na linha para marcar ou desmarcar. Vídeos: com status 'ok'. Landm.: com landmarks "
                "válidos. Sinaliz.: pessoas com vídeos. Janelas: trechos de 1 s para o treino."
            )
        )
        self.bt_todas = botao("Marcar todas as elegíveis", acao=self.marcar_elegiveis)
        self.bt_limpar = botao("Limpar seleção", acao=self.limpar_selecao)
        lay.addLayout(linha(self.bt_todas, self.bt_limpar))
        self.lb_sel = rotulo("", "secundario", quebra=True)
        self.lb_sel.setToolTip(POR_QUE_MINIMO_CLASSES)
        lay.addWidget(self.lb_sel)
        esq.addWidget(sec_cls)
        self.sec_cfg = Secao("Configuração", aberta=False)
        lay = self.sec_cfg.conteudo
        f = formulario()
        self.cb_modo = QComboBox()
        for k, v in MODOS.items():
            self.cb_modo.addItem(v, k)
        self.cb_prot = QComboBox()
        self.cb_prot.addItems(["grupos_sinalizador", "loso", "aleatorio_por_video"])
        self.seeds = QLineEdit()
        self.epocas = QSpinBox()
        self.epocas.setRange(1, 2000)
        f.addRow("Modo", self.cb_modo)
        f.addRow("Protocolo", self.cb_prot)
        f.addRow("Seeds", self.seeds)
        f.addRow("Épocas (máx.)", self.epocas)
        lay.addLayout(f)
        self.ck_ensaio = QCheckBox("Treinar mesmo assim (ENSAIO: não usar no TCC)")
        self.ck_ensaio.toggled.connect(lambda _: self._avaliar())
        lay.addWidget(self.ck_ensaio)
        esq.addWidget(self.sec_cfg)
        self.sec_aug = Secao("Data augmentation (só no treino)", aberta=False)
        lay = self.sec_aug.conteudo
        self.cks = {}
        la = linha()
        for k, t in FAMILIAS:
            self.cks[k] = QCheckBox(t)
            self.cks[k].toggled.connect(self._resumir)
            la.addWidget(self.cks[k])
        lay.addLayout(la)
        self.cb_int = QComboBox()
        self.cb_int.addItems(["fraca", "media", "forte"])
        lay.addLayout(linha(rotulo("Intensidade"), self.cb_int))
        esq.addWidget(self.sec_aug)
        esq.addStretch(1)
        for w in (self.cb_modo, self.cb_prot, self.cb_int):
            w.currentIndexChanged.connect(self._resumir)
        self.seeds.textChanged.connect(self._resumir)
        self.epocas.valueChanged.connect(self._resumir)
        self.ck_ensaio.toggled.connect(self._resumir)
        h.addLayout(esq, 5)

        # barra de acoes fixa: iniciar e acompanhar sem rolar a pagina
        self.bt_iniciar = botao("Iniciar treino", primario=True, acao=self.iniciar)
        self.bt_parar = botao("Parar", perigo=True, acao=self.parar)
        self.bt_parar.setEnabled(False)
        self.barra = QProgressBar()
        self.barra.setMinimumWidth(220)
        self.barra.setVisible(False)  # aparece quando um treino comeca
        self.lb_status = rotulo("", "secundario", quebra=True)
        self.lb_status.setMinimumWidth(260)
        self.adicionar_acao(
            self.bt_iniciar,
            self.bt_parar,
            botao("Ver resultados", variante="fantasma", acao=lambda: self.app.ir_para("resultados")),
            self.barra,
            self.lb_status,
        )

        dir_ = QTabWidget()
        w1 = QWidget()
        l1 = QVBoxLayout(w1)
        self.cb_curva = QComboBox()
        self.cb_curva.setMinimumContentsLength(18)
        self.cb_curva.currentIndexChanged.connect(self._desenhar_curva)
        self.ck_suavizar = QCheckBox("Suavizar (média móvel de 5 épocas)")
        self.ck_suavizar.toggled.connect(lambda _: self._desenhar_curva())
        l1.addLayout(linha(rotulo("Partição"), self.cb_curva, self.ck_suavizar))
        l1.addWidget(dica("Sem suavizar, a linha é o valor real de cada época. O teste não aparece nesta curva."))
        self.grafico = GraficoCurvas()
        l1.addWidget(self.grafico, 1)
        dir_.addTab(w1, "Curvas por época")
        w2 = QWidget()
        l2 = QVBoxLayout(w2)
        self.t_teste = Tabela(
            [
                "Seed",
                "Partição",
                "Testado em",
                "Acurácia (janela)",
                "F1 (janela)",
                "Acurácia (vídeo)",
                "F1 (vídeo)",
                "Épocas",
                "Melhor época",
            ]
        )
        self.t_teste.itemSelectionChanged.connect(self._desenhar_matriz)
        l2.addWidget(self.t_teste, 1)
        self.matriz = GraficoMatriz()
        l2.addWidget(self.matriz, 2)
        dir_.addTab(w2, "Teste por partição")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        dir_.addTab(self.log, "Log")
        h.addWidget(dir_, 6)
        self.raiz.addWidget(h, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._ler_progresso)
        self._resumir()

    # ------------------------------------------------------------------
    def _resumir(self, *_a) -> None:
        cfg = [self.cb_modo.currentText(), self.cb_prot.currentText()]
        cfg.append(f"seeds {self.seeds.text().strip() or '-'}")
        cfg.append(f"até {self.epocas.value()} épocas")
        if self.ck_ensaio.isChecked():
            cfg.append("ENSAIO")
        self.sec_cfg.definir_resumo(" · ".join(cfg))
        fam = [c.text().split(" (")[0].lower() for c in self.cks.values() if c.isChecked()]
        self.sec_aug.definir_resumo(
            (", ".join(fam) + f" · intensidade {self.cb_int.currentText()}") if fam else "nenhuma (baseline)"
        )

    def ao_entrar(self) -> None:
        if self.proc and self.proc.rodando():
            return
        cfg = self.cfg
        self.seeds.setText(" ".join(str(s) for s in cfg["treino"]["seeds"]))
        self.epocas.setValue(int(cfg["treino"]["epocas"]))
        self.cb_prot.setCurrentText(cfg["avaliacao"]["protocolo"])
        self.cb_int.setCurrentText(cfg["augmentation"]["intensidade"])
        ativo = cfg["augmentation"].get("ativo", False)
        for k, ck in self.cks.items():
            ck.setChecked(bool(ativo and cfg["augmentation"][k].get("ativo", False)))
        self.lb_sel.setText("Contando vídeos e janelas por classe...")
        self._contar()

    def preparar_modelo_final(self) -> None:
        """Chamado pela aba Tempo real quando nao ha modelo: deixa o modo 'Modelo final' escolhido."""
        self.cb_modo.setCurrentIndex(self.cb_modo.findData("final"))
        self.lb_status.setText(
            "Modo 'Modelo final' escolhido: confira as classes e clique em Iniciar treino. "
            "Ao terminar, o modelo aparece na aba Tempo real."
        )

    def _contar(self) -> None:
        cfg = self.cfg

        def trabalho(p, c):
            from ...selecao import tabela_classes

            return tabela_classes(cfg)

        iniciar_tarefa(self, trabalho, self._mostrar_tabela, descricao="Contando vídeos e janelas por classe...")

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"manifesto", "vocabulario", "manual"} and not (self.proc and self.proc.rodando()):
            self._contar()

    def encerrar(self) -> None:
        if self.proc and self.proc.rodando():
            self.proc.parar()

    def pode_fechar(self) -> bool:
        if self.proc and self.proc.rodando():
            if not perguntar(self, "Há um treino em andamento. Interromper e sair?"):
                return False
        return True

    def _mostrar_tabela(self, tabela) -> None:
        """Classes sem videos utilizaveis ficam desabilitadas e desmarcadas. A selecao salva e limpa
        em silencio do que nao existe, esta arquivado ou nao tem videos (item 6)."""
        from ...config import salvar_local
        from ...selecao import elegiveis, limpar_selecao, minimos

        self.tabela_cls = tabela
        mins = minimos(self.cfg)
        com_videos = [l.classe for l in tabela if l.videos_com_landmarks > 0]
        salva = (self.cfg.get("treino") or {}).get("classes") or []
        limpa = limpar_selecao(self.cfg, salva, com_videos)
        if salva and limpa != list(salva):
            salvar_local(self.cfg, {"treino": {"classes": limpa or None}})
            self.app.recarregar_config()
        self.marcadas = set(limpa) if limpa else set(elegiveis(tabela, mins))
        self._preencher_classes()

    def _preencher_classes(self) -> None:
        from PySide6.QtGui import QColor

        from ...selecao import minimos, situacao

        mins = minimos(self.cfg)
        cinza = QColor(tema.CORES["texto3"])
        self.t_cls.setRowCount(0)
        for i, l in enumerate(self.tabela_cls):
            self.t_cls.insertRow(i)
            sit, habilitada = situacao(l, mins)
            marcada = habilitada and l.classe in self.marcadas
            valores = (
                l.classe,
                l.videos_ok,
                l.videos_com_landmarks,
                l.sinalizadores,
                "-" if l.janelas is None else l.janelas,
                sit,
            )
            for j, v in enumerate(valores):
                it = QTableWidgetItem(str(v))
                it.setData(Qt.UserRole, l.classe)
                if j == 0:
                    # a caixa e controlada pelo clique na linha (inclusive no texto), nao pelo Qt
                    it.setCheckState(Qt.Checked if marcada else Qt.Unchecked)
                    it.setFlags((it.flags() | Qt.ItemIsEnabled) & ~Qt.ItemIsUserCheckable)
                if j == 5:
                    it.setToolTip(sit)
                if not habilitada:
                    it.setFlags(it.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
                    it.setForeground(cinza)
                    it.setToolTip("Sem vídeos utilizáveis: grave e aprove takes desta palavra.")
                elif "abaixo" in sit:
                    it.setForeground(QColor(tema.CORES["aviso"]))
                self.t_cls.setItem(i, j, it)
        from PySide6.QtWidgets import QHeaderView

        cab = self.t_cls.horizontalHeader()
        for j in range(1, 5):
            cab.setSectionResizeMode(j, QHeaderView.ResizeToContents)
        cab.setSectionResizeMode(0, QHeaderView.Interactive)
        fm = self.t_cls.fontMetrics()
        cab.resizeSection(0, max([fm.horizontalAdvance(l.classe) for l in self.tabela_cls] + [60]) + 52)
        cab.setSectionResizeMode(5, QHeaderView.Stretch)
        self._avaliar()

    def _linha_por_classe(self, classe: str):
        return next((l for l in self.tabela_cls if l.classe == classe), None)

    def _clique_classe(self, r: int, _c: int) -> None:
        it = self.t_cls.item(r, 0)
        if it is not None:
            self.alternar_classe(it.data(Qt.UserRole))

    def alternar_classe(self, classe: str) -> None:
        """Desmarcar e sempre permitido; marcar respeita o minimo (exceto em ENSAIO)."""
        from ...selecao import minimos, pode_marcar

        l = self._linha_por_classe(classe)
        if l is None or self.proc and self.proc.rodando():
            return
        if classe in self.marcadas:
            self.marcadas.discard(classe)
        else:
            ok, motivo = pode_marcar(l, minimos(self.cfg), self.ck_ensaio.isChecked())
            if not ok:
                self.avisar(motivo)
                return
            self.marcadas.add(classe)
        self._preencher_classes()
        self._salvar_selecao()

    def marcar_elegiveis(self) -> None:
        from ...selecao import elegiveis, minimos

        self.marcadas = set(elegiveis(self.tabela_cls, minimos(self.cfg), self.ck_ensaio.isChecked()))
        self._preencher_classes()
        self._salvar_selecao()

    def limpar_selecao(self) -> None:
        self.marcadas = set()
        self._preencher_classes()
        self._salvar_selecao()

    def _salvar_selecao(self) -> None:
        """A selecao vale tambem para a aba Dataset e a linha de comando (configs/local.yaml)."""
        from ...config import salvar_local

        salvar_local(self.cfg, {"treino": {"classes": self._selecionadas() or None}})
        self.app.recarregar_config()

    def avisar(self, texto: str) -> None:
        self.lb_sel.setText(texto)

    def _selecionadas(self) -> list[str]:
        return [l.classe for l in self.tabela_cls if l.classe in self.marcadas and l.videos_com_landmarks > 0]

    def _avaliar(self):
        from ...selecao import avaliar_selecao

        if not self.tabela_cls:
            return None
        av = avaliar_selecao(
            self.cfg, classes=self._selecionadas(), tabela=self.tabela_cls, ensaio=self.ck_ensaio.isChecked()
        )
        msgs = list(av.erros)
        if av.bloqueios and not av.ensaio:
            msgs.append(
                "Classes marcadas abaixo do mínimo de sinalizadores: "
                + ", ".join(av.bloqueios)
                + ". Desmarque-as, grave com outra pessoa ou marque ENSAIO."
            )
        msgs += av.avisos
        n = len(av.classes)
        cab = f"{n} classe(s) marcada(s): {', '.join(av.classes) or 'nenhuma'}."
        self.lb_sel.setText(cab + ("\n" + "\n".join(msgs) if msgs else "\nSeleção válida."))
        self.bt_iniciar.setEnabled(av.pode_treinar and not (self.proc and self.proc.rodando()))
        return av

    # ------------------------------------------------------------------
    def iniciar(self) -> None:
        av = self._avaliar()
        if av is None:
            erro(self, "Aguarde a contagem por classe.")
            return
        if av.erros:
            erro(self, "\n".join(av.erros))
            return
        if av.bloqueios and not self.ck_ensaio.isChecked():
            erro(self, "Classes abaixo do mínimo:\n" + "\n".join(f"- {c}: {m}" for c, m in av.bloqueios.items()))
            return
        if av.bloqueios and not perguntar(
            self, "Treinar como ENSAIO? Todas as saídas serão marcadas e os números não devem ir para o TCC."
        ):
            return
        try:
            seeds = [int(x) for x in self.seeds.text().replace(",", " ").split()]
        except ValueError:
            erro(self, "Seeds devem ser números inteiros separados por espaço.")
            return
        from ...config import salvar_local

        classes = self._selecionadas()
        salvar_local(self.cfg, {"treino": {"classes": classes}})  # a selecao vale tambem para a linha de comando
        self.app.recarregar_config()
        modo = self.cb_modo.currentData()
        fams = [k for k, ck in self.cks.items() if ck.isChecked()]
        self.arq_prog = self.cfg.caminho("runs") / "progresso" / f"{carimbo()}_{modo}.jsonl"
        self.arq_prog.parent.mkdir(parents=True, exist_ok=True)
        self._offset = 0
        self.curvas = Curvas()
        args = ["--config", str(self.cfg.arquivo), "--classes", *classes, "--progresso", str(self.arq_prog)]
        if self.ck_ensaio.isChecked() and av.bloqueios:
            args.append("--ensaio")
        if modo == "ablacao":
            script = "ablacao.py"
            if seeds != list(self.cfg["ablacao"]["seeds"]):
                args += ["--seeds", *map(str, seeds)]
            if self.epocas.value() != int(self.cfg["treino"]["epocas"]):
                args += ["--epocas", str(self.epocas.value())]
            args.append("--gerar-splits")
        else:
            script = "treinar.py"
            args += [
                "--silencioso",
                "--seeds",
                *map(str, seeds),
                "--epocas",
                str(self.epocas.value()),
                "--aug",
                *(fams or ["nenhuma"]),
                "--intensidade",
                self.cb_int.currentText(),
                "--nome",
                "final" if modo == "final" else ("baseline" if not fams else "aug_" + "_".join(fams)),
            ]
            if modo == "final":
                args.append("--final")
            else:
                args += ["--gerar-splits", "--protocolo", self.cb_prot.currentText()]
        self.log.clear()
        self.t_teste.setRowCount(0)
        self.cb_curva.clear()
        self.grafico.limpar("Carregando TensorFlow e montando as janelas...")
        self.proc = Subprocesso(self)
        self.proc.linha.connect(self.log.appendPlainText)
        self.proc.terminou.connect(self._fim)
        self.proc.iniciar_script(script, args)
        self.bt_iniciar.setEnabled(False)
        self.bt_parar.setEnabled(True)
        self.barra.setVisible(True)
        self.barra.setRange(0, 0)  # indeterminada ate o primeiro evento de inicio
        self.barra.setFormat("%p%")
        self.lb_status.setText(f"Iniciado: {MODOS[modo]}.")
        self.timer.start(500)

    def parar(self) -> None:
        if self.proc and self.proc.rodando() and perguntar(self, "Interromper o treino? A execução fica incompleta."):
            self.proc.parar()

    def _fim(self, codigo: int) -> None:
        self.timer.stop()
        self._ler_progresso()
        self.bt_iniciar.setEnabled(True)
        self._avaliar()
        self.bt_parar.setEnabled(False)
        self.barra.setRange(0, 1)
        self.barra.setValue(1 if codigo == 0 else 0)
        self.barra.setFormat("Concluído" if codigo == 0 else "Interrompido")
        if codigo == 0 and self.curvas.terminou:
            self.lb_status.setText(f"Concluído. Resultados em {self.curvas.terminou.get('pasta')}.")
        elif codigo == 0:
            self.lb_status.setText("Concluído.")
        else:
            self.lb_status.setText(f"Interrompido ou com erro (código {codigo}). Veja a aba Log.")

    # ------------------------------------------------------------------
    def _ler_progresso(self) -> None:
        if not self.arq_prog or not self.arq_prog.exists():
            return
        with open(self.arq_prog, "rb") as f:
            f.seek(self._offset)
            bruto = f.read()
        corte = bruto.rfind(b"\n") + 1  # so consome linhas completas
        self._offset += corte
        novo = bruto[:corte].decode("utf-8", "replace")
        mudou_lista = False
        for l in novo[:corte].splitlines():
            try:
                ev = json.loads(l)
            except json.JSONDecodeError:
                continue
            antes = len(self.curvas.series)
            self.curvas.aplicar(ev)
            if len(self.curvas.series) != antes:
                mudou_lista = True
            if ev.get("tipo") == "fim_particao":
                self._linha_teste(ev)
            if ev.get("tipo") == "inicio_particao":
                cond = (
                    f"Condição {len(self.curvas.execucoes)} da ablação. "
                    if self.cb_modo.currentData() == "ablacao" and self.curvas.execucoes
                    else ""
                )
                self.lb_status.setText(
                    f"{cond}Treinando: seed {ev['seed']}, partição {ev['fold']} (teste em "
                    f"{', '.join(ev.get('sinalizadores_teste', [])) or 'validação'})."
                )
        self._atualizar_barra()
        if mudou_lista:
            atual = self.cb_curva.currentData()
            self.cb_curva.blockSignals(True)
            self.cb_curva.clear()
            import re

            for k in self.curvas.series:
                nome = re.sub(r"^\d{8}_\d{6}_", "", k[2] or "")
                rot = f"{nome} | seed {k[0]} | partição {k[1]}" if nome else f"seed {k[0]} | partição {k[1]}"
                self.cb_curva.addItem(rot, k)
            self.cb_curva.setCurrentIndex(
                self.cb_curva.count() - 1
                if atual is None or atual not in self.curvas.series
                else list(self.curvas.series).index(atual)
            )
            self.cb_curva.blockSignals(False)
        self._desenhar_curva()

    def _atualizar_barra(self) -> None:
        if self.proc is None or not self.proc.rodando():
            return  # ao terminar, _fim decide a barra
        import time

        feitas, total = self.curvas.progresso()
        if total:
            self.barra.setRange(0, total)
            self.barra.setValue(feitas)
            resta = self.curvas.restante_s(time.time())
            self.barra.setFormat(
                f"%v de %m partições · {texto_restante(resta)}" if resta is not None else "%v de %m partições"
            )

    def _desenhar_curva(self) -> None:
        k = self.cb_curva.currentData()
        if k is None or k not in self.curvas.series:
            return
        s = self.curvas.series[k]
        self.grafico.desenhar(
            s,
            self.curvas.melhor_epoca(k),
            f"{self.cb_curva.currentText()} | {len(s['epoca'])} épocas",
            suavizar=self.ck_suavizar.isChecked(),
        )

    def _linha_teste(self, ev: dict) -> None:
        i = self.t_teste.rowCount()
        self.t_teste.insertRow(i)
        vals = [
            ev["seed"],
            ev["fold"],
            " ".join(ev.get("sinalizadores_teste", [])) or "-",
            f"{ev['teste_janela']['acuracia']:.3f}",
            f"{ev['teste_janela']['f1_macro']:.3f}",
            f"{ev['teste_video']['acuracia']:.3f}",
            f"{ev['teste_video']['f1_macro']:.3f}",
            ev.get("epocas"),
            ev.get("melhor_epoca"),
        ]
        k = (ev["seed"], ev["fold"], ev.get("execucao", ""))
        for j, v in enumerate(vals):
            it = QTableWidgetItem(str(v))
            it.setData(Qt.UserRole, k)
            self.t_teste.setItem(i, j, it)
        if self.curvas.inicio.get("protocolo") == "final":
            self.t_teste.item(i, 2).setText("validação (modelo final)")
        self.t_teste.resizeColumnsToContents()

    def _desenhar_matriz(self) -> None:
        r = self.t_teste.currentRow()
        if r < 0:
            return
        k = self.t_teste.item(r, 0).data(Qt.UserRole)
        ev = self.curvas.fim.get(k)
        if ev:
            self.matriz.desenhar(
                ev["matriz_video"],
                ev.get("classes") or self.curvas.classes,
                f"Teste (vídeos) | seed {k[0]} | partição {k[1]}",
            )
