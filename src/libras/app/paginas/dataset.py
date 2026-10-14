"""Dataset: extrair landmarks, augmentation de video, validar, integridade e splits."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPlainTextEdit, QProgressBar

from ...ui.dinamica import Colunas
from ..base import Pagina, Subprocesso, botao, erro, iniciar_tarefa, linha, painel, rotulo


class PaginaClasse(Pagina):
    titulo = "Dataset"
    somente_desktop = True
    subtitulo = (
        "Extração dos landmarks que faltarem, augmentation de vídeo (opcional), relatório de validação, "
        "verificação de integridade e divisão por sinalizador (splits)."
    )

    def __init__(self, app):
        super().__init__(app)
        self.proc: Subprocesso | None = None
        h = Colunas(900)
        h.setSpacing(14)
        p, lay = painel()
        lay.addWidget(rotulo("Landmarks", "titulo_painel"))
        lay.addWidget(
            rotulo(
                "Takes gravados e importados já saem com landmarks. Use isto para vídeos antigos, "
                "pendentes aprovados depois, ou após mudar a configuração do MediaPipe.",
                "secundario",
                quebra=True,
            )
        )
        self.ck_pend = QCheckBox("Incluir pendentes")
        self.bt_extrair = botao("Extrair landmarks", primario=True, acao=self.extrair)
        lay.addLayout(linha(self.bt_extrair, self.ck_pend))
        lay.addWidget(rotulo("Augmentation de vídeo (família D)", "titulo_painel"))
        lay.addWidget(
            rotulo(
                "Degrada os vídeos e reextrai os landmarks. Custa uma extração completa por variante.",
                "secundario",
                quebra=True,
            )
        )
        from ...augmentation.video import VARIANTES

        self.cks_var = {}
        lv = QHBoxLayout()
        for v in VARIANTES:
            ck = QCheckBox(v)
            ck.setChecked(v in self.cfg["augmentation"]["video"]["variantes"])
            self.cks_var[v] = ck
            lv.addWidget(ck)
        lv.addStretch(1)
        lay.addLayout(lv)
        self.bt_aug = botao("Gerar variantes", acao=self.aumentar)
        self.bt_cancelar = botao("Cancelar", acao=self.cancelar)
        self.bt_cancelar.setEnabled(False)
        lay.addLayout(linha(self.bt_aug, self.bt_cancelar))
        self.barra = QProgressBar()
        lay.addWidget(self.barra)
        lay.addWidget(rotulo("Verificações", "titulo_painel"))
        lay.addLayout(
            linha(
                botao("Relatório de validação", acao=self.validar),
                botao("Verificar integridade", acao=self.integridade),
            )
        )
        lay.addWidget(rotulo("Splits (divisão por sinalizador)", "titulo_painel"))
        self.cb_prot = QComboBox()
        self.cb_prot.addItems(["grupos_sinalizador", "loso", "aleatorio_por_video"])
        lay.addLayout(linha(self.cb_prot, botao("Gerar split", acao=self.splits)))
        lay.addWidget(
            rotulo(
                "O split usa as classes selecionadas na aba Treino. Ao iniciar um treino, ele é "
                "gerado automaticamente para a seleção atual.",
                "secundario",
                quebra=True,
            )
        )
        lay.addStretch(1)
        h.addWidget(p, 2)
        p2, l2 = painel()
        l2.addWidget(rotulo("Saída", "titulo_painel"))
        self.saida = QPlainTextEdit()
        self.saida.setReadOnly(True)
        l2.addWidget(self.saida, 1)
        h.addWidget(p2, 3)
        self.raiz.addWidget(h, 1)

    def ao_entrar(self) -> None:
        self.cb_prot.setCurrentText(self.cfg["avaliacao"]["protocolo"])

    def encerrar(self) -> None:
        if self.proc:
            self.proc.parar()

    def pode_fechar(self) -> bool:
        if self.proc and self.proc.rodando():
            from ..base import perguntar

            if not perguntar(self, "Há uma extração em andamento. Cancelar e sair?"):
                return False
        return True

    def _cfg_args(self) -> list[str]:
        return ["--config", str(self.cfg.arquivo)]

    def _rodar(self, script: str, args: list[str]) -> None:
        if self.proc and self.proc.rodando():
            erro(self, "Já há uma tarefa em andamento.")
            return
        self.saida.clear()
        self.barra.setValue(0)
        self.proc = Subprocesso(self)
        self.proc.linha.connect(self.saida.appendPlainText)
        self.proc.progresso.connect(lambda i, n: (self.barra.setRange(0, n), self.barra.setValue(i)))
        self.proc.terminou.connect(self._fim)
        for b in (self.bt_extrair, self.bt_aug):
            b.setEnabled(False)
        self.bt_cancelar.setEnabled(True)
        self.proc.iniciar_script(script, args + self._cfg_args())

    def _fim(self, codigo: int) -> None:
        for b in (self.bt_extrair, self.bt_aug):
            b.setEnabled(True)
        self.bt_cancelar.setEnabled(False)
        self.saida.appendPlainText(
            f"\n[concluido, codigo {codigo}]" if codigo in (0, 1) else f"\n[interrompido ou com erro, codigo {codigo}]"
        )

    def extrair(self) -> None:
        self._rodar("extrair.py", ["--pendentes"] if self.ck_pend.isChecked() else [])

    def aumentar(self) -> None:
        vs = [v for v, ck in self.cks_var.items() if ck.isChecked()]
        if not vs:
            erro(self, "Marque pelo menos uma variante.")
            return
        self._rodar("aumentar_videos.py", ["--variantes", *vs])

    def cancelar(self) -> None:
        if self.proc:
            self.proc.parar()

    def validar(self) -> None:
        cfg = self.cfg
        self.saida.setPlainText("Gerando relatório...")

        def trabalho(p, c):
            from ...validacao import gerar_relatorio, salvar_relatorio

            r = gerar_relatorio(cfg)
            if len(r.linhas) > 3:
                salvar_relatorio(cfg, r)
            return r.texto()

        iniciar_tarefa(self, trabalho, self.saida.setPlainText)

    def integridade(self) -> None:
        cfg = self.cfg

        def trabalho(p, c):
            from ...coleta.manifesto import Manifesto
            from ...coleta.sinalizadores import CadastroSinalizadores

            m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
            avisos = "\n".join(m.avisos + cad.avisos)
            return (avisos + "\n\n" if avisos else "") + m.verificar_integridade(cad.ids(), cad.ids_com_tcle()).texto()

        iniciar_tarefa(self, trabalho, self.saida.setPlainText)

    def splits(self) -> None:
        cfg, prot = self.cfg, self.cb_prot.currentText()

        def trabalho(p, c):
            from ...splits import gerar_splits, resumo_split, salvar_split, videos_elegiveis

            linhas, _ = videos_elegiveis(cfg)
            sp = gerar_splits(cfg, linhas, protocolo=prot)
            caminho = salvar_split(cfg, sp)
            return resumo_split(sp, {l["video_id"]: l for l in linhas}) + f"\n\nSalvo em {caminho}"

        iniciar_tarefa(self, trabalho, self.saida.setPlainText)
