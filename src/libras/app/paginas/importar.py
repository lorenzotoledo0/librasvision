"""Dialogo de importacao de videos ja gravados (copia, valida, deduplica, registra e extrai landmarks)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFileDialog, QLineEdit, QProgressBar, QVBoxLayout

from ...coleta.importacao import MetaPadrao, importar, parse_mapa
from ...coleta.sinalizadores import CadastroSinalizadores
from ...ui import tema
from ..base import Tabela, botao, erro, formulario, iniciar_tarefa, linha, rotulo


class DialogoImportar(QDialog):
    def __init__(self, pagina):
        super().__init__(pagina)
        self.cfg = pagina.cfg
        self.setWindowTitle("Importar vídeos existentes")
        self.resize(980, 640)
        self.entradas: list[Path] = []
        lay = QVBoxLayout(self)
        self.lb_n = rotulo("Nenhum arquivo selecionado.", "secundario")
        lay.addLayout(
            linha(
                botao("Adicionar arquivos...", acao=self._arquivos),
                botao("Adicionar pasta...", acao=self._pasta),
                botao("Limpar", acao=self._limpar),
                self.lb_n,
            )
        )
        f = formulario()
        self.ck_pasta = QCheckBox("Usar o nome da pasta como palavra (ex.: .../biblioteca/S01_1.mp4)")
        self.ck_pasta.setChecked(True)
        self.cb_padrao = QComboBox()
        self.cb_padrao.addItems(["(nenhum)"] + list(self.cfg["importacao"]["padroes_nome"].keys()))
        self.mapa = QLineEdit()
        self.mapa.setPlaceholderText("rótulos antigos: usuario1=S03 usuario2=S04")
        self.cb_cls = QComboBox()
        self.cb_cls.addItems(["(da pasta/nome)"] + self.cfg.classes)
        self.cb_sig = QComboBox()
        self.cb_sig.addItems([""] + CadastroSinalizadores(self.cfg).ids_com_tcle())
        op = self.cfg["coleta"]["opcoes"]
        self.cb_ilum, self.cb_dist, self.cb_fundo = QComboBox(), QComboBox(), QComboBox()
        for cb, k in ((self.cb_ilum, "iluminacao"), (self.cb_dist, "distancia"), (self.cb_fundo, "fundo")):
            cb.addItems([""] + op[k])
        self.ck_padronizar = QCheckBox("Padronizar formato (mantém o original intacto)")
        f.addRow("", self.ck_pasta)
        f.addRow("Padrão do nome", self.cb_padrao)
        f.addRow("Mapa de rótulos", self.mapa)
        f.addRow("Palavra (se não vier do nome/pasta)", self.cb_cls)
        f.addRow("Sinalizador (se não vier do nome)", self.cb_sig)
        f.addRow("Iluminação / distância / fundo", linha(self.cb_ilum, self.cb_dist, self.cb_fundo))
        f.addRow("", self.ck_padronizar)
        lay.addLayout(f)
        self.barra = QProgressBar()
        self.bt_sim = botao("Simular (só validar)", acao=lambda: self._executar(True))
        self.bt_imp = botao("Importar", primario=True, acao=lambda: self._executar(False))
        self.lb_res = rotulo("", "secundario")
        lay.addLayout(linha(self.bt_sim, self.bt_imp, self.barra, self.lb_res))
        self.tabela = Tabela(["Situação", "Arquivo", "Palavra", "Sinalizador", "video_id", "Motivo / avisos"])
        lay.addWidget(self.tabela, 1)
        self._itens = []

    def _arquivos(self) -> None:
        exts = " ".join(f"*{e}" for e in self.cfg["importacao"]["extensoes"])
        arqs, _ = QFileDialog.getOpenFileNames(self, "Vídeos", "", f"Vídeos ({exts})")
        self.entradas += [Path(a) for a in arqs]
        self.lb_n.setText(f"{len(self.entradas)} item(ns) selecionado(s).")

    def _pasta(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pasta com vídeos")
        if d:
            self.entradas.append(Path(d))
            self.lb_n.setText(f"{len(self.entradas)} item(ns) selecionado(s).")

    def _limpar(self) -> None:
        self.entradas = []
        self.tabela.setRowCount(0)
        self.lb_n.setText("Nenhum arquivo selecionado.")

    def _executar(self, simular: bool) -> None:
        if not self.entradas:
            erro(self, "Adicione arquivos ou uma pasta primeiro.")
            return
        try:
            mapa = parse_mapa(self.mapa.text().split())
        except Exception as e:  # noqa: BLE001
            erro(self, e)
            return
        cls = self.cb_cls.currentText()
        padrao = MetaPadrao(
            classe=None if cls.startswith("(") else cls,
            signer_id=self.cb_sig.currentText() or None,
            iluminacao=self.cb_ilum.currentText(),
            distancia=self.cb_dist.currentText(),
            fundo=self.cb_fundo.currentText(),
        )
        nome = self.cb_padrao.currentText()
        kw = dict(
            usar_pasta_classe=self.ck_pasta.isChecked(),
            padrao_nome=None if nome.startswith("(") else nome,
            mapa=mapa,
            padronizar=self.ck_padronizar.isChecked(),
            simular=simular,
        )
        self.tabela.setRowCount(0)
        self._itens = []
        self.bt_imp.setEnabled(False)
        self.bt_sim.setEnabled(False)
        entradas = list(self.entradas)
        cfg = self.cfg

        def trabalho(prog, canc):
            return importar(cfg, entradas, padrao, progresso=lambda i, n, it: prog(i, n, ""), **kw)

        def fim(r):
            itens, rel = r
            self.bt_imp.setEnabled(True)
            self.bt_sim.setEnabled(True)
            linhas, cores = [], []
            for it in itens:
                motivo = it.motivo or ", ".join(it.avisos)
                linhas.append([it.situacao, it.origem.name, it.classe or "", it.signer_id or "", it.video_id, motivo])
                cores.append(
                    {"rejeitado": tema.CORES["erro"], "ignorado": tema.CORES["texto2"]}.get(
                        it.situacao, tema.CORES["aviso"] if it.avisos else None
                    )
                )
            self.tabela.preencher(linhas, cores)
            c = Counter(i.situacao for i in itens)
            self.lb_res.setText(
                ", ".join(f"{k}: {v}" for k, v in sorted(c.items()))
                + (f" | relatório: {rel.name}" if rel else " (simulação)")
            )

        def falha(e):
            self.bt_imp.setEnabled(True)
            self.bt_sim.setEnabled(True)
            erro(self, e)

        iniciar_tarefa(self, trabalho, fim, falha, lambda i, n, t: (self.barra.setRange(0, n), self.barra.setValue(i)))
