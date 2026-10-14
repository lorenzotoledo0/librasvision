"""Resultados: execucoes de treino, ablacoes (com robustez) e modelos finais."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...ui import tema
from ...ui.dinamica import Colunas
from ..base import EstadoVazio, Pagina, Tabela, botao, dica, erro, info, linha, painel, perguntar, rotulo
from ..graficos import GraficoCurvas, ler_historico, melhor_epoca


def listar_execucoes(runs: Path) -> list[dict]:
    out = []
    if not runs.exists():
        return out
    for d in sorted(runs.iterdir(), reverse=True):
        if not d.is_dir() or d.name in ("logs", "progresso", "tempo_real"):
            continue
        if (d / "resultados.json").exists() or (d / "plano.json").exists():
            tipo, ok = "ablação", (d / "RELATORIO.md").exists()
        elif (d / "metadados.json").exists():
            tipo, ok = "modelo final", True
        elif (d / "ambiente.json").exists():
            tipo, ok = "avaliação", (d / "resumo.json").exists()
        else:
            continue
        resumo, ensaio, metrica = "", False, ""
        try:
            if (d / "resumo.json").exists():
                r = json.loads((d / "resumo.json").read_text(encoding="utf-8"))
                ensaio = bool(r.get("ensaio"))
                metrica = f"F1 vídeo {r.get('f1_macro_video_media', float('nan')):.3f}"
                resumo = ", ".join(r.get("classes") or [])
            elif (d / "metadados.json").exists():
                r = json.loads((d / "metadados.json").read_text(encoding="utf-8"))
                ensaio = bool(r.get("ensaio"))
                resumo = ", ".join(r.get("classes") or [])
        except Exception:  # noqa: BLE001 - arquivo parcial
            pass
        out.append(
            {
                "pasta": d,
                "nome": d.name,
                "tipo": tipo,
                "concluida": ok,
                "ensaio": ensaio,
                "metrica": metrica,
                "classes": resumo,
            }
        )
    return out


class PaginaClasse(Pagina):
    titulo = "Resultados"
    subtitulo = (
        "Execuções de treino e avaliação, ablações (com robustez e tabelas para o TCC) e modelos finais. "
        "Execuções marcadas como ENSAIO não devem ir para o TCC."
    )

    def __init__(self, app):
        super().__init__(app)
        h = Colunas(900)
        p, lay = painel()
        self.tabela = Tabela(["Execução", "Tipo", "Situação", "Métrica", "Classes"])
        self.tabela.itemSelectionChanged.connect(self._mostrar)
        lay.addWidget(self.tabela, 1)
        lay.addLayout(linha(botao("Abrir pasta", acao=self._abrir)))
        # pacote para o app real (13.2): so modelo final, nunca ENSAIO
        self.bt_publicar = botao(
            "Publicar modelo",
            primario=True,
            acao=self.publicar,
            dica="Grava o pacote em publicados/ para o app LibrasVision (sem dados de pessoas)",
        )
        self.bt_exportar = botao(
            "Exportar pacote (.zip)", acao=self.exportar, dica="Gera um .zip do pacote para levar a outro computador"
        )
        lay.addLayout(linha(self.bt_publicar, self.bt_exportar))
        self.lb_publicar = rotulo("", "secundario", quebra=True)
        lay.addWidget(self.lb_publicar)
        lay.addWidget(
            dica("ENSAIO: treino abaixo dos mínimos; não usar no TCC. Incompleta: interrompida ou em andamento.")
        )
        h.addWidget(p, 2)
        self.abas = QTabWidget()
        # curvas por epoca no estilo do sistema (as mesmas da aba Treino), lidas do historico de cada particao
        wc = QWidget()
        lc = QVBoxLayout(wc)
        self.cb_part = QComboBox()
        self.cb_part.setMinimumContentsLength(14)
        self.cb_part.currentIndexChanged.connect(self._desenhar)
        self.ck_suavizar = QCheckBox("Suavizar (média móvel de 5 épocas)")
        self.ck_suavizar.toggled.connect(lambda _: self._desenhar())
        lc.addLayout(linha(rotulo("Partição"), self.cb_part, self.ck_suavizar))
        self.grafico = GraficoCurvas()
        self.grafico.aviso("Selecione uma execução com histórico por época.")
        lc.addWidget(self.grafico, 1)
        self.abas.addTab(wc, "Curvas")
        self.texto = QPlainTextEdit()
        self.texto.setReadOnly(True)
        self.abas.addTab(self.texto, "Relatório")
        self.area_fig = QScrollArea()
        self.area_fig.setWidgetResizable(True)
        self.galeria = QWidget()
        self.lg = QVBoxLayout(self.galeria)
        self.area_fig.setWidget(self.galeria)
        self.abas.addTab(self.area_fig, "Figuras do TCC")
        self.aviso_ablacao = EstadoVazio(
            "Ablação e robustez: só no computador",
            "As tabelas e figuras da ablação e da robustez precisam de uma janela "
            "mais larga. Abra esta execução no computador.",
        )
        self.aviso_ablacao.hide()
        d = QVBoxLayout()
        d.addWidget(self.aviso_ablacao)
        d.addWidget(self.abas, 1)
        h.addLayout(d, 3)
        self.raiz.addWidget(h, 1)
        self.itens: list[dict] = []
        self.historicos: dict[str, Path] = {}
        self.faixa = "expandida"

    def aplicar_faixa(self, faixa: str) -> None:
        super().aplicar_faixa(faixa)
        self.faixa = faixa
        self._mostrar()

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"runs", "manual"}:
            self.ao_entrar()

    def ao_entrar(self) -> None:
        self.itens = listar_execucoes(self.cfg.caminho("runs"))
        linhas, cores = [], []
        for it in self.itens:
            sit = ("ENSAIO" if it["ensaio"] else "concluída") if it["concluida"] else "incompleta"
            linhas.append([it["nome"], it["tipo"], sit, it["metrica"], it["classes"]])
            cores.append(tema.CORES["aviso"] if it["ensaio"] else (None if it["concluida"] else tema.CORES["texto3"]))
        self.tabela.preencher(linhas, cores)
        self._atualizar_publicar()

    def _atual(self) -> dict | None:
        r = self.tabela.currentRow()
        return self.itens[r] if 0 <= r < len(self.itens) else None

    def _atualizar_publicar(self) -> None:
        it = self._atual()
        final = bool(it) and it["tipo"] == "modelo final"
        pode = final and not it["ensaio"]
        self.bt_publicar.setEnabled(pode)
        self.bt_exportar.setEnabled(pode)
        if not final:
            self.lb_publicar.setText("Publicar e exportar: selecione um modelo final.")
        elif it["ensaio"]:
            self.lb_publicar.setText("Modelo ENSAIO: não pode ser publicado para o app real.")
        else:
            self.lb_publicar.setText(
                "Pronto para publicar no app real. Vai só o modelo, os metadados e o cartão; nenhum dado de pessoa."
            )

    def _preparar_publicacao(self, acao: str):
        from ...publicacao import descrever, preparar

        it = self._atual()
        if not it:
            return None
        try:
            prep = preparar(self.cfg, it["pasta"])
        except Exception as e:  # noqa: BLE001 - mostrado ao usuario
            erro(self, e, "Não foi possível publicar")
            return None
        if not perguntar(self, descrever(prep) + f"\n\n{acao}?", "Publicar modelo para o app real"):
            return None
        return it

    def publicar(self):
        from ...publicacao import publicar

        it = self._preparar_publicacao("Publicar agora em publicados/")
        if not it:
            return None
        try:
            destino = publicar(self.cfg, it["pasta"])
        except Exception as e:  # noqa: BLE001
            erro(self, e, "Não foi possível publicar")
            return None
        info(self, f"Modelo publicado como {destino.name}. O app LibrasVision oferece a troca ao abrir.")
        return destino

    def exportar(self, destino: str | None = None):
        from ...publicacao import exportar_zip

        it = self._preparar_publicacao("Gerar o arquivo .zip")
        if not it:
            return None
        if destino is None:
            destino, _ = QFileDialog.getSaveFileName(
                self, "Salvar pacote", f"librasvision_{it['nome']}.zip", "Pacote (*.zip)"
            )
            if not destino:
                return None
        try:
            p = exportar_zip(self.cfg, it["pasta"], Path(destino))
        except Exception as e:  # noqa: BLE001
            erro(self, e, "Não foi possível exportar")
            return None
        info(self, f"Pacote salvo em {p.name}.")
        return p

    def _mostrar(self) -> None:
        self._atualizar_publicar()
        it = self._atual()
        while self.lg.count():
            w = self.lg.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.cb_part.blockSignals(True)
        self.cb_part.clear()
        self.historicos = {}
        if not it:
            self.cb_part.blockSignals(False)
            return
        ablacao = it["tipo"] == "ablação"
        so_desktop = ablacao and self.faixa == "compacta"
        self.aviso_ablacao.setVisible(so_desktop)
        self.abas.setVisible(not so_desktop)
        d = it["pasta"]
        for nome in ("RELATORIO.md", "LEIA-ME.txt", "metadados.json"):
            if (d / nome).exists():
                self.texto.setPlainText((d / nome).read_text(encoding="utf-8"))
                break
        else:
            self.texto.setPlainText("Execução incompleta (interrompida ou em andamento).")
        for hcsv in (
            sorted(d.glob("seed_*/fold_*/historico.csv"))
            + sorted(d.glob("treino/historico.csv"))
            + sorted(d.glob("*/seed_*/fold_*/historico.csv"))[:40]
        ):
            rel = hcsv.parent.relative_to(d).as_posix().replace("seed_", "seed ").replace("fold_", "partição ")
            self.historicos[rel.replace("/", " | ")] = hcsv
        for nome in self.historicos:
            self.cb_part.addItem(nome)
        self.cb_part.blockSignals(False)
        if self.historicos:
            self._desenhar()
        else:
            self.grafico.limpar("Esta execução não tem histórico por época.")
        figs = sorted(d.glob("fig_*.png")) + sorted(d.glob("matriz_confusao_*_total.png"))
        if not figs and (d / "treino" / "curvas.png").exists():
            figs = [d / "treino" / "curvas.png"]
        for f in figs[:8]:
            self.lg.addWidget(rotulo(f.name, "secundario"))
            lb = QLabel()
            pm = QPixmap(str(f))
            lb.setPixmap(pm.scaledToWidth(min(760, pm.width()), Qt.SmoothTransformation))
            self.lg.addWidget(lb)
        self.lg.addStretch(1)

    def _desenhar(self) -> None:
        caminho = self.historicos.get(self.cb_part.currentText())
        if not caminho:
            return
        s = ler_historico(caminho)
        if not s["epoca"]:
            self.grafico.limpar("Histórico vazio.")
            return
        it = self._atual()
        self.grafico.desenhar(
            s,
            melhor_epoca(s),
            f"{it['nome'] if it else ''} | {self.cb_part.currentText()}",
            suavizar=self.ck_suavizar.isChecked(),
        )

    def _abrir(self) -> None:
        it = self._atual()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(it["pasta"] if it else self.cfg.caminho("runs"))))
