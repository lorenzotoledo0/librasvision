"""Graficos da interface (Treino e Resultados), com estilo vindo dos tokens (tema.estilo_matplotlib).

Curva por epoca no estilo da imagem de referencia (docs/referencia_curva.png):
- validacao: um ponto por epoca, cor percorrendo a rampa ao longo das epocas, ligado por haste fina
  a linha de base; uma linha fina liga os pontos (o dado real, sem suavizar);
- treino: linha tracejada com quadrados vazados, em cor neutra (se distingue por traco e marcador,
  nao so pela cor);
- melhor epoca (menor perda de validacao): anel e rotulo;
- "suavizar" e opcional, visivel e rotulado: desenha a media movel e as hastes passam a ligar cada
  ponto real a curva suavizada; os pontos continuam sendo os valores reais.
O teste nunca entra nesta curva (D13).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..ui import tema
from .base import rotulo

JANELA_SUAVIZACAO = 5


def media_movel(v, k: int = JANELA_SUAVIZACAO) -> np.ndarray:
    """Media movel centrada que ignora NaN e mantem o tamanho (bordas com janela menor)."""
    v = np.asarray(v, float)
    out = np.full_like(v, np.nan)
    m = k // 2
    for i in range(len(v)):
        trecho = v[max(0, i - m) : i + m + 1]
        trecho = trecho[np.isfinite(trecho)]
        if len(trecho):
            out[i] = trecho.mean()
    return out


def ler_historico(caminho: Path) -> dict:
    """historico.csv de uma particao (treino.py) -> series por epoca."""
    s = {"epoca": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    with open(caminho, encoding="utf-8") as f:
        for l in csv.DictReader(f):
            for k in s:
                try:
                    s[k].append(float(l.get(k, "nan")) if k != "epoca" else int(l["epoca"]))
                except (TypeError, ValueError):
                    s[k].append(np.nan)
    return s


def melhor_epoca(s: dict) -> int | None:
    v = np.array(s.get("val_loss", []), float)
    base = v if np.isfinite(v).any() else np.array(s.get("loss", []), float)
    if not len(base) or not np.isfinite(base).any():
        return None
    return int(s["epoca"][int(np.nanargmin(base))])


class _GraficoPreguicoso(QWidget):
    """O matplotlib so e criado quando ha algo para desenhar (item 7: a primeira pintura da figura
    vazia custava cerca de 200 ms na troca de aba). Ate la, um texto simples."""

    def __init__(self, texto: str):
        super().__init__()
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._aviso = rotulo(texto, "secundario", quebra=True)
        self._aviso.setAlignment(Qt.AlignCenter)
        self._lay.addWidget(self._aviso, 1)
        self.fig = self.canvas = None
        self.setMinimumHeight(260)

    def _garantir(self, figsize) -> None:
        if self.fig is not None:
            return
        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        with matplotlib.rc_context(tema.estilo_matplotlib()):
            self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self._aviso.hide()
        self._lay.addWidget(self.canvas, 1)

    def aviso(self, texto: str) -> None:
        if self.fig is None:
            self._aviso.setText(texto)


class GraficoCurvas(_GraficoPreguicoso):
    def __init__(self):
        super().__init__("Aguardando o treino")
        self.ultimo: tuple | None = None

    def limpar(self, texto: str = "Aguardando o treino") -> None:
        self.ultimo = None
        if self.fig is None:
            self.aviso(texto)
            return
        import matplotlib

        with matplotlib.rc_context(tema.estilo_matplotlib()):
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, texto, transform=ax.transAxes, ha="center", color=tema.CORES["texto2"])
        self.canvas.draw_idle()

    def desenhar(self, s: dict, melhor: int | None, titulo: str, suavizar: bool = False) -> None:
        import matplotlib

        self._garantir((7, 4))
        self.ultimo = (s, melhor, titulo)
        c = tema.CORES
        with matplotlib.rc_context(tema.estilo_matplotlib()):
            self.fig.clear()
            axs = [self.fig.add_subplot(1, 2, i + 1) for i in range(2)]
            for ax, (m, rot) in zip(axs, (("loss", "Perda"), ("accuracy", "Acurácia"))):
                self._painel(ax, s, m, rot, melhor, suavizar)
            alcas, rotulos = axs[0].get_legend_handles_labels()
            leg = self.fig.legend(
                alcas,
                rotulos,
                loc="lower center",
                ncol=min(2, len(rotulos)),
                handlelength=2.4,
                bbox_to_anchor=(0.5, 0.0),
            )
            for t in leg.get_texts():
                t.set_color(c["texto"])
            self.fig.suptitle(titulo, color=c["texto2"], fontsize=tema.px("grafico") * 0.75, x=0.01, ha="left")
            self.fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        self.canvas.draw_idle()

    @staticmethod
    def _painel(ax, s: dict, m: str, rot: str, melhor: int | None, suavizar: bool) -> None:
        c = tema.CORES
        ep = np.asarray(s["epoca"], float)
        tr = np.asarray(s[m], float)
        va = np.asarray(s.get(f"val_{m}", []), float)
        tem_val = len(va) == len(ep) and np.isfinite(va).any()
        valores = np.concatenate([tr[np.isfinite(tr)], va[np.isfinite(va)] if tem_val else []])
        lo, hi = (float(valores.min()), float(valores.max())) if len(valores) else (0.0, 1.0)
        pad = (hi - lo) * 0.08 or 0.05
        base = lo - pad
        ax.set_ylim(base, hi + pad * 1.6)
        ax.set_title(rot, loc="left")
        ax.set_xlabel("Época")
        ax.plot(
            ep,
            tr,
            ls=(0, (5, 3)),
            lw=1.4,
            color=c["texto2"],
            marker="s",
            ms=4,
            mfc="none",
            mec=c["texto2"],
            label="treino (tracejado, quadrados)",
        )
        if tem_val:
            cores = tema.rampa(len(ep))
            n = len(ep)
            haste = dict(colors=cores, lw=0.9 if n <= 60 else 0.6, alpha=0.75, linestyles=(0, (2, 2)))
            if suavizar:
                sm = media_movel(va)
                ax.plot(
                    ep,
                    sm,
                    color=tema.COR["curva_dados"],
                    lw=1.8,
                    label=f"validação suavizada (média móvel de {JANELA_SUAVIZACAO})",
                )
                ax.vlines(ep, sm, va, **haste)
            else:
                ax.plot(ep, va, color=tema.COR["curva_dados"], lw=1.0, alpha=0.85)
                ax.vlines(ep, base, va, **haste)
            ax.scatter(
                ep,
                va,
                c=cores,
                s=24,
                zorder=3,
                edgecolors=c["painel"],
                linewidths=0.6,
                label=f"validação: valor real (cor = época 1 a {n})",
            )
            if melhor and melhor in s["epoca"]:
                i = s["epoca"].index(melhor)
                ax.scatter([ep[i]], [va[i]], s=150, facecolors="none", edgecolors=c["texto"], linewidths=1.6, zorder=4)
                ax.annotate(
                    f"melhor época {melhor}",
                    (ep[i], va[i]),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    color=c["texto"],
                    fontsize=tema.px("grafico") * 0.75,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=c["painel2"], ec=c["borda_controle"], lw=0.8),
                )


class GraficoMatriz(_GraficoPreguicoso):
    def __init__(self):
        super().__init__("Selecione uma partição concluída")

    def desenhar(self, cm, classes: list[str], titulo: str) -> None:
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap

        self._garantir((4, 4))
        c = tema.CORES
        cmap = LinearSegmentedColormap.from_list("rampa", [c["painel2"], tema.RAMPA[1], tema.RAMPA[0]])
        with matplotlib.rc_context(tema.estilo_matplotlib()):
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.grid(False)
            cm = np.asarray(cm)
            ax.imshow(cm, cmap=cmap)
            ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
            ax.set_yticks(range(len(classes)), classes)
            ax.set_xlabel("Previsto")
            ax.set_ylabel("Verdadeiro")
            mx = cm.max() if cm.size and cm.max() else 1
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    fundo = tema.hex_de([x * 255 for x in cmap(cm[i, j] / mx)])
                    cor = c["sobre_acento"] if tema.contraste(c["sobre_acento"], fundo) >= 4.5 else c["matriz_escuro"]
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color=cor,
                        fontweight="bold" if i == j else "normal",
                    )
            ax.set_title(titulo, loc="left")
            self.fig.tight_layout()
        self.canvas.draw_idle()
