"""Janela da previa de data augmentation (aba Revisao): o take original e 5 variacoes, animados.

So visualizacao (libras.augmentation.previa). Nada daqui muda o treino, o config ou os dados.
"Salvar figura" grava o mosaico do quadro atual em runs/augmentation_previa/ (so esqueletos, sem imagem
da pessoa), para o texto do TCC.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QGridLayout, QVBoxLayout

from ..augmentation import FAMILIAS_ONLINE, INTENSIDADES
from ..augmentation.previa import Previa, desenhar, gerar_previa, mosaico
from ..erros import ErroLibras
from .base import TelaVideo, botao, erro, info, linha, rotulo

NOMES_FAMILIAS = {
    "espacial": "Espacial (rotação, proporções, profundidade)",
    "temporal": "Temporal",
    "ruido": "Ruído e falhas",
}
N_VARIACOES = 5


class DialogoPreviaAugmentation(QDialog):
    def __init__(self, parent, cfg, linha_manifesto: dict):
        super().__init__(parent)
        self.cfg, self.linha = cfg, linha_manifesto
        self.previa: Previa | None = None
        self.seed = 0
        self.quadro = 0
        self.setWindowTitle(f"Prévia de augmentation: {linha_manifesto['video_id']}")
        lay = QVBoxLayout(self)
        lay.addWidget(
            rotulo(
                "Janela de 1 s com mais movimento das mãos, normalizada como no treino. Cada variação aplica "
                "sempre as famílias marcadas; no treino, cada família entra com a probabilidade do config e "
                "nada desta janela é salvo.",
                "secundario",
                quebra=True,
            )
        )
        self.ck = {}
        controles = []
        for f in FAMILIAS_ONLINE:
            c = QCheckBox(NOMES_FAMILIAS[f])
            c.setChecked(bool(cfg["augmentation"][f].get("ativo", True)))
            c.toggled.connect(self.sortear)
            self.ck[f] = c
            controles.append(c)
        self.cb_int = QComboBox()
        self.cb_int.addItems(list(INTENSIDADES))
        self.cb_int.setCurrentText(cfg["augmentation"].get("intensidade", "media"))
        self.cb_int.currentIndexChanged.connect(self.sortear)
        lay.addLayout(linha(*controles))
        lay.addLayout(
            linha(
                rotulo("Intensidade"),
                self.cb_int,
                botao("Sortear de novo", acao=self.novo_sorteio),
                botao("Salvar figura", acao=self.salvar_figura),
            )
        )
        grade = QGridLayout()
        self.telas = []
        for k in range(N_VARIACOES + 1):
            t = TelaVideo("", proporcao=320 / 300)
            t.setMinimumSize(200, 188)
            grade.addWidget(t, k // 3, k % 3)
            self.telas.append(t)
        lay.addLayout(grade, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._passo)
        self.resize(1040, 760)
        self.sortear()

    def familias(self) -> list[str]:
        return [f for f, c in self.ck.items() if c.isChecked()]

    def novo_sorteio(self) -> None:
        self.seed += 1
        self.sortear()

    def sortear(self) -> None:
        try:
            self.previa = gerar_previa(
                self.cfg, self.linha, N_VARIACOES, self.seed, self.familias(), self.cb_int.currentText()
            )
        except ErroLibras as e:
            self.previa = None
            self.timer.stop()
            erro(self, e, "Prévia de augmentation")
            return
        self.quadro = 0
        self.timer.start(66)  # cerca de 15 quadros por segundo, devagar o bastante para comparar
        self._passo()

    def _passo(self) -> None:
        if self.previa is None:
            return
        i = self.quadro % self.previa.n_frames
        self.quadro += 1
        cx = self.previa.caixa
        self.telas[0].mostrar(desenhar(self.previa.original, i, titulo="original", caixa=cx))
        for k, v in enumerate(self.previa.variacoes, 1):
            self.telas[k].mostrar(desenhar(v, i, titulo=f"variacao {k}", caixa=cx))

    def salvar_figura(self) -> None:
        import cv2

        from ..utils import carimbo

        if self.previa is None:
            return
        i = (self.quadro - 1) % self.previa.n_frames
        pasta = self.cfg.caminho("runs") / "augmentation_previa"
        pasta.mkdir(parents=True, exist_ok=True)
        arq = pasta / f"{self.previa.video_id}_{'-'.join(self.previa.familias) or 'nenhuma'}_{carimbo()}.png"
        cv2.imwrite(str(arq), mosaico(self.previa, i))
        info(self, f"Figura salva em {arq.relative_to(self.cfg.raiz).as_posix()} (quadro {i + 1}).")

    def done(self, r) -> None:  # noqa: N802 - fecha o timer junto com a janela
        self.timer.stop()
        super().done(r)
