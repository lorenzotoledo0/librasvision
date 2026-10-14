"""Configuracoes: ajustes mais usados, gravados em configs/local.yaml (o config.yaml nao e reescrito)."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from ...config import salvar_local
from ...ui.dinamica import Colunas
from ..base import Pagina, botao, dica, erro, formulario, info, linha, painel, perguntar, rotulo

ESCALAS = [1.0, 1.1, 1.25, 1.5]  # tamanho do texto: 100% a 150% (item 12.6)

# (secao, chave, rotulo, tipo, minimo, maximo)
CAMPOS = [
    (
        "Câmera e gravação",
        [
            (("coleta", "camera", "indice"), "Índice da câmera", int, 0, 9),
            (("coleta", "camera", "largura"), "Largura (px)", int, 160, 3840),
            (("coleta", "camera", "altura"), "Altura (px)", int, 120, 2160),
            (("coleta", "camera", "fps"), "FPS pedido", int, 5, 120),
            (("coleta", "gravacao", "duracao_s"), "Duração do take (s)", float, 0.5, 30),
            (("coleta", "gravacao", "contagem_regressiva_s"), "Contagem regressiva (s)", int, 0, 10),
            (("coleta", "qualidade", "fps_minimo"), "FPS mínimo (aviso)", int, 1, 60),
            (("coleta", "qualidade", "taxa_minima_mao"), "Detecção mínima de mão (aviso)", float, 0, 1),
            (("coleta", "metas", "takes_por_classe_por_sinalizador"), "Meta padrão de novas palavras", int, 1, 500),
        ],
    ),
    (
        "Treino",
        [
            (("treino", "epocas"), "Épocas (máx.)", int, 1, 2000),
            (("treino", "paciencia"), "Paciência do early stopping", int, 1, 500),
            (("treino", "minimos", "sinalizadores_por_classe"), "Mín. sinalizadores por classe", int, 1, 50),
            (("treino", "minimos", "classes"), "Mín. de classes", int, 2, 100),
            (("avaliacao", "n_folds"), "Partições (grupos por sinalizador)", int, 2, 20),
        ],
    ),
    (
        "Tempo real",
        [
            (("tempo_real", "limiar_confianca"), "Limiar de confiança", float, 0, 1),
            (("tempo_real", "suavizacao"), "Suavização (predições)", int, 1, 30),
            (("tempo_real", "frames_estaveis"), "Predições iguais para aceitar", int, 1, 30),
        ],
    ),
]


def _ler(cfg, caminho):
    v = cfg
    for k in caminho:
        v = v[k]
    return v


def _aninhar(caminho, valor) -> dict:
    d = valor
    for k in reversed(caminho):
        d = {k: d}
    return d


class PaginaClasse(Pagina):
    titulo = "Configurações"
    subtitulo = (
        "Ajustes gravados em configs/local.yaml, que prevalece sobre o config.yaml. As justificativas dos "
        "valores padrão estão em docs/decisoes.md."
    )

    def __init__(self, app):
        super().__init__(app)
        # ---- aparencia e acessibilidade (item 12.6): vale na hora e fica em configs/local.yaml
        pa, la = painel()
        la.addWidget(rotulo("Aparência e acessibilidade", "titulo_painel"))
        fa = formulario()
        self.cb_escala = QComboBox()
        for v in ESCALAS:
            self.cb_escala.addItem(f"{int(round(v * 100))}%", v)
        self.cb_escala.currentIndexChanged.connect(self._aplicar_aparencia)
        fa.addRow("Tamanho do texto", self.cb_escala)
        self.ck_movimento = QCheckBox("Reduzir movimento")
        self.ck_movimento.toggled.connect(self._aplicar_aparencia)
        fa.addRow("", self.ck_movimento)
        la.addLayout(fa)
        la.addWidget(
            dica(
                "Reduzir movimento desliga a transição entre abas e as animações dos avisos. "
                "O aplicativo não emite som: todos os avisos são visuais."
            )
        )
        la.addLayout(
            linha(
                botao("Atalhos de teclado", acao=lambda: self.app.mostrar_atalhos()),
                botao("Galeria de componentes", acao=lambda: self.app.abrir_galeria()),
            )
        )
        self.raiz.addWidget(pa)
        h = Colunas(1050)
        h.setSpacing(14)
        self.widgets = {}
        for titulo, campos in CAMPOS:
            p, lay = painel()
            lay.addWidget(rotulo(titulo, "titulo_painel"))
            f = formulario()
            for caminho, rot, tipo, mn, mx in campos:
                w = QSpinBox() if tipo is int else QDoubleSpinBox()
                w.setRange(mn, mx)
                if tipo is float:
                    w.setDecimals(2)
                    w.setSingleStep(0.05 if mx <= 1 else 0.5)
                f.addRow(rot, w)
                self.widgets[caminho] = (w, tipo)
            lay.addLayout(f)
            if titulo == "Treino":
                self.seeds = QLineEdit()
                f.addRow("Seeds", self.seeds)
            lay.addStretch(1)
            h.addWidget(p, 1)
        self.raiz.addWidget(h, 1)
        self.raiz.addStretch(1)
        # barra fixa: salvar fica sempre a vista, mesmo com a pagina rolada
        self.adicionar_acao(
            botao("Salvar", primario=True, acao=self.salvar),
            botao("Restaurar padrão", acao=self.restaurar),
            botao(
                "Abrir pasta de logs",
                variante="fantasma",
                acao=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.cfg.caminho("logs")))),
            ),
        )

    def _aplicar_aparencia(self) -> None:
        if getattr(self, "_carregando", False):
            return
        escala = float(self.cb_escala.currentData() or 1.0)
        reduzir = self.ck_movimento.isChecked()
        salvar_local(self.cfg, {"interface": {"escala_fonte": escala, "reduzir_movimento": reduzir}})
        self.app.recarregar_config()
        self.app.aplicar_preferencias()

    def ao_entrar(self) -> None:
        pref = self.cfg.get("interface") or {}
        self._carregando = True
        i = self.cb_escala.findData(float(pref.get("escala_fonte", 1.0)))
        self.cb_escala.setCurrentIndex(i if i >= 0 else 0)
        self.ck_movimento.setChecked(bool(pref.get("reduzir_movimento", False)))
        self._carregando = False
        for caminho, (w, tipo) in self.widgets.items():
            try:
                w.setValue(tipo(_ler(self.cfg, caminho)))
            except (KeyError, TypeError):
                pass
        self.seeds.setText(" ".join(map(str, self.cfg["treino"]["seeds"])))

    def salvar(self) -> None:
        ajustes: dict = {}
        from ...config import mesclar

        for caminho, (w, tipo) in self.widgets.items():
            mesclar(ajustes, _aninhar(caminho, tipo(w.value())))
        try:
            seeds = [int(x) for x in self.seeds.text().replace(",", " ").split()]
        except ValueError:
            erro(self, "Seeds devem ser números inteiros.")
            return
        mesclar(ajustes, {"treino": {"seeds": seeds}})
        arq = salvar_local(self.cfg, ajustes)
        self.app.recarregar_config()
        info(self, f"Configurações salvas em {arq}.")

    def restaurar(self) -> None:
        arq = self.cfg.arquivo.with_name("local.yaml")
        if arq.exists() and perguntar(self, "Apagar configs/local.yaml e voltar aos valores do config.yaml?"):
            arq.unlink()
            self.app.recarregar_config()
            self.ao_entrar()
